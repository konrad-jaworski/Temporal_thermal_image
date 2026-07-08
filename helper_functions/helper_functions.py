import numpy as np
import random
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math
import pandas as pd

# Augmentation: -----------------------------------------------------------------------------------

class NoiseAdditionExperiment:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, bscan, depth):
        noise = torch.randn_like(bscan) * self.sigma
        bscan_noisy = (bscan + noise).clamp_min(0.0)
        return bscan_noisy, depth
    
class RandomHorizontalFlipBscan:
    """
    Randomly flips B-scan data along spatial width (W).
    Applies consistently to:
      - X: [T, W]
      - mask: [W]
      - depth: [W]
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X, mask):
        """
        Parameters:
        -----------
        X : torch.Tensor
            Shape [T, W]
        mask : torch.Tensor
            Shape [W]

        Returns:
        --------
        X, mask, depth : torch.Tensor
            Possibly flipped tensors
        """
        if random.random() < self.p:
            X = torch.flip(X, dims=[-1])      # flip W
            mask = torch.flip(mask, dims=[-1])
        return X, mask   

class HorizontalShift:
    """
    Shift sample horizontally using reflect padding.
    X: (H, W)
    mask: (W,)
    """

    def __init__(self, p=0.5, min_shift=1, max_shift=64):
        self.p = p
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, X, mask):
        if random.random() >= self.p:
            return X, mask

        H, W = X.size()
        idx = int(torch.randint(self.min_shift, self.max_shift, (1,)).item())
        direction = random.choice(["left", "right"])

        if direction == "left":
            # pad right, crop left
            X_pad = F.pad(X.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, idx:idx+W]

            mask_pad = F.pad(mask.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[idx:idx+W]

        else:  # right
            # pad left, crop right
            X_pad = F.pad(X.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, :W]

            mask_pad = F.pad(mask.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[:W]

        return X_shifted, mask_shifted
   

# Derivatives approximation --------------------------------------------------------------------------------------------------------

def d1_dy(X: torch.Tensor) -> torch.Tensor:
    """
    First derivative along height (H axis / rows).
    X: (H, W) -> (H, W)
    """
    d = torch.zeros_like(X)
    d[1:-1, :] = 0.5 * (X[2:, :] - X[:-2, :])  # central difference
    d[0, :]    = X[1, :] - X[0, :]             # forward difference
    d[-1, :]   = X[-1, :] - X[-2, :]           # backward difference
    return d

def d2_dy2(X: torch.Tensor) -> torch.Tensor:
    """
    Second derivative along height (H axis / rows).
    X: (H, W) -> (H, W)
    """
    d2 = torch.zeros_like(X)
    d2[1:-1, :] = X[2:, :] - 2 * X[1:-1, :] + X[:-2, :]
    d2[0, :]    = d2[1, :]
    d2[-1, :]   = d2[-2, :]
    return d2


def d1_dx(X: torch.Tensor) -> torch.Tensor:
    """
    First derivative along width (spatial axis).
    X: (H_time, W_space)
    """
    d = torch.zeros_like(X)

    # central difference
    d[:, 1:-1] = (X[:, 2:] - X[:, :-2]) * 0.5

    # boundaries
    d[:, 0]  = X[:, 1] - X[:, 0]
    d[:, -1] = X[:, -1] - X[:, -2]

    return d


def d2_dx2(X: torch.Tensor)->torch.Tensor:
    """
    Second derivative along width (W axis/ columns)
    """
    d2 = torch.zeros_like(X)
    d2[:, 1:-1] = X[:,2:] - 2 * X[ :,1:-1] + X[:, :-2]
    d2[:, 0] = d2[:, 1] 
    d2[:, -1]= d2[:, -2]

    return d2

# Interpolation and extrapolation methods: -----------------------------------------------------------------------------------

class Interpolate:
    """
    Interpolate data to a new size using PyTorch's interpolate function. It will interpolate both dimmensions of bscan
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        
        data_interpolated = torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0)
    
class Interpolate_mask:
    """
    Interpolate mask, it will use nearest neighbours to resize mask to fit into the bscan data.
    We should input already torch format data.
    """

    def __init__(self,size,mode='nearest'):
        self.size=size
        self.mode=mode

    def __call__(self,data):
        data_resize=F.interpolate(data[None,None,:],
                                  size=self.size,
                                  mode=self.mode
                                  ).squeeze(0).squeeze(0)
        
        return data_resize

    
# Validation metrics: -----------------------------------------------------------------------------------

def visualize_global_prediction(pred_all,mask_all,n):
    """
    It takes all of the predictions and masks plot individual simulation with Absolute error between them

    """
    plt.figure(figsize=(15,12))
    plt.subplot(1,3,1)
    plt.imshow(pred_all[(n-1)*512:n*512,:],vmin=0,vmax=1)
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.title('Predicted Field of view')

    plt.subplot(1,3,2)
    plt.imshow(mask_all[(n-1)*512:n*512,:],vmin=0,vmax=1)
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.title('Ground truth Field of view')

    plt.subplot(1,3,3)
    plt.imshow(torch.abs(pred_all[(n-1)*512:n*512,:]-mask_all[(n-1)*512:n*512,:]),vmin=0,vmax=1)
    plt.title('Absolute error')
    plt.colorbar(label='Normalized depth error',shrink=0.3)
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.tight_layout()

def visualize_bscan_prediction(pred_all,mask_all,n,level):
    """
    Function which visualize prediction over single B-scan at the same time presenting the depth seen by the network
    """
    n=n-1
    plt.plot(pred_all[level+512*n,:])
    plt.plot(mask_all[level+512*n,:])
    plt.grid(alpha=0.3)

    W = 512
    x = np.arange(W)

    for y_val in np.arange(0.1, 1.0, 0.1):
        y = np.full(W, y_val)
        plt.plot(x, y, color='red', linewidth=2,alpha=0.3)

    plt.xlabel('Frame width [px]')
    plt.ylabel('Normalized depth')
    plt.title('Single B-scan evaluation')


def compute_dataset_depth_metrics(
    pred_all,
    gt_all,
    depth_levels,
    tol=1e-6,
    thickness_mm=None,
    include_background=True,
):
    """
    Compute dataset-level metrics from concatenated tensors [N, W].

    Parameters
    ----------
    pred_all : torch.Tensor
        Predictions, shape [N, W]
    gt_all : torch.Tensor
        Ground truth, shape [N, W]
    depth_levels : list
        Possible depth levels, e.g. [0.1, 0.3, 0.5, 0.7, 0.9]
    tol : float
        Float tolerance for matching depth levels
    thickness_mm : float or None
        If provided, also report metrics in mm
    include_background : bool
        If True, also compute background MAE / MedAE / RMSE

    Returns
    -------
    results : dict
        Dataset-level global and per-level metrics
    """
    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.ndim != 2 or gt_all.ndim != 2:
        raise ValueError(f"Expected [N, W] tensors, got pred {pred_all.shape}, gt {gt_all.shape}")

    if pred_all.shape != gt_all.shape:
        raise ValueError(f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}")

    # ------------------------------
    # Compute absolute and squared errors over the whole dataset
    # ------------------------------
    abs_err = (pred_all - gt_all).abs()
    sq_err = (pred_all - gt_all) ** 2

    def _safe_mean(x):
        return x.mean().item() if x.numel() > 0 else float("nan")

    def _safe_median(x):
        return x.median().item() if x.numel() > 0 else float("nan")

    def _safe_rmse(x):
        return torch.sqrt(x.mean()).item() if x.numel() > 0 else float("nan")

    # -----------------------------
    # Global defect-only metrics over defect regions
    # -----------------------------
    roi_mask = gt_all > 0
    roi_abs = abs_err[roi_mask]
    roi_sq = sq_err[roi_mask]

    results = {
        "global": {
            "mae": _safe_mean(roi_abs),
            "medae": _safe_median(roi_abs),
            "rmse": _safe_rmse(roi_sq),
            "count": int(roi_mask.sum().item()),
        },
        "per_level": {},
        "dataset": {
            "num_bscans": int(pred_all.shape[0]),
            "width": int(pred_all.shape[1]),
            "num_columns_total": int(pred_all.numel()),
        }
    }

    # -----------------------------
    # Per-depth-level metrics
    # -----------------------------
    for level in depth_levels:
        level = float(level)
        level_mask = torch.abs(gt_all - level) < tol

        level_abs = abs_err[level_mask]
        level_sq = sq_err[level_mask]

        results["per_level"][level] = {
            "mae": _safe_mean(level_abs),
            "medae": _safe_median(level_abs),
            "rmse": _safe_rmse(level_sq),
            "count": int(level_mask.sum().item()),
        }

    # -----------------------------
    # Background metrics
    # -----------------------------
    if include_background:
        bg_mask = torch.abs(gt_all) < tol
        bg_abs = abs_err[bg_mask]
        bg_sq = sq_err[bg_mask]

        results["background"] = {
            "mae": _safe_mean(bg_abs),
            "medae": _safe_median(bg_abs),
            "rmse": _safe_rmse(bg_sq),
            "count": int(bg_mask.sum().item()),
        }

    # -----------------------------
    # Convert to mm if requested
    # -----------------------------
    if thickness_mm is not None:
        results["global_mm"] = {
            "mae": results["global"]["mae"] * thickness_mm if results["global"]["mae"] == results["global"]["mae"] else float("nan"),
            "medae": results["global"]["medae"] * thickness_mm if results["global"]["medae"] == results["global"]["medae"] else float("nan"),
            "rmse": results["global"]["rmse"] * thickness_mm if results["global"]["rmse"] == results["global"]["rmse"] else float("nan"),
        }

        results["per_level_mm"] = {}
        for level, vals in results["per_level"].items():
            results["per_level_mm"][level] = {
                "mae": vals["mae"] * thickness_mm if vals["mae"] == vals["mae"] else float("nan"),
                "medae": vals["medae"] * thickness_mm if vals["medae"] == vals["medae"] else float("nan"),
                "rmse": vals["rmse"] * thickness_mm if vals["rmse"] == vals["rmse"] else float("nan"),
                "count": vals["count"],
            }

        if include_background and "background" in results:
            results["background_mm"] = {
                "mae": results["background"]["mae"] * thickness_mm if results["background"]["mae"] == results["background"]["mae"] else float("nan"),
                "medae": results["background"]["medae"] * thickness_mm if results["background"]["medae"] == results["background"]["medae"] else float("nan"),
                "rmse": results["background"]["rmse"] * thickness_mm if results["background"]["rmse"] == results["background"]["rmse"] else float("nan"),
                "count": results["background"]["count"],
            }

    return results

def print_dataset_depth_metrics(results, thickness_mm=None):
    def fmt(x):
        return "nan" if not (x == x) else f"{x:.6f}"

    print("=== DATASET GLOBAL ROI METRICS ===")
    print(f"MAE:   {fmt(results['global']['mae'])}")
    print(f"MedAE: {fmt(results['global']['medae'])}")
    print(f"RMSE:  {fmt(results['global']['rmse'])}")
    print(f"Count: {results['global']['count']}")

    if thickness_mm is not None and "global_mm" in results:
        print("\n=== DATASET GLOBAL ROI METRICS [mm] ===")
        print(f"MAE:   {fmt(results['global_mm']['mae'])} mm")
        print(f"MedAE: {fmt(results['global_mm']['medae'])} mm")
        print(f"RMSE:  {fmt(results['global_mm']['rmse'])} mm")

    print("\n=== PER-DEPTH-LEVEL ROI METRICS ===")
    for level, vals in results["per_level"].items():
        print(
            f"Depth={level:.3f} | "
            f"MAE={fmt(vals['mae'])} | "
            f"MedAE={fmt(vals['medae'])} | "
            f"RMSE={fmt(vals['rmse'])} | "
            f"Count={vals['count']}"
        )

    if thickness_mm is not None and "per_level_mm" in results:
        print("\n=== PER-DEPTH-LEVEL ROI METRICS [mm] ===")
        for level, vals in results["per_level_mm"].items():
            print(
                f"Depth={level:.3f} | "
                f"MAE={fmt(vals['mae'])} mm | "
                f"MedAE={fmt(vals['medae'])} mm | "
                f"RMSE={fmt(vals['rmse'])} mm | "
                f"Count={vals['count']}"
            )

    if "background" in results:
        print("\n=== BACKGROUND METRICS ===")
        print(f"MAE:   {fmt(results['background']['mae'])}")
        print(f"MedAE: {fmt(results['background']['medae'])}")
        print(f"RMSE:  {fmt(results['background']['rmse'])}")
        print(f"Count: {results['background']['count']}")

        if thickness_mm is not None and "background_mm" in results:
            print("\n=== BACKGROUND METRICS [mm] ===")
            print(f"MAE:   {fmt(results['background_mm']['mae'])} mm")
            print(f"MedAE: {fmt(results['background_mm']['medae'])} mm")
            print(f"RMSE:  {fmt(results['background_mm']['rmse'])} mm")


def make_inner_1d_mask(mask, erosion_pixels=3, min_pixels=1):
    """
    Shrink each contiguous GT defect region along width direction.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean tensor, shape [N, W].

    erosion_pixels : int
        Number of pixels removed from left and right edge of each GT segment.

    min_pixels : int
        Minimum number of pixels required to keep a segment.

    Returns
    -------
    inner_mask : torch.Tensor
        Boolean tensor, shape [N, W].
    """

    if mask.ndim != 2:
        raise ValueError(f"Expected [N, W] mask, got {mask.shape}")

    mask = mask.bool()
    inner_mask = torch.zeros_like(mask)

    N, W = mask.shape

    for i in range(N):
        row = mask[i]

        if row.sum() == 0:
            continue

        idx = torch.where(row)[0]

        # Split into contiguous components
        gaps = torch.where(idx[1:] - idx[:-1] > 1)[0]

        starts = torch.cat([idx[:1], idx[gaps + 1]])
        ends = torch.cat([idx[gaps], idx[-1:]])

        for start, end in zip(starts, ends):
            new_start = start + erosion_pixels
            new_end = end - erosion_pixels

            if new_end >= new_start:
                if (new_end - new_start + 1) >= min_pixels:
                    inner_mask[i, new_start:new_end + 1] = True

    return inner_mask

def compute_dataset_depth_metrics_inner_gt(
    pred_all,
    gt_all,
    depth_levels,
    tol=1e-6,
    thickness_mm=3.5,
    include_background=False,
    erosion_pixels=3,
    min_pixels=1,
):
    """
    Compute dataset-level pixel-wise metrics using a smaller inner GT mask.

    This keeps the original matrix-subtraction logic:

        error = pred_all - gt_all

    but evaluates ROI metrics only on an eroded/shrunk GT defect region.
    This removes many edge pixels where shape mismatch dominates the error.

    Parameters
    ----------
    pred_all : torch.Tensor
        Predictions, shape [N, W].

    gt_all : torch.Tensor
        Ground truth, shape [N, W].

    depth_levels : list
        Possible depth levels, e.g. [0.25, 0.45, 0.65, 0.85].

    tol : float
        Float tolerance for matching depth levels.

    thickness_mm : float or None
        If provided, also report metrics in mm.

    include_background : bool
        If True, also compute background metrics.

    erosion_pixels : int
        Number of GT pixels removed from both left and right edge
        of each defect segment.

    min_pixels : int
        Minimum number of pixels required to keep an eroded segment.

    Returns
    -------
    results : dict
        Dataset-level global and per-level metrics.
    """

    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.ndim != 2 or gt_all.ndim != 2:
        raise ValueError(f"Expected [N, W] tensors, got pred {pred_all.shape}, gt {gt_all.shape}")

    if pred_all.shape != gt_all.shape:
        raise ValueError(f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}")

    # ------------------------------
    # Same old matrix subtraction
    # ------------------------------
    abs_err = (pred_all - gt_all).abs()
    sq_err = (pred_all - gt_all) ** 2

    def _safe_mean(x):
        return x.mean().item() if x.numel() > 0 else float("nan")

    def _safe_median(x):
        return x.median().item() if x.numel() > 0 else float("nan")

    def _safe_rmse(x):
        return torch.sqrt(x.mean()).item() if x.numel() > 0 else float("nan")

    # -----------------------------
    # Smaller / inner GT ROI mask
    # -----------------------------
    gt_defect_mask = gt_all > 0

    roi_mask = make_inner_1d_mask(
        gt_defect_mask,
        erosion_pixels=erosion_pixels,
        min_pixels=min_pixels,
    )

    roi_abs = abs_err[roi_mask]
    roi_sq = sq_err[roi_mask]

    results = {
        "global": {
            "mae": _safe_mean(roi_abs),
            "medae": _safe_median(roi_abs),
            "rmse": _safe_rmse(roi_sq),
            "count": int(roi_mask.sum().item()),
        },
        "per_level": {},
        "dataset": {
            "num_bscans": int(pred_all.shape[0]),
            "width": int(pred_all.shape[1]),
            "num_columns_total": int(pred_all.numel()),
            "original_roi_count": int(gt_defect_mask.sum().item()),
            "inner_roi_count": int(roi_mask.sum().item()),
            "erosion_pixels": int(erosion_pixels),
        }
    }

    # -----------------------------
    # Per-depth-level metrics
    # -----------------------------
    for level in depth_levels:
        level = float(level)

        # Important:
        # Level mask is also restricted to the smaller inner ROI.
        level_mask = (torch.abs(gt_all - level) < tol) & roi_mask

        level_abs = abs_err[level_mask]
        level_sq = sq_err[level_mask]

        results["per_level"][level] = {
            "mae": _safe_mean(level_abs),
            "medae": _safe_median(level_abs),
            "rmse": _safe_rmse(level_sq),
            "count": int(level_mask.sum().item()),
        }

    # -----------------------------
    # Background metrics unchanged
    # -----------------------------
    if include_background:
        bg_mask = torch.abs(gt_all) < tol
        bg_abs = abs_err[bg_mask]
        bg_sq = sq_err[bg_mask]

        results["background"] = {
            "mae": _safe_mean(bg_abs),
            "medae": _safe_median(bg_abs),
            "rmse": _safe_rmse(bg_sq),
            "count": int(bg_mask.sum().item()),
        }

    # -----------------------------
    # Convert to mm if requested
    # -----------------------------
    if thickness_mm is not None:
        results["global_mm"] = {
            "mae": results["global"]["mae"] * thickness_mm if results["global"]["mae"] == results["global"]["mae"] else float("nan"),
            "medae": results["global"]["medae"] * thickness_mm if results["global"]["medae"] == results["global"]["medae"] else float("nan"),
            "rmse": results["global"]["rmse"] * thickness_mm if results["global"]["rmse"] == results["global"]["rmse"] else float("nan"),
        }

        results["per_level_mm"] = {}

        for level, vals in results["per_level"].items():
            results["per_level_mm"][level] = {
                "mae": vals["mae"] * thickness_mm if vals["mae"] == vals["mae"] else float("nan"),
                "medae": vals["medae"] * thickness_mm if vals["medae"] == vals["medae"] else float("nan"),
                "rmse": vals["rmse"] * thickness_mm if vals["rmse"] == vals["rmse"] else float("nan"),
                "count": vals["count"],
            }

        if include_background and "background" in results:
            results["background_mm"] = {
                "mae": results["background"]["mae"] * thickness_mm if results["background"]["mae"] == results["background"]["mae"] else float("nan"),
                "medae": results["background"]["medae"] * thickness_mm if results["background"]["medae"] == results["background"]["medae"] else float("nan"),
                "rmse": results["background"]["rmse"] * thickness_mm if results["background"]["rmse"] == results["background"]["rmse"] else float("nan"),
                "count": results["background"]["count"],
            }

    return results

def compute_iou_over_concatenated_sims(
    pred_all,
    gt_all,
    sim_size=512,
    pred_threshold=0.03,
    gt_threshold=0.0,
    eps=1e-8,
    depth_levels=None,
    tol=1e-6,
    gt_values_are_removed_depth=True,
    depth_levels_are_removed_depth=None,
):
    """
    Compute:
        1. Global binary IoU
        2. Per-simulation binary IoU
        3. Per-depth-level recall

    Depth nomenclature
    ------------------
    If gt_values_are_removed_depth=True, positive GT values are interpreted as
    removed-depth / mask-depth values and converted to physical burial depth:

        physical_depth = 1.0 - mask_value

    Background stays background. This is important:

        mask value 0.0 -> background, not depth 1.0

    Parameters
    ----------
    pred_all : torch.Tensor
        Predicted depth tensor, shape [N, W]
    gt_all : torch.Tensor
        Ground-truth depth tensor, shape [N, W]
    sim_size : int
        Number of rows belonging to one simulation block
    pred_threshold : float
        Threshold for predicted defect mask
    gt_threshold : float
        Threshold for GT defect mask
    eps : float
        Small constant for numerical safety
    depth_levels : list or None
        Depth levels to evaluate recall for.

        If gt_values_are_removed_depth=True and depth_levels_are_removed_depth=True,
        then depth_levels are also converted using 1.0 - level.

    tol : float
        Tolerance for matching depth levels
    gt_values_are_removed_depth : bool
        If True, positive GT values are converted as 1.0 - gt
    depth_levels_are_removed_depth : bool or None
        If None, uses the same convention as gt_values_are_removed_depth.

    Returns
    -------
    results : dict
        Original IoU fields are preserved:

        {
            "global_iou",
            "global_intersection",
            "global_union",
            "num_sims",
            "per_sim_iou",
            "per_sim_intersection",
            "per_sim_union",
        }

        New per-depth recall fields:

        {
            "depth_levels",
            "per_depth_recall",
            "per_depth_gt_count",
            "per_depth_detected_count",
            "per_depth_missed_count",
            "per_level_recall",
        }
    """

    pred_all = torch.as_tensor(pred_all).float()
    gt_all_raw = torch.as_tensor(gt_all).float()

    if pred_all.ndim != 2 or gt_all_raw.ndim != 2:
        raise ValueError(
            f"Expected [N, W], got pred {pred_all.shape}, gt {gt_all_raw.shape}"
        )

    if pred_all.shape != gt_all_raw.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_all.shape}, gt {gt_all_raw.shape}"
        )

    n_rows = pred_all.shape[0]
    if n_rows % sim_size != 0:
        raise ValueError(
            f"Number of rows ({n_rows}) is not divisible by sim_size ({sim_size})"
        )

    # -------------------------------------------------
    # GT binary defect mask must be defined from raw GT
    # before any 1-depth conversion.
    # -------------------------------------------------
    gt_mask = gt_all_raw > gt_threshold

    # -------------------------------------------------
    # Convert GT values to physical burial-depth convention
    # only inside defect regions. Background remains 0.
    # -------------------------------------------------
    if gt_values_are_removed_depth:
        gt_depth = torch.where(
            gt_mask,
            1.0 - gt_all_raw,
            torch.zeros_like(gt_all_raw),
        )
    else:
        gt_depth = gt_all_raw

    pred_mask = pred_all > pred_threshold

    # -------------------------------------------------
    # Global binary IoU
    # -------------------------------------------------
    global_intersection = (pred_mask & gt_mask).sum().item()
    global_union = (pred_mask | gt_mask).sum().item()
    global_iou = global_intersection / (global_union + eps)

    # -------------------------------------------------
    # Per-simulation binary IoU
    # -------------------------------------------------
    num_sims = n_rows // sim_size

    per_sim_iou = []
    per_sim_intersection = []
    per_sim_union = []

    for i in range(num_sims):
        start = i * sim_size
        end = (i + 1) * sim_size

        pred_chunk = pred_mask[start:end]
        gt_chunk = gt_mask[start:end]

        inter = (pred_chunk & gt_chunk).sum().item()
        union = (pred_chunk | gt_chunk).sum().item()
        iou = inter / (union + eps)

        per_sim_intersection.append(int(inter))
        per_sim_union.append(int(union))
        per_sim_iou.append(float(iou))

    # -------------------------------------------------
    # Prepare physical depth levels
    # -------------------------------------------------
    if depth_levels_are_removed_depth is None:
        depth_levels_are_removed_depth = gt_values_are_removed_depth

    if depth_levels is None:
        positive_depths = gt_depth[gt_mask]

        if positive_depths.numel() > 0:
            # Rounded unique values are safer for floating-point labels.
            rounded = torch.round(positive_depths / tol) * tol
            depth_levels_physical = torch.unique(rounded).detach().cpu().tolist()
            depth_levels_physical = sorted([float(x) for x in depth_levels_physical])
        else:
            depth_levels_physical = []
    else:
        depth_levels_physical = []

        for level in depth_levels:
            level = float(level)

            if depth_levels_are_removed_depth:
                level = 1.0 - level

            depth_levels_physical.append(float(level))

    # -------------------------------------------------
    # Per-depth recall
    # -------------------------------------------------
    per_depth_recall = []
    per_depth_gt_count = []
    per_depth_detected_count = []
    per_depth_missed_count = []
    per_level_recall = {}

    for level in depth_levels_physical:
        level = float(level)

        level_gt_mask = (torch.abs(gt_depth - level) < tol) & gt_mask

        gt_count = level_gt_mask.sum().item()
        detected_count = (pred_mask & level_gt_mask).sum().item()
        missed_count = gt_count - detected_count

        recall = detected_count / (gt_count + eps)

        per_depth_recall.append(float(recall))
        per_depth_gt_count.append(int(gt_count))
        per_depth_detected_count.append(int(detected_count))
        per_depth_missed_count.append(int(missed_count))

        per_level_recall[level] = {
            "recall": float(recall),
            "gt_count": int(gt_count),
            "detected_count": int(detected_count),
            "missed_count": int(missed_count),
        }

    results = {
        # Original global IoU outputs
        "global_iou": float(global_iou),
        "global_intersection": int(global_intersection),
        "global_union": int(global_union),

        # Original per-simulation IoU outputs
        "num_sims": int(num_sims),
        "per_sim_iou": per_sim_iou,
        "per_sim_intersection": per_sim_intersection,
        "per_sim_union": per_sim_union,

        # New per-depth recall outputs
        "depth_levels": depth_levels_physical,
        "per_depth_recall": per_depth_recall,
        "per_depth_gt_count": per_depth_gt_count,
        "per_depth_detected_count": per_depth_detected_count,
        "per_depth_missed_count": per_depth_missed_count,
        "per_level_recall": per_level_recall,
    }

    return results

def print_iou_results(results):
    print("=== GLOBAL IoU ===")
    print(f"IoU:          {results['global_iou']:.6f}")
    print(f"Intersection: {results['global_intersection']}")
    print(f"Union:        {results['global_union']}")
    print(f"Num sims:     {results['num_sims']}")

    print("\n=== PER-SIMULATION IoU ===")
    for i, (iou, inter, union) in enumerate(
        zip(
            results["per_sim_iou"],
            results["per_sim_intersection"],
            results["per_sim_union"],
        )
    ):
        print(
            f"Sim {i:02d} | "
            f"IoU={iou:.6f} | "
            f"Intersection={inter} | "
            f"Union={union}"
        )

    if "per_level_recall" in results:
        print("\n=== PER-DEPTH-LEVEL RECALL ===")
        print("(Depth values are printed in physical burial-depth convention.)")

        for level, vals in results["per_level_recall"].items():
            print(
                f"Depth={level:.3f} | "
                f"Recall={vals['recall']:.6f} | "
                f"Detected={vals['detected_count']} | "
                f"GT count={vals['gt_count']} | "
                f"Missed={vals['missed_count']}"
            )

def plot_error_histograms_with_levels(
    pred_all,
    gt_all,
    depth_levels,
    bins=700,
    tol=1e-6,
    roi_only=True,
    plot_absolute=False,
    max_cols=4,
    figsize_per_plot=(5, 4),
    xlim=(-0.075, 0.075),          # zoomed view around +/-0.05
    reference_lines=(-0.05, 0.05),
    log_y=False,                   # set True if tails are hard to see
):
    """
    Plot global and per-depth-level error histograms.

    Parameters
    ----------
    pred_all : torch.Tensor
        Prediction tensor, shape [N, W] or compatible
    gt_all : torch.Tensor
        Ground-truth tensor, same shape as pred_all
    depth_levels : list
        Possible defect depth levels, e.g. [0.1, 0.3, 0.5, 0.7, 0.9]
    bins : int
        Number of histogram bins
    tol : float
        Tolerance for matching depth level
    roi_only : bool
        If True, global histogram is computed only for ROI (gt > 0)
    plot_absolute : bool
        If True, also plot absolute error histograms
    max_cols : int
        Max subplot columns
    figsize_per_plot : tuple
        Size per subplot block
    xlim : tuple or None
        X-axis limits for zooming, e.g. (-0.075, 0.075).
        Use None to disable zoom.
    reference_lines : tuple or None
        Extra vertical reference lines, e.g. (-0.05, 0.05).
    log_y : bool
        If True, use logarithmic y-axis for histograms.
    """

    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.shape != gt_all.shape:
        raise ValueError(f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}")

    err_all = pred_all - gt_all
    abs_err_all = err_all.abs()

    # -------------------------
    # Internal setting:
    # statistics are computed only from the central error region.
    # This avoids long-tail undetected pixels dominating mean/median.
    # -------------------------
    central_stats_range = (-0.1, 0.1)

    def get_central_error_values(err_values):
        """
        Return errors inside the central statistics window.
        Works with numpy arrays.
        """
        err_values = np.asarray(err_values)
        lo, hi = central_stats_range
        return err_values[(err_values >= lo) & (err_values <= hi)]

    def safe_np_mean(x):
        return float(np.mean(x)) if len(x) > 0 else float("nan")

    def safe_np_median(x):
        return float(np.median(x)) if len(x) > 0 else float("nan")

    def safe_np_std(x):
        """
        Standard deviation around the mean.
        """
        return float(np.std(x)) if len(x) > 0 else float("nan")

    def safe_np_std_about_median(x):
        """
        RMS spread around the median instead of the mean.

        std_about_median = sqrt(mean((x - median(x))^2))

        This is useful when the mean is slightly biased by an asymmetric tail,
        but the median still represents the main histogram peak.
        """
        if len(x) == 0:
            return float("nan")

        med = np.median(x)
        return float(np.sqrt(np.mean((x - med) ** 2)))

    def fmt(x):
        return "nan" if not (x == x) else f"{x:.6f}"

    # -------------------------
    # Global mask
    # -------------------------
    if roi_only:
        global_mask = gt_all > 0
        global_title_suffix = "ROI only"
    else:
        global_mask = torch.ones_like(gt_all, dtype=torch.bool)
        global_title_suffix = "All pixels"

    global_err = err_all[global_mask].cpu().numpy()
    global_abs_err = abs_err_all[global_mask].cpu().numpy()

    # -------------------------
    # Helper for vertical lines
    # -------------------------
    def add_error_reference_lines(ax, err_values):
        ax.axvline(
            0.0,
            linestyle='--',
            color='black',
            linewidth=1.8,
            label='Zero error'
        )

        if reference_lines is not None:
            for ref in reference_lines:
                ax.axvline(
                    ref,
                    linestyle='--',
                    color='gray',
                    linewidth=1.8,
                    label=f'{ref:+.2f} reference'
                )

        central_err_values = get_central_error_values(err_values)

        if len(central_err_values) > 0:
            mean_to_plot = central_err_values.mean()
            median_to_plot = np.median(central_err_values)
        else:
            # Fallback only in case no samples fall inside [-0.1, 0.1]
            mean_to_plot = err_values.mean()
            median_to_plot = np.median(err_values)

        ax.axvline(
            mean_to_plot,
            linestyle='-',
            color='red',
            linewidth=2.0,
            label='Mean error [-0.1, 0.1]'
        )

        ax.axvline(
            median_to_plot,
            linestyle='-',
            color='orange',
            linewidth=2.0,
            label='Median error [-0.1, 0.1]'
        )

        if xlim is not None:
            ax.set_xlim(xlim)

        if log_y:
            ax.set_yscale("log")

    # -------------------------
    # Global signed histogram
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(global_err, bins=bins, log=False)
    add_error_reference_lines(ax, global_err)

    ax.set_title(f"Global signed error histogram ({global_title_suffix})")
    ax.set_xlabel("Signed error = pred - gt")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Global absolute histogram
    # -------------------------
    if plot_absolute:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.hist(global_abs_err, bins=bins, log=log_y)

        if xlim is not None:
            ax.set_xlim(0.0, max(abs(xlim[0]), abs(xlim[1])))

        if reference_lines is not None:
            for ref in reference_lines:
                ax.axvline(
                    abs(ref),
                    linestyle='--',
                    color='gray',
                    linewidth=1.8,
                    label=f'|{ref:+.2f}| reference'
                )

        ax.set_title(f"Global absolute error histogram ({global_title_suffix})")
        ax.set_xlabel("Absolute error = |pred - gt|")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    # -------------------------
    # Per-level signed histograms
    # -------------------------
    n_levels = len(depth_levels)
    ncols = min(max_cols, n_levels)
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    axes = np.array(axes).flatten() if n_levels > 1 else [axes]

    for i, level in enumerate(depth_levels):
        level = float(level)
        ax = axes[i]

        level_mask = torch.abs(gt_all - level) < tol
        level_err = err_all[level_mask].cpu().numpy()

        if len(level_err) > 0:
            ax.hist(level_err, bins=bins, log=False)

            add_error_reference_lines(ax, level_err)

            ax.set_title(f"Depth={level:.3f}\nN={len(level_err)}")
            ax.set_xlabel("pred - gt")
            ax.set_ylabel("Count")

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=8)

        else:
            ax.set_title(f"Depth={level:.3f}\nNo samples")
            ax.axis("off")

    for j in range(len(depth_levels), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Signed error histograms by depth level", y=1.02)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Per-level absolute histograms
    # -------------------------
    if plot_absolute:
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
        )

        axes = np.array(axes).flatten() if n_levels > 1 else [axes]

        for i, level in enumerate(depth_levels):
            level = float(level)
            ax = axes[i]

            level_mask = torch.abs(gt_all - level) < tol
            level_abs = abs_err_all[level_mask].cpu().numpy()

            if len(level_abs) > 0:
                ax.hist(level_abs, bins=bins, log=log_y)

                if xlim is not None:
                    ax.set_xlim(0.0, max(abs(xlim[0]), abs(xlim[1])))

                if reference_lines is not None:
                    for ref in reference_lines:
                        ax.axvline(
                            abs(ref),
                            linestyle='--',
                            color='gray',
                            linewidth=1.8,
                            label=f'|{ref:+.2f}| reference'
                        )

                ax.set_title(f"Depth={level:.3f}\nN={len(level_abs)}")
                ax.set_xlabel("|pred - gt|")
                ax.set_ylabel("Count")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(fontsize=8)

            else:
                ax.set_title(f"Depth={level:.3f}\nNo samples")
                ax.axis("off")

        for j in range(len(depth_levels), len(axes)):
            axes[j].axis("off")

        fig.suptitle("Absolute error histograms by depth level", y=1.02)
        plt.tight_layout()
        plt.show()

    # -------------------------
    # Print summary statistics
    # -------------------------
    print("=== GLOBAL ERROR STATS ===")
    print(f"Central statistics window: [{central_stats_range[0]:+.3f}, {central_stats_range[1]:+.3f}]")

    if len(global_err) > 0:
        global_err_central = get_central_error_values(global_err)

        global_std_full = safe_np_std(global_err)
        global_std_full_median = safe_np_std_about_median(global_err)

        global_std_central = safe_np_std(global_err_central)
        global_std_central_median = safe_np_std_about_median(global_err_central)

        print(f"Mean signed error central:          {fmt(safe_np_mean(global_err_central))}")
        print(f"Median signed error central:        {fmt(safe_np_median(global_err_central))}")
        print(f"Std signed error central:           {fmt(global_std_central)}")
        print(f"Std around median central:          {fmt(global_std_central_median)}")
        print(f"Std signed error full:              {fmt(global_std_full)}")
        print(f"Std around median full:             {fmt(global_std_full_median)}")
        print(f"Central count:                      {len(global_err_central)} / {len(global_err)}")
        print(f"Mean absolute error:                {global_abs_err.mean():.6f}")
    else:
        print("No global samples found.")

    print("\n=== PER-LEVEL ERROR STATS ===")
    print(f"Central statistics window: [{central_stats_range[0]:+.3f}, {central_stats_range[1]:+.3f}]")

    for level in depth_levels:
        level = float(level)
        level_mask = torch.abs(gt_all - level) < tol
        level_err = err_all[level_mask]
        level_abs = abs_err_all[level_mask]

        if level_err.numel() > 0:
            level_err_np = level_err.cpu().numpy()
            level_err_central = get_central_error_values(level_err_np)

            central_mean = safe_np_mean(level_err_central)
            central_median = safe_np_median(level_err_central)

            # Standard deviation around mean
            central_std = safe_np_std(level_err_central)
            full_std = safe_np_std(level_err_np)

            # Standard-deviation-like spread around median
            central_std_median = safe_np_std_about_median(level_err_central)
            full_std_median = safe_np_std_about_median(level_err_np)

            print(
                f"Depth={level:.3f} | "
                f"Mean signed central={fmt(central_mean)} | "
                f"Median signed central={fmt(central_median)} | "
                f"Std central={fmt(central_std)} | "
                f"Std around median central={fmt(central_std_median)} | "
                f"Std full={fmt(full_std)} | "
                f"Std around median full={fmt(full_std_median)} | "
                f"Mean abs={level_abs.mean().item():.6f} | "
                f"Central count={len(level_err_central)}/{level_err.numel()} | "
                f"Count={level_err.numel()}"
            )
        else:
            print(f"Depth={level:.3f} | No samples")


# Erosion sensitivity analysis

def sweep_inner_gt_erosion_metrics(
    pred_all,
    gt_all,
    depth_levels,
    erosion_values=None,
    tol=1e-6,
    thickness_mm=3.5,
    include_background=False,
    min_pixels=1,
):
    """
    Sweep different erosion_pixels values and collect global + per-depth metrics.

    This function does not modify your existing functions.
    It repeatedly calls:

        compute_dataset_depth_metrics_inner_gt(...)

    Purpose
    -------
    Helps decide how many edge pixels can be removed before the metric becomes
    too optimistic or too statistically unstable.

    Parameters
    ----------
    pred_all : torch.Tensor
        Predictions, shape [N, W].

    gt_all : torch.Tensor
        Ground truth, shape [N, W].

    depth_levels : list
        Possible depth levels, e.g. [0.25, 0.45, 0.65, 0.85].

    erosion_values : list or None
        List of erosion values to test. If None, uses range(0, 16).

    tol : float
        Float tolerance for matching depth levels.

    thickness_mm : float or None
        If provided, also reports metrics in mm.

    include_background : bool
        Passed into compute_dataset_depth_metrics_inner_gt.

    min_pixels : int
        Minimum number of pixels required to keep an eroded segment.

    Returns
    -------
    df : pandas.DataFrame
        Table with global and per-depth metrics for each erosion value.
    """

    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.ndim != 2 or gt_all.ndim != 2:
        raise ValueError(
            f"Expected [N, W] tensors, got pred {pred_all.shape}, gt {gt_all.shape}"
        )

    if pred_all.shape != gt_all.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}"
        )

    if erosion_values is None:
        erosion_values = list(range(0, 16))

    # -----------------------------
    # Original counts before erosion
    # -----------------------------
    gt_defect_mask = gt_all > 0
    original_global_count = int(gt_defect_mask.sum().item())

    original_level_counts = {}

    for level in depth_levels:
        level = float(level)
        level_mask = torch.abs(gt_all - level) < tol
        original_level_counts[level] = int(level_mask.sum().item())

    rows = []

    # -----------------------------
    # Sweep erosion values
    # -----------------------------
    for erosion_pixels in erosion_values:
        results = compute_dataset_depth_metrics_inner_gt(
            pred_all=pred_all,
            gt_all=gt_all,
            depth_levels=depth_levels,
            tol=tol,
            thickness_mm=thickness_mm,
            include_background=include_background,
            erosion_pixels=erosion_pixels,
            min_pixels=min_pixels,
        )

        inner_global_count = results["dataset"]["inner_roi_count"]

        global_retention = (
            inner_global_count / original_global_count
            if original_global_count > 0
            else np.nan
        )

        global_row = {
            "erosion_pixels": int(erosion_pixels),
            "scope": "global",
            "depth": np.nan,
            "mae": results["global"]["mae"],
            "medae": results["global"]["medae"],
            "rmse": results["global"]["rmse"],
            "count": results["global"]["count"],
            "original_count": original_global_count,
            "retention": global_retention,
        }

        if thickness_mm is not None and "global_mm" in results:
            global_row.update(
                {
                    "mae_mm": results["global_mm"]["mae"],
                    "medae_mm": results["global_mm"]["medae"],
                    "rmse_mm": results["global_mm"]["rmse"],
                }
            )

        rows.append(global_row)

        # -----------------------------
        # Per-depth metrics
        # -----------------------------
        for level in depth_levels:
            level = float(level)
            vals = results["per_level"][level]

            original_count = original_level_counts[level]
            inner_count = vals["count"]

            retention = (
                inner_count / original_count
                if original_count > 0
                else np.nan
            )

            row = {
                "erosion_pixels": int(erosion_pixels),
                "scope": "per_depth",
                "depth": level,
                "mae": vals["mae"],
                "medae": vals["medae"],
                "rmse": vals["rmse"],
                "count": inner_count,
                "original_count": original_count,
                "retention": retention,
            }

            if thickness_mm is not None and "per_level_mm" in results:
                vals_mm = results["per_level_mm"][level]

                row.update(
                    {
                        "mae_mm": vals_mm["mae"],
                        "medae_mm": vals_mm["medae"],
                        "rmse_mm": vals_mm["rmse"],
                    }
                )

            rows.append(row)

    df = pd.DataFrame(rows)

    # -----------------------------
    # Add changes relative to previous erosion level
    # -----------------------------
    df["group_id"] = df.apply(
        lambda r: "global"
        if r["scope"] == "global"
        else f"depth_{r['depth']:.6f}",
        axis=1,
    )

    df = df.sort_values(["group_id", "erosion_pixels"]).reset_index(drop=True)

    metric_cols = ["mae", "medae", "rmse"]

    if thickness_mm is not None:
        metric_cols += ["mae_mm", "medae_mm", "rmse_mm"]

    for col in metric_cols:
        if col in df.columns:
            df[f"{col}_delta_prev"] = df.groupby("group_id")[col].diff()

    df["retention_delta_prev"] = df.groupby("group_id")["retention"].diff()

    # Restore more readable sorting
    df = df.sort_values(["erosion_pixels", "scope", "depth"]).reset_index(drop=True)

    return df


def print_erosion_sweep_summary(
    df,
    metric="mae_mm",
    show_per_depth=True,
):
    """
    Print compact erosion sensitivity summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from sweep_inner_gt_erosion_metrics.

    metric : str
        Metric to print, e.g. "mae", "medae", "rmse", "mae_mm".

    show_per_depth : bool
        If True, also print per-depth erosion summary.
    """

    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Falling back to 'mae'.")
        metric = "mae"

    print("=== GLOBAL EROSION SWEEP ===")

    global_df = df[df["scope"] == "global"].copy()
    global_df = global_df.sort_values("erosion_pixels")

    for _, row in global_df.iterrows():
        metric_val = row[metric]

        print(
            f"Erosion={int(row['erosion_pixels']):02d} px | "
            f"{metric}={metric_val:.6f} | "
            f"Count={int(row['count'])}/{int(row['original_count'])} | "
            f"Retention={100.0 * row['retention']:.2f}%"
        )

    if show_per_depth:
        print("\n=== PER-DEPTH EROSION SWEEP ===")

        depth_df = df[df["scope"] == "per_depth"].copy()
        depth_df = depth_df.sort_values(["erosion_pixels", "depth"])

        for erosion in sorted(depth_df["erosion_pixels"].unique()):
            sub = depth_df[depth_df["erosion_pixels"] == erosion]

            print(f"\n--- Erosion={int(erosion)} px ---")

            for _, row in sub.iterrows():
                metric_val = row[metric]

                print(
                    f"Depth={row['depth']:.3f} | "
                    f"{metric}={metric_val:.6f} | "
                    f"Count={int(row['count'])}/{int(row['original_count'])} | "
                    f"Retention={100.0 * row['retention']:.2f}%"
                )


def plot_erosion_sensitivity(
    df,
    metric="mae_mm",
    plot_per_depth=True,
):
    """
    Plot metric and retained pixel percentage versus erosion_pixels.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from sweep_inner_gt_erosion_metrics.

    metric : str
        Metric to plot, e.g. "mae", "medae", "rmse", "mae_mm".

    plot_per_depth : bool
        If True, also plot per-depth curves.
    """

    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Falling back to 'mae'.")
        metric = "mae"

    # -----------------------------
    # Global metric vs erosion
    # -----------------------------
    global_df = df[df["scope"] == "global"].copy()
    global_df = global_df.sort_values("erosion_pixels")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        global_df["erosion_pixels"],
        global_df[metric],
        marker="o",
        label=f"Global {metric}",
    )

    ax.set_xlabel("Erosion pixels")
    ax.set_ylabel(metric)
    ax.set_title(f"Global {metric} vs erosion pixels")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Global retained ROI vs erosion
    # -----------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        global_df["erosion_pixels"],
        100.0 * global_df["retention"],
        marker="o",
        label="Global retained ROI",
    )

    ax.set_xlabel("Erosion pixels")
    ax.set_ylabel("Retained ROI [%]")
    ax.set_title("Global retained GT ROI vs erosion pixels")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Per-depth plots
    # -----------------------------
    if plot_per_depth:
        depth_df = df[df["scope"] == "per_depth"].copy()
        depth_df = depth_df.sort_values(["depth", "erosion_pixels"])

        # Metric per depth
        fig, ax = plt.subplots(figsize=(8, 5))

        for depth in sorted(depth_df["depth"].dropna().unique()):
            sub = depth_df[depth_df["depth"] == depth]

            ax.plot(
                sub["erosion_pixels"],
                sub[metric],
                marker="o",
                label=f"Depth={depth:.3f}",
            )

        ax.set_xlabel("Erosion pixels")
        ax.set_ylabel(metric)
        ax.set_title(f"Per-depth {metric} vs erosion pixels")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Retention per depth
        fig, ax = plt.subplots(figsize=(8, 5))

        for depth in sorted(depth_df["depth"].dropna().unique()):
            sub = depth_df[depth_df["depth"] == depth]

            ax.plot(
                sub["erosion_pixels"],
                100.0 * sub["retention"],
                marker="o",
                label=f"Depth={depth:.3f}",
            )

        ax.set_xlabel("Erosion pixels")
        ax.set_ylabel("Retained pixels [%]")
        ax.set_title("Per-depth retained GT pixels vs erosion pixels")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()


def suggest_erosion_from_sweep(
    df,
    metric="mae_mm",
    min_global_retention=0.75,
    min_per_depth_retention=0.50,
    relative_change_threshold=0.05,
):
    """
    Suggest a reasonable erosion value based on stability and pixel retention.

    This is not meant to blindly optimize the metric.
    It tries to find the smallest erosion where:

        1. Global retention is still high enough.
        2. Every depth level keeps enough pixels.
        3. Metric improvement relative to previous erosion is small.

    Parameters
    ----------
    df : pandas.DataFrame
        Output from sweep_inner_gt_erosion_metrics.

    metric : str
        Metric used for stability check, e.g. "mae_mm" or "mae".

    min_global_retention : float
        Minimum global retained ROI fraction.

    min_per_depth_retention : float
        Minimum retained fraction for every depth level.

    relative_change_threshold : float
        Relative metric change threshold.
        Example: 0.05 means less than 5% change from previous erosion.

    Returns
    -------
    suggestion : dict or None
        Suggested erosion information.
    """

    if metric not in df.columns:
        print(f"Metric '{metric}' not found. Falling back to 'mae'.")
        metric = "mae"

    global_df = df[df["scope"] == "global"].copy()
    depth_df = df[df["scope"] == "per_depth"].copy()

    global_df = global_df.sort_values("erosion_pixels")

    candidates = []

    for _, row in global_df.iterrows():
        erosion = int(row["erosion_pixels"])

        if erosion == 0:
            continue

        current_metric = row[metric]
        current_retention = row["retention"]

        prev_rows = global_df[global_df["erosion_pixels"] == erosion - 1]

        if len(prev_rows) == 0:
            continue

        prev_metric = prev_rows.iloc[0][metric]

        if not np.isfinite(current_metric) or not np.isfinite(prev_metric):
            continue

        if abs(prev_metric) < 1e-12:
            relative_change = np.nan
        else:
            relative_change = abs(current_metric - prev_metric) / abs(prev_metric)

        sub_depth = depth_df[depth_df["erosion_pixels"] == erosion]

        if len(sub_depth) == 0:
            continue

        min_depth_retention = sub_depth["retention"].min()

        passes = (
            current_retention >= min_global_retention
            and min_depth_retention >= min_per_depth_retention
            and np.isfinite(relative_change)
            and relative_change <= relative_change_threshold
        )

        candidates.append(
            {
                "erosion_pixels": erosion,
                metric: current_metric,
                "global_retention": current_retention,
                "min_per_depth_retention": min_depth_retention,
                "relative_change_from_previous": relative_change,
                "passes": passes,
            }
        )

    valid_candidates = [c for c in candidates if c["passes"]]

    if len(valid_candidates) == 0:
        print("No erosion value satisfies all selected criteria.")
        print("You may need to relax retention or relative-change thresholds.")
        return None

    suggestion = valid_candidates[0]

    print("=== SUGGESTED EROSION VALUE ===")
    print(f"Erosion pixels: {suggestion['erosion_pixels']}")
    print(f"{metric}: {suggestion[metric]:.6f}")
    print(f"Global retention: {100.0 * suggestion['global_retention']:.2f}%")
    print(
        f"Minimum per-depth retention: "
        f"{100.0 * suggestion['min_per_depth_retention']:.2f}%"
    )
    print(
        f"Relative change from previous erosion: "
        f"{100.0 * suggestion['relative_change_from_previous']:.2f}%"
    )

    return suggestion



def tomografic_reconstruction(pred_all,step=1,elev=35,azim=-135):
    Z = torch.as_tensor(pred_all).detach().cpu().numpy()

    # optional downsampling for speed / cleaner view
    step = 1
    Z = Z[::step, ::step]

    H, W = Z.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        linewidth=0,
        antialiased=True
    )

    fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.08, label='Predicted value')

    ax.set_title("3D surface of predicted depth field")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_zlabel("Value")

    # nice viewing angle
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.show()
