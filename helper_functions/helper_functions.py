import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import imageio
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import math
from tqdm import tqdm

# Augmentation: -----------------------------------------------------------------------------------

class NoiseAddition:
    def __init__(self, sigma_mean=0.065, sigma_std=0.01, sigma_min=0.04, sigma_max=0.09):
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, bscan, depth):
        sigma = torch.normal(
            mean=torch.tensor(self.sigma_mean),
            std=torch.tensor(self.sigma_std)
        ).item()
        sigma = max(self.sigma_min, min(self.sigma_max, sigma))

        noise = torch.randn_like(bscan) * sigma
        return (bscan + noise).clamp_min(0.0), depth
    
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
    Interpolate data to a new size using PyTorch's interpolate function.
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        
        data_interpolated = torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0)
    
# Validation metrics: -----------------------------------------------------------------------------------

class NoiseAdditionExperiment:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, bscan, depth):
        noise = torch.randn_like(bscan) * self.sigma
        bscan_noisy = (bscan + noise).clamp_min(0.0)
        return bscan_noisy, depth

def visualize_global_prediction(pred_all,mask_all,n):
    """
    It takes all of the predictions and masks plot individual simulation with Absolute error between them

    """
    plt.figure(figsize=(15,12))
    plt.subplot(1,3,1)
    plt.imshow(pred_all[(n-1)*512:n*512,:])
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.title('Predicted Field of view')

    plt.subplot(1,3,2)
    plt.imshow(mask_all[(n-1)*512:n*512,:])
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.title('Ground truth Field of view')

    plt.subplot(1,3,3)
    plt.imshow(torch.abs(pred_all[(n-1)*512:n*512,:]-mask_all[(n-1)*512:n*512,:]))
    plt.title('Absolute error')
    plt.colorbar(label='Normalized depth error',shrink=0.3)
    plt.xlabel('Pixel width')
    plt.ylabel('Pixel height')
    plt.tight_layout()

def visualize_bscan_prediction(pred_all,mask_all,n,level):
    """
    Function which visualize prediction over single B-scan at the same time presenting the depth seen by the network
    """
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


def compute_iou_over_concatenated_sims(
    pred_all,
    gt_all,
    sim_size=512,
    pred_threshold=0.03,
    gt_threshold=0.0,
    eps=1e-8,
):
    """
    Compute IoU globally and per simulation from concatenated tensors.

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
        Threshold for GT defect mask (usually 0.0 if gt>0 defines ROI)
    eps : float
        Small constant for numerical safety

    Returns
    -------
    results : dict
        {
            "global_iou": float,
            "global_intersection": int,
            "global_union": int,
            "num_sims": int,
            "per_sim_iou": list,
            "per_sim_intersection": list,
            "per_sim_union": list,
        }
    """
    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.ndim != 2 or gt_all.ndim != 2:
        raise ValueError(f"Expected [N, W], got pred {pred_all.shape}, gt {gt_all.shape}")

    if pred_all.shape != gt_all.shape:
        raise ValueError(f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}")

    n_rows = pred_all.shape[0]
    if n_rows % sim_size != 0:
        raise ValueError(
            f"Number of rows ({n_rows}) is not divisible by sim_size ({sim_size})"
        )

    pred_mask = pred_all > pred_threshold
    gt_mask = gt_all > gt_threshold

    # -------- global IoU --------
    global_intersection = (pred_mask & gt_mask).sum().item()
    global_union = (pred_mask | gt_mask).sum().item()
    global_iou = global_intersection / (global_union + eps)

    # -------- per-simulation IoU --------
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

    results = {
        "global_iou": float(global_iou),
        "global_intersection": int(global_intersection),
        "global_union": int(global_union),
        "num_sims": int(num_sims),
        "per_sim_iou": per_sim_iou,
        "per_sim_intersection": per_sim_intersection,
        "per_sim_union": per_sim_union,
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

def network_eval_run(model, loader, non_linear_pred=False, device='cuda'):
    """
    Simple eval loop for prediction gathering.

    Returns
    -------
    pred_all : torch.Tensor
        Concatenated predictions
    mask_all : torch.Tensor
        Concatenated ground truth
    err_all : torch.Tensor
        Signed error = pred - gt
    abs_err_all : torch.Tensor
        Absolute error = |pred - gt|
    """
    model.eval()

    pred_all = []
    mask_all = []
    err_all = []
    abs_err_all = []

    with torch.no_grad():
        for X, mask in tqdm(loader):
            X = X.to(device)
            mask = mask.to(device)

            pred = model(X)

            if non_linear_pred:
                pred = pred.clamp(0, 1)
                pred = pred ** 2
                pred = pred.clamp(0, 1)

            err = pred - mask
            abs_err = err.abs()

            pred_all.append(pred.cpu())
            mask_all.append(mask.cpu())
            err_all.append(err.cpu())
            abs_err_all.append(abs_err.cpu())

    pred_all = torch.cat(pred_all, dim=0)
    mask_all = torch.cat(mask_all, dim=0)
    err_all = torch.cat(err_all, dim=0)
    abs_err_all = torch.cat(abs_err_all, dim=0)

    return pred_all, mask_all, err_all, abs_err_all

def plot_error_histograms_with_levels(
    pred_all,
    gt_all,
    depth_levels,
    bins=80,
    tol=1e-6,
    roi_only=True,
    plot_absolute=True,
    max_cols=3,
    figsize_per_plot=(5, 4),
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
    """
    pred_all = torch.as_tensor(pred_all).float()
    gt_all = torch.as_tensor(gt_all).float()

    if pred_all.shape != gt_all.shape:
        raise ValueError(f"Shape mismatch: pred {pred_all.shape}, gt {gt_all.shape}")

    err_all = pred_all - gt_all
    abs_err_all = err_all.abs()

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
    # Global signed histogram
    # -------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(global_err, bins=bins)
    plt.axvline(0.0, linestyle='--')
    plt.axvline(global_err.mean(), linestyle='-')
    plt.title(f"Global signed error histogram ({global_title_suffix})")
    plt.xlabel("Signed error = pred - gt")
    plt.ylabel("Count")
    plt.show()

    # -------------------------
    # Global absolute histogram
    # -------------------------
    if plot_absolute:
        plt.figure(figsize=(7, 5))
        plt.hist(global_abs_err, bins=bins)
        plt.title(f"Global absolute error histogram ({global_title_suffix})")
        plt.xlabel("Absolute error = |pred - gt|")
        plt.ylabel("Count")
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
    axes = axes.flatten() if n_levels > 1 else [axes]

    for i, level in enumerate(depth_levels):
        level = float(level)
        ax = axes[i]

        level_mask = torch.abs(gt_all - level) < tol
        level_err = err_all[level_mask].cpu().numpy()

        if len(level_err) > 0:
            ax.hist(level_err, bins=bins)
            ax.axvline(0.0, linestyle='--')
            ax.axvline(level_err.mean(), linestyle='-')
            ax.set_title(f"Depth={level:.3f}\nN={len(level_err)}")
            ax.set_xlabel("pred - gt")
            ax.set_ylabel("Count")
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
        axes = axes.flatten() if n_levels > 1 else [axes]

        for i, level in enumerate(depth_levels):
            level = float(level)
            ax = axes[i]

            level_mask = torch.abs(gt_all - level) < tol
            level_abs = abs_err_all[level_mask].cpu().numpy()

            if len(level_abs) > 0:
                ax.hist(level_abs, bins=bins)
                ax.set_title(f"Depth={level:.3f}\nN={len(level_abs)}")
                ax.set_xlabel("|pred - gt|")
                ax.set_ylabel("Count")
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
    if len(global_err) > 0:
        print(f"Mean signed error:   {global_err.mean():.6f}")
        print(f"Median signed error: {torch.tensor(global_err).median().item():.6f}")
        print(f"Mean absolute error: {global_abs_err.mean():.6f}")
    else:
        print("No global samples found.")

    print("\n=== PER-LEVEL ERROR STATS ===")
    for level in depth_levels:
        level = float(level)
        level_mask = torch.abs(gt_all - level) < tol
        level_err = err_all[level_mask]
        level_abs = abs_err_all[level_mask]

        if level_err.numel() > 0:
            print(
                f"Depth={level:.3f} | "
                f"Mean signed={level_err.mean().item():.6f} | "
                f"Median signed={level_err.median().item():.6f} | "
                f"Mean abs={level_abs.mean().item():.6f} | "
                f"Count={level_err.numel()}"
            )
        else:
            print(f"Depth={level:.3f} | No samples")