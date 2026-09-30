"""
Evaluation metrics and error plots for predicted defect depths.

Predictions and ground truth are depth maps in normalised units (fraction of
the specimen thickness, 0 = sound material). They can be a single depth
profile [W] from one B-scan, or a 2D map [rows, W] made by stacking the
profiles of consecutive rows of one specimen.

Only defect pixels (ground truth > 0) are evaluated. Defects cover a small
part of the specimen, so including the background, which is easy to predict
as 0, would make the errors look better than they are.

Functions:
  calculate_errors                  - pixel-wise MAE / MedAE / RMSE over the
                                      defect pixels.
  calculate_median_depth_errors     - one depth estimate per defect (median
                                      prediction in the defect interior),
                                      then MAE / MedAE / RMSE.
  plot_signed_errors_by_depth       - histograms of (prediction - GT) for
                                      each ground-truth depth.
  calculate_metrics_per_simulation  - calculate_median_depth_errors for every
                                      test simulation, optionally in mm.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from matplotlib.ticker import MultipleLocator, PercentFormatter
import warnings

def calculate_errors(pred, mask):
    """
    Pixel-wise depth errors over the ground-truth defect pixels.

    Parameters
    ----------
    pred, mask : np.ndarray or torch.Tensor, same shape
        Predicted and ground-truth depth.

    Returns
    -------
    dict
        mae   : mean absolute error,
        medae : median absolute error (less sensitive to a few large
                errors, e.g. at defect edges),
        rmse  : root mean squared error (weights large errors more),
        count : number of defect pixels evaluated.
        Errors are in the same units as the inputs.
    """
    # Accepts both torch tensors (moved to CPU and detached from the graph)
    # and numpy arrays / lists.
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred)
    mask = to_numpy(mask)

    if pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {mask.shape}")

    # Evaluate defect pixels only; the background would dominate the pixel
    # count and lower the errors artificially.
    roi = mask > 0

    if not roi.any():
        raise ValueError("The mask contains no defect pixels.")

    error = pred[roi] - mask[roi]

    return {
        "mae": float(np.mean(np.abs(error))),
        "medae": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "count": int(roi.sum()),
    }


def calculate_median_depth_errors(
    pred,
    mask,
    erosion_pixels=1,
    inner_percent=100,
):
    """
    Defect-level depth error: one depth estimate per defect.

    Close to the edge of a defect the thermal response is blurred by lateral
    heat diffusion, so the predicted depth there is less reliable than in the
    middle. This function therefore estimates the depth of each defect from
    its interior only, and compares that single value with the ground truth:

      1. Find the separate defects in the ground truth (connected groups of
         non-zero pixels, diagonal neighbours included).
      2. Remove `erosion_pixels` layers of pixels from the edge of each
         defect.
      3. From what is left, keep the `inner_percent` % of pixels that are
         farthest from the defect edge.
      4. Take the median prediction over those pixels as the depth of the
         defect.
      5. Assign this value to every pixel of the original (not eroded)
         defect.
      6. Compute MAE / MedAE / RMSE between these values and the ground
         truth over all defect pixels, so large defects weigh more than
         small ones.

    Parameters
    ----------
    pred, mask : np.ndarray or torch.Tensor, same shape, 1D or 2D
        Predicted and ground-truth depth. A 2D input must be one specimen
        (consecutive rows), so that defects are connected across rows.
    erosion_pixels : int >= 0
        Number of pixel layers removed from the defect edge. 0 = no erosion.
    inner_percent : float in [0, 100]
        Share of the remaining pixels used for the median (by pixel count).
        100 uses all of them; 0 uses only the single most central pixel.
        At least one pixel per defect is always used.

    Returns
    -------
    metrics : dict
        mae, medae, rmse, in the same units as the inputs.
    median_prediction : np.ndarray
        Same shape as the input; each defect filled with its estimated
        depth, 0 elsewhere.
    overlay : np.ndarray, uint8
        Map of which pixels were used, for plotting:
        0 = background, 1 = defect pixel not used for the median,
        2 = defect pixel used for the median.

    If erosion removes a whole (small) defect, the pixels of that defect
    farthest from its edge are used instead, so every defect is still
    evaluated; a warning reports how often this happened. Pixels at the same
    distance from the edge are ordered by their distance to the defect
    centre.
    """
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred)
    mask = to_numpy(mask)

    # Input checks.
    if pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {mask.shape}")
    if pred.ndim not in (1, 2):
        raise ValueError("Expected matching 1D or 2D arrays.")
    if (
        not isinstance(erosion_pixels, (int, np.integer))
        or erosion_pixels < 0
    ):
        raise ValueError("erosion_pixels must be a nonnegative integer.")
    if not np.isfinite(inner_percent) or not 0 <= inner_percent <= 100:
        raise ValueError("inner_percent must be in [0, 100].")
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains non-finite values.")

    roi = mask > 0
    if not roi.any():
        raise ValueError("The mask contains no defect pixels.")

    # Label connected defects. A full 3 x 3 structure (3 in 1D) counts
    # diagonal neighbours as connected.
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    labels, count = ndi.label(roi, structure=structure)

    median_prediction = np.zeros_like(pred, dtype=float)
    overlay = np.zeros(mask.shape, dtype=np.uint8)
    overlay[roi] = 1

    # Number of defects that disappeared completely after erosion.
    fallback_count = 0

    for label in range(1, count + 1):
        region = labels == label

        # Distance of every defect pixel to the nearest non-defect pixel.
        # The one-pixel pad makes the image border count as background, so a
        # defect touching the border is not treated as infinitely deep
        # inside.
        padded = np.pad(region, 1, mode="constant", constant_values=False)
        crop = (slice(1, -1),) * region.ndim
        distance = ndi.distance_transform_edt(padded)[crop]

        eroded = (
            ndi.binary_erosion(
                region,
                structure=structure,
                iterations=erosion_pixels,
                border_value=0,
            )
            if erosion_pixels > 0
            else region.copy()
        )

        # Small defects can vanish after erosion; fall back to their most
        # central pixel(s).
        if not eroded.any():
            fallback_count += 1
            eroded = region & (distance == distance[region].max())

        coords = np.argwhere(eroded)
        n_keep = max(
            1,
            int(np.ceil(len(coords) * inner_percent / 100.0)),
        )

        # Rank the candidate pixels: first by distance from the original
        # defect edge (largest first), then by distance to the defect centre
        # (smallest first), then by index so the order is deterministic.
        # np.lexsort uses the last key as the primary one.
        centroid = np.argwhere(region).mean(axis=0)
        centroid_distance_sq = np.sum((coords - centroid) ** 2, axis=1)
        boundary_distance = distance[tuple(coords.T)]

        order = np.lexsort((
            np.arange(len(coords)),
            centroid_distance_sq,
            -boundary_distance,
        ))
        chosen = coords[order[:n_keep]]
        selected_indices = tuple(chosen.T)

        values = pred[selected_indices]
        if not np.isfinite(values).all():
            raise ValueError(
                "A selected interior contains non-finite predictions."
            )

        # The median is robust to a few outlying predictions inside the
        # defect.
        median_prediction[region] = np.median(values)
        overlay[selected_indices] = 2

    error = median_prediction[roi] - mask[roi]

    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "medae": float(np.median(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }

    if fallback_count:
        warnings.warn(
            f"Erosion removed {fallback_count} defect(s); used their "
            "deepest original pixels as fallback candidates.",
            stacklevel=2,
        )

    return metrics, median_prediction, overlay

def plot_signed_errors_by_depth(
    pred,
    mask,
    depth_levels=(0.1, 0.2, 0.3, 0.4, 0.5),
    tol=1e-6,
    bins=100,
):
    """
    Histograms of the signed error (prediction - ground truth), one panel
    per ground-truth depth.

    The sign shows whether defects of a given depth are systematically
    predicted too deep (positive) or too shallow (negative). Depths are
    fractions of the thickness (0.1 = 10 %), so the error is in percentage
    points of the thickness. The red dashed lines at +/-0.10 mark
    +/-10 percentage points, not +/-10 % of each depth.

    Parameters
    ----------
    pred, mask : np.ndarray or torch.Tensor, same shape
    depth_levels : sequence of float
        Ground-truth depths to plot; one panel each.
    tol : float
        Tolerance used to match mask values to a depth level.
    bins : int
        Number of histogram bins.

    Returns
    -------
    fig, axes, statistics
        statistics is a list with, per depth level, the number of pixels
        and the mean and median signed error.
    """
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred)
    mask = to_numpy(mask)

    if pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {mask.shape}")
    if len(depth_levels) == 0:
        raise ValueError("Provide at least one depth level.")

    # Errors of all pixels whose ground truth equals each depth level
    # (NaN / inf values are skipped).
    errors_by_depth = [
        pred[selected] - mask[selected]
        for level in depth_levels
        for selected in [
            np.isclose(mask, level, atol=tol, rtol=0)
            & np.isfinite(mask)
            & np.isfinite(pred)
        ]
    ]

    # The same x-range and bins in every panel, so they can be compared.
    # The range is the largest error rounded up to a multiple of 0.05, and
    # at least +/-0.15.
    largest_error = max(
        (float(np.max(np.abs(e))) for e in errors_by_depth if e.size),
        default=0.0,
    )
    limit = max(0.15, np.ceil(largest_error / 0.05) * 0.05)
    bin_edges = np.linspace(-limit, limit, bins + 1)

    fig, axes = plt.subplots(
        len(depth_levels),
        1,
        figsize=(11, 3 * len(depth_levels)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    statistics = []

    for ax, level, errors in zip(axes, depth_levels, errors_by_depth):
        # Zero-error line and the +/-10 pp reference lines.
        ax.axvline(0, color="gray", linewidth=1)
        ax.axvline(
            -0.10, color="red", linestyle="--",
            linewidth=1.5, label="Reference: ±10 percentage points",
        )
        ax.axvline(0.10, color="red", linestyle="--", linewidth=1.5)

        if errors.size:
            mean = float(np.mean(errors))
            median = float(np.median(errors))

            ax.hist(
                errors,
                bins=bin_edges,
                color="steelblue",
                edgecolor="white",
                alpha=0.8,
            )
            ax.axvline(
                mean, color="darkorange", linewidth=2,
                label=f"Mean: {mean * 100:+.2f} pp",
            )
            ax.axvline(
                median, color="purple", linestyle=":",
                linewidth=2.5,
                label=f"Median: {median * 100:+.2f} pp",
            )
        else:
            mean = median = np.nan
            ax.text(
                0.5, 0.5, "No matching pixels",
                transform=ax.transAxes, ha="center",
            )

        statistics.append({
            "depth": float(level),
            "count": int(errors.size),
            "mean_signed_error": mean,
            "median_signed_error": median,
        })

        # x-axis shown in percent, with ticks every 5 pp; labels are
        # repeated on every panel even though the axis is shared.
        ax.set_title(f"GT depth: {level:.1f} ({level:.0%})")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(-limit, limit)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=9)

    axes[-1].set_xlabel(
        "Signed error: prediction − GT [percentage points]"
    )
    plt.show()

    return fig, axes, statistics


def calculate_metrics_per_simulation(
    pred_all,
    mask_all,
    rows_per_sim=174,
    num_simulations=6,
    thickness_mm=5.0,
):
    """
    Defect-level errors for each test simulation separately, and their mean.

    The predictions of all test B-scans are stacked row by row: the first
    `rows_per_sim` rows belong to simulation 1, the next ones to simulation 2,
    and so on. Each block is evaluated with calculate_median_depth_errors
    (erosion_pixels=1, inner_percent=100).

    Parameters
    ----------
    pred_all, mask_all : np.ndarray or torch.Tensor, [total_rows, W]
        Stacked predicted and ground-truth depth profiles, normalised.
    rows_per_sim : int
        Number of rows (B-scans) per simulation.
    num_simulations : int
        Number of simulations; total_rows must equal
        rows_per_sim * num_simulations.
    thickness_mm : float or None
        Specimen thickness. The normalised errors are multiplied by it to
        give millimetres. None keeps the normalised units.

    Returns
    -------
    per_simulation : list of dict
        mae, medae, rmse for every simulation.
    mean_metrics : dict
        Mean of each metric over the simulations (every simulation has the
        same weight).
    """
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred_all)
    mask = to_numpy(mask_all)

    if pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {mask.shape}")
    if pred.ndim != 2:
        raise ValueError("Expected arrays with shape (total_rows, width).")

    for name, value in (
        ("rows_per_sim", rows_per_sim),
        ("num_simulations", num_simulations),
    ):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")

    # Guards against splitting the stack at the wrong places.
    expected_rows = rows_per_sim * num_simulations
    if pred.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows "
            f"({num_simulations} simulations × {rows_per_sim}), "
            f"received {pred.shape[0]}."
        )

    if thickness_mm is not None:
        if not np.isfinite(thickness_mm) or thickness_mm <= 0:
            raise ValueError("thickness_mm must be finite and positive.")

    # Normalised depth x thickness = depth in mm.
    scale = 1.0 if thickness_mm is None else thickness_mm
    unit = "" if thickness_mm is None else " mm"
    per_simulation = []

    for i in range(num_simulations):
        # Rows of simulation i.
        start = i * rows_per_sim
        stop = start + rows_per_sim

        metrics, _, _ = calculate_median_depth_errors(
            pred[start:stop],
            mask[start:stop],
            erosion_pixels=1,
            inner_percent=100,
        )

        result = {
            "simulation": i + 1,
            "mae": metrics["mae"] * scale,
            "medae": metrics["medae"] * scale,
            "rmse": metrics["rmse"] * scale,
        }
        per_simulation.append(result)

        print(
            f"Simulation {i + 1} | "
            f"MAE: {result['mae']:.4f}{unit} | "
            f"MedAE: {result['medae']:.4f}{unit} | "
            f"RMSE: {result['rmse']:.4f}{unit}"
        )

    mean_metrics = {
        name: float(np.mean([result[name] for result in per_simulation]))
        for name in ("mae", "medae", "rmse")
    }

    print(
        f"\nMean across {num_simulations} simulations | "
        f"MAE: {mean_metrics['mae']:.4f}{unit} | "
        f"MedAE: {mean_metrics['medae']:.4f}{unit} | "
        f"RMSE: {mean_metrics['rmse']:.4f}{unit}"
    )

    return per_simulation, mean_metrics