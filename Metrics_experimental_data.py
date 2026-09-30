import numpy as np
from matplotlib import pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator
from helper_functions.helper_functions import resize_tensor
from scipy import ndimage as ndi
from matplotlib.ticker import MultipleLocator, PercentFormatter
import warnings

def calculate_errors(pred, mask):
    """Calculate MAE, MedAE, and RMSE only at GT defect pixels."""
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred)
    mask = to_numpy(mask)

    if pred.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {mask.shape}")

    roi = mask > 0 # We selecte only defect and disregard BG since defect are rather sparse and we could artificially improve the metrics

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
    Estimate defect values from progressively smaller interior regions.

    Steps
    -----
    1. Identify connected GT defects.
    2. Erode each defect by erosion_pixels.
    3. Keep inner_percent of the remaining pixels, prioritizing pixels
       furthest from the ORIGINAL defect boundary.
    4. Calculate the median prediction over those selected pixels.
    5. Assign that median across the ORIGINAL defect extent.
    6. Calculate MAE, MedAE, and RMSE over all original GT defect pixels.

    Parameters
    ----------
    pred, mask : matching 1D or 2D arrays / PyTorch tensors
        For 2D inputs, use one spatial scene.
    erosion_pixels : int
        Number of erosion iterations.
    inner_percent : float in [0, 100]
        Percentage of eroded pixels to retain.
        100 uses the entire eroded region.
        0 selects one deepest pixel.
        At least one pixel is retained per defect.

    Returns
    -------
    metrics : dict
        Pixel-weighted errors in the same units as the inputs.
    median_prediction : ndarray
        Estimated median assigned across each original GT defect.
    overlay : uint8 ndarray
        0 = background
        1 = excluded GT pixels
        2 = selected pixels used for median estimation

    Notes
    -----
    If erosion removes a defect, selection falls back to its deepest
    original pixel(s), preserving every defect in the evaluation.

    Percentage refers to pixel count, not width or height.
    Equal-distance ties prioritize proximity to the region centroid.
    """
    def to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=float)

    pred = to_numpy(pred)
    mask = to_numpy(mask)

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

    structure = np.ones((3,) * mask.ndim, dtype=bool)
    labels, count = ndi.label(roi, structure=structure)

    median_prediction = np.zeros_like(pred, dtype=float)
    overlay = np.zeros(mask.shape, dtype=np.uint8)
    overlay[roi] = 1

    fallback_count = 0

    for label in range(1, count + 1):
        region = labels == label

        # Padding treats locations outside the image as background.
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

        if not eroded.any():
            fallback_count += 1
            eroded = region & (distance == distance[region].max())

        coords = np.argwhere(eroded)
        n_keep = max(
            1,
            int(np.ceil(len(coords) * inner_percent / 100.0)),
        )

        # Primary ranking: greatest distance from the original boundary.
        # Tie-breaker: closest to the original region's centroid.
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
    Plot signed error distributions grouped by ground-truth depth.

    Inputs use fractions: 0.1 = 10%.
    Error = prediction - ground truth.
    Reference lines at ±0.10 indicate ±10 percentage points,
    not relative errors of ±10% of each depth.

    Returns
    -------
    fig, axes, statistics
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

    errors_by_depth = [
        pred[selected] - mask[selected]
        for level in depth_levels
        for selected in [
            np.isclose(mask, level, atol=tol, rtol=0)
            & np.isfinite(mask)
            & np.isfinite(pred)
        ]
    ]

    # Shared limits and bins make the panels comparable.
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

import numpy as np


def calculate_metrics_per_simulation(
    pred_all,
    mask_all,
    rows_per_sim=174,
    num_simulations=6,
    thickness_mm=5.0,
):
    """
    Use calculate_median_depth_errors separately for each simulation,
    with erosion_pixels=1 and inner_percent=100.

    Inputs contain normalized fractions. Set thickness_mm=None to keep
    those units; otherwise, results are converted to millimetres.

    Returns
    -------
    per_simulation : list of dict
        MAE, MedAE, and RMSE for each simulation.
    mean_metrics : dict
        Arithmetic mean of each metric across simulations.
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

    scale = 1.0 if thickness_mm is None else thickness_mm
    unit = "" if thickness_mm is None else " mm"
    per_simulation = []

    for i in range(num_simulations):
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

