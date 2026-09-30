import numpy as np
from matplotlib import pyplot as plt
import torch
from pathlib import Path
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi

def TSR_fitting(
    data,
    verbose=True,
    plot=True,
    *,
    time=None,
    min_order=1,
    max_order=10,
    error_tolerance=0.01,
    random_state=None,
):
    """
    Fit thermographic data using Thermographic Signal Reconstruction (TSR).

    Each pixel is fitted independently:
        log(temperature) = polynomial(log(time))

    A common polynomial order is selected for the entire sequence: the
    lowest order satisfying the relative reconstruction RMSE tolerance.

    Parameters
    ----------
    data : array-like, shape (T, H, W)
        Positive, finite temperature values, preferably temperature rise
        above the baseline.
    verbose : bool
        Print reconstruction errors and plot error versus polynomial order.
    plot : bool
        Compare original and reconstructed curves at four random pixels,
        or all pixels if fewer than four are available.
    time : array-like, shape (T,), optional
        Positive, strictly increasing sampling times. Defaults to 1, ..., T.
        For physical sampling times, exclude the t=0 frame before fitting.
    min_order, max_order : int
        Inclusive polynomial-order search range.
    error_tolerance : float
        Maximum relative RMSE in the original temperature scale:
            ||reconstruction - data||_2 / ||data||_2
        For example, 0.01 means 1%.
    random_state : int, optional
        Random seed for selecting plotted pixels.

    Returns
    -------
    filtered_data : ndarray, shape (T, H, W)
        Reconstructed sequence. If no order meets the tolerance, returns
        the reconstruction with the lowest measured error.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must be a nonempty array with shape (T, H, W).")
    if not np.all(np.isfinite(data)) or np.any(data <= 0):
        raise ValueError(
            "Log-log TSR requires finite, strictly positive data. "
            "Select the positive temperature-rise interval before fitting."
        )
    if (
        not isinstance(min_order, (int, np.integer))
        or not isinstance(max_order, (int, np.integer))
        or not 0 <= min_order <= max_order
    ):
        raise ValueError("Require integer orders: 0 <= min_order <= max_order.")
    if not np.isfinite(error_tolerance) or error_tolerance < 0:
        raise ValueError("error_tolerance must be finite and nonnegative.")

    T, H, W = data.shape
    if T < 2 or min_order >= T:
        raise ValueError("Require T >= 2 and min_order < T.")

    if time is None:
        time = np.arange(1, T + 1, dtype=np.float64)
    else:
        time = np.asarray(time, dtype=np.float64)

    if (
        time.shape != (T,)
        or not np.all(np.isfinite(time))
        or np.any(time <= 0)
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "time must have shape (T,) and be finite, positive, "
            "and strictly increasing."
        )

    # Normalize log(time) to [-1, 1] for numerical stability.
    log_time = np.log(time)
    x = 2 * (log_time - log_time[0]) / (
        log_time[-1] - log_time[0]
    ) - 1

    # Scaling before taking logs does not change the polynomial model.
    scale = np.max(data)
    values = (data / scale).reshape(T, -1)
    log_values = np.log(data.reshape(T, -1)) - np.log(scale)
    reference_norm = np.linalg.norm(values)

    orders = []
    errors = []
    best_error = np.inf
    selected_order = None
    best_reconstruction = None
    tolerance_met = False

    for order in range(min_order, min(max_order, T - 1) + 1):
        # A Chebyshev basis spans the same polynomial space while improving
        # conditioning. All pixels are fitted simultaneously.
        design = np.polynomial.chebyshev.chebvander(x, order)
        coefficients = np.linalg.lstsq(
            design, log_values, rcond=None
        )[0]

        with np.errstate(over="ignore", invalid="ignore"):
            reconstruction = np.exp(design @ coefficients)
            error = np.linalg.norm(reconstruction - values) / reference_norm

        if not np.isfinite(error):
            error = np.inf

        orders.append(order)
        errors.append(error)

        if verbose:
            print(f"Order {order:2d}: relative RMSE = {error:.3%}")

        if error < best_error:
            best_error = error
            selected_order = order
            best_reconstruction = reconstruction

        if error <= error_tolerance:
            tolerance_met = True
            break

    if best_reconstruction is None:
        raise RuntimeError("All polynomial fits produced nonfinite errors.")

    filtered_data = (
        best_reconstruction.reshape(T, H, W) * scale
    )

    if verbose:
        status = "tolerance met" if tolerance_met else "best available fit"
        print(
            f"Selected order {selected_order}: {status}; "
            f"relative RMSE = {best_error:.3%}"
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(orders, 100 * np.asarray(errors), "o-")
        ax.axhline(
            100 * error_tolerance,
            color="red",
            linestyle="--",
            label=f"Tolerance: {error_tolerance:.1%}",
        )
        ax.set(
            xlabel="Polynomial order",
            ylabel="Relative reconstruction RMSE (%)",
            title="TSR reconstruction error",
            xticks=orders,
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

    if plot:
        rng = np.random.default_rng(random_state)
        pixels = rng.choice(H * W, size=min(4, H * W), replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, pixel in zip(axes.flat, pixels):
            row, col = divmod(int(pixel), W)
            ax.plot(time, data[:, row, col], label="Original", alpha=0.7)
            ax.plot(
                time,
                filtered_data[:, row, col],
                "--",
                label=f"TSR, order {selected_order}",
            )
            ax.set(
                xlabel="Time",
                ylabel="Temperature signal",
                title=f"Pixel ({row}, {col})",
            )
            ax.grid(alpha=0.3)
            ax.legend()

        for ax in axes.flat[len(pixels):]:
            ax.set_visible(False)

        fig.tight_layout()

    if verbose or plot:
        plt.show()

    return filtered_data


def find_heating_stop(data):
    """
    Parameters
    ----------
    data : array-like, shape (T, H, W)
        Thermographic sequence.

    Returns
    -------
    frame_index : int
        Index of the frame with the highest mean temperature.
        If multiple frames share the maximum, returns the first.
    """
    data = np.asarray(data)

    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must be a nonempty array with shape (T, H, W).")

    mean_temperature = np.mean(data, axis=(1, 2))

    if not np.all(np.isfinite(mean_temperature)):
        raise ValueError("data must contain only finite values.")

    return int(np.argmax(mean_temperature))


def PCT(data, n_components):
    """
    Principal Component Thermography.

    Parameters
    ----------
    data : np.ndarray
        Thermal sequence with shape [T, H, W].
    n_components : int
        Number of spatial EOF components to return.

    Returns
    -------
    eofs : np.ndarray
        Spatial EOF maps with shape [n_components, H, W].
        Each eofs[i] is directly plottable as a 2D image.
    """
    data = np.asarray(data, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(
            f"Expected data with shape [T, H, W], got {data.shape}."
        )

    if not np.isfinite(data).all():
        raise ValueError("Data contains NaN or infinite values.")

    time, height, width = data.shape
    maximum_components = min(time, height * width)

    if not 1 <= n_components <= maximum_components:
        raise ValueError(
            f"n_components must be between 1 and "
            f"{maximum_components}."
        )

    # [T, H, W] → [T, H*W]
    matrix = data.reshape(time, height * width)

    # Remove each pixel's temporal mean.
    matrix = matrix - matrix.mean(axis=0, keepdims=True)

    # Spatial modes are contained in Vt.
    _, _, Vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    # [components, H*W] → [components, H, W]
    eofs = Vt[:n_components].reshape(
        n_components,
        height,
        width,
    )

    return eofs


def PPT(data, fps):
    """
    Pulsed Phase Thermography using a temporal Fourier transform.

    Parameters
    ----------
    data : array-like, shape (T, H, W)
        Real, finite thermographic sequence with uniformly spaced frames.
    fps : float
        Camera sampling rate in frames per second.

    Returns
    -------
    amplitude : ndarray, shape (F, H, W)
        Single-sided amplitude spectrum, normalized by T.
        Positive-frequency amplitudes are doubled, except the Nyquist bin.
        Units match the input temperature units.
    phase : ndarray, shape (F, H, W)
        Phase in radians, between -pi and pi.
        Phase is undefined wherever the Fourier amplitude is zero.
    frequencies : ndarray, shape (F,)
        Nonnegative frequency bins in Hz, including the DC bin at 0 Hz.
        F = T // 2 + 1.

    Notes
    -----
    No windowing, detrending, or baseline subtraction is applied.
    Phase is referenced to the first frame of the supplied sequence.
    """
    data = np.asarray(data)

    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must be a nonempty array with shape (T, H, W).")
    if np.iscomplexobj(data):
        raise ValueError("data must contain real temperature values.")

    data = data.astype(np.float64, copy=False)

    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite values.")
    if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive, finite scalar.")

    T = data.shape[0]
    if T < 2:
        raise ValueError("At least two frames are required.")

    # Transform each pixel's time history.
    spectrum = np.fft.rfft(data, axis=0)
    frequencies = np.fft.rfftfreq(T, d=1.0 / fps)

    amplitude = np.abs(spectrum) / T
    phase = np.angle(spectrum)

    # Convert to a single-sided amplitude spectrum.
    if T % 2 == 0:
        amplitude[1:-1] *= 2  # Exclude DC and Nyquist.
    else:
        amplitude[1:] *= 2   # Odd T has no Nyquist bin.

    return amplitude, phase, frequencies


