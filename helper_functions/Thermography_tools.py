"""
Classical thermography processing tools.

TSR_fitting
    Thermographic Signal Reconstruction (TSR, Shepard). Every pixel's
    cooling curve is replaced by a low-order polynomial fit in log-log
    space. This removes temporal noise while keeping the shape of the curve,
    which carries the depth information.

find_heating_stop
    Finds the frame at which heating ends (the hottest frame on average),
    used to separate the heating and cooling phases of a sequence.

Sequences are numpy arrays of shape [T, H, W] (frames, rows, columns).
"""

import numpy as np
from matplotlib import pyplot as plt

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
    Thermographic Signal Reconstruction of a thermal sequence.

    For a semi-infinite body heated by a short pulse, the surface
    temperature falls as t^(-1/2), a straight line in log-log coordinates.
    A subsurface defect bends this line from the moment the heat reaches it.
    TSR fits, for every pixel independently,

        log(Delta T) = polynomial of order n in log(t)

    and replaces the measured curve by the fit. A low order follows the
    physical curve but not the noise.

    The same order n is used for all pixels. Orders are tried from
    min_order upwards; the first one whose reconstruction error is at most
    error_tolerance is kept. If none reaches the tolerance, the order with
    the lowest error is used.

    Parameters
    ----------
    data : array-like, [T, H, W]
        Temperature signal. All values must be finite and > 0 (the log is
        taken), so use the temperature rise above the baseline and only the
        frames where it is positive, typically the cooling phase.
    verbose : bool
        Print the error for every order tried and plot error vs. order.
    plot : bool
        Plot the original and reconstructed curves of 4 random pixels (all
        pixels if there are fewer than 4).
    time : array-like, [T], optional
        Time of every frame; positive and strictly increasing. Defaults to
        1, 2, ..., T (frame numbers). With real times, the t = 0 frame must
        be left out because log(0) is undefined.
    min_order, max_order : int
        Range of polynomial orders tried (both included). The order is also
        limited to T - 1, the highest order T points can determine.
    error_tolerance : float
        Accepted relative reconstruction error, measured on the temperature
        (not the log):
            ||reconstruction - data||_2 / ||data||_2
        e.g. 0.01 = 1 %.
    random_state : int, optional
        Seed for choosing the pixels shown when plot=True.

    Returns
    -------
    filtered_data : np.ndarray, [T, H, W]
        Reconstructed sequence, in the same units as `data`.
    """
    # float64 for the least-squares fit.
    data = np.asarray(data, dtype=np.float64)

    # Input checks.
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

    # Map log(t) linearly onto [-1, 1]. High powers of large numbers would
    # make the fit numerically unstable; on [-1, 1] they stay bounded.
    log_time = np.log(time)
    x = 2 * (log_time - log_time[0]) / (
        log_time[-1] - log_time[0]
    ) - 1

    # Divide by the global maximum so the values are <= 1. In log space
    # this is only a constant offset, which the constant term of the
    # polynomial absorbs, so the fit itself is not affected.
    # values     : data / scale, [T, H*W]; used to measure the error.
    # log_values : log(data / scale), [T, H*W]; what is fitted.
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
        # Chebyshev polynomials up to `order`, evaluated at x: design matrix
        # [T, order + 1]. They describe exactly the same polynomials as the
        # powers 1, x, x^2, ... but give a much better conditioned
        # least-squares problem. Every column of log_values is one pixel,
        # so a single lstsq call fits all pixels at once.
        design = np.polynomial.chebyshev.chebvander(x, order)
        coefficients = np.linalg.lstsq(
            design, log_values, rcond=None
        )[0]

        # Back from log space to temperature. A poor high-order fit can
        # overflow in exp; such an order simply gets an infinite error.
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

        # Stop at the lowest order that is good enough.
        if error <= error_tolerance:
            tolerance_met = True
            break

    if best_reconstruction is None:
        raise RuntimeError("All polynomial fits produced nonfinite errors.")

    # Undo the scaling and restore the [T, H, W] layout.
    filtered_data = (
        best_reconstruction.reshape(T, H, W) * scale
    )

    if verbose:
        status = "tolerance met" if tolerance_met else "best available fit"
        print(
            f"Selected order {selected_order}: {status}; "
            f"relative RMSE = {best_error:.3%}"
        )

        # Error vs. polynomial order, with the tolerance as a dashed line.
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
        # Visual check of the fit at a few random pixels.
        rng = np.random.default_rng(random_state)
        pixels = rng.choice(H * W, size=min(4, H * W), replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, pixel in zip(axes.flat, pixels):
            # Flat pixel index -> (row, column).
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

        # Hide unused panels when fewer than 4 pixels are plotted.
        for ax in axes.flat[len(pixels):]:
            ax.set_visible(False)

        fig.tight_layout()

    if verbose or plot:
        plt.show()

    return filtered_data


def find_heating_stop(data):
    """
    Index of the frame at which heating stops.

    While the specimen is heated its mean surface temperature rises; once
    the heat source is switched off it starts to fall. The frame with the
    highest mean temperature (averaged over all pixels) is therefore taken
    as the end of heating / start of cooling.

    Parameters
    ----------
    data : array-like, [T, H, W]
        Thermal sequence.

    Returns
    -------
    frame_index : int
        Index of the hottest frame on average. If several frames share the
        maximum, the first one is returned.
    """
    data = np.asarray(data)

    if data.ndim != 3 or any(size == 0 for size in data.shape):
        raise ValueError("data must be a nonempty array with shape (T, H, W).")

    # Mean temperature of every frame -> [T].
    mean_temperature = np.mean(data, axis=(1, 2))

    if not np.all(np.isfinite(mean_temperature)):
        raise ValueError("data must contain only finite values.")

    return int(np.argmax(mean_temperature))