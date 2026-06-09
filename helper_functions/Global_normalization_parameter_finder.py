import numpy as np
import glob
import torch
from tqdm import tqdm

from helper_functions import d1_dy, d2_dx2, d1_dx, d2_dy2


train_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/open_source_data_npz_rb/*.npz"
files = sorted(glob.glob(train_folder))

# =========================================================
# Configuration
# =========================================================
log_scaling = False

use_cooling_only = True
cooling_frame = 11

compute_derivative_scales = True

print(f"Found {len(files)} training cubes")
print(f"Mode: {'cooling only' if use_cooling_only else 'full sequence'}")

if use_cooling_only:
    print(f"Cooling starts from frame: {cooling_frame}")


# =========================================================
# Helper: sequence selection
# =========================================================
def select_sequence(d, use_cooling_only=False, cooling_frame=0):
    """
    Select full sequence or cooling-only sequence.

    Parameters
    ----------
    d : np.ndarray
        Thermal sequence with shape [T, H, W].

    use_cooling_only : bool
        If True, only frames from cooling_frame onward are used.

    cooling_frame : int
        First frame of cooling phase.

    Returns
    -------
    d : np.ndarray
        Selected sequence.
    """

    if use_cooling_only:
        d = d[cooling_frame:, :, :]

    return d


# =========================================================
# Helper: input transform
# =========================================================
def transform_data(d, log_scaling=False):
    """
    Transform baseline-removed thermal data.

    Negative values are clipped to zero because log1p is intended here
    for positive temperature rise.

    Parameters
    ----------
    d : np.ndarray
        Baseline-removed thermal sequence.

    log_scaling : bool
        If True, applies log1p after clipping to non-negative values.

    Returns
    -------
    d : np.ndarray
        Transformed sequence.
    """

    d = np.maximum(d, 0.0)

    if log_scaling:
        d = np.log1p(d)

    return d.astype(np.float32)


# =========================================================
# PASS 1: maximum input scale
# =========================================================
scale_temp = 0.0

for f in tqdm(files, desc="Pass 1: temperature maximum"):
    d = np.load(f)["data"].astype(np.float32)

    d = select_sequence(
        d,
        use_cooling_only=use_cooling_only,
        cooling_frame=cooling_frame,
    )

    d_for_input = transform_data(d, log_scaling=log_scaling)

    scale_temp = max(scale_temp, float(d_for_input.max()))

scale_temp = max(scale_temp, 1e-12)

print("\nComputed temperature/input scale:", scale_temp)


# =========================================================
# PASS 2: derivative maxima
# =========================================================
scale_dt = 1.0
scale_dxx = 1.0
scale_dx = 1.0
scale_dtt = 1.0

if compute_derivative_scales:
    dt_max = 0.0
    dxx_max = 0.0
    dx_max = 0.0
    dtt_max = 0.0

    for f in tqdm(files, desc="Pass 2: derivative maxima"):
        d = np.load(f)["data"].astype(np.float32)

        d = select_sequence(
            d,
            use_cooling_only=use_cooling_only,
            cooling_frame=cooling_frame,
        )

        # ---------------------------------------------------------
        # Important:
        # Derivatives are computed on raw baseline-removed data,
        # not on d / scale_temp.
        #
        # This keeps derivative channels in raw physical scale.
        # ---------------------------------------------------------
        d_for_derivatives = np.maximum(d, 0.0)

        T, H, W = d_for_derivatives.shape

        for row in tqdm(range(H), leave=False):
            bscan = torch.from_numpy(d_for_derivatives[:, row, :]).float()

            dt = d1_dy(bscan).abs()
            dxx = d2_dx2(bscan).abs()
            dx = d1_dx(bscan).abs()
            dtt = d2_dy2(bscan).abs()

            dt_max = max(dt_max, float(dt.max()))
            dxx_max = max(dxx_max, float(dxx.max()))
            dx_max = max(dx_max, float(dx.max()))
            dtt_max = max(dtt_max, float(dtt.max()))

    scale_dt = max(dt_max, 1e-12)
    scale_dxx = max(dxx_max, 1e-12)
    scale_dx = max(dx_max, 1e-12)
    scale_dtt = max(dtt_max, 1e-12)

    print("\nComputed max |dT/dt|:", scale_dt)
    print("Computed max |d2T/dx2|:", scale_dxx)
    print("Computed max |dT/dx|:", scale_dx)
    print("Computed max |d2T/dt2|:", scale_dtt)


# =========================================================
# SAVE
# =========================================================
np.savez(
    r"normalization_params_C_opendataset_max.npz",
    scale=scale_temp,
    scale_dt=scale_dt,
    scale_dxx=scale_dxx,
    scale_dx=scale_dx,
    scale_dtt=scale_dtt,
    log_scaling=log_scaling,
    use_cooling_only=use_cooling_only,
    cooling_frame=cooling_frame,
    derivative_scale_mode="raw_max",
    temperature_scale_mode="max",
)

print("\nSaved normalization parameters.")