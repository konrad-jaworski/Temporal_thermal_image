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

# ---------------------------------------------------------
# Output file depends on whether the scale is computed from
# cooling-only sequence or heating+cooling full sequence.
# ---------------------------------------------------------
sequence_mode = "cooling_only" if use_cooling_only else "heating_and_cooling"

output_file = rf"normalization_params_{sequence_mode}.npz"

print(f"Found {len(files)} training cubes")
print(f"Mode: {'cooling only' if use_cooling_only else 'heating + cooling / full sequence'}")
print(f"Output file: {output_file}")

if use_cooling_only:
    print(f"Cooling starts from frame: {cooling_frame}")

if len(files) == 0:
    raise RuntimeError(f"No files found in folder pattern: {train_folder}")


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
# PASS 1: input min/max scale
# =========================================================
# Non-log projected temperature values
temp_min = np.inf
temp_max = -np.inf
temp_abs_max = 0.0

# log1p-projected temperature values
temp_log1p_min = np.inf
temp_log1p_max = -np.inf
temp_log1p_abs_max = 0.0

for f in tqdm(files, desc="Pass 1: temperature and log1p min/max"):
    d = np.load(f)["data"].astype(np.float32)

    d = select_sequence(
        d,
        use_cooling_only=use_cooling_only,
        cooling_frame=cooling_frame,
    )

    # -----------------------------------------------------
    # Raw clipped input: max(DeltaT, 0)
    # -----------------------------------------------------
    d_for_input = transform_data(d, log_scaling=False)

    current_min = float(d_for_input.min())
    current_max = float(d_for_input.max())
    current_abs_max = float(np.abs(d_for_input).max())

    temp_min = min(temp_min, current_min)
    temp_max = max(temp_max, current_max)
    temp_abs_max = max(temp_abs_max, current_abs_max)

    # -----------------------------------------------------
    # log1p projected input: log1p(max(DeltaT, 0))
    # -----------------------------------------------------
    d_for_input_log1p = transform_data(d, log_scaling=True)

    current_log1p_min = float(d_for_input_log1p.min())
    current_log1p_max = float(d_for_input_log1p.max())
    current_log1p_abs_max = float(np.abs(d_for_input_log1p).max())

    temp_log1p_min = min(temp_log1p_min, current_log1p_min)
    temp_log1p_max = max(temp_log1p_max, current_log1p_max)
    temp_log1p_abs_max = max(temp_log1p_abs_max, current_log1p_abs_max)


# For your current pipeline, temperature is clipped to >= 0,
# so scale_temp should be equivalent to temp_max.
# Keeping the key name "scale" preserves compatibility with later code.
scale_temp = max(temp_abs_max, 1e-12)

# New log1p scale.
# This is the maximum of log1p(max(DeltaT, 0)).
scale_log1p = max(temp_log1p_abs_max, 1e-12)

print("\nComputed raw input minimum:", temp_min)
print("Computed raw input maximum:", temp_max)
print("Computed raw input max absolute scale:", scale_temp)

print("\nComputed log1p input minimum:", temp_log1p_min)
print("Computed log1p input maximum:", temp_log1p_max)
print("Computed log1p input max absolute scale:", scale_log1p)


# =========================================================
# PASS 2: derivative min/max and max-absolute scales
# =========================================================
scale_dt = 1.0
scale_dxx = 1.0
scale_dx = 1.0
scale_dtt = 1.0

dt_min = np.inf
dt_max = -np.inf
dxx_min = np.inf
dxx_max = -np.inf
dx_min = np.inf
dx_max = -np.inf
dtt_min = np.inf
dtt_max = -np.inf

dt_abs_max = 0.0
dxx_abs_max = 0.0
dx_abs_max = 0.0
dtt_abs_max = 0.0

if compute_derivative_scales:
    for f in tqdm(files, desc="Pass 2: derivative min/max"):
        d = np.load(f)["data"].astype(np.float32)

        d = select_sequence(
            d,
            use_cooling_only=use_cooling_only,
            cooling_frame=cooling_frame,
        )

        # ---------------------------------------------------------
        # Important:
        # Derivatives are computed on raw baseline-removed data,
        # not on d / scale_temp and not on log1p(d).
        #
        # This keeps derivative channels in raw physical scale.
        # ---------------------------------------------------------
        d_for_derivatives = np.maximum(d, 0.0)

        T, H, W = d_for_derivatives.shape

        for row in tqdm(range(H), leave=False):
            bscan = torch.from_numpy(d_for_derivatives[:, row, :]).float()

            dt = d1_dy(bscan)
            dxx = d2_dx2(bscan)
            dx = d1_dx(bscan)
            dtt = d2_dy2(bscan)

            # -----------------------------
            # True min/max values
            # -----------------------------
            dt_min = min(dt_min, float(dt.min()))
            dt_max = max(dt_max, float(dt.max()))

            dxx_min = min(dxx_min, float(dxx.min()))
            dxx_max = max(dxx_max, float(dxx.max()))

            dx_min = min(dx_min, float(dx.min()))
            dx_max = max(dx_max, float(dx.max()))

            dtt_min = min(dtt_min, float(dtt.min()))
            dtt_max = max(dtt_max, float(dtt.max()))

            # -----------------------------
            # Max-absolute values for scaling
            # -----------------------------
            dt_abs_max = max(dt_abs_max, float(dt.abs().max()))
            dxx_abs_max = max(dxx_abs_max, float(dxx.abs().max()))
            dx_abs_max = max(dx_abs_max, float(dx.abs().max()))
            dtt_abs_max = max(dtt_abs_max, float(dtt.abs().max()))

    # Keep these names unchanged because your later code expects them.
    # These are max-absolute scales, intended for x / scale_x normalization.
    scale_dt = max(dt_abs_max, 1e-12)
    scale_dxx = max(dxx_abs_max, 1e-12)
    scale_dx = max(dx_abs_max, 1e-12)
    scale_dtt = max(dtt_abs_max, 1e-12)

    print("\nComputed dT/dt min:", dt_min)
    print("Computed dT/dt max:", dt_max)
    print("Computed max |dT/dt|:", scale_dt)

    print("\nComputed d2T/dx2 min:", dxx_min)
    print("Computed d2T/dx2 max:", dxx_max)
    print("Computed max |d2T/dx2|:", scale_dxx)

    print("\nComputed dT/dx min:", dx_min)
    print("Computed dT/dx max:", dx_max)
    print("Computed max |dT/dx|:", scale_dx)

    print("\nComputed d2T/dt2 min:", dtt_min)
    print("Computed d2T/dt2 max:", dtt_max)
    print("Computed max |d2T/dt2|:", scale_dtt)


# =========================================================
# SAVE
# =========================================================
np.savez(
    output_file,

    # -----------------------------------------------------
    # Original keys preserved
    # -----------------------------------------------------
    scale=scale_temp,
    scale_dt=scale_dt,
    scale_dxx=scale_dxx,
    scale_dx=scale_dx,
    scale_dtt=scale_dtt,

    # -----------------------------------------------------
    # New key for log1p-scaled input
    # -----------------------------------------------------
    scale_log1p=scale_log1p,

    # -----------------------------------------------------
    # Raw clipped temperature diagnostics
    # -----------------------------------------------------
    temp_min=temp_min,
    temp_max=temp_max,
    temp_abs_max=temp_abs_max,

    # -----------------------------------------------------
    # log1p-projected temperature diagnostics
    # -----------------------------------------------------
    temp_log1p_min=temp_log1p_min,
    temp_log1p_max=temp_log1p_max,
    temp_log1p_abs_max=temp_log1p_abs_max,

    # -----------------------------------------------------
    # Derivative diagnostics
    # -----------------------------------------------------
    dt_min=dt_min,
    dt_max=dt_max,
    dt_abs_max=dt_abs_max,

    dxx_min=dxx_min,
    dxx_max=dxx_max,
    dxx_abs_max=dxx_abs_max,

    dx_min=dx_min,
    dx_max=dx_max,
    dx_abs_max=dx_abs_max,

    dtt_min=dtt_min,
    dtt_max=dtt_max,
    dtt_abs_max=dtt_abs_max,

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------
    log_scaling=log_scaling,
    use_cooling_only=use_cooling_only,
    cooling_frame=cooling_frame,
    sequence_mode=sequence_mode,
    derivative_scale_mode="raw_max_abs",
    temperature_scale_mode="raw_and_log1p_max_abs_after_clipping",
)

print("\nSaved normalization parameters.")
print(f"Saved file: {output_file}")