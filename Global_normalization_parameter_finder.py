import numpy as np
import glob
import torch
from helper_functions.helper_functions import d1_dy, d2_dx2

train_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/*.npz"

all_temp_values = []
all_dt_values = []
all_dxx_values = []

# --------------------------------------------------
# First pass: compute temperature scale from log1p(T)
# --------------------------------------------------
for f in glob.glob(train_folder):
    d = np.load(f)["data"]   # shape: [T, H, W]

    # clamp negatives just in case
    d = np.maximum(d, 0.0)

    log_d = np.log1p(d)
    all_temp_values.append(log_d.ravel())

all_temp_values = np.concatenate(all_temp_values)
scale_temp = float(np.percentile(all_temp_values, 99.5))

print("Computed temperature scale:", scale_temp)

# --------------------------------------------------
# Second pass: compute derivative scales from
# normalized B-scans: X = log1p(T) / scale_temp
# --------------------------------------------------
for f in glob.glob(train_folder):
    d = np.load(f)["data"]   # shape: [T, H, W]
    d = np.maximum(d, 0.0)

    # log + normalize
    X = np.log1p(d) / scale_temp   # [T, H, W]

    T, H, W = X.shape

    # Each row in H gives one B-scan of shape [T, W]
    for row in range(H):
        bscan = X[:, row, :]   # [T, W]

        bscan_t = torch.from_numpy(bscan).float()

        dt = d1_dy(bscan_t)     # first derivative over time axis
        dxx = d2_dx2(bscan_t)   # second derivative over width axis

        all_dt_values.append(dt.abs().numpy().ravel())
        all_dxx_values.append(dxx.abs().numpy().ravel())

all_dt_values = np.concatenate(all_dt_values)
all_dxx_values = np.concatenate(all_dxx_values)

scale_dt = float(np.percentile(all_dt_values, 99.5))
scale_dxx = float(np.percentile(all_dxx_values, 99.5))

print("Computed dT/dt scale:", scale_dt)
print("Computed d2T/dx2 scale:", scale_dxx)

# --------------------------------------------------
# Save all scales
# --------------------------------------------------
np.savez(
    r"normalization_params.npz",
    scale=scale_temp,
    scale_dt=scale_dt,
    scale_dxx=scale_dxx
)