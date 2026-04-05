import numpy as np
import glob
import torch
from tqdm import tqdm
from helper_functions import d1_dy, d2_dx2, d1_dx, d2_dy2

train_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/training_rb/*.npz"
files = sorted(glob.glob(train_folder))
log_scaling = False  # Set to True if you want to compute scales based on log-transformed data, False for raw data

print(f"Found {len(files)} training cubes")


# =========================================================
# Helper: percentile from histogram
# =========================================================
def percentile_from_hist(hist, bin_edges, q):
    cdf = np.cumsum(hist)
    target = q / 100.0 * cdf[-1]
    idx = np.searchsorted(cdf, target)
    idx = min(idx, len(bin_edges) - 2)
    return bin_edges[idx]


# =========================================================
# Helper: data transform
# =========================================================
def transform_data(d, log_scaling=True):
    d = np.maximum(d, 0.0)
    if log_scaling:
        d = np.log1p(d)
    return d


# =========================================================
# PASS 1: find temperature scale
# =========================================================
temp_max = 0.0

for f in tqdm(files, desc="Pass 1a/3: estimate temp max"):
    d = np.load(f)["data"]
    d = transform_data(d, log_scaling=log_scaling)
    temp_max = max(temp_max, float(d.max()))

n_bins_temp = 20000
temp_edges = np.linspace(0.0, temp_max, n_bins_temp + 1)
temp_hist = np.zeros(n_bins_temp, dtype=np.int64)

for f in tqdm(files, desc="Pass 1b/3: histogram temperature"):
    d = np.load(f)["data"]
    d = transform_data(d, log_scaling=log_scaling)

    h, _ = np.histogram(d, bins=temp_edges)
    temp_hist += h

scale_temp = float(percentile_from_hist(temp_hist, temp_edges, 99.5))
print("\nComputed temperature scale:", scale_temp)


# =========================================================
# PASS 2: estimate derivative maxima
# =========================================================
dt_max = 0.0
dxx_max = 0.0
dtt_max = 0.0
dx_max = 0.0

for f in tqdm(files, desc="Pass 2/3: estimate derivative maxima"):
    d = np.load(f)["data"]
    d = transform_data(d, log_scaling=log_scaling)

    X = d / scale_temp
    T, H, W = X.shape

    for row in tqdm(range(H), leave=False):
        bscan = torch.from_numpy(X[:, row, :]).float()

        dt = d1_dy(bscan).abs()
        dxx = d2_dx2(bscan).abs()
        dx = d1_dx(bscan).abs()
        dtt = d2_dy2(bscan).abs()

        dt_max = max(dt_max, float(dt.max()))
        dxx_max = max(dxx_max, float(dxx.max()))
        dx_max = max(dx_max, float(dx.max()))
        dtt_max = max(dtt_max, float(dtt.max()))

print("\nEstimated max |dT/dt|:", dt_max)
print("Estimated max |d2T/dx2|:", dxx_max)
print("Estimated max |d2T/dt2|:", dtt_max)
print("Estimated max |dT/dx|:", dx_max)


# =========================================================
# PASS 3: histogram accumulation
# =========================================================
n_bins_dt = 20000
n_bins_dxx = 20000
n_bins_dx = 20000
n_bins_dtt = 20000

dt_max = max(dt_max, 1e-12)
dxx_max = max(dxx_max, 1e-12)
dtt_max = max(dtt_max, 1e-12)
dx_max = max(dx_max, 1e-12)

dt_edges = np.linspace(0.0, dt_max, n_bins_dt + 1)
dxx_edges = np.linspace(0.0, dxx_max, n_bins_dxx + 1)
dtt_edges = np.linspace(0.0, dtt_max, n_bins_dtt + 1)
dx_edges = np.linspace(0.0, dx_max, n_bins_dx + 1)

dt_hist = np.zeros(n_bins_dt, dtype=np.int64)
dxx_hist = np.zeros(n_bins_dxx, dtype=np.int64)
dtt_hist = np.zeros(n_bins_dtt, dtype=np.int64)
dx_hist = np.zeros(n_bins_dx, dtype=np.int64)

for f in tqdm(files, desc="Pass 3/3: histogram derivatives"):
    d = np.load(f)["data"]
    d = transform_data(d, log_scaling=log_scaling)

    X = d / scale_temp
    T, H, W = X.shape

    for row in tqdm(range(H), leave=False):
        bscan = torch.from_numpy(X[:, row, :]).float()

        dt = d1_dy(bscan).abs().numpy()
        dxx = d2_dx2(bscan).abs().numpy()
        dtt = d2_dy2(bscan).abs().numpy()
        dx = d1_dx(bscan).abs().numpy()

        h_dt, _ = np.histogram(dt, bins=dt_edges)
        h_dxx, _ = np.histogram(dxx, bins=dxx_edges)
        h_dtt, _ = np.histogram(dtt, bins=dtt_edges)
        h_dx, _ = np.histogram(dx, bins=dx_edges)

        dt_hist += h_dt
        dxx_hist += h_dxx
        dtt_hist += h_dtt
        dx_hist += h_dx

scale_dt = float(percentile_from_hist(dt_hist, dt_edges, 99.5))
scale_dxx = float(percentile_from_hist(dxx_hist, dxx_edges, 99.5))
scale_dtt = float(percentile_from_hist(dtt_hist, dtt_edges, 99.5))
scale_dx = float(percentile_from_hist(dx_hist, dx_edges, 99.5))

print("\nComputed dT/dt scale:", scale_dt)
print("Computed d2T/dx2 scale:", scale_dxx)
print("Computed d2T/dt2 scale:", scale_dtt)
print("Computed dT/dx scale:", scale_dx)


# =========================================================
# SAVE
# =========================================================
np.savez(
    r"normalization_params.npz",
    scale=scale_temp,
    scale_dt=scale_dt,
    scale_dxx=scale_dxx,
    scale_dx=scale_dx,
    scale_dtt=scale_dtt,
    log_scaling=log_scaling
)

print("\nSaved normalization parameters to normalization_params.npz")