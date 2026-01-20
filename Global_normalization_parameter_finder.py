import numpy as np
import glob

mins = []
maxs = []

for f in glob.glob(r"E:\Simulated_and_experimental_data\Synthetic_data\all_data/*.npz"):
    d = np.load(f)["data"]   # adapt key name
    mins.append(d.min())
    maxs.append(d.max())

T_min_global = float(np.min(mins))
T_max_global = float(np.max(maxs))

np.savez(r"normalization_params.npz",
         T_min=T_min_global,
         T_max=T_max_global)