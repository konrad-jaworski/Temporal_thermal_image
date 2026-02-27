# import numpy as np
# import glob

# mins = []
# maxs = []

# for f in glob.glob(r"E:\Simulated_and_experimental_data\Synthetic_data\all_data_same_length_no_base_add_noise/*.npz"):
#     d = np.load(f)["data"]   # adapt key name
#     mins.append(d.min())
#     maxs.append(d.max())
    

# T_min_global = float(np.min(mins))
# T_max_global = float(np.max(maxs))

# np.savez(r"normalization_params.npz",
#          T_min=T_min_global,
#          T_max=T_max_global)

import numpy as np
import glob

train_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/*.npz"

all_values = []

for f in glob.glob(train_folder):
    d = np.load(f)["data"]  # adjust key if needed
    
    # if needed, clamp negatives
    d = np.maximum(d, 0.0)
    
    log_d = np.log1p(d)
    all_values.append(log_d.ravel())

# concatenate all flattened arrays
all_values = np.concatenate(all_values)

# compute global 99.5 percentile
scale = float(np.percentile(all_values, 99.5))

print("Computed global scale:", scale)

np.savez(r"normalization_params.npz",
         scale=scale)