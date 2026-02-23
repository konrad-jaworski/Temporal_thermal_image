import os
import numpy as np
import torch

from helper_functions.helper_functions import TSR_extrapolation

extrapolator=TSR_extrapolation()

# Loop which will go through all .npz files in the input folder, and perform TSR with extrapolation on each of them saving the results in the output folder but output files are still .npz file with new extrapolated data

input_folder='/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data'
output_folder='/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated'

os.makedirs(output_folder, exist_ok=True)

# Helper: safe load .npz into a normal dict (keeps arrays + pickled python objects if present)
def load_npz_as_dict(path: str) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}

# Helper: save dict back to .npz
def save_dict_as_npz(path: str, d: dict):
    # np.savez can store arrays; for object arrays (e.g., meta as list) it will store as dtype=object
    np.savez(path, **d)

# Process all npz files
npz_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".npz")])

print(f"Found {len(npz_files)} .npz files in: {input_folder}")
print(f"Saving extrapolated files to: {output_folder}")

for i, fname in enumerate(npz_files, 1):
    in_path = os.path.join(input_folder, fname)
    out_path = os.path.join(output_folder, fname)

    try:
        data = load_npz_as_dict(in_path)

        # Expecting the extrapolator to return extrapolated data.
        # Common patterns:
        #   - returns just Y_extra
        #   - or returns (Y_hat, Y_extra, Y_centered)
        # We'll support both without guessing too much.
        Y_hat,_,Y_extra = extrapolator(data)

        # Convert torch -> numpy if needed
        if torch.is_tensor(Y_extra):
            Y_extra_np = Y_extra.detach().cpu().numpy()
        else:
            Y_extra_np = np.asarray(Y_extra)

        # Build output dict: keep everything, replace/augment data
        out = dict(data)

        # Keep original under a new key (so you can compare later)
        if "data" in out:
            out["data_original"] = out["data"]

        # Store extrapolated data under 'data'
        out["data"] = Y_extra_np

        # Optionally store reconstruction (if provided)
        if Y_hat is not None:
            if torch.is_tensor(Y_hat):
                out["data_reconstructed"] = Y_hat.detach().cpu().numpy()
            else:
                out["data_reconstructed"] = np.asarray(Y_hat)

        save_dict_as_npz(out_path, out)

        print(f"[{i}/{len(npz_files)}] OK: {fname} -> saved")

    except Exception as e:
        print(f"[{i}/{len(npz_files)}] ERROR on {fname}: {e}")

print("Done.")