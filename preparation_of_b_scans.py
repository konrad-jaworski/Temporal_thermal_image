import os
import numpy as np
import glob

def extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    output_depth_folder,
    lower_bound,
    upper_bound
):
    """
    Extract row-wise B-scan data and corresponding targets from .npz files.

    For each row index i in [lower_bound, upper_bound):
      - Save data[:, i, :] as one training sample
      - Save mask[i] as classification target
      - Save mask[i] * depth_factor as regression target
    """

    # Creation of the specific folders
    os.makedirs(output_X_folder, exist_ok=True)
    os.makedirs(output_depth_folder, exist_ok=True)

    # Checking if folder contain npz file
    files = glob.glob(os.path.join(input_folder, "*.npz"))
    if not files:
        print("No .npz files found!")
        return

    print(f"Processing {len(files)} files...")

    sample_counter = 0

    for fpath in files:
        base_name = os.path.basename(fpath).replace(".npz", "")
        npz = np.load(fpath, allow_pickle=True)

        if not all(k in npz for k in ['data', 'mask', 'meta']):
            print(f"Skipping {base_name}: missing required keys")
            continue

        data = npz['data']      # [T, H, W]
        mask = npz['mask']      # [H]
        meta = npz['meta']

        # Creating depth factor which is inverted due to the naming convection describing amount of material removed.
        depth_factor = (100.0 - float(meta[40][1])) / 100.0

        for i in range(lower_bound, upper_bound):

            # --- Input (B-scan row) ---
            X = data[:, i, :]                   # [T, W]

            # --- Classification target ---
            Y = np.array(mask[i], dtype=np.uint8)

            # --- Regression target ---
            depth_target = np.array(
                mask[i] * depth_factor,
                dtype=np.float32
            )

            # unique filename per row
            fname = f"{base_name}_row_{i:04d}"

            np.save(os.path.join(output_X_folder, fname + ".npy"), X)
            np.save(os.path.join(output_depth_folder, fname + ".npy"), depth_target)

            sample_counter += 1

    print(f"Done. Saved {sample_counter} row-wise samples.")


input_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation"
output_X_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/data"
output_depth_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/depth"

lower_bound = 400
upper_bound = 500

extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    output_depth_folder,
    lower_bound,
    upper_bound
)