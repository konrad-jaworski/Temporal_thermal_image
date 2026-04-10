import os
import numpy as np
import glob

def extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound,
    upper_bound
):
    """
    Extract row-wise B-scan data and corresponding targets from .npz files.

    For each row index i in [lower_bound, upper_bound):
      - Save data[:, i, :] as one training sample
      - Save mask[i] * depth_factor as regression target
    """

    # Creation of the specific folders
    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_bscan_folder, exist_ok=True)

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
            depth_target = np.array(mask[i], dtype=np.float32)

            # unique filename per row
            fname = f"{base_name}_row_{i:04d}"

            np.save(os.path.join(output_bscan_folder, fname + ".npy"), X)
            np.save(os.path.join(output_depth_folder, fname + ".npy"), depth_target)

            sample_counter += 1

    print(f"Done. Saved {sample_counter} row-wise samples.")


input_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/validation"
output_bscan_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/validation_bscan_full"
output_depth_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/validation_mask_full"

lower_bound = 0
upper_bound = 512

extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound,
    upper_bound
)