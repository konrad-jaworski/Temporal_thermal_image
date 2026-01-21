import os
import numpy as np
import glob

def extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    output_Y_folder,
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

    os.makedirs(output_X_folder, exist_ok=True)
    os.makedirs(output_Y_folder, exist_ok=True)
    os.makedirs(output_depth_folder, exist_ok=True)

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
            np.save(os.path.join(output_Y_folder, fname + ".npy"), Y)
            np.save(os.path.join(output_depth_folder, fname + ".npy"), depth_target)

            sample_counter += 1

    print(f"Done. Saved {sample_counter} row-wise samples.")


input_folder = r"E:\Simulated_and_experimental_data\Synthetic_data\all_data_same_length_no_base_add_noise_test"
output_X_folder = r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_test\data"
output_Y_folder = r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_test\detection"
output_depth_folder = r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_test\depth"

lower_bound = 180
upper_bound = 230

extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    output_Y_folder,
    output_depth_folder,
    lower_bound,
    upper_bound
)