import os
import numpy as np
import glob

def extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound,
    upper_bound,
    trim_width=None,
    experimental=False
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


        if experimental:
            if not all(k in npz for k in ['data', 'meta']):
                print(f"Skipping {base_name}: missing required keys")
                continue

            if trim_width is not None:
                data=npz['data'][:, :, trim_width:-trim_width]  # Trim width if specified
            else:
                data = npz['data']      # [T, H, W]
            
            for i in range(lower_bound, upper_bound):

                # --- Input (B-scan row) ---
                X = data[:, i, :]                   # [T, W]

                # --- Classification target ---
                depth_target = np.zeros_like(X[0,:], dtype=np.float32) # dummy target to learn backgrond , only subset will be selected for training.

                # unique filename per row
                fname = f"{base_name}_row_{i:04d}"

                np.save(os.path.join(output_bscan_folder, fname + ".npy"), X)
                np.save(os.path.join(output_depth_folder, fname + ".npy"), depth_target)

                sample_counter += 1
        else:    
            if not all(k in npz for k in ['data', 'mask', 'meta']):
                print(f"Skipping {base_name}: missing required keys")
                continue

        
            if trim_width is not None:
                data=npz['data'][:, :, trim_width:-trim_width]  # Trim width if specified
                mask=npz['mask'][:,trim_width:-trim_width]  # Mask width is also trimmed accordingly
            else:
                data = npz['data']      # [T, H, W]
                mask = npz['mask']      # [H]
            
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


input_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/Experimental_sample/exp_rb"
output_bscan_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/Experimental_sample/exp_bscans"
output_depth_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/Experimental_sample/exp_masks"

lower_bound = 0
upper_bound = 480

extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound,
    upper_bound,
    trim_width=64,
    experimental=True
)