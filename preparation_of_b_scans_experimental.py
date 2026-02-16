import os
import numpy as np
import glob

def center_frame(data,trimming_val):
    """
    Simple function which will crop the frame to 512 pixels, depending on triming_val
    """

    return data[:,:,trimming_val:-trimming_val]

def extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    lower_bound,
    upper_bound,
    trim_width=True,
    trim_val=64
):
    """
    Extract row-wise B-scan data and corresponding targets from .npz files.

    For each row index i in [lower_bound, upper_bound):
      - Save data[:, i, :] as one sample
    """

    os.makedirs(output_X_folder, exist_ok=True)
    
    files = glob.glob(os.path.join(input_folder, "*.npz"))
    if not files:
        print("No .npz files found!")
        return

    print(f"Processing {len(files)} files...")

    sample_counter = 0

    for fpath in files:
        base_name = os.path.basename(fpath).replace(".npz", "")
        npz = np.load(fpath, allow_pickle=True)

        if not all(k in npz for k in ['data']):
            print(f"Skipping {base_name}: missing required keys")
            continue

        data = npz['data']      # [T, H, W]

        if trim_width:
            data=center_frame(data,trim_val)

        for i in range(lower_bound, upper_bound):

            # --- Input (B-scan row) ---
            X = data[:, i, :]                   # [T, W]

            # unique filename per row
            fname = f"{base_name}_row_{i:04d}"

            np.save(os.path.join(output_X_folder, fname + ".npy"), X)
    
            sample_counter += 1

    print(f"Done. Saved {sample_counter} row-wise samples.")


input_folder = r"/Volumes/KINGSTON/Models_lc_and_test_data/all_data_same_length_no_base_add_noise_test/Test_sample_sim_4mm"
output_X_folder = r"/Volumes/KINGSTON/Models_lc_and_test_data/all_data_same_length_no_base_add_noise_test/Test_sample_sim_4mm"

lower_bound = 0
upper_bound = 511

extract_rowwise_bscan_and_targets(
    input_folder,
    output_X_folder,
    lower_bound,
    upper_bound,
    trim_width=False
)