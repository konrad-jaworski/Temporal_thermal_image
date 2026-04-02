import os
import numpy as np
import glob

def preprocess_deltaT(
    input_folder, 
    output_folder, 
    baseline_frames=4
):
    """
    Preprocess all .npz files: compute baseline from first `baseline_frames` frames,
    subtract it from sequence, optionally add noise, and save new .npz files in output_folder
    preserving all original keys.
    """
    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(os.path.join(input_folder, "*.npz"))
    if not files:
        print("No .npz files found in input folder!")
        return

    print(f"Processing {len(files)} files...")

    for fpath in files:
        fname = os.path.basename(fpath)
        data_npz = np.load(fpath, allow_pickle=True)
        
        if 'data' not in data_npz:
            print(f"Skipping {fname}: no 'data' key found")
            continue

        data_tr = data_npz['data']  # [T,H,W] shape
        if baseline_frames > data_tr.shape[0]:
            raise ValueError(f"{fname}: baseline_frames={baseline_frames} > sequence length={data_tr.shape[0]}")

        # compute baseline (average of first baseline_frames)
        T0 = data_tr[0:baseline_frames, :, :].mean(axis=0)

        # subtract baseline
        deltaT = data_tr - T0

        # Prepare dictionary to save
        save_dict = {key: data_npz[key] for key in data_npz.files}  # copy all keys
        save_dict['data'] = deltaT  # replace 'data' key

        # Save to new folder
        out_path = os.path.join(output_folder, fname)
        np.savez(out_path, **save_dict)
    
    print(f"Preprocessing done. Files saved to {output_folder}")

input_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/testing"
output_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/testing_rb"
baseline_frames = 1  # you can change this depending on how many initial frames you want to consider as baseline

preprocess_deltaT(input_folder, output_folder, baseline_frames)