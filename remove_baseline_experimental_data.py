import os
import numpy as np
import glob

# Simple function to convert raw data to degrees Celsius, if needed accomodate to your specific data format
def convert_to_degC(data):
    return data / 100.0 - 273.15

def preprocess_deltaT(
    input_folder, 
    output_folder, 
    baseline_frames=12,
    conversion_to_degC=True
):
    """
    Preprocess all .npz files: compute baseline from first `baseline_frames` frames,
    subtract it from sequence, optionally add noise, and save new .npz files in output_folder
    preserving all original keys.
    """
    os.makedirs(output_folder, exist_ok=True) # create output folder if it doesn't exist

    files = glob.glob(os.path.join(input_folder, "*.npz")) # get list of all .npz files in input folder
    if not files:
        print("No .npz files found in input folder!")
        return

    print(f"Processing {len(files)} files...")

    for fpath in files:
        fname = os.path.basename(fpath)
        data_npz = np.load(fpath, allow_pickle=True)
        
        if 'data' not in data_npz: # check if 'data' key exists
            print(f"Skipping {fname}: no 'data' key found")
            continue

        data_tr = data_npz['data']  # [T,H,W] shape

        if conversion_to_degC:
            data_tr = convert_to_degC(data_tr) # convert to degrees Celsius if needed

        if baseline_frames > data_tr.shape[0]: # check if baseline_frames is greater than sequence length
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

input_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Two_real_samples/thick_sample_10s"
output_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Two_real_samples/thick_sample_10s_remove_base"
baseline_frames = 12  # you can change this

preprocess_deltaT(input_folder, output_folder, baseline_frames)