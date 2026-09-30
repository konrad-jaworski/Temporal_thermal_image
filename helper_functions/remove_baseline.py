"""
Converts raw thermal sequences into temperature-rise sequences (Delta T).

A raw infrared recording contains the absolute temperature of the specimen,
which also includes the ambient temperature and small differences in
emissivity and initial temperature across the surface. Defect detection
only needs the temperature change caused by the heating. For every pixel
the average of the first frames, recorded before the heating starts, is
taken as its baseline and subtracted from the whole sequence:

    Delta T(t, y, x) = T(t, y, x) - mean(T(0 .. baseline_frames-1, y, x))

Optionally the result is also converted from centi-Kelvin to Celsius,
cropped in time and clipped to non-negative values.

Input:
    A folder of .npz files with a 'data' key holding the sequence [T, H, W].
Output:
    Files with the same names in output_folder, with every original key
    copied and 'data' replaced by Delta T. These files are the input of
    preparation_of_b_scans.py and Global_normalization_parameter_finder.py.

The block at the bottom of the file sets the paths and runs the
preprocessing when the script is executed.
"""

import os
import numpy as np
import glob

def preprocess_deltaT(
    input_folder,
    output_folder,
    baseline_frames=4,
    convert_to_C=False,
    shift_to=None,
    cliping=False
):
    """
    Subtracts the per-pixel baseline from every .npz sequence in a folder.

    Parameters
    ----------
    input_folder : str
        Folder with the raw .npz sequences.
    output_folder : str
        Where the processed files are written (created if missing).
    baseline_frames : int
        Number of frames at the start of the recording, before heating,
        that are averaged to give the baseline of each pixel.
    convert_to_C : bool
        If True, the raw values are treated as centi-Kelvin (the camera
        output format) and converted to degrees Celsius before the baseline
        is computed.
    shift_to : int or None
        If given, frames before this index are dropped after the baseline
        subtraction, so the sequence starts at the same moment as in the
        simulations. The baseline is still computed from the original first
        frames.
    cliping : bool
        If True, negative Delta T values (noise below the baseline) are set
        to 0.

    Every key of the input file is copied to the output file; only 'data'
    is replaced. Files without a 'data' key are skipped.
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

        data_tr = data_npz['data']  # [T, H, W]

        # Camera values in centi-Kelvin -> Kelvin (/100) -> Celsius (-273.15).
        # Only the /100 changes Delta T; the offset cancels out in the
        # baseline subtraction.
        if convert_to_C:
            data_tr = data_tr / 100 - 273.15

        if baseline_frames > data_tr.shape[0]:
            raise ValueError(f"{fname}: baseline_frames={baseline_frames} > sequence length={data_tr.shape[0]}")

        # Baseline of every pixel: its mean over the first baseline_frames
        # frames. Averaging several frames reduces the effect of camera
        # noise on the baseline. T0 has shape [H, W].
        T0 = data_tr[:baseline_frames, :, :].mean(axis=0)

        # Broadcasting subtracts T0 from every frame.
        deltaT = data_tr - T0

        # Drop the first shift_to frames so the time axis lines up with the
        # simulated sequences.
        if shift_to is not None:
            deltaT=deltaT[shift_to:,:,:]

        # Values below 0 are noise around the baseline, not a physical
        # temperature rise.
        if cliping==True:
            deltaT = np.clip(deltaT, a_min=0, a_max=None)

        # Copy every original key (e.g. the mask or recording metadata) and
        # replace only the thermal data.
        save_dict = {key: data_npz[key] for key in data_npz.files}
        save_dict['data'] = deltaT

        out_path = os.path.join(output_folder, fname)
        np.savez(out_path, **save_dict)

    print(f"Preprocessing done. Files saved to {output_folder}")

# Paths and settings for the run. baseline_frames is the number of frames
# recorded before the heating starts in these measurements.
input_folder = r""
output_folder = r""
baseline_frames = 20

preprocess_deltaT(input_folder, output_folder,
                  baseline_frames,
                  convert_to_C=True,
                  shift_to=25,
                  cliping=False)