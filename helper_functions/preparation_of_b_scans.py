"""
Cuts 3D thermal sequences into 2D B-scans and saves them with their depth
targets, in the format read by BScanDepthDataset (data/data_operators.py).

Input:
    A folder of .npz files, one per specimen / simulation, each with
      'data' : thermal sequence [T, H, W] (baseline-removed Delta T),
      'mask' : depth map [H, W] (0 = sound material, otherwise the defect
               depth); not needed for experimental data without ground truth.

Output:
    For every row (or column) of every sequence, two .npy files with the same
    name, "<sequence>_row_0042.npy" or "<sequence>_col_0042.npy":
      output_bscan_folder / name : B-scan, [T, W] for rows, [T, H] for columns
      output_depth_folder / name : depth target, [W] or [H], float32

A B-scan is the temperature along one line of the image followed over time.
The network predicts the depth profile along that line, so a whole 3D
sequence becomes H (or W) independent training samples.

The block at the bottom of the file sets the paths and runs the extraction
when the script is executed.
"""

import os
import glob
import numpy as np


def extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound=0,
    upper_bound=None,
    trim_width=None,
    experimental=False,
    scan_direction="rows",
):
    """
    Extracts B-scans and depth targets from every .npz file in a folder.

    For scan_direction="rows":
        X      = data[:, row, :]   -> [T, W]
        target = mask[row, :]      -> [W]

    For scan_direction="columns":
        X      = data[:, :, col]   -> [T, H]
        target = mask[:, col]      -> [H]

    Parameters
    ----------
    input_folder : str
        Folder with the .npz sequences.
    output_bscan_folder, output_depth_folder : str
        Where the B-scans and the depth targets are written. Created if
        they do not exist.
    lower_bound, upper_bound : int, int or None
        Only rows (or columns) with index in [lower_bound, upper_bound) are
        extracted. upper_bound=None means up to the last one. The same range
        is used for every file.
    trim_width : int or None
        Number of columns removed from both the left and the right edge of
        every sequence (and mask) before extraction, e.g. to cut away the
        specimen border. None or 0 keeps the full width.
    experimental : bool
        True for measured data without ground truth. No 'mask' is required
        and an all-zero target is saved for each B-scan, so the dataset
        class (which expects a target file for every B-scan) can still load
        the samples for inference.
    scan_direction : {"rows", "columns"}
        Whether B-scans are taken along image rows or image columns.
    """

    # Validate the arguments before any file is read or written.
    if scan_direction not in {"rows", "columns"}:
        raise ValueError(
            'scan_direction must be either "rows" or "columns".'
        )

    trim = 0 if trim_width is None else trim_width

    if not isinstance(trim, (int, np.integer)) or trim < 0:
        raise ValueError(
            "trim_width must be None or a non-negative integer."
        )

    # Sorted so the files are always processed in the same order.
    files = sorted(glob.glob(os.path.join(input_folder, "*.npz")))

    if not files:
        print("No .npz files found!")
        return

    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_bscan_folder, exist_ok=True)

    # Simulated data must carry a ground-truth mask; experimental data only
    # the thermal sequence.
    required_keys = {"data"}

    if not experimental:
        required_keys.add("mask")

    print(
        f"Processing {len(files)} files along {scan_direction}..."
    )

    sample_counter = 0

    for fpath in files:
        # File name without extension; used as prefix of every sample saved
        # from this sequence, so each B-scan can be traced back to it.
        base_name = os.path.splitext(os.path.basename(fpath))[0]

        # The arrays are read inside the `with` block so the file is closed
        # right after loading. Files without the required keys are skipped
        # with a message rather than stopping the whole run.
        with np.load(fpath, allow_pickle=True) as npz:
            if not required_keys.issubset(npz.files):
                print(
                    f"Skipping {base_name}: missing required keys"
                )
                continue

            data = npz["data"]

            if experimental:
                mask = None
            else:
                mask = npz["mask"]

        # Shape checks: the sequence must be 3D and non-empty, and the mask
        # must cover exactly the same image area.
        if data.ndim != 3 or 0 in data.shape:
            raise ValueError(
                f"{base_name}: data must have shape [T, H, W]."
            )

        if mask is not None and mask.shape != data.shape[1:]:
            raise ValueError(
                f"{base_name}: mask shape {mask.shape} does not "
                f"match data spatial shape {data.shape[1:]}."
            )

        # Remove `trim` columns on the left and on the right, from the
        # sequence and the mask alike. Only the width is trimmed, never the
        # height.
        if trim:
            if 2 * trim >= data.shape[2]:
                raise ValueError(
                    f"{base_name}: trim_width removes the entire width."
                )

            data = data[:, :, trim:-trim]

            if mask is not None:
                mask = mask[:, trim:-trim]

        # In data [T, H, W], axis 1 indexes rows and axis 2 indexes columns.
        scan_axis = 1 if scan_direction == "rows" else 2
        number_of_scans = data.shape[scan_axis]

        stop = (
            number_of_scans
            if upper_bound is None
            else upper_bound
        )

        if not 0 <= lower_bound < stop <= number_of_scans:
            raise ValueError(
                f"{base_name}: bounds must satisfy "
                f"0 <= lower_bound < upper_bound <= "
                f"{number_of_scans}."
            )

        for i in range(lower_bound, stop):

            # Take one line of the image over all frames, together with the
            # matching line of the depth mask.
            if scan_direction == "rows":
                X = data[:, i, :]       # [T, W]
                target = (
                    None
                    if experimental
                    else mask[i, :]
                )
                label = "row"

            else:
                X = data[:, :, i]       # [T, H]
                target = (
                    None
                    if experimental
                    else mask[:, i]
                )
                label = "col"

            # Placeholder target of zeros for experimental data (see the
            # `experimental` parameter above).
            if experimental:
                depth_target = np.zeros(
                    X.shape[1],
                    dtype=np.float32,
                )
            else:
                depth_target = np.asarray(
                    target,
                    dtype=np.float32,
                )

            # Zero-padded index so the files sort in scan order.
            fname = f"{base_name}_{label}_{i:04d}.npy"

            # The B-scan is saved in the dtype of the source data; the
            # dataset class converts it to float32 when loading.
            np.save(
                os.path.join(output_bscan_folder, fname),
                X,
            )

            np.save(
                os.path.join(output_depth_folder, fname),
                depth_target,
            )

            sample_counter += 1

    print(
        f"Done. Saved {sample_counter} "
        f"{scan_direction}-wise samples."
    )


# Paths and settings for the run. The current values cut the test split of
# the open-source dataset into column-wise B-scans.
input_folder = r"/home/jaworskj/projects/thermal_B_scan/open_source_dataset/testing"
output_bscan_folder = r"/home/jaworskj/projects/thermal_B_scan/open_source_dataset/testing/data_bscans_columns"
output_depth_folder = r"/home/jaworskj/projects/thermal_B_scan/open_source_dataset/testing/data_masks_columns"

lower_bound = 0
upper_bound = None

extract_rowwise_bscan_and_targets(
    input_folder,
    output_bscan_folder,
    output_depth_folder,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    trim_width=None,
    experimental=False,
    scan_direction="columns",
)