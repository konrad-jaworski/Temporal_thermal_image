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
    Extract B-scans from data with shape [T, H, W].

    scan_direction="rows":
        X = data[:, row, :]       -> [T, W]
        target = mask[row, :]     -> [W]

    scan_direction="columns":
        X = data[:, :, column]    -> [T, H]
        target = mask[:, column]  -> [H]

    upper_bound=None processes all available rows or columns.
    """

    if scan_direction not in {"rows", "columns"}:
        raise ValueError(
            'scan_direction must be either "rows" or "columns".'
        )

    trim = 0 if trim_width is None else trim_width

    if not isinstance(trim, (int, np.integer)) or trim < 0:
        raise ValueError(
            "trim_width must be None or a non-negative integer."
        )

    files = sorted(glob.glob(os.path.join(input_folder, "*.npz")))

    if not files:
        print("No .npz files found!")
        return

    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_bscan_folder, exist_ok=True)

    required_keys = {"data"}

    if not experimental:
        required_keys.add("mask")

    print(
        f"Processing {len(files)} files along {scan_direction}..."
    )

    sample_counter = 0

    for fpath in files:
        base_name = os.path.splitext(os.path.basename(fpath))[0]

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

        if data.ndim != 3 or 0 in data.shape:
            raise ValueError(
                f"{base_name}: data must have shape [T, H, W]."
            )

        if mask is not None and mask.shape != data.shape[1:]:
            raise ValueError(
                f"{base_name}: mask shape {mask.shape} does not "
                f"match data spatial shape {data.shape[1:]}."
            )

        # Remove columns from the left and right sides.
        if trim:
            if 2 * trim >= data.shape[2]:
                raise ValueError(
                    f"{base_name}: trim_width removes the entire width."
                )

            data = data[:, :, trim:-trim]

            if mask is not None:
                mask = mask[:, trim:-trim]

        # Axis 1 contains rows; axis 2 contains columns.
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

            fname = f"{base_name}_{label}_{i:04d}.npy"

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