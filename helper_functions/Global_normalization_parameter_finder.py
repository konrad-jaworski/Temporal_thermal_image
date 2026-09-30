"""
Computes the global normalisation constant for the B-net input.

The network is not fed raw temperatures. Each B-scan is divided by one fixed
number, `scale`, which is the largest temperature rise (Delta T, after
baseline removal) found anywhere in the training set. Using one constant for
the whole dataset, instead of normalising every sample separately, keeps the
relative amplitude between samples: a deep defect gives a weaker thermal
response than a shallow one, and that difference has to survive
normalisation.

Input:
    Baseline-removed thermal sequences (.npz files with the key 'data',
    shape [T, H, W]), as produced by remove_baseline.py.

Output:
    normalization_params_experimental_<sequence_mode>.npz, written in the
    current working directory, with a single key 'scale'. It is read by
    BScanDepthDataset (data/data_operators.py).

The script runs top to bottom when executed; there is no main() function.
"""

import numpy as np
import glob
from tqdm import tqdm


# Only the training split is used, so no information from the validation or
# test data leaks into the normalisation.
train_folder = r"/home/jaworskj/projects/thermal_B_scan/open_source_dataset/training/*.npz"
files = sorted(glob.glob(train_folder))

# Settings.
# use_cooling_only / cooling_frame: compute the maximum on frames
#   [cooling_frame:] only. With cooling_frame = 0 the whole sequence is used.
use_cooling_only = True
cooling_frame = 0

# The sequence mode is written into the output file name, so the scales for
# cooling-only and full sequences can be stored side by side.
sequence_mode = "cooling_only" if use_cooling_only else "heating_and_cooling"

output_file = rf"normalization_params_experimental_{sequence_mode}.npz"

print(f"Found {len(files)} training cubes")
print(f"Mode: {'cooling only' if use_cooling_only else 'heating + cooling / full sequence'}")
print(f"Output file: {output_file}")

if use_cooling_only:
    print(f"Cooling starts from frame: {cooling_frame}")

if len(files) == 0:
    raise RuntimeError(f"No files found in folder pattern: {train_folder}")


def select_sequence(d, use_cooling_only=False, cooling_frame=0):
    """
    Returns either the full sequence or only its cooling part.

    Parameters
    ----------
    d : np.ndarray
        Thermal sequence, shape [T, H, W].
    use_cooling_only : bool
        If True, frames before `cooling_frame` are dropped.
    cooling_frame : int
        Index of the first frame after the heat source is switched off.

    Returns
    -------
    np.ndarray
        [T', H, W] with T' = T - cooling_frame if use_cooling_only,
        otherwise the input unchanged.
    """

    if use_cooling_only:
        d = d[cooling_frame:, :, :]

    return d


# Every training sequence is loaded one at a time (the full set does not have
# to fit in memory) and the running maximum of Delta T is updated.
temp_max = -np.inf

for f in tqdm(files, desc="Maximum temperature rise"):
    d = np.load(f)["data"].astype(np.float32)

    d = select_sequence(
        d,
        use_cooling_only=use_cooling_only,
        cooling_frame=cooling_frame,
    )

    temp_max = max(temp_max, float(d.max()))


# The lower bound of 1e-12 only protects against division by zero if the
# data contained no positive temperature rise at all.
scale = max(temp_max, 1e-12)

print("\nMaximum temperature rise (scale):", scale)

np.savez(output_file, scale=scale)

print(f"Saved file: {output_file}")