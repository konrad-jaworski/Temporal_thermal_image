import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image



import numpy as np


def replace_negative_outliers_outside_peak_3d(
    data,
    threshold_sigma=6.0,
    percentile=5,
    peak_buffer_left=10,
    peak_buffer_right=30,
    verbose=True,
):
    """
    Detect and replace negative corrupted frames in a 3D thermal sequence.

    The input tensor must have time on the first axis:

        data.shape = [T, H, W]

    The method is designed for corrupted frames that appear as sudden negative
    drops. It avoids modifying the real heating peak by excluding a protected
    frame window around the global maximum of the mean temperature curve.

    Detection logic
    ---------------
    For every internal frame t, an expected frame is estimated from its direct
    temporal neighbours:

        expected[t] = 0.5 * (data[t - 1] + data[t + 1])

    Then the residual is computed:

        residual[t] = data[t] - expected[t]

    Negative corrupted frames produce strongly negative residuals. Therefore,
    the detection score is based only on the negative side of the residual:

        score[t] = -percentile(residual[t], percentile)

    A high score means that at least part of the frame is much colder than
    expected from neighbouring frames.

    Replacement logic
    -----------------
    Every detected corrupted frame is replaced by the mean of the nearest
    available non-corrupted frame before it and the nearest available
    non-corrupted frame after it:

        corrected[t] = 0.5 * (corrected[left] + corrected[right])

    This avoids copying only the left or right frame.

    Parameters
    ----------
    data : np.ndarray
        Input thermal tensor with shape [T, H, W].

    threshold_sigma : float
        Robust z-score threshold for outlier detection.
        Higher value means less sensitive detection.

    percentile : float
        Percentile of residual values used to detect localized negative drops.
        Lower value makes the method more sensitive to small corrupted regions.

    peak_buffer_left : int
        Number of frames before the detected heating peak excluded from
        correction.

    peak_buffer_right : int
        Number of frames after the detected heating peak excluded from
        correction.

    verbose : bool
        If True, prints detected peak and corrupted frame indices.

    Returns
    -------
    corrected : np.ndarray
        Corrected tensor with shape [T, H, W].

    bad_frames : list[int]
        List of detected and replaced frame indices.

    scores : np.ndarray
        Negative anomaly score for every frame. Boundary frames are NaN.

    robust_z : np.ndarray
        Robust z-score for every frame. Protected and boundary frames are NaN.

    peak_frame : int
        Detected heating peak frame index.
    """

    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D tensor [T, H, W], got shape {data.shape}")

    T, H, W = data.shape

    if T < 3:
        raise ValueError("At least 3 frames are required.")

    corrected = data.copy().astype(np.float32)

    # ------------------------------------------------------------------
    # Detect the global heating peak.
    #
    # The mean curve compresses every frame into one scalar value.
    # The peak frame is the frame with the largest mean temperature.
    # This region usually contains real high temporal curvature and should
    # not be modified by the outlier correction.
    # ------------------------------------------------------------------
    mean_curve = data.mean(axis=(1, 2))
    peak_frame = int(np.argmax(mean_curve))

    protected_start = max(1, peak_frame - peak_buffer_left)
    protected_end = min(T - 2, peak_frame + peak_buffer_right)

    protected_mask = np.zeros(T, dtype=bool)
    protected_mask[protected_start:protected_end + 1] = True

    # ------------------------------------------------------------------
    # Compute negative anomaly scores.
    #
    # Boundary frames are skipped because frame 0 has no previous frame and
    # frame T-1 has no next frame.
    # ------------------------------------------------------------------
    scores = np.full(T, np.nan, dtype=np.float32)

    for t in range(1, T - 1):
        expected = 0.5 * (data[t - 1] + data[t + 1])
        residual = data[t] - expected

        # Only negative deviations are important.
        # Example:
        #   residual percentile = -2.0  -> score = 2.0
        #   residual percentile =  0.1  -> score = -0.1
        neg_score = -np.percentile(residual, percentile)

        scores[t] = neg_score

    # ------------------------------------------------------------------
    # Estimate robust statistics only outside the protected peak region.
    #
    # Median and MAD are used instead of mean and standard deviation because
    # they are less sensitive to occasional corrupted frames.
    # ------------------------------------------------------------------
    valid_detection_frames = np.array(
        [t for t in range(1, T - 1) if not protected_mask[t]],
        dtype=int,
    )

    if len(valid_detection_frames) == 0:
        raise ValueError("No frames available for detection outside peak buffer.")

    valid_scores = scores[valid_detection_frames]

    median_score = np.nanmedian(valid_scores)
    mad_score = np.nanmedian(np.abs(valid_scores - median_score)) + 1e-12

    robust_z = np.full(T, np.nan, dtype=np.float32)
    robust_z[valid_detection_frames] = (
        0.6745 * (valid_scores - median_score) / mad_score
    )

    # ------------------------------------------------------------------
    # Select corrupted frames.
    #
    # Only frames outside the protected region can be selected.
    # ------------------------------------------------------------------
    bad_frames = valid_detection_frames[
        robust_z[valid_detection_frames] > threshold_sigma
    ].tolist()

    bad_set = set(bad_frames)

    # ------------------------------------------------------------------
    # Replace corrupted frames.
    #
    # For every corrupted frame t:
    #   1. Search left for nearest frame that is not corrupted.
    #   2. Search right for nearest frame that is not corrupted.
    #   3. Replace frame t with the mean of these two valid frames.
    #
    # If only one valid neighbour exists, the frame is copied from that side.
    # This fallback is needed only near boundaries or in unusual long corrupted
    # blocks.
    # ------------------------------------------------------------------
    for t in bad_frames:
        left = t - 1
        right = t + 1

        while left in bad_set and left > 0:
            left -= 1

        while right in bad_set and right < T - 1:
            right += 1

        left_is_valid = left >= 0 and left not in bad_set
        right_is_valid = right < T and right not in bad_set

        if left_is_valid and right_is_valid:
            corrected[t] = 0.5 * (corrected[left] + corrected[right])
        elif left_is_valid:
            corrected[t] = corrected[left]
        elif right_is_valid:
            corrected[t] = corrected[right]

    if verbose:
        print(f"Detected heating peak frame: {peak_frame}")
        print(f"Protected frame range: {protected_start} to {protected_end}")
        print(f"Detected corrupted frames: {bad_frames}")
        print(f"Number of replaced frames: {len(bad_frames)}")

        for t in bad_frames:
            print(
                f"Frame {t}: "
                f"score={scores[t]:.6f}, "
                f"robust_z={robust_z[t]:.2f}"
            )

    return corrected, bad_frames, scores, robust_z, peak_frame


def load_mat_array(path, key="imageArray"):
    mat = sio.loadmat(path)

    if key not in mat:
        available_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Key '{key}' not found in {path}. "
            f"Available keys: {available_keys}"
        )

    return np.asarray(mat[key])


def load_png_mask(path):
    """
    Load grayscale PNG mask as uint8 array.
    """
    return np.array(Image.open(path).convert("L"))


def fahrenheit_raw_to_celsius(data_raw):
    """
    Convert centi-Fahrenheit-like values to Celsius.

    Example:
        7600 -> 76.00 °F -> 24.44 °C

    """
    divisor = 100.0

    data_f = data_raw.astype(np.float32) / divisor
    data_c = (data_f - 32.0) * 5.0 / 9.0

    return data_c.astype(np.float32)


def convert_mask_to_removed_depth_normalized(mask, thickness_mm=5.0):
    pixel_to_distance_mm = {
        0: 0.0,
        51: 0.5,
        102: 1.0,
        153: 1.5,
        204: 2.0,
        255: 2.5,
    }

    mask = np.asarray(mask)
    mask_out = np.zeros(mask.shape, dtype=np.float32)

    unique_values = np.unique(mask).astype(int)

    for value in unique_values:
        if value not in pixel_to_distance_mm:
            raise ValueError(
                f"Unknown mask pixel value: {value}. "
                f"Expected one of {list(pixel_to_distance_mm.keys())}."
            )

        if value == 0:
            mask_out[mask == value] = 0.0
        else:
            distance_from_surface_mm = pixel_to_distance_mm[value]
            removed_depth_mm = thickness_mm - distance_from_surface_mm
            normalized_depth = removed_depth_mm / thickness_mm
            mask_out[mask == value] = normalized_depth

    return mask_out.astype(np.float32)


def preprocess_mat_dataset(data_folder, mask_folder, output_folder):
    """
    Data:
        .mat files with key 'imageArray', shape [H, W, T]

    Mask:
        .png grayscale files with same base name, shape [H, W]

    Output:
        .npz files with:
            data : [T, W, H], Celsius
            mask : [W, H], normalized removed-material depth
    """

    data_folder = Path(data_folder)
    mask_folder = Path(mask_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(data_folder.glob("*.mat"))

    if len(data_paths) == 0:
        raise FileNotFoundError(f"No .mat files found in data folder: {data_folder}")

    print(f"Found {len(data_paths)} data files.")

    for data_path in data_paths:
        name = data_path.stem
        mask_path = mask_folder / f"{name}.png"

        if not mask_path.exists():
            print(f"[SKIP] No matching PNG mask for {data_path.name}")
            continue

        data_raw = load_mat_array(data_path, key="imageArray")
        mask_raw = load_png_mask(mask_path)

        data_raw = np.squeeze(data_raw)
        mask_raw = np.squeeze(mask_raw)

        if data_raw.ndim != 3:
            raise ValueError(
                f"Expected data shape [H, W, T] in {data_path.name}, "
                f"but got shape {data_raw.shape}"
            )

        if mask_raw.ndim != 2:
            raise ValueError(
                f"Expected mask shape [H, W] in {mask_path.name}, "
                f"but got shape {mask_raw.shape}"
            )

        data_c = fahrenheit_raw_to_celsius(data_raw)

        # [H, W, T] -> [T, H, W] -> [T, W, H]
        data_out = np.moveaxis(data_c, -1, 0)
        data_out = np.transpose(data_out, (0, 2, 1))

        # [H, W] -> [W, H]
        mask_rot = mask_raw.T

        # Now we need to crop out the mask and data to ROI and remove outliers
        r_start=85
        r_end=-70

        c_start=45
        c_end=-45

        data_out=data_out[:,r_start:r_end,c_start:c_end]
        mask_rot=mask_rot[r_start:r_end,c_start:c_end]

        # Removal of outliers
        data_out,_,_,_,_ = replace_negative_outliers_outside_peak_3d(
        data_out,
        threshold_sigma=15.0,
        percentile=5,
        peak_buffer_left=15,
        peak_buffer_right=30,
        verbose=True,
        )

        mask_out = convert_mask_to_removed_depth_normalized(
            mask_rot,
            thickness_mm=5.0,
        )

        if data_out.shape[1:] != mask_out.shape:
            raise ValueError(
                f"Shape mismatch after transform for {name}: "
                f"data spatial shape = {data_out.shape[1:]}, "
                f"mask shape = {mask_out.shape}"
            )

        output_path = output_folder / f"{name}.npz"

        np.savez_compressed(
            output_path,
            data=data_out.astype(np.float32),
            mask=mask_out.astype(np.float32),
        )

        print(
            f"[OK] {name}.npz | "
            f"data {data_out.shape}, "
            f"mask {mask_out.shape}, "
            f"mask values {np.unique(mask_out)}"
        )

    print(f"\nDone. Output saved to: {output_folder}")

data_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/archive/data"
mask_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/archive/labels/automated_mask"
output_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/archive/labels/open_source_data_npz"

preprocess_mat_dataset(
    data_folder=data_folder,
    mask_folder=mask_folder,
    output_folder=output_folder,
)