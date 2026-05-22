import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image


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

        # Now we need to crop out the mask and data to ROI

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