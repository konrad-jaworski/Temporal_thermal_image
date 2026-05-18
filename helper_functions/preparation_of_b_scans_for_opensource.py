import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


# -------------------------------------------------------------------------
# Original physical meaning:
# distance from measured surface to defect face [mm]
#
# After preprocessing:
# removed_depth = 5.0 - surface_depth
# normalized_removed_depth = removed_depth / 5.0
# -------------------------------------------------------------------------

SURFACE_DEPTH_TO_NORM_REMOVED = {
    0.5: 0.9,
    1.0: 0.8,
    1.5: 0.7,
    2.0: 0.6,
    2.5: 0.5,
}

NORM_REMOVED_TO_SURFACE_DEPTH = {
    0.9: 0.5,
    0.8: 1.0,
    0.7: 1.5,
    0.6: 2.0,
    0.5: 2.5,
}

KNOWN_LEVELS = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
KNOWN_DEFECT_LEVELS = np.array([0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)


def depth_to_folder_string(depth):
    return str(depth).replace(".", "p")


def snap_mask_to_known_levels(mask, tol=1e-3):
    mask = mask.astype(np.float32)

    flat = mask.reshape(-1)

    distances = np.abs(flat[:, None] - KNOWN_LEVELS[None, :])
    nearest_idx = np.argmin(distances, axis=1)
    nearest_values = KNOWN_LEVELS[nearest_idx]
    nearest_distances = distances[np.arange(flat.size), nearest_idx]

    max_error = float(nearest_distances.max())

    if max_error > tol:
        bad_value = float(flat[np.argmax(nearest_distances)])
        nearest = float(nearest_values[np.argmax(nearest_distances)])
        raise ValueError(
            f"Mask contains value not close to known levels. "
            f"Bad value: {bad_value}, nearest level: {nearest}, "
            f"difference: {max_error}. Increase tol only if this is expected."
        )

    snapped = nearest_values.reshape(mask.shape).astype(np.float32)

    return snapped


def unique_nonzero_levels(mask_1d, tol=1e-6):
    values = np.unique(mask_1d.astype(np.float32))
    values = values[np.abs(values) > tol]

    return set(round(float(v), 6) for v in values)


def save_bscan_npy_pair(data_dir, mask_dir, source_name, row_idx, data_bscan, mask_bscan):
    """
    Save one B-scan as two separate .npy files.

    data:
        [T, H]

    mask:
        [H]

    The filenames are identical apart from the folder, so your dataset
    can match them easily.
    """

    filename = f"{source_name}_row_{row_idx:04d}.npy"

    data_path = data_dir / filename
    mask_path = mask_dir / filename

    np.save(data_path, data_bscan.astype(np.float32))
    np.save(mask_path, mask_bscan.astype(np.float32))


def split_npz_bscans_by_surface_depth(
    npz_folder,
    output_root,
    val_depth_surface_mm,
    test_depth_surface_mm,
    background_ratio=1.0,
    seed=42,
    tol=1e-3,
    overwrite=False,
):
    """
    Split preprocessed .npz files into train/val/test B-scan folders.

    Input .npz format:
        data : [T, W, H]
        mask : [W, H]

    Output folder format:
        config_folder/
            training_data/
            training_mask/
            validation_data/
            validation_mask/
            testing_data/
            testing_mask/
            split_summary.json

    Saved files:
        .npy data files with shape [T, H]
        .npy mask files with shape [H]
    """

    val_depth_surface_mm = float(val_depth_surface_mm)
    test_depth_surface_mm = float(test_depth_surface_mm)

    if val_depth_surface_mm == test_depth_surface_mm:
        raise ValueError("Validation and test depth must be different.")

    if val_depth_surface_mm not in SURFACE_DEPTH_TO_NORM_REMOVED:
        raise ValueError(
            f"Unknown validation surface depth: {val_depth_surface_mm}. "
            f"Allowed: {list(SURFACE_DEPTH_TO_NORM_REMOVED.keys())}"
        )

    if test_depth_surface_mm not in SURFACE_DEPTH_TO_NORM_REMOVED:
        raise ValueError(
            f"Unknown test surface depth: {test_depth_surface_mm}. "
            f"Allowed: {list(SURFACE_DEPTH_TO_NORM_REMOVED.keys())}"
        )

    val_level = round(float(SURFACE_DEPTH_TO_NORM_REMOVED[val_depth_surface_mm]), 6)
    test_level = round(float(SURFACE_DEPTH_TO_NORM_REMOVED[test_depth_surface_mm]), 6)

    all_defect_levels = set(round(float(x), 6) for x in KNOWN_DEFECT_LEVELS)
    train_levels = all_defect_levels - {val_level, test_level}

    npz_folder = Path(npz_folder)
    output_root = Path(output_root)

    config_name = (
        f"val_surface_{depth_to_folder_string(val_depth_surface_mm)}mm"
        f"_test_surface_{depth_to_folder_string(test_depth_surface_mm)}mm"
    )

    output_dir = output_root / config_name

    training_data_dir = output_dir / "training_data"
    training_mask_dir = output_dir / "training_mask"

    validation_data_dir = output_dir / "validation_data"
    validation_mask_dir = output_dir / "validation_mask"

    testing_data_dir = output_dir / "testing_data"
    testing_mask_dir = output_dir / "testing_mask"

    if output_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(output_dir)

    for d in [
        training_data_dir,
        training_mask_dir,
        validation_data_dir,
        validation_mask_dir,
        testing_data_dir,
        testing_mask_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(npz_folder.glob("*.npz"))

    if len(npz_paths) == 0:
        raise FileNotFoundError(f"No .npz files found in {npz_folder}")

    train_defect_items = []
    val_items = []
    test_items = []
    background_items = []
    discarded_items = []

    print("=" * 100)
    print(f"Creating split: {config_name}")
    print(f"Validation surface depth: {val_depth_surface_mm} mm -> level {val_level}")
    print(f"Test surface depth:       {test_depth_surface_mm} mm -> level {test_level}")
    print(f"Train levels:             {sorted(train_levels)}")
    print(
        "Train surface depths:     "
        f"{[NORM_REMOVED_TO_SURFACE_DEPTH[round(float(d), 6)] for d in sorted(train_levels)]} mm"
    )
    print("=" * 100)

    for path in tqdm(npz_paths, desc="Scanning npz files"):
        arr = np.load(path)

        if "data" not in arr or "mask" not in arr:
            raise KeyError(f"{path.name} must contain keys 'data' and 'mask'.")

        data = arr["data"].astype(np.float32)
        mask = arr["mask"].astype(np.float32)

        if data.ndim != 3:
            raise ValueError(f"{path.name}: expected data [T, W, H], got {data.shape}")

        if mask.ndim != 2:
            raise ValueError(f"{path.name}: expected mask [W, H], got {mask.shape}")

        T, W, H = data.shape

        if mask.shape != (W, H):
            raise ValueError(
                f"{path.name}: data spatial shape is {(W, H)}, "
                f"but mask shape is {mask.shape}"
            )

        mask = snap_mask_to_known_levels(mask, tol=tol)

        source_name = path.stem

        for row_idx in range(W):
            mask_bscan = mask[row_idx, :]
            levels_present = unique_nonzero_levels(mask_bscan)

            is_background = len(levels_present) == 0

            has_val = val_level in levels_present
            has_test = test_level in levels_present
            has_train = len(levels_present & train_levels) > 0

            contains_only_train_levels = (
                len(levels_present) > 0
                and levels_present.issubset(train_levels)
            )

            contains_only_val_level = levels_present == {val_level}

            item = {
                "source_file": str(path),
                "source_name": source_name,
                "row_idx": row_idx,
            }

            if is_background:
                background_items.append(item)

            elif contains_only_train_levels:
                train_defect_items.append(item)

            elif contains_only_val_level:
                val_items.append(item)

            elif has_test:
                test_items.append(item)

            else:
                discarded_items.append({
                    **item,
                    "levels_present": sorted(levels_present),
                    "has_val": has_val,
                    "has_test": has_test,
                    "has_train": has_train,
                })

    rng = random.Random(seed)

    n_train_defect = len(train_defect_items)
    n_background_target = int(round(background_ratio * n_train_defect))
    n_background_target = min(n_background_target, len(background_items))

    sampled_background_items = rng.sample(background_items, n_background_target)

    train_items = train_defect_items + sampled_background_items
    rng.shuffle(train_items)

    print("\nCOUNTS BEFORE SAVING")
    print(f"Train defect B-scans:       {len(train_defect_items)}")
    print(f"Background candidates:      {len(background_items)}")
    print(f"Sampled background B-scans: {len(sampled_background_items)}")
    print(f"Final train B-scans:        {len(train_items)}")
    print(f"Strict validation B-scans:  {len(val_items)}")
    print(f"Test B-scans:               {len(test_items)}")
    print(f"Discarded B-scans:          {len(discarded_items)}")

    def save_items(items, data_dir, mask_dir, split_name):
        for item in tqdm(items, desc=f"Saving {split_name}"):
            arr = np.load(item["source_file"])

            data = arr["data"].astype(np.float32)
            mask = arr["mask"].astype(np.float32)
            mask = snap_mask_to_known_levels(mask, tol=tol)

            row_idx = item["row_idx"]

            data_bscan = data[:, row_idx, :]   # [T, H]
            mask_bscan = mask[row_idx, :]      # [H]

            save_bscan_npy_pair(
                data_dir=data_dir,
                mask_dir=mask_dir,
                source_name=item["source_name"],
                row_idx=row_idx,
                data_bscan=data_bscan,
                mask_bscan=mask_bscan,
            )

    save_items(
        train_items,
        training_data_dir,
        training_mask_dir,
        "training",
    )

    save_items(
        val_items,
        validation_data_dir,
        validation_mask_dir,
        "validation",
    )

    save_items(
        test_items,
        testing_data_dir,
        testing_mask_dir,
        "testing",
    )

    summary = {
        "config_name": config_name,
        "val_depth_surface_mm": val_depth_surface_mm,
        "test_depth_surface_mm": test_depth_surface_mm,
        "val_level_norm_removed": val_level,
        "test_level_norm_removed": test_level,
        "train_levels_norm_removed": sorted(train_levels),
        "train_depths_surface_mm": [
            NORM_REMOVED_TO_SURFACE_DEPTH[round(float(d), 6)]
            for d in sorted(train_levels)
        ],
        "background_ratio": background_ratio,
        "seed": seed,
        "counts": {
            "train_defect_bscans": len(train_defect_items),
            "background_candidates": len(background_items),
            "sampled_background_bscans": len(sampled_background_items),
            "final_train_bscans": len(train_items),
            "strict_validation_bscans": len(val_items),
            "test_bscans": len(test_items),
            "discarded_bscans": len(discarded_items),
        },
        "folders": {
            "training_data": str(training_data_dir),
            "training_mask": str(training_mask_dir),
            "validation_data": str(validation_data_dir),
            "validation_mask": str(validation_mask_dir),
            "testing_data": str(testing_data_dir),
            "testing_mask": str(testing_mask_dir),
        },
        "internal_known_levels": [float(x) for x in KNOWN_LEVELS],
        "rules": {
            "train": "rows containing only train levels and/or background",
            "validation": "rows containing only validation level and background",
            "test": "rows containing test level; mixed levels allowed",
            "background": "sampled into training to match defect train count",
        },
        "saved_format": {
            "data": ".npy, shape [T, H]",
            "mask": ".npy, shape [H]",
        },
    }

    summary_path = output_dir / "split_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nDone. Saved split to: {output_dir}")
    print(f"Summary saved to: {summary_path}")

    return summary

npz_folder = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/open_source_data_npz_rb"

output_root = r"/home/kjaworski/Pulpit/Temporal_thermal_imaging/Open_Source_Dataset/bscan_splits"

summary = split_npz_bscans_by_surface_depth(
    npz_folder=npz_folder,
    output_root=output_root,
    val_depth_surface_mm=1.0,
    test_depth_surface_mm=2.0,
    background_ratio=1.0,
    seed=42,
    tol=1e-3,
    overwrite=True,
)