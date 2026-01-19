import os
import shutil
import hashlib


def file_hash(path, chunk_size=8192):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_npz_files(source_root, destination_root):
    """
    Recursively collect all valid .npz files from source_root
    into destination_root, excluding macOS metadata files (._*).
    """

    os.makedirs(destination_root, exist_ok=True)

    # Hash existing destination files
    existing_hashes = {}
    for fname in os.listdir(destination_root):
        if fname.endswith(".npz") and not fname.startswith("._"):
            full_path = os.path.join(destination_root, fname)
            existing_hashes[file_hash(full_path)] = fname

    copied = 0
    skipped_duplicates = 0
    skipped_metadata = 0

    for root, _, files in os.walk(source_root):
        for file in files:

            # Exclude macOS metadata files
            if file.startswith("._"):
                skipped_metadata += 1
                continue

            if not file.endswith(".npz"):
                continue

            src_path = os.path.join(root, file)
            src_hash = file_hash(src_path)

            # Duplicate content check
            if src_hash in existing_hashes:
                skipped_duplicates += 1
                continue

            # Resolve name collisions
            dst_name = file
            dst_path = os.path.join(destination_root, dst_name)
            base, ext = os.path.splitext(file)
            counter = 1

            while os.path.exists(dst_path):
                dst_name = f"{base}_{counter}{ext}"
                dst_path = os.path.join(destination_root, dst_name)
                counter += 1

            shutil.copy2(src_path, dst_path)
            existing_hashes[src_hash] = dst_name
            copied += 1

    print(f"Copied files: {copied}")
    print(f"Skipped duplicates: {skipped_duplicates}")
    print(f"Skipped macOS metadata files: {skipped_metadata}")

collect_npz_files(r'E:\Simulated_and_experimental_data\Synthetic_data\Gaussian_heating',
                  r'E:\Simulated_and_experimental_data\Synthetic_data\all_data')

collect_npz_files(r'E:\Simulated_and_experimental_data\Synthetic_data\Uniform_heating',
                  r'E:\Simulated_and_experimental_data\Synthetic_data\all_data')