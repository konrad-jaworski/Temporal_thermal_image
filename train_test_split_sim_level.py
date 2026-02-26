import os
import shutil
from pathlib import Path

# ===== USER SETTINGS =====
source_dir = Path("/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated")

required_phrase = "thickness4mm"   # only consider files containing this

rules = {
    "training_deep_2_shallow":    ["depth20pct", "depth30pct","depth40pct","depth50pct"],
    "validation_deep_2_shallow":  ["depth60pct"],
    "test_deep_2_shallow":        ["depth70pct","depth80pct","depth90pct"],
}

dry_run = False   # set to False to actually copy
# =========================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

for p in source_dir.iterdir():
    if not p.is_file() or p.suffix != ".npz":
        continue

    name = p.name

    if required_phrase not in name:
        continue

    matched = False
    for folder, phrases in rules.items():
        if any(ph in name for ph in phrases):
            dst_dir = source_dir / folder
            ensure_dir(dst_dir)
            dst = dst_dir / p.name

            print(f"{'WOULD COPY' if dry_run else 'COPY'}: {p.name} -> {folder}/")
            if not dry_run:
                shutil.copy2(p, dst)
            matched = True
            break

    if not matched:
        print(f"SKIP (no rule match): {p.name}")