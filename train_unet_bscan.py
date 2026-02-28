import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper_functions.helper_functions import NoiseAddition, HorizontalShift, DefectSlopeDropout
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetTiny

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")

# -------------------------
# Transforms
# -------------------------
train_transforms = ComposeBScanTransforms([
    HorizontalShift(p=0.3),
    NoiseAddition(),                 
    DefectSlopeDropout(p=0.1)
])

val_noisy_transforms = ComposeBScanTransforms([
    NoiseAddition(sigma_min=0.02, sigma_max=0.06)
])

# -------------------------
# Datasets
# -------------------------
train_dataset = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/data",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/depth",
    transform=train_transforms,
    normalization_path="/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/normalization_params.npz"
)

# CLEAN validation (controls early stopping + best model)
val_dataset_clean = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/data",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/depth",
    transform=None,  # <-- important: no augmentation
    normalization_path="/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/normalization_params.npz"
)

# NOISY validation (robustness metric only)
val_dataset_noisy = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/data",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/depth",
    transform=val_noisy_transforms,
    normalization_path="/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/normalization_params.npz"
)

# -------------------------
# Loaders
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=24,
    pin_memory=pin_memory
)

val_loader_clean = DataLoader(
    val_dataset_clean,
    batch_size=16,
    shuffle=False,
    num_workers=24,
    pin_memory=pin_memory
)

val_loader_noisy = DataLoader(
    val_dataset_noisy,
    batch_size=16,
    shuffle=False,
    num_workers=24,
    pin_memory=pin_memory
)

# -------------------------
# Model / loss / optimizer
# -------------------------
model = BnetTiny().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------
# Save paths
# -------------------------
main_path = "/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/models_logs"
model_name = "tinynet"
model_dir = os.path.join(main_path, model_name)
os.makedirs(model_dir, exist_ok=True)

best_path = os.path.join(model_dir, "best_model_clean.pth")
last_path = os.path.join(model_dir, "last_model.pth")

# -------------------------
# Training config
# -------------------------
num_epochs = 500
best_clean_loss = float("inf")

patience = 50
min_delta = 1e-3 
counter = 0

train_log = []
val_clean_log = []
val_noisy_log = []

# -------------------------
# Eval helper
# -------------------------
def evaluate(loader):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for bscan, depth in loader:
            bscan = bscan.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            output = model(bscan)
            loss = criterion(output, depth)

            bs = bscan.size(0)
            total += loss.item() * bs
            n += bs
    return total / max(1, n)

# -------------------------
# Training loop
# -------------------------
for epoch in tqdm(range(num_epochs), desc="Epochs", leave=False):
    model.train()
    running_loss = 0.0

    for bscan, depth in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
        bscan = bscan.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(bscan)
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * bscan.size(0)

    train_epoch_loss = running_loss / len(train_loader.dataset)
    train_log.append(train_epoch_loss)

    # ---- Two validation passes ----
    clean_loss = evaluate(val_loader_clean)      # controls early stopping
    noisy_loss = evaluate(val_loader_noisy)      # robustness metric

    val_clean_log.append(clean_loss)
    val_noisy_log.append(noisy_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train: {train_epoch_loss:.6f} | "
        f"Val(clean): {clean_loss:.6f} | "
        f"Val(noisy): {noisy_loss:.6f}"
    )

    # always save last
    torch.save(model.state_dict(), last_path)

    # early stopping + best checkpoint ONLY on clean val
    if clean_loss < best_clean_loss - min_delta:
        best_clean_loss = clean_loss
        counter = 0
        torch.save(model.state_dict(), best_path)
        print(f"Saved BEST (clean) model: {best_clean_loss:.6f} -> {best_path}")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered (based on clean validation).")
        break

# -------------------------
# Save logs
# -------------------------
torch.save(train_log, os.path.join(model_dir, "train_log.pt"))
torch.save(val_clean_log, os.path.join(model_dir, "val_clean_log.pt"))
torch.save(val_noisy_log, os.path.join(model_dir, "val_noisy_log.pt"))
print("Saved logs + checkpoints in:", model_dir)