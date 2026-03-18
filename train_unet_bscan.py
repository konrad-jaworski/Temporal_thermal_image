import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from helper_functions.helper_functions import HorizontalShift,NoiseAddition,DefectSlopeDropout
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernelSmarter,BnetSmallKernelSmarterRefine,BnetMean


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic-ish conv behavior (can slow down)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # This enforces deterministic algorithms where possible (may error for some ops)
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int):
    # Make each worker seed deterministic
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

SEED = 123
seed_everything(SEED, deterministic=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


# -------------------------
# Loss: ROI-weighted, count-normalized SmoothL1 for sparse masks
# -------------------------
class WeightedSmoothL1Sparse(nn.Module):
    """
    For sparse targets where gt==0 for background and gt>0 inside defect ROI.
    Computes loss separately on ROI and background, normalizes by element counts,
    and weights ROI higher.
    """
    def __init__(self, w_roi: float = 10.0, w_bg: float = 1.0, beta: float = 0.05, roi_threshold: float = 0.0):
        super().__init__()
        self.w_roi = w_roi
        self.w_bg = w_bg
        self.beta = beta
        self.roi_threshold = roi_threshold

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: [B, W]
        """
        eps = 1e-6
        roi = (gt > self.roi_threshold).float()
        bg  = 1.0 - roi

        # SmoothL1 per-element (Huber)
        # reduction='none' to do our own count-normalization
        per_elem = F.smooth_l1_loss(pred, gt, reduction="none", beta=self.beta)

        roi_sum = (per_elem * roi).sum()
        bg_sum  = (per_elem * bg).sum()

        roi_cnt = roi.sum()
        bg_cnt  = bg.sum()

        loss_roi = roi_sum / (roi_cnt + eps)
        loss_bg  = bg_sum  / (bg_cnt + eps)

        return self.w_roi * loss_roi + self.w_bg * loss_bg


# -------------------------
# Transforms
# -------------------------
train_transforms = ComposeBScanTransforms([
    HorizontalShift(p=0.3),  # keep as baseline invariance
    NoiseAddition()
])


# -------------------------
# Datasets
# -------------------------
train_dataset = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/data",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/training/depth",
    transform=train_transforms,
    normalization_path="/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/normalization_params.npz",
    derivative_mode=None
)

val_dataset_clean = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/data",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/all_data_extrapolated/validation/depth",
    transform=None,
    normalization_path="/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/normalization_params.npz",
    derivative_mode=None
)


# -------------------------
# Loaders
# -------------------------
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=24,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True if 24 > 0 else False
)

val_loader_clean = DataLoader(
    val_dataset_clean,
    batch_size=16,
    shuffle=False,
    num_workers=24,
    pin_memory=pin_memory,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=True if 24 > 0 else False
)


# -------------------------
# Model / loss / optimizer
# -------------------------
# model = BnetSmallKernelSmarter().to(device)
# model = BnetSmallKernelSmarterRefine().to(device)
model = BnetMean().to(device)

# NEW LOSS (Stage 1 ablation)
# criterion = WeightedSmoothL1Sparse(
#     w_roi=10.0,   # start here; try 5.0 if BG starts drifting
#     w_bg=1.0,
#     beta=0.05,    # Huber transition; ~0.02–0.1 reasonable on [0,1] targets
#     roi_threshold=0.0
# )

# MSE loss (Stage 1 baseline)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4) # This was previous configuration

# optimizer = optim.Adam([
#     {"params": model.unet.parameters(), "lr": 1e-4},
#     {"params": model.vertical_proj.parameters(), "lr": 1e-3},
#     {"params": model.regressor_smarter.parameters(), "lr": 1e-3}
# ])

# Optional: stable training if you see spikes
GRAD_CLIP_NORM = 1.0  # set e.g. 1.0 if needed

# Stage 2: non-linear prediction

non_linear_pred=False
EPS = 1e-6

def depth_to_u(depth):
    return torch.sqrt(depth.clamp_min(0.0) + EPS)

def u_to_depth(u):
    return (u.clamp(0.0, 1.0) ** 2)



# -------------------------
# Save paths
# -------------------------
main_path = "/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/models_logs"
model_name = "mean_net_noisy"
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
min_delta = 1e-5
counter = 0

train_log = []
val_clean_log = []


# -------------------------
# Eval helper
# -------------------------
def evaluate(loader,non_linear_pred=non_linear_pred):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for bscan, depth in loader:
            bscan = bscan.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            
            if non_linear_pred==False:
                output = model(bscan)
                loss = criterion(output, depth)
            else:
                output_u = model(bscan)                  # predicts u in [0,1]
                target_u = depth_to_u(depth)             # sqrt(depth)
                loss = criterion(output_u, target_u)

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
        
        if non_linear_pred==False:
            output = model(bscan)
            loss = criterion(output, depth)
        else:
            output_u = model(bscan)                  # predicts u in [0,1]
            target_u = depth_to_u(depth)             # sqrt(depth)
            loss = criterion(output_u, target_u)

        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        running_loss += loss.item() * bscan.size(0)

    train_epoch_loss = running_loss / len(train_loader.dataset)
    train_log.append(train_epoch_loss)

    # ---- Validation pass ----
    clean_loss = evaluate(val_loader_clean,non_linear_pred)
    val_clean_log.append(clean_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train: {train_epoch_loss:.6f} | "
        f"Val(clean): {clean_loss:.6f}"
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
# Save logs + run config
# -------------------------
torch.save(train_log, os.path.join(model_dir, "train_log.pt"))
torch.save(val_clean_log, os.path.join(model_dir, "val_clean_log.pt"))

# Save meta so you can reproduce the run exactly
run_config = {
    "seed": SEED,
    "batch_size": 16,
    "num_workers": 24,
    "lr": 1e-4,
    "loss": "MSE",
    "channels": "Repeated",
    "derivative_mode": "None",
    "Model":"mean type",
    "hshift_p": 0.3,
    "patience": patience,
}
torch.save(run_config, os.path.join(model_dir, "run_config.pt"))

print("Saved logs + checkpoints in:", model_dir)