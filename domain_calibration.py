import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.data_operators import BScanDepthDataset
from networks.Unets import BnetSmallKernelSmarterRefine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")


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
g = torch.Generator()
g.manual_seed(SEED)


# We are using the sound regions of the experimental sample to calibrate the model to non-uniform heating
train_dataset = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/Experimental_sample/exp_bscans_domain",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/Experimental_sample/exp_masks_domain",
    transform=None,
    normalization_path="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/normalization_params.npz",
    derivative_mode=None,
    log_scaling=True,
    cooling_phase=False
)

# We control the model over validaton set in order to not drop the perfomance in depth regression
val_dataset_clean = BScanDepthDataset(
    bscan_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/validation_bscan",
    depth_dir="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/validation_mask",
    transform=None,
    normalization_path="/home/kjaworski/Pulpit/Temporal_thermal_imaging/Bscan_thermography_dataset/normalization_params.npz",
    derivative_mode=None,
    log_scaling=True,
    cooling_phase=False
)

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


model = BnetSmallKernelSmarterRefine().to(device)
model_path = '/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/models_logs_official/smart_net_heating_and_cooling/best_model_clean.pth'
state_dict=torch.load(model_path)
model.load_state_dict(state_dict)


for param in model.unet.parameters():
    param.requires_grad = False

for param in model.vertical_proj.parameters():
    param.requires_grad = False

for param in model.regressor_smarter.parameters():
    param.requires_grad = True

for param in model.refinement.parameters():
    param.requires_grad = True



criterion = nn.MSELoss()

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam([
    {"params": model.regressor_smarter.parameters(), "lr": 1e-5},
    {"params": model.refinement.parameters(), "lr": 1e-5},
])

def set_frozen_bn_eval(module):
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.eval()


GRAD_CLIP_NORM = 1.0

# -------------------------
# Save paths
# -------------------------
main_path = "/home/kjaworski/Pulpit/Themporal_thermal_imaging_code/Temporal_thermal_image/models_logs_official"
model_name = "smart_net_H_C_domain"
model_dir = os.path.join(main_path, model_name)
os.makedirs(model_dir, exist_ok=True)

best_path = os.path.join(model_dir, "best_model_clean.pth")


# -------------------------
# Training config
# -------------------------
num_epochs = 30 # In a form of a sweep
best_clean_loss = float("inf")

min_delta = 1e-5 # in case if somehow this will produce better performance on validation dataset

train_log = []
val_clean_log = []

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

    model.unet.eval()
    model.vertical_proj.eval()
    model.unet.apply(set_frozen_bn_eval)
    model.vertical_proj.apply(set_frozen_bn_eval)

    running_loss = 0.0

    for bscan, depth in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
        bscan = bscan.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        output = model(bscan)
        loss = criterion(output, depth)
        
        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)

        optimizer.step()
        running_loss += loss.item() * bscan.size(0)

    train_epoch_loss = running_loss / len(train_loader.dataset)
    train_log.append(train_epoch_loss)

    # ---- Validation pass ----
    clean_loss = evaluate(val_loader_clean)
    val_clean_log.append(clean_loss)

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train: {train_epoch_loss:.6f} | "
        f"Val(clean): {clean_loss:.6f}"
    )

    # We save all of the models to further check their performance
    last_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), last_path)

    if clean_loss < best_clean_loss - min_delta:
        best_clean_loss = clean_loss
        torch.save(model.state_dict(), best_path)
        print(f"Saved BEST (clean) model: {best_clean_loss:.6f} -> {best_path}")
   

  
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
    "lr": 1e-5,
    "loss": "MSE",
    "channels": "Repeated",
    "derivative_mode": "None",
    "Model":"Smartnet H+C domain adaptation"
}
torch.save(run_config, os.path.join(model_dir, "run_config.pt"))

print("Saved logs + checkpoints in:", model_dir)
