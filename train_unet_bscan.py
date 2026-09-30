import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper_functions.helper_functions import HorizontalShift,NoiseAdditionExperiment,RandomHorizontalFlipBscan
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernelSmarterRefine,BnetSmallKernelSmarter,BnetSmallKernel,BnetMean,BnetSwinTransformer

# The device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_memory = (device.type == "cuda")

# -------------------------
# Transforms
# -------------------------
train_transforms = ComposeBScanTransforms([
    # Detectore wise noise addition, for experimental data we remove it ! NoiseAdditionExperiment(sigma=0.065)
    RandomHorizontalFlipBscan(p=0.2), # keep as abseline invariance
    HorizontalShift(p=0.2)  # keep as baseline invariance
])

# Data loaders configuration

cooling_phase=False
main_path = "/home/jaworskj/projects/thermal_B_scan/open_source_dataset/Model_performance_open_source_publication_fixed_encoder_fixed_lr"
normalization_path="/home/jaworskj/projects/thermal_B_scan/normalization_params_experimental_cooling_only.npz"
# -------------------------
# Datasets
# -------------------------
train_dataset = BScanDepthDataset(
    bscan_dir="/home/jaworskj/projects/thermal_B_scan/open_source_dataset/training/data_bscans",
    depth_dir="/home/jaworskj/projects/thermal_B_scan/open_source_dataset/training/data_masks",
    transform=train_transforms,
    normalization_path=normalization_path,
    cooling_phase=cooling_phase,   
)

val_dataset = BScanDepthDataset(
    bscan_dir="/home/jaworskj/projects/thermal_B_scan/open_source_dataset/validation/data_bscans",
    depth_dir="/home/jaworskj/projects/thermal_B_scan/open_source_dataset/validation/data_masks",
    transform=None,
    normalization_path=normalization_path,
    cooling_phase=cooling_phase
)

# -------------------------
# Loaders
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader_clean = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)




# -------------------------
# Model / loss / optimizer
# -------------------------

models = [
    BnetMean()
]

models_names = [
    "Bnet_mean"
]
# models = [
#     BnetSwinTransformer(),
#     BnetMean(),
#     BnetSmallKernel(),
#     BnetSmallKernelSmarter(),
#     BnetSmallKernelSmarterRefine()
# ]

# models_names = [
#     "Bnet_SwinTransformer",
#     "Bnet_mean",
#     "Bnet_Projection",
#     "Bnet_Deeper_regression",
#     "Bnet_refined",
# ]

# MSE loss (Stage 1 baseline)
criterion = nn.MSELoss()

# Optional: stable training if you see spikes
GRAD_CLIP_NORM = 1.0  # set e.g. 1.0 if needed

# -------------------------
# Training config
# -------------------------
num_epochs = 500
patience = 30 
min_delta = 1e-5
# Learning rate
lr=1e-4

# -------------------------
# Eval helper
# -------------------------
def evaluate(model, loader):
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
# Train all models one by one
# -------------------------
for i in range(len(models)):

    model = models[i].to(device)
    model_name = models_names[i]

    print("\n" + "=" * 80)
    print(f"Training model {i+1}/{len(models)}: {model_name}")
    print("=" * 80)

    # Freeze the pretrained encoder.
    # for parameter in model.unet.encoder.parameters():
    #     parameter.requires_grad = False
    #     parameter.grad = None

    # optimizer = torch.optim.Adam(
    #     (p for p in model.parameters() if p.requires_grad),
    #     lr=lr,
    # )
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=num_epochs,
    #     eta_min=1e-6,
    # )

    optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)
    
    best_clean_loss = float("inf")
    counter = 0
    train_log = []
    val_clean_log = []

    # -------------------------
    # Save paths
    # -------------------------
    
    model_dir = os.path.join(main_path, model_name)
    best_path = os.path.join(model_dir, "best_model_clean.pth")
    last_path = os.path.join(model_dir, "last_model.pth")

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in tqdm(range(num_epochs), desc=f"{model_name} epochs", leave=False):
        model.train()

        # Depending if we want to train the encoder
        # model.unet.encoder.eval()
        running_loss = 0.0

        for bscan, depth in tqdm(
            train_loader,
            desc=f"{model_name} | Epoch {epoch+1} Train",
            leave=False
        ):
            bscan = bscan.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(bscan)
            loss = criterion(output, depth)

            loss.backward()

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            running_loss += loss.item() * bscan.size(0)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_log.append(train_epoch_loss)

        # ---- Validation pass ----
        clean_loss = evaluate(model, val_loader_clean)
        val_clean_log.append(clean_loss)

        print(
            f"{model_name} | "
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train: {train_epoch_loss:.6f} | "
            f"Val(clean): {clean_loss:.6f}"
        )

        # Always save last
        torch.save(model.state_dict(), last_path)

        # Early stopping + best checkpoint
        if clean_loss < best_clean_loss - min_delta:
            best_clean_loss = clean_loss
            counter = 0

            torch.save(model.state_dict(), best_path)

            print(
                f"Saved BEST model for {model_name}: "
                f"{best_clean_loss:.6f} -> {best_path}"
            )
        else:
            counter += 1

        # scheduler.step()
        if counter >= patience:
            print(f"Early stopping triggered for {model_name}.")
            break

    # -------------------------
    # Save logs + run config
    # -------------------------
    torch.save(train_log, os.path.join(model_dir, "train_log.pt"))
    torch.save(val_clean_log, os.path.join(model_dir, "val_clean_log.pt"))

    run_config = {
        "batch_size": 8,
        "lr": lr,
        "loss": "MSE",
        "model": model_name,
        "patience": patience,
        "min_delta": min_delta,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "best_clean_loss": best_clean_loss,
    }

    torch.save(run_config, os.path.join(model_dir, "run_config.pt"))

    print(f"Saved logs + checkpoints in: {model_dir}")

    # Move previous model to CPU and clear CUDA cache
    model = model.to("cpu")
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()