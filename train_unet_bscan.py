"""
Training script for the B-net models.

Trains one or more B-net variants (networks/Unets.py) to predict the defect
depth profile [512] from a thermal B-scan [3, 512, 512], using the B-scans
and depth targets written by helper_functions/preparation_of_b_scans.py.

For every model in `models` the script:
  - trains with Adam and an MSE loss for up to num_epochs epochs,
  - evaluates the validation set after every epoch,
  - saves the last weights every epoch and the best weights (lowest
    validation loss) whenever they improve,
  - stops early when the validation loss has not improved by more than
    min_delta for `patience` epochs,
  - saves the loss curves and the training settings.

Output, per model, in <main_path>/<model name>/:
  best_model_clean.pth  weights with the lowest validation loss
  last_model.pth        weights after the last epoch
  train_log.pt          training loss per epoch (list)
  val_clean_log.pt      validation loss per epoch (list)
  run_config.pt         settings of the run (dict)

The script runs top to bottom when executed; the paths below have to be set
to the local copy of the dataset.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from helper_functions.helper_functions import HorizontalShift,NoiseAdditionExperiment,RandomHorizontalFlipBscan
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernelSmarterRefine,BnetSmallKernelSmarter,BnetSmallKernel,BnetMean

# Use the GPU if there is one.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentations, applied to training samples only. Each B-scan has a 20 %
# chance to be mirrored and a 20 % chance to be shifted sideways; the depth
# target is transformed in the same way.
# NoiseAdditionExperiment(sigma=0.065) adds camera-like noise; it can be added
# to the list for training on simulated data, and is left out for
# experimental data, which already contains real noise.
train_transforms = ComposeBScanTransforms([
    RandomHorizontalFlipBscan(p=0.2),
    HorizontalShift(p=0.2)
])

# cooling_phase=False: the whole sequence (heating and cooling) is used as
# input.
# main_path: folder where the trained models and logs are written.
# normalization_path: file with the global scale from
# Global_normalization_parameter_finder.py.
cooling_phase=False
main_path = "/home/jaworskj/projects/thermal_B_scan/open_source_dataset/trained_models"
normalization_path="/home/jaworskj/projects/thermal_B_scan/normalization_params_experimental_cooling_only.npz"

# Training and validation data. Both use the same normalisation constant
# (computed on the training set); only the training set is augmented.
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

# Batches of 8 B-scans. Training data is shuffled every epoch; validation
# data is kept in a fixed order.
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


# Models to train, one after another, and the folder name used for each.
# To compare all variants, replace the two lists with the commented ones.
models = [
    BnetMean()
]

models_names = [
    "Bnet_mean"
]
# models = [
#     BnetMean(),
#     BnetSmallKernel(),
#     BnetSmallKernelSmarter(),
#     BnetSmallKernelSmarterRefine()
# ]

# models_names = [
#     "Bnet_mean",
#     "Bnet_Projection",
#     "Bnet_Deeper_regression",
#     "Bnet_refined",
# ]

# Mean squared error between the predicted and true depth profiles
# (normalised depth, averaged over all positions and samples in a batch).
criterion = nn.MSELoss()

# Gradients are rescaled so that their total norm is at most this value,
# which prevents single bad batches from causing large weight updates.
# None disables clipping.
GRAD_CLIP_NORM = 1.0

# num_epochs: upper limit on the number of epochs.
# patience / min_delta: early stopping; training stops when the validation
# loss has not dropped by more than min_delta for `patience` epochs in a row.
# lr: Adam learning rate (constant).
num_epochs = 500
patience = 30
min_delta = 1e-5
lr=1e-4

def evaluate(model, loader):
    """
    Mean loss of `model` over all samples of `loader`.

    The model is put in evaluation mode (batch normalisation uses its stored
    statistics) and gradients are not computed. The loss of every batch is
    weighted by its size, so a smaller last batch does not bias the mean.
    """
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


# Train every model in `models` separately, from scratch.
for i in range(len(models)):

    model = models[i].to(device)
    model_name = models_names[i]

    print("\n" + "=" * 80)
    print(f"Training model {i+1}/{len(models)}: {model_name}")
    print("=" * 80)

    # Alternative setups tested during development, kept for reference:
    # a frozen pretrained encoder (only the decoder and heads are trained)
    # and a cosine learning-rate schedule. As set now, the whole network is
    # trained with Adam at a constant learning rate.
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

    # Early-stopping state and loss history of this model.
    best_clean_loss = float("inf")
    counter = 0
    train_log = []
    val_clean_log = []

    # Output files of this model.
    model_dir = os.path.join(main_path, model_name)
    best_path = os.path.join(model_dir, "best_model_clean.pth")
    last_path = os.path.join(model_dir, "last_model.pth")

    # Create the output folder if it does not exist yet.
    os.makedirs(model_dir, exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc=f"{model_name} epochs", leave=False):
        # Training mode: batch normalisation updates its statistics.
        model.train()

        # With a frozen encoder, its batch-norm layers should also stay in
        # evaluation mode:
        # model.unet.encoder.eval()
        running_loss = 0.0

        for bscan, depth in tqdm(
            train_loader,
            desc=f"{model_name} | Epoch {epoch+1} Train",
            leave=False
        ):
            bscan = bscan.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            # Standard training step: clear gradients, forward pass, loss,
            # backward pass, optional clipping, weight update.
            optimizer.zero_grad(set_to_none=True)

            output = model(bscan)
            loss = criterion(output, depth)

            loss.backward()

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()

            # Sum of per-sample losses, to get the epoch mean below.
            running_loss += loss.item() * bscan.size(0)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_log.append(train_epoch_loss)

        # Validation loss after this epoch.
        clean_loss = evaluate(model, val_loader_clean)
        val_clean_log.append(clean_loss)

        print(
            f"{model_name} | "
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train: {train_epoch_loss:.6f} | "
            f"Val(clean): {clean_loss:.6f}"
        )

        # The latest weights are always saved, so training can be inspected
        # or resumed even if it is interrupted.
        torch.save(model.state_dict(), last_path)

        # A new best model must improve the validation loss by more than
        # min_delta; otherwise the patience counter goes up.
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

    # Loss curves and the settings of this run, saved next to the weights.
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

    # Free the GPU memory before the next model is trained.
    model = model.to("cpu")
    del optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()