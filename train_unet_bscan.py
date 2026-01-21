import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from helper_functions.helper_functions import RandomHorizontalFlipBscan, NoiseAddition
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_transforms = ComposeBScanTransforms([
        RandomHorizontalFlipBscan(p=0.5),
        NoiseAddition(noise_level=0.05),
    ])

    train_dataset = BScanDepthDataset(
        bscan_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\data",
        depth_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\depth",
        transform=train_transforms,
        normalization_path=r"C:\Users\stone\Temporal_thermal_image\normalization_params.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Your UNet-based regressor
    model = BnetSmallKernel()
    model.to(device)

    # Loss function (per-column regression)
    criterion = nn.MSELoss()  # Could use L1Loss or HuberLoss if preferred

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    best_loss = float('inf')
    save_path = r"C:\Users\stone\Temporal_thermal_image\Unet_small_kernel.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)

        for bscan, depth in loop:
            bscan = bscan.to(device)   # [B,3,H,W]
            depth = depth.to(device)   # [B,W]

            optimizer.zero_grad()
            output = model(bscan)      # [B,W]

            loss = criterion(output, depth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bscan.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with loss {best_loss:.6f} at {save_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()