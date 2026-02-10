import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from helper_functions.helper_functions import RandomHorizontalFlipBscan, NoiseAddition
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernel, BnetBigKernel, BnetMean,CompactBnet
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_transforms = ComposeBScanTransforms([
        RandomHorizontalFlipBscan(p=0.5),
        NoiseAddition(noise_level=0.05),
    ])

    val_transforms = ComposeBScanTransforms([
        # NoiseAddition(noise_level=0.05)
    ])

    train_dataset = BScanDepthDataset(
        bscan_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\data",
        depth_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\depth",
        transform=train_transforms,
        normalization_path=r"C:\Users\stone\Temporal_thermal_image\normalization_params.npz"
    )

    val_dataset = BScanDepthDataset(
        bscan_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_val\data",
        depth_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_val\depth",
        transform=val_transforms,
        normalization_path=r"C:\Users\stone\Temporal_thermal_image\normalization_params.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Your UNet-based regressor
    model = BnetSmallKernel()
    # model= BnetBigKernel()
    # model = BnetMean()
    # model=CompactBnet()
    model.to(device)

    # Loss function (per-column regression)
    criterion = nn.MSELoss()  # Could use L1Loss or HuberLoss if preferred
    # criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 500
    best_loss = float('inf')
    save_path = r"C:\Users\stone\Temporal_thermal_image\Unet_small_kernel_l2_both_classes.pth"

    # Early stopping parameters
    patience = 100        # epochs to wait
    min_delta = 1e-4      # minimum improvement
    counter = 0

    train_log=[]
    val_log=[]



    for epoch in tqdm(range(num_epochs), desc=f"Epochs", leave=False):
        model.train()
       
        running_loss = 0.0

        for bscan, depth in tqdm(train_loader, desc=f"Epoch {epoch+1} Batches", leave=False):
            bscan = bscan.to(device)   # [B,3,H,W]
            depth = depth.to(device)   # [B,W]

            optimizer.zero_grad()
            output = model(bscan)      # [B,W]
            loss = criterion(output, depth)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bscan.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_log.append(epoch_loss)
        

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bscan, depth in tqdm(val_loader, desc=f"Epoch {epoch+1} Val Batches", leave=False):
                bscan = bscan.to(device)
                depth = depth.to(device)

                output = model(bscan)
                loss = criterion(output, depth)
                val_loss += loss.item() * bscan.size(0)
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_log.append(val_epoch_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.6f}, Val Loss: {val_epoch_loss:.6f}")

        if val_epoch_loss < best_loss - min_delta:
            best_loss = val_epoch_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with loss {best_loss:.6f} at {save_path}")
        else:
            counter+=1

        if counter >= patience:
            print("Early stopping triggered.")
            break    
        
    torch.save(train_log,'train_log_unet_small_kernel_big_dataset.pt')
    torch.save(val_log, 'val_log_unet_small_kernel_big_dataset.pt')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()