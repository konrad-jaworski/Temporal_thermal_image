import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from helper_functions.helper_functions import RandomHorizontalFlipBscan, NoiseAddition
from data.data_operators import BScanDepthDataset, ComposeBScanTransforms
from networks.Unets import BnetSmallKernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


train_transforms = ComposeBScanTransforms([
    RandomHorizontalFlipBscan(p=0.5),
    NoiseAddition(noise_level=0.05),
])

train_dataset = BScanDepthDataset(
    bscan_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\data",
    depth_dir=r"E:\Simulated_and_experimental_data\Synthetic_data\B-scans_train\depth",
    transform=train_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)