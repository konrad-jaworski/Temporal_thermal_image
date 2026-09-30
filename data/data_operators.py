import os
import numpy as np
import torch
from torch.utils.data import Dataset

from helper_functions.helper_functions import (
    Interpolate,
    Interpolate_mask,
)


class ComposeBScanTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, bscan, depth):
        for t in self.transforms:
            bscan, depth = t(bscan, depth)
        return bscan, depth


class BScanDepthDataset(Dataset):
    def __init__(
        self,
        bscan_dir,
        depth_dir,
        transform=None,
        normalization_path=None,
        cooling_phase=True,
        cooling_frame=11,
        resize_size=512,
        dtype=torch.float32
    ):
        """
        Dataset for column/row-wise B-scan depth regression.

        Expected file format
        --------------------
        bscan_dir:
            .npy files containing B-scans with shape [T, W]

        depth_dir:
            .npy files containing depth vectors with shape [W]

        Returned tensors
        ----------------
        x:
            [3, 512, 512]

        depth:
            [512]

        Preprocessing logic
        -------------------
        1. Optional cooling cut:
            bscan = bscan[cooling_frame:, :]

        2. Optional augmentation:
            Applied before scaling/normalization.
        
        Parameters
        ----------
        bscan_dir : str
            Folder with B-scan .npy files.

        depth_dir : str
            Folder with corresponding mask/depth .npy files.

        transform : callable or None
            Optional transform applied to bscan and depth before scaling.

        dtype : torch.dtype
            Tensor dtype.

        normalization_path : str
            Path to .npz file containing normalization parameters.

        resize_size : int
            Target interpolation size.
        """

        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.normalization_path = normalization_path
        self.cooling_phase = cooling_phase
        self.cooling_frame = cooling_frame
        self.dtype = dtype
       
        
        if self.normalization_path is not None:
            config = np.load(self.normalization_path, allow_pickle=True)
            if "scale" not in config:
                raise KeyError("Normalization file must contain key 'scale'.")
            self.scale = float(config["scale"]) 
            
        # Resising of the bscan to fit into the network
        self.resize = Interpolate(size=resize_size)
        self.resize_mask = Interpolate_mask(size=resize_size)

        self.files = sorted(
            f for f in os.listdir(self.bscan_dir)
            if f.endswith(".npy")
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.bscan_dir}")

        for f in self.files:
            depth_path = os.path.join(self.depth_dir, f)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Missing depth file for {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # We grab the folder
        fname = self.files[idx]

        # From it we grab bscan and the depth mask
        bscan_path = os.path.join(self.bscan_dir, fname)
        depth_path = os.path.join(self.depth_dir, fname)

        # Numpy arrays
        bscan = np.load(bscan_path)   # [T, W]
        depth = np.load(depth_path)   # [W]

        # Projecting it into the torch tensor
        bscan = torch.from_numpy(bscan).to(self.dtype)
        depth = torch.from_numpy(depth).to(self.dtype)

        # --------------------------------------------------
        # Sequence selection
        # --------------------------------------------------
        if self.cooling_phase:
            if self.cooling_frame < 0 or self.cooling_frame >= bscan.shape[0]:
                raise ValueError(
                    f"Invalid cooling_frame={self.cooling_frame} for "
                    f"B-scan with {bscan.shape[0]} frames."
                )

            bscan = bscan[self.cooling_frame:, :]

        # --------------------------------------------------
        # Augmentation before scaling
        # --------------------------------------------------
        if self.transform is not None:
            bscan, depth = self.transform(bscan, depth)

        bscan_base = bscan / self.scale
    
        x = bscan_base.unsqueeze(0)     # [1, T, W]
        x = self.resize(x)              # [1, 512, 512]
        x = x.repeat(3, 1, 1)           # [3, 512, 512]

        depth = self.resize_mask(depth) # [512]

        return x.float(), depth.float() 

        

        