import os
import numpy as np
import torch
from torch.utils.data import Dataset
from helper_functions.helper_functions import Interpolate

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
        transform=None,   # ← augmentation hook
        dtype=torch.float32,
        normalization_path=None
    ):
        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.dtype = dtype
        self.path=normalization_path

        # Normalization config
        config=np.load(self.path,allow_pickle=True)
        # deltaT_max=config['T_max']
        # deltaT_min=config['T_min']
        # self.normalizer=DataNormalization(deltaT_min,deltaT_max)

        self.scale=config['scale']

        # Interpolation to fixed size of the B-scan
        self.resize = Interpolate(size=512)

        # Sorted files for the __len__ method
        self.files = sorted(os.listdir(bscan_dir))

        # Safety check
        for f in self.files:
            if not os.path.exists(os.path.join(depth_dir, f)):
                raise FileNotFoundError(f"Missing depth file for {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load numpy arrays
        bscan = np.load(os.path.join(self.bscan_dir, fname))   # (H, W)
        depth = np.load(os.path.join(self.depth_dir, fname))   # (W,)

        # Sanity checks (important during research)
        assert bscan.ndim == 2, "B-scan must be 2D"
        assert depth.ndim == 1, "Depth must be 1D"
        assert bscan.shape[1] == depth.shape[0], \
            "Width mismatch between bscan and depth"

        # Convert to torch
        bscan = torch.from_numpy(bscan).to(self.dtype)
        depth = torch.from_numpy(depth).to(self.dtype)

        # Convert to the log1p scale
        bscan=torch.log1p(bscan)


        # Add channel dimension for CNN
        # (you can later stack to 3 channels if needed)
        #- Normalize
        #- Add channel dim 
        #- Interpolate
        #- Copy the our image to same channels dimmensions 

        bscan = bscan/self.scale  # We normalize the data by firstly mapping it to log1p scale and then divide it by 99.5 % percentail of trainnig data

        bscan = bscan.unsqueeze(0)  # [ 1, H, W]
        bscan = self.resize(bscan)               # [ 1, 512, W] 
        
        # Space for the augmentation in the future research, we need to do this in order to cooperate with pre_trained network.
        bscan=bscan.repeat(3,1,1)             # [ 3, 512, W]
        bscan = bscan.float()

        # --------------------------------------------------
        # AUGMENTATION HOOK (data + depth together)
        # --------------------------------------------------
        if self.transform is not None:
            bscan, depth = self.transform(bscan, depth)

        depth=depth.float()
        
        return bscan, depth
    
class BScanDepthDatasetExp(Dataset):
    """
    Custom Dataset class for the experimental data which will not provide the ground truth
    """
    def __init__(
        self,
        bscan_dir,
        dtype=torch.float32,
        normalization_path=None
    ):
        self.bscan_dir = bscan_dir
        self.dtype = dtype
        self.path=normalization_path

        # Normalization config
        config=np.load(self.path,allow_pickle=True)
        deltaT_max=config['T_max']
        deltaT_min=config['T_min']
        self.normalizer=DataNormalization(deltaT_min,deltaT_max)

        # Interpolation to fixed size of the B-scan
        self.resize = Interpolate(size=512)

        self.files = sorted(os.listdir(bscan_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # Load numpy arrays
        bscan = np.load(os.path.join(self.bscan_dir, fname))   # (H, W)

        # Sanity checks (important during research)
        assert bscan.ndim == 2, "B-scan must be 2D"
    
        # Convert to torch
        bscan = torch.from_numpy(bscan).to(self.dtype)

        # Add channel dimension for CNN
        # (you can later stack to 3 channels if needed)
        #- Normalize
        #- Add channel dim 
        #- Interpolate
        #- Copy the our image to smae channels dimmensions 
        bscan = self.normalizer.normalize(bscan)   
        bscan = bscan.unsqueeze(0)  # [ 1, H, W]
        bscan = self.resize(bscan)               # [ 1, 512, W]
        bscan=bscan.repeat(3,1,1)             # [ 3, 512, W]
        bscan = bscan.float()
    
        return bscan