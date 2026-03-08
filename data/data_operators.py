import os
import numpy as np
import torch
from torch.utils.data import Dataset
from helper_functions.helper_functions import Interpolate,d1_dx,d1_dy,d2_dx2,d2_dy2

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
        normalization_path=None,
        derivative_mode=None # time-> both derivative are with respect to time, space-> both derivatives are with respect to space, mixed-> first derivative is with respect to time and second with respect to space.
    ):
        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.dtype = dtype
        self.path=normalization_path
        self.derivative_mode=derivative_mode

        # Normalization config
        config=np.load(self.path,allow_pickle=True)
        self.scale=config['scale']

        # Additional parameters for scaling depended on analyses mode
        if derivative_mode != None:
            self.scale_d_dt=config['scale_dt']
            self.scale_d2_dx2=config['scale_dxx']

        # Interpolation to fixed size of the B-scan
        self.resize = Interpolate(size=512) # Fixed for our network design

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


        # Applying transformation to the bscan
        if self.transform is not None:
            bscan, depth = self.transform(bscan, depth)


        if self.derivative_mode=='mixed':
            
            d1=d1_dy(bscan) # First time derivative
            d1=d1/self.scale_d_dt # Normalization of time derivative

            
            d2=d2_dx2(bscan) # Second space derivative
            d2=d2/self.scale_d2_dx2 # Normalization of space derivative
        elif self.derivative_mode=='time':

            d1=d1_dy(bscan) # First time derivative
            d1=d1/self.scale_d_dt # Normalization of time derivative

            
            d2=d2_dy2(bscan) # Second time derivative
            d2=d2/self.scale_d2_dt2 # For now not implemented
        elif self.derivative_mode=='space':
            
            d1=d1_dx(bscan) # First space derivative
            d1=d1/self.scale_d_dx # For now not implemented 

            d2=d2_dx2(bscan) # Second space derivative
            d2=d2/self.scale_d2_dx2 # Normalization of space derivative
        else:
            # Formulating channels to martch RGB pretrained encoder
            bscan=bscan.repeat(3,1,1)             # [ 3, 512, W] We repeat thermal values to 3 channels
    
        if self.derivative_mode != None:
            d1 = d1.unsqueeze(0)  # [ 1, H, W]
            d1 = self.resize(d1)

            d2 = d2.unsqueeze(0)  # [ 1, H, W]
            d2 = self.resize(d2)


        bscan = bscan.unsqueeze(0)  # [ 1, H, W]
        bscan = self.resize(bscan)

        bscan=torch.concatenate((bscan,d1,d2),dim=0)
        
        bscan = bscan.float()
        depth=depth.float()
        
        return bscan, depth
    
