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
        transform=None,
        dtype=torch.float32,
        normalization_path=None,
        derivative_mode=None  # None, 'mixed','time','space'
    ):
        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.dtype = dtype
        self.path = normalization_path
        self.derivative_mode = derivative_mode

        config = np.load(self.path, allow_pickle=True)
        self.scale = config["scale"]

        if self.derivative_mode is not None:
            self.scale_d_dt = config["scale_dt"]
            self.scale_d2_dx2 = config["scale_dxx"]
            self.scale_d_dx = config["scale_dx"]
            self.scale_d2_dt2 = config['scale_dtt']

        self.resize = Interpolate(size=512)
        self.files = sorted(os.listdir(bscan_dir))

        for f in self.files:
            if not os.path.exists(os.path.join(depth_dir, f)):
                raise FileNotFoundError(f"Missing depth file for {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        bscan = np.load(os.path.join(self.bscan_dir, fname))   # [H, W]
        depth = np.load(os.path.join(self.depth_dir, fname))   # [W]

        assert bscan.ndim == 2, "B-scan must be 2D"
        assert depth.ndim == 1, "Depth must be 1D"
        assert bscan.shape[1] == depth.shape[0], "Width mismatch between bscan and depth"

        bscan = torch.from_numpy(bscan).to(self.dtype)
        depth = torch.from_numpy(depth).to(self.dtype)

        # Augmentation on base channel only, before derivatives
        if self.transform is not None:
            bscan, depth = self.transform(bscan, depth)

        # Base preprocessing
        bscan = torch.log1p(bscan)
        bscan = bscan / self.scale

        # --------------------------------------------------
        # No derivative channels: repeat base image to 3 channels
        # --------------------------------------------------
        if self.derivative_mode is None:
            bscan = bscan.unsqueeze(0)     # [1, H, W]
            bscan = self.resize(bscan)     # [1, 512, 512] if your resize does both axes
            bscan = bscan.repeat(3, 1, 1)  # [3, 512, 512]

            return bscan.float(), depth.float()

        # --------------------------------------------------
        # Mixed derivative mode: [X, dX/dt, d2X/dx2]
        # --------------------------------------------------
        if self.derivative_mode == "mixed":
            d1 = d1_dy(bscan) / self.scale_d_dt
            d2 = d2_dx2(bscan) / self.scale_d2_dx2

        elif self.derivative_mode== "time":
            d1=d1_dy(bscan)/self.scale_d_dt
            d2=d2_dy2(bscan)/self.scale_d2_dt2

        elif self.derivative_mode=="space":
            d1=d1_dx(bscan)/self.scale_d_dx
            d2=d2_dx2(bscan)/self.scale_d2_dx2

        else:
            raise NotImplementedError(
                f"derivative_mode='{self.derivative_mode}' not implemented yet. "
                "Use None or 'mixed' or 'time' or 'space'."
            )

        # Resize each channel separately
        bscan = bscan.unsqueeze(0)   # [1, H, W]
        d1 = d1.unsqueeze(0)         # [1, H, W]
        d2 = d2.unsqueeze(0)         # [1, H, W]

        bscan = self.resize(bscan)
        d1 = self.resize(d1)
        d2 = self.resize(d2)

        x = torch.cat((bscan, d1, d2), dim=0)   # [3, 512, 512]

        return x.float(), depth.float()
    
