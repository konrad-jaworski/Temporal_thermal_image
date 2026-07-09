import os
import numpy as np
import torch
from torch.utils.data import Dataset

from helper_functions.helper_functions import (
    Interpolate,
    Interpolate_mask,
    d1_dx,
    d1_dy,
    d2_dx2,
    d2_dy2,
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
        dtype=torch.float32,
        normalization_path=None,
        projection_mode=None, # 'log1p','cos'
        derivative_mode=None,  # None, 'time', 'space', 'phase', 'phase_cos'
        cooling_phase=True,
        cooling_frame=11,
        log_scaling=True,
        resize_size=512,
        eps=1e-8,
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

        3. Negative baseline-removed values are clipped:
            bscan_pos = clamp(bscan, min=0)

        4. Base channel:
            if log_scaling:
                base = log1p(bscan_pos)
            else:
                base = bscan_pos / scale

        5. Derivatives:
            computed from bscan_pos, not from normalized/log base channel.
            derivative channels stay signed and are divided by positive
            derivative scales.

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

        derivative_mode : str or None
            None:
                repeat base channel 3 times.

            'time':
                channels = base, dT/dt, d2T/dt2

            'space':
                channels = base, dT/dx, d2T/dx2

            'phase':
                channels = base, base, phase/pi

            'phase_cos':
                channels = base, base, cos(phase)

        cooling_phase : bool
            If True, use only frames from cooling_frame onward.

        cooling_frame : int
            First frame of cooling phase.

        log_scaling : bool
            If True, use log1p transform and do not divide base channel
            by percentile/max scale.

        resize_size : int
            Target interpolation size.

        eps : float
            Minimum allowed normalization scale.
        """

        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.dtype = dtype
        self.normalization_path = normalization_path
        self.derivative_mode = derivative_mode
        self.cooling_phase = cooling_phase
        self.cooling_frame = cooling_frame
        self.log_scaling = log_scaling
        self.eps = eps
        self.projection_mode=projection_mode

        allowed_modes = [None, "time", "space", "phase", "phase_cos"]
        if self.derivative_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported derivative_mode={self.derivative_mode}. "
                f"Allowed modes are {allowed_modes}."
            )

        if self.normalization_path is not None:
            config = np.load(self.normalization_path, allow_pickle=True)
            if "scale" not in config:
                raise KeyError("Normalization file must contain key 'scale'.")
            
            self.scale = float(config["scale"]) # Only temperature scale is implemented
            self.scale_log1p=float(config["scale_log1p"])
            self._check_scale(self.scale, "scale")

        if self.derivative_mode in ["time", "space"]:
            required_keys = ["scale_dt", "scale_dxx", "scale_dx", "scale_dtt"]
            for key in required_keys:
                if key not in config:
                    raise KeyError(
                        f"Normalization file is missing key '{key}', "
                        f"required for derivative_mode='{self.derivative_mode}'."
                    )

            self.scale_d_dt = float(config["scale_dt"])
            self.scale_d2_dx2 = float(config["scale_dxx"])
            self.scale_d_dx = float(config["scale_dx"])
            self.scale_d2_dt2 = float(config["scale_dtt"])

            self._check_scale(self.scale_d_dt, "scale_dt")
            self._check_scale(self.scale_d2_dx2, "scale_dxx")
            self._check_scale(self.scale_d_dx, "scale_dx")
            self._check_scale(self.scale_d2_dt2, "scale_dtt")

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

    def _check_scale(self, value, name):
        """
        Check whether normalization scale is numerically meaningful.
        """

        if not np.isfinite(value):
            raise ValueError(f"{name} is not finite: {value}")

        if value < self.eps:
            raise ValueError(
                f"{name} is too small: {value}. "
                f"This suggests the scale was incorrectly computed or the "
                f"corresponding channel is nearly zero."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # We grab the folder
        fname = self.files[idx]
        # From it we grab bscan and the depth mask
        bscan_path = os.path.join(self.bscan_dir, fname)
        depth_path = os.path.join(self.depth_dir, fname)
        bscan = np.load(bscan_path)   # [T, W]
        depth = np.load(depth_path)   # [W]



        if bscan.ndim != 2:
            raise ValueError(f"B-scan must be 2D [T, W], got {bscan.shape} in {fname}")

        if depth.ndim != 1:
            raise ValueError(f"Depth must be 1D [W], got {depth.shape} in {fname}")

        if bscan.shape[1] != depth.shape[0]:
            raise ValueError(
                f"Width mismatch in {fname}: "
                f"bscan width={bscan.shape[1]}, depth width={depth.shape[0]}"
            )

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

        # --------------------------------------------------
        # Base channel preprocessing.
        #
        # If log_scaling=True:
        #     log1p already compresses dynamic range, so no scale division.
        #
        # If log_scaling=False:
        #     raw positive temperature rise is divided by global scale.
        # --------------------------------------------------


        if self.projection_mode==None:
            bscan_base = bscan / self.scale
        elif self.projection_mode=='log1p':
            bscan_base = torch.log1p(bscan)/self.scale_log1p
        elif self.projection_mode=='cos':
            bscan_base=(torch.cos(bscan)+1)/2  # Here we do not have anythig to normalize since cosine already produce values from -1 to 1 we just use common scale for every case
        else:
            raise ValueError(
                    f"Invalid projection mode!"
                )

        # --------------------------------------------------
        # No derivative channels: repeat base image to 3 channels.
        # --------------------------------------------------
        if self.derivative_mode is None:
            x = bscan_base.unsqueeze(0)     # [1, T, W]
            x = self.resize(x)              # [1, 512, 512]
            x = x.repeat(3, 1, 1)           # [3, 512, 512]

            depth = self.resize_mask(depth) # [512]

            return x.float(), depth.float() # it ends here if we would not investigate any channel augmentation.

        # --------------------------------------------------
        # Additional channels, in current version not used.
        # --------------------------------------------------
        if self.derivative_mode == "time":
            # Signed derivatives from raw positive baseline-removed data.
            d1 = d1_dy(bscan_pos) / self.scale_d_dt
            d2 = d2_dy2(bscan_pos) / self.scale_d2_dt2

        elif self.derivative_mode == "space":
            # Signed derivatives from raw positive baseline-removed data.
            d1 = d1_dx(bscan_pos) / self.scale_d_dx
            d2 = d2_dx2(bscan_pos) / self.scale_d2_dx2

        elif self.derivative_mode == "phase":
            # Phase calculated from raw positive baseline-removed data.
            # Phase is normalized to [-1, 1] by division by pi.
            phase_channel = torch.fft.fft(bscan_pos, dim=0)

            d1 = bscan_base
            d2 = torch.angle(phase_channel) / torch.pi

        elif self.derivative_mode == "phase_cos":
            # Cosine phase encoding.
            # torch.angle returns radians, so cosine should be applied directly.
            phase_channel = torch.fft.fft(bscan_pos, dim=0)

            d1 = bscan_base
            d2 = torch.cos(torch.angle(phase_channel))

        else:
            raise NotImplementedError(
                f"derivative_mode='{self.derivative_mode}' not implemented."
            )

        # --------------------------------------------------
        # Resize channels to network input size.
        # --------------------------------------------------
        bscan_base = bscan_base.unsqueeze(0)  # [1, T, W]
        d1 = d1.unsqueeze(0)                  # [1, T, W]
        d2 = d2.unsqueeze(0)                  # [1, T, W]

        bscan_base = self.resize(bscan_base)
        d1 = self.resize(d1)
        d2 = self.resize(d2)

        x = torch.cat((bscan_base, d1, d2), dim=0)  # [3, 512, 512]

        depth = self.resize_mask(depth)

        return x.float(), depth.float()