"""
PyTorch dataset for training B-net on thermal B-scans.

A thermal B-scan is a 2D slice taken out of a 3D thermographic sequence
[T, H, W]: one spatial line (a row or a column of the camera image) followed
over time, which gives an array of shape [T, W]. Every B-scan has a matching
depth vector of shape [W] holding the defect depth under each pixel of that
line (0 where there is no defect). Both are stored as .npy files with the same
file name in two separate folders; they are produced by
helper_functions/preparation_of_b_scans.py.

For each sample the dataset:
  1. loads the B-scan and its depth vector,
  2. optionally drops the heating frames so only the cooling phase is kept,
  3. applies the optional augmentations (flip / shift / noise),
  4. divides the B-scan by a single global scale computed on the training set
     (see helper_functions/Global_normalization_parameter_finder.py),
  5. resizes the B-scan to [resize_size, resize_size] and the depth vector to
     [resize_size], and copies the B-scan into 3 channels so it can be fed to
     ImageNet-pretrained encoders.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from helper_functions.helper_functions import (
    Interpolate,
    Interpolate_mask,
)


class ComposeBScanTransforms:
    """
    Chains several augmentations that act on a (B-scan, depth) pair.

    torchvision's Compose only passes a single image through the transforms.
    Here every transform receives and returns both the B-scan [T, W] and the
    depth vector [W], so geometric augmentations (e.g. a horizontal flip) are
    applied identically to the input and to its target.
    """

    def __init__(self, transforms):
        # List of callables with the signature t(bscan, depth) -> (bscan, depth).
        self.transforms = transforms

    def __call__(self, bscan, depth):
        # Transforms are applied in the order they were given in the list.
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
        Dataset returning (B-scan, depth profile) pairs for depth regression.

        Parameters
        ----------
        bscan_dir : str
            Folder with the B-scans, one .npy file per sample, shape [T, W]
            (T = number of frames, W = number of pixels along the scan line).
        depth_dir : str
            Folder with the depth vectors, shape [W]. File names must be
            identical to the ones in bscan_dir.
        transform : callable or None
            Augmentation applied to (bscan, depth) after the cooling cut and
            before normalisation, usually a ComposeBScanTransforms instance.
        normalization_path : str
            .npz file with the key 'scale': the maximum temperature rise
            found in the training set. Every B-scan is divided by it.
        cooling_phase : bool
            If True, frames before `cooling_frame` (the heating phase) are
            removed and only the cooling part of the sequence is used.
        cooling_frame : int
            Index of the first cooling frame (the frame at which the heat
            source is switched off).
        resize_size : int
            Size of the network input. The B-scan becomes
            [resize_size, resize_size] and the depth vector [resize_size].
        dtype : torch.dtype
            dtype used when converting the loaded numpy arrays.

        Returned sample
        ---------------
        x : torch.Tensor, [3, resize_size, resize_size]
            Normalised B-scan, time along dim 1, space along dim 2, repeated
            over 3 channels.
        depth : torch.Tensor, [resize_size]
            Depth profile along the scan line.
        """

        self.bscan_dir = bscan_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.normalization_path = normalization_path
        self.cooling_phase = cooling_phase
        self.cooling_frame = cooling_frame
        self.dtype = dtype

        # One global scale for the whole dataset, not a per-sample one. This
        # keeps the absolute amplitude of the temperature rise, which carries
        # depth information (deeper defects give a weaker thermal contrast).
        if self.normalization_path is not None:
            config = np.load(self.normalization_path, allow_pickle=True)
            if "scale" not in config:
                raise KeyError("Normalization file must contain key 'scale'.")
            self.scale = float(config["scale"])

        # B-scans do not share the same size (the number of frames depends on
        # the recording and cooling cut, the width on the specimen), so they
        # are all brought to a fixed square size for the network.
        # The B-scan is resized bilinearly. The depth vector uses
        # nearest-neighbour interpolation, so no artificial depth values are
        # created at the edges of a defect.
        self.resize = Interpolate(size=resize_size)
        self.resize_mask = Interpolate_mask(size=resize_size)

        # Sorted list of sample names, so the order is the same on every run.
        self.files = sorted(
            f for f in os.listdir(self.bscan_dir)
            if f.endswith(".npy")
        )

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files found in {self.bscan_dir}")

        # Check at start-up that every B-scan has its depth vector, rather
        # than failing in the middle of an epoch.
        for f in self.files:
            depth_path = os.path.join(self.depth_dir, f)
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Missing depth file for {f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # The B-scan and its depth vector share the same file name.
        fname = self.files[idx]

        bscan_path = os.path.join(self.bscan_dir, fname)
        depth_path = os.path.join(self.depth_dir, fname)

        bscan = np.load(bscan_path)   # [T, W]
        depth = np.load(depth_path)   # [W]

        bscan = torch.from_numpy(bscan).to(self.dtype)
        depth = torch.from_numpy(depth).to(self.dtype)

        # Keep only the cooling phase. The index is checked first so that a
        # wrong cooling_frame raises a clear error instead of silently
        # returning an empty tensor.
        if self.cooling_phase:
            if self.cooling_frame < 0 or self.cooling_frame >= bscan.shape[0]:
                raise ValueError(
                    f"Invalid cooling_frame={self.cooling_frame} for "
                    f"B-scan with {bscan.shape[0]} frames."
                )

            bscan = bscan[self.cooling_frame:, :]

        # Augmentations run on the raw temperature values, before scaling,
        # so e.g. the noise level in NoiseAdditionExperiment is expressed in
        # the same units as the measured data.
        if self.transform is not None:
            bscan, depth = self.transform(bscan, depth)

        # Global normalisation with the training-set maximum -> values roughly
        # in [0, 1].
        bscan_base = bscan / self.scale

        # Add a channel axis, resize to the network input size, then copy the
        # single channel 3 times because the encoders are ImageNet-pretrained
        # and expect RGB-like input.
        x = bscan_base.unsqueeze(0)     # [1, T, W]
        x = self.resize(x)              # [1, 512, 512]
        x = x.repeat(3, 1, 1)           # [3, 512, 512]

        # Depth vector resized to the same width as the B-scan.
        depth = self.resize_mask(depth) # [512]

        return x.float(), depth.float()