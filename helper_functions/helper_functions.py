"""
Data augmentations and resizing utilities for thermal B-scans.

All augmentations take and return a pair (B-scan, depth target):
    B-scan : torch.Tensor [T, W]  (time x position along the scan line)
    depth  : torch.Tensor [W]     (defect depth under each position)
Geometric augmentations (flip, shift) move the B-scan and its target
together, so the label stays aligned with the image. They are combined with
data.data_operators.ComposeBScanTransforms and applied inside
BScanDepthDataset, on raw temperature values, before normalisation.

The resizing classes bring B-scans of any size to the fixed input size of
the network: bilinear interpolation for the B-scan, nearest-neighbour for
the depth target.
"""

import random
import torch
import torch.nn.functional as F


class NoiseAdditionExperiment:
    """
    Adds Gaussian noise to every pixel of the B-scan, to imitate detector
    noise of a real infrared camera.

    The noise is applied before normalisation, so `sigma` is in the same
    units as the temperature rise (Delta T). After adding the noise, negative
    values are clipped to 0, matching the clipping of Delta T used elsewhere
    in the pipeline. The depth target is returned unchanged.
    """

    def __init__(self, sigma):
        # Standard deviation of the noise, in temperature units.
        self.sigma = sigma

    def __call__(self, bscan, depth):
        noise = torch.randn_like(bscan) * self.sigma
        bscan_noisy = (bscan + noise).clamp_min(0.0)
        return bscan_noisy, depth

class RandomHorizontalFlipBscan:
    """
    Mirrors the B-scan and its depth target along the spatial axis (W) with
    probability p.

    A defect seen from the other side of the scan line produces the mirrored
    thermal response, so this adds new valid samples without changing the
    physics. The time axis is never flipped, since the cooling curve has a
    fixed direction in time.
    """

    def __init__(self, p=0.5):
        # Probability that a given sample is flipped.
        self.p = p

    def __call__(self, X, mask):
        """
        Parameters
        ----------
        X : torch.Tensor
            B-scan, shape [T, W].
        mask : torch.Tensor
            Depth target, shape [W].

        Returns
        -------
        X, mask : torch.Tensor
            Both flipped along W, or both unchanged.
        """
        if random.random() < self.p:
            # The last dimension is the spatial one for both tensors.
            X = torch.flip(X, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        return X, mask

class HorizontalShift:
    """
    Moves the B-scan and its depth target sideways by a random number of
    pixels, with probability p.

    This teaches the network that a defect can appear at any position along
    the scan line. The width W is kept: the side that is emptied by the shift
    is filled with a mirror image of the neighbouring pixels (reflect
    padding), so no artificial zero-temperature band is created. The same
    padding is applied to the target, so image and label stay aligned.

    X: B-scan [T, W]
    mask: depth target [W]
    """

    def __init__(self, p=0.5, min_shift=1, max_shift=64):
        self.p = p
        # The shift is drawn from [min_shift, max_shift - 1] pixels
        # (torch.randint excludes the upper bound).
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, X, mask):
        if random.random() >= self.p:
            return X, mask

        H, W = X.size()
        idx = int(torch.randint(self.min_shift, self.max_shift, (1,)).item())
        direction = random.choice(["left", "right"])

        # F.pad in "reflect" mode needs a leading batch/channel dimension,
        # hence unsqueeze(0) before padding and squeeze(0) after it.
        if direction == "left":
            # Content moves left by idx: pad idx mirrored columns on the
            # right, then drop the first idx columns.
            X_pad = F.pad(X.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, idx:idx+W]

            mask_pad = F.pad(mask.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[idx:idx+W]

        else:
            # Content moves right by idx: pad idx mirrored columns on the
            # left, then keep the first W columns.
            X_pad = F.pad(X.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, :W]

            mask_pad = F.pad(mask.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[:W]

        return X_shifted, mask_shifted


def resize_tensor(tensor, height, width):
    """
    Bilinear resize of a 2D image, or a batch of them, to [height, width].

    F.interpolate only accepts 4D input [B, C, H, W], so 2D [H, W] and 3D
    [C, H, W] tensors are expanded to 4D first and returned in their
    original number of dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        [H, W], [C, H, W] or [B, C, H, W].
    height, width : int
        Output size.

    Returns
    -------
    torch.Tensor
        Same number of dimensions as the input, float32, with the last two
        dimensions equal to (height, width).
    """

    original_ndim = tensor.ndim

    if original_ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif original_ndim != 4:
        raise ValueError("Tensor must have 2, 3, or 4 dimensions.")

    tensor = F.interpolate(
        tensor.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    if original_ndim == 2:
        return tensor.squeeze(0).squeeze(0)

    if original_ndim == 3:
        return tensor.squeeze(0)

    return tensor


class Interpolate:
    """
    Resizes a B-scan to the network input size.

    Both axes (time and space) are interpolated, bilinearly by default. With
    an integer `size` the output is square, e.g. [C, T, W] -> [C, 512, 512].
    This is how B-scans with different numbers of frames and widths are all
    given the same shape.
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        # data: [C, T, W]. A batch dimension is added for F.interpolate and
        # removed again afterwards.
        data_interpolated = torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0)

class Interpolate_mask:
    """
    Resizes a 1D depth target [W] to the same width as the resized B-scan.

    Nearest-neighbour interpolation is used, so every output value is one of
    the input depths. Linear interpolation would create intermediate depths
    at defect edges that do not exist in the specimen.
    Input must already be a torch tensor.
    """

    def __init__(self,size,mode='nearest'):
        self.size=size
        self.mode=mode

    def __call__(self,data):
        # [W] -> [1, 1, W] (the 3D layout F.interpolate expects for 1D
        # signals) -> [1, 1, size] -> [size].
        data_resize=F.interpolate(data[None,None,:],
                                  size=self.size,
                                  mode=self.mode
                                  ).squeeze(0).squeeze(0)

        return data_resize