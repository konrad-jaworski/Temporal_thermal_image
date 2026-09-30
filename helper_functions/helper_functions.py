import random
import torch
import torch.nn.functional as F

# Augmentation: -----------------------------------------------------------------------------------

class NoiseAdditionExperiment:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, bscan, depth):
        noise = torch.randn_like(bscan) * self.sigma
        bscan_noisy = (bscan + noise).clamp_min(0.0)
        return bscan_noisy, depth
    
class RandomHorizontalFlipBscan:
    """
    Randomly flips B-scan data along spatial width (W).
    Applies consistently to:
      - X: [T, W]
      - mask: [W]
      - depth: [W]
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, X, mask):
        """
        Parameters:
        -----------
        X : torch.Tensor
            Shape [T, W]
        mask : torch.Tensor
            Shape [W]

        Returns:
        --------
        X, mask, depth : torch.Tensor
            Possibly flipped tensors
        """
        if random.random() < self.p:
            X = torch.flip(X, dims=[-1])      # flip W
            mask = torch.flip(mask, dims=[-1])
        return X, mask   

class HorizontalShift:
    """
    Shift sample horizontally using reflect padding.
    X: (H, W)
    mask: (W,)
    """

    def __init__(self, p=0.5, min_shift=1, max_shift=64):
        self.p = p
        self.min_shift = min_shift
        self.max_shift = max_shift

    def __call__(self, X, mask):
        if random.random() >= self.p:
            return X, mask

        H, W = X.size()
        idx = int(torch.randint(self.min_shift, self.max_shift, (1,)).item())
        direction = random.choice(["left", "right"])

        if direction == "left":
            # pad right, crop left
            X_pad = F.pad(X.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, idx:idx+W]

            mask_pad = F.pad(mask.unsqueeze(0), (0, idx), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[idx:idx+W]

        else:  # right
            # pad left, crop right
            X_pad = F.pad(X.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            X_shifted = X_pad[:, :W]

            mask_pad = F.pad(mask.unsqueeze(0), (idx, 0), mode="reflect").squeeze(0)
            mask_shifted = mask_pad[:W]

        return X_shifted, mask_shifted
   
# Interpolation and extrapolation methods: -----------------------------------------------------------------------------------

def resize_tensor(tensor, height, width):
    """Resize [H,W], [C,H,W], or [B,C,H,W] tensor."""

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
    Interpolate data to a new size using PyTorch's interpolate function. It will interpolate both dimmensions of bscan
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        
        data_interpolated = torch.nn.functional.interpolate(data.unsqueeze(0), size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0)
    
class Interpolate_mask:
    """
    Interpolate mask, it will use nearest neighbours to resize mask to fit into the bscan data.
    We should input already torch format data.
    """

    def __init__(self,size,mode='nearest'):
        self.size=size
        self.mode=mode

    def __call__(self,data):
        data_resize=F.interpolate(data[None,None,:],
                                  size=self.size,
                                  mode=self.mode
                                  ).squeeze(0).squeeze(0)
        
        return data_resize
