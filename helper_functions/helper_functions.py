import numpy as np
import random
import torch

class NoiseAddition:
    """
    Class to add Gaussian noise to data.
    """
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise
    
class DataNormalization:
    """
    Normalize and denormalize data based on provided min and max values.
    """
    def __init__(self,deltaT_min,deltaT_max):
        self.deltaT_min=deltaT_min
        self.deltaT_max=deltaT_max

    def normalize(self, data):
        return (data - self.deltaT_min) / (self.deltaT_max - self.deltaT_min)
    
    def denormalize(self, data_normalized):
        return data_normalized * (self.deltaT_max - self.deltaT_min) + self.deltaT_min
    
class Interpolate:
    """
    Interpolate data to a new size using PyTorch's interpolate function.
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def __call__(self, data):
        
        data_interpolated = torch.nn.functional.interpolate(data, size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated
    
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
        depth : torch.Tensor
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