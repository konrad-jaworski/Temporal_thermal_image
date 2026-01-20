import numpy as np
import torch

class NoiseAddition:
    """
    Class to add Gaussian noise to data.
    """
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def add_noise(self, data):
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
        return (data - self.T_min) / (self.T_max - self.T_min)
    
    def denormalize(self, data_normalized):
        return data_normalized * (self.T_max - self.T_min) + self.T_min
    
class Interpolate:
    """
    Interpolate data to a new size using PyTorch's interpolate function.
    """
    def __init__(self, size, mode='bilinear'):
        self.size = size
        self.mode = mode

    def interpolate(self, data):
        data = data.unsqueeze(0).unsqueeze(0)  # add batch and channel dimensions
        data_interpolated = torch.nn.functional.interpolate(data, size=self.size, mode=self.mode, align_corners=False)
        return data_interpolated.squeeze(0).squeeze(0)  # remove batch and channel dimensions
    