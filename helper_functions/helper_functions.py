import numpy as np
import torch

class NoiseAddition:
    """
    Class to add Gaussian noise to data.
    """
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def add_noise(self, data):
        noise = torch.randn_like(data) * self.noise_level
        return data + noise
    
class DataNormalization:
    """
    
    """
    def __init__(self,deltaT_min,deltaT_max):
        self.deltaT_min=deltaT_min
        self.deltaT_max=deltaT_max

    def normalize(self, data):
        return (data - self.T_min) / (self.T_max - self.T_min)
    
    def denormalize(self, data_normalized):
        return data_normalized * (self.T_max - self.T_min) + self.T_min