import numpy as np
from pathlib import Path
import torch
import torch.utils.data as data


class USGrid:
    def __init__(self, grid_dict: dict):

        # 1. Extract and Flatten (Remove the (1, N) dimension)
        self.x_sos = grid_dict['x_sos'].flatten()
        self.z_sos = grid_dict['z_sos'].flatten()

        self.x_dt = grid_dict['x_dt'].flatten()
        self.z_dt = grid_dict['z_dt'].flatten()

        # 2. Store Resolutions
        self.res_sos = (len(self.z_sos), len(self.x_sos)) # (64, 64)
        self.res_dt  = (len(self.z_dt), len(self.x_dt))   # (128, 128)

        # 3. Compute Boundaries for Normalization
        self.x_min, self.x_max = self.x_sos.min(), self.x_sos.max()
        self.z_min, self.z_max = self.z_sos.min(), self.z_sos.max()


    def normalize(self, x, z):
        """ 
            Normalize coordinates to [-1, 1] range
            Formula: 2 * (val - min) / (max - min) - 1
        """
        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        z_norm = 2 * (z - self.z_min) / (self.z_max - self.z_min) - 1
        return x_norm, z_norm
    

    def denormalize(self, x, z):
        """ 
            Denormalize coordinates from [-1, 1] range back to original
            Formula: 0.5 * (val + 1) * (max - min) + min
        """
        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        z_norm = 2 * (z - self.z_min) / (self.z_max - self.z_min) - 1
        return x_norm, z_norm
    

    def __str__(self) -> str:
         return f"SoS Grid: {self.res_sos} | Range X: [{self.x_min:.4f}, {self.x_max:.4f}]" + \
                f"DT  Grid: {self.res_dt} | Range Z: [{self.z_min:.4f}, {self.z_max:.4f}]"