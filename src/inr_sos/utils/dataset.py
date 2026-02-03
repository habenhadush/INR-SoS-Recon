import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from inr_sos.io.utils import load_mat  
from inr_sos.utils.params import USGrid 
class USDataset(Dataset):
    def __init__(self, data_path, grid_path, matrix_path=None):
        """
        PyTorch Dataset for Speed-of-Sound Inversion.
        
        Parameters
        ----------
        data_path : str
            Path to 'train-VS-8pairs-IC-081225.mat'
        grid_path : str
            Path to 'grid_parameters.mat'
        matrix_path : str (Optional)
            Path to 'L.mat' or derived from data file. 
            If None, attempts to load 'A' from data_path.
        """
        self.data_path = Path(data_path)
        
        # 1. SETUP GEOMETRY (Fixed)
        # Load grid parameters to handle coordinate normalization
        grid_dict = load_mat(grid_path)
        # Map raw keys to USGrid expected keys if necessary
        clean_grid_dict = {
            'x_sos': grid_dict.get('xax_sos', grid_dict.get('x_sos')),
            'z_sos': grid_dict.get('zax_sos', grid_dict.get('z_sos')),
            'x_dt':  grid_dict.get('xDT', grid_dict.get('x_dt')),
            'z_dt':  grid_dict.get('zDT', grid_dict.get('z_dt'))
        }
        self.grid = USGrid(clean_grid_dict)
        
        # 2. PRE-CALCULATE NORMALIZED COORDINATES (Inputs)
        # The INR needs (x, z) inputs in range [-1, 1]. 
        # Since these are fixed, we generate them once here.
        # Shape: (4096, 2)
        X_mesh, Z_mesh = np.meshgrid(self.grid.x_sos, self.grid.z_sos)
        x_flat = X_mesh.flatten()
        z_flat = Z_mesh.flatten()
        
        x_norm, z_norm = self.grid.normalize(x_flat, z_flat)
        # Stack into (N, 2) tensor: [[x1, z1], [x2, z2], ...]
        self.coords_norm = torch.tensor(np.stack([x_norm, z_norm], axis=1), dtype=torch.float32)

        # 3. LOAD MATRIX L (Fixed)
        # Load 'A' matrix once. It fits in RAM (131k x 4k float32 ~= 2GB? No, 500MB)
        with h5py.File(self.data_path, 'r') as f:
            if 'A' in f:
                # Transpose needed: (4096, 131072) -> (131072, 4096)
                print("Loading and Transposing L-Matrix (A)...")
                self.L_matrix = torch.tensor(f['A'][:].T, dtype=torch.float32)
            else:
                # Fallback if A is in a separate file
                print("L-matrix not found in data file. Loading external...")
                # Add external loading logic here if needed
                self.L_matrix = None
                
        # 4. GET DATASET LENGTH
        with h5py.File(self.data_path, 'r') as f:
            self.length = f['measmnts'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Lazy load specific sample.
        Returns:
            coords: (4096, 2) - Input (x,z)
            s_gt:   (4096, 1) - Target Slowness
            d_meas: (131072, 1) - Input Data
            mask:   (131072, 1) - Validity Mask
        """
        with h5py.File(self.data_path, 'r') as f:
            # A. Load Ground Truth (Batch, 64, 64) -> Flatten -> (4096, 1)
            # Note: h5py slicing is fast.
            s_raw = f['imgs_gt'][idx] 
            s_vec = torch.tensor(s_raw.flatten(), dtype=torch.float32).unsqueeze(1)
            
            # B. Load Measurements (Batch, 131072) -> (131072, 1)
            d_vec = torch.tensor(f['measmnts'][idx], dtype=torch.float32).unsqueeze(1)
            
            # C. Load Mask (Batch, 131072) -> Invert -> (131072, 1)
            nan_vals = f['nanidx'][idx]
            mask_vec = torch.tensor(1.0 - nan_vals, dtype=torch.float32).unsqueeze(1)

        # Return dictionary for clarity
        return {
            'coords': self.coords_norm, # The "Question" (x,z)
            's_gt': s_vec,              # The "Answer" (Slowness) - Used for Validation
            'd_meas': d_vec,            # The "Observation" (Data) - Used for Loss
            'mask': mask_vec,           # The "Filter"
            'idx': idx
        }
    
    def get_L_matrix(self):
        """Helper to get L matrix to GPU memory"""
        return self.L_matrix