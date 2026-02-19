import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from inr_sos.io.utils import load_mat  
from inr_sos.utils.params import USGrid 
import logging

class USDataset(Dataset):

    def __init__(self, data_path, grid_path, matrix_path=None, use_external_L_matrix=False, paths_per_batch=None):
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
        self.h5_file = None  # Will be used for lazy loading
        self.paths_per_batch = paths_per_batch

        # 1. SETUP GEOMETRY
        # Load grid parameters to handle coordinate normalization
        grid_dict = load_mat(grid_path)
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

        # 3. LOAD MATRIX L (The Logic Update)
        if use_external_L_matrix:
            if matrix_path is None:
                logging.error("matrix_path must be provided when use_external_L_matrix is True.")
            logging.info("Loading L-Matrix from external file: {}".format(matrix_path))
            L_data = load_mat(matrix_path)
            if 'L' not in L_data:
                logging.error("Matrix 'L' not found in the provided matrix file.")
            self.L_matrix = torch.tensor(L_data['L'], dtype=torch.float32)
            logging.info("L-Matrix loaded successfully with shape: {}".format(self.L_matrix.shape))
        else:
            logging.info(f"Loading L-Matrix from data file: {self.data_path}")
            with h5py.File(self.data_path, 'r') as f:
                if 'A' in f:
                    logging.info("Loading L-Matrix from data file...")
                    A_data = np.array(f['A'])
                    if A_data.shape == (4096, 131072):
                        A_data = A_data.T
                    self.L_matrix = torch.tensor(A_data, dtype=torch.float32)
                    logging.info("L-Matrix loaded successfully with shape: {}".format(self.L_matrix.shape))

                else:
                    raise KeyError("Matrix 'A' not found in file and no matrix_path provided.")
                
        # 4. GET DATASET LENGTH
        with h5py.File(self.data_path, 'r') as f:
            self.length = f['measmnts'].shape[0]
            logging.info("Dataset initialized with {} samples.".format(self.length))

    def _open_h5_file(self):
        """Helper to open HDF5 file for lazy loading."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
       

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
        self._open_h5_file()

        # A. Load Ground Truth (4096, 1)
        s_raw = self.h5_file['imgs_gt'][idx] 
        s_vec = torch.tensor(s_raw.flatten(), dtype=torch.float32).unsqueeze(1)
        
        s_mean = s_vec.mean()
        s_std = s_vec.std()
        s_normalized = (s_vec - s_mean) / (s_std + 1e-8)
        
        # B. Load Measurements and Mask
        d_vec = torch.tensor(self.h5_file['measmnts'][idx], dtype=torch.float32).unsqueeze(1)
        nan_vals = self.h5_file['nanidx'][idx]
        mask_vec = torch.tensor(1.0 - nan_vals, dtype=torch.float32).unsqueeze(1)
    
        return {
            'coords': self.coords_norm, 
            's_gt_raw': s_vec, 
            's_gt_normalized': s_normalized,       
            's_stats': (s_mean, s_std),       
            'd_meas': d_vec,            
            'mask': mask_vec,           
            'L_matrix': self.L_matrix, 
            'idx': idx
        }      


class RayDataset(Dataset):
    def __init__(self, L_matrix, displacement_filed, mask) -> None:
        super().__init__()
        self.L_matrix = L_matrix
        self.displacement = displacement_filed  
        self.mask = mask 
        self.number_of_paths = L_matrix.shape[0]   
    
    def __len__(self):
        return self.number_of_paths
    
    def __getitem__(self, idx):
        L_row = self.L_matrix[idx, :]
        displacement_value = self.displacement[idx]
        mask_value = self.mask[idx]
        
        return {
            'L_row': L_row,
            'displacement': displacement_value,
            'mask': mask_value
        }
