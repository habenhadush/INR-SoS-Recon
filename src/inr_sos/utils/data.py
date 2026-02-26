import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from scipy.sparse import csc_matrix
from inr_sos.io.utils import load_mat, load_metadata
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
                    node = f['A']
                    if isinstance(node, h5py.Group):
                        # Sparse Group: reconstruct CSC matrix
                        keys = list(node.keys())
                        if 'data' in keys and 'ir' in keys and 'jc' in keys:
                            logging.info("Reconstructing sparse L-Matrix from data file...")
                            data = node['data'][:]
                            ir = node['ir'][:]
                            jc = node['jc'][:]
                            n_cols = len(jc) - 1
                            n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
                            sparse_A = csc_matrix((data, ir, jc), shape=(n_rows, n_cols))
                            A_data = sparse_A.toarray()
                        else:
                            raise ValueError(f"'A' is a Group but missing sparse keys. Found: {keys}")
                    else:
                        # Dense Dataset
                        logging.info("Loading dense L-Matrix from data file...")
                        A_data = np.array(node)
                        if A_data.shape[0] < A_data.shape[1]:
                            A_data = A_data.T
                    self.L_matrix = torch.tensor(A_data, dtype=torch.float32)
                    logging.info("L-Matrix loaded successfully with shape: {}".format(self.L_matrix.shape))
                elif matrix_path is not None:
                    logging.info("'A' not in data file, loading from matrix_path: {}".format(matrix_path))
                    L_data = load_mat(matrix_path)
                    if 'L' not in L_data:
                        logging.error("Matrix 'L' not found in the provided matrix file.")
                    self.L_matrix = torch.tensor(L_data['L'], dtype=torch.float32)
                    logging.info("L-Matrix loaded successfully with shape: {}".format(self.L_matrix.shape))
                else:
                    logging.warning("Matrix 'A' not found in data file and no matrix_path provided. L_matrix set to None.")
                    self.L_matrix = None

        # 4. GET DATASET LENGTH & LOAD OPTIONAL FIELDS
        with h5py.File(self.data_path, 'r') as f:
            self.length = f['measmnts'].shape[0]
            logging.info("Dataset initialized with {} samples.".format(self.length))

            # Optional benchmark reconstructions
            if 'all_slowness_recons_l1' in f:
                self.benchmarks_l1 = np.array(f['all_slowness_recons_l1']).T if f['all_slowness_recons_l1'].ndim >= 2 else np.array(f['all_slowness_recons_l1'])
                logging.info(f"L1 benchmarks loaded with shape: {self.benchmarks_l1.shape}")
            else:
                self.benchmarks_l1 = None

            if 'all_slowness_recons_l2' in f:
                self.benchmarks_l2 = np.array(f['all_slowness_recons_l2']).T if f['all_slowness_recons_l2'].ndim >= 2 else np.array(f['all_slowness_recons_l2'])
                logging.info(f"L2 benchmarks loaded with shape: {self.benchmarks_l2.shape}")
            else:
                self.benchmarks_l2 = None

            # Optional correlation vectors
            if 'all_correlation_vector' in f:
                self.correlation_vectors = np.array(f['all_correlation_vector']).T if f['all_correlation_vector'].ndim >= 2 else np.array(f['all_correlation_vector'])
                logging.info(f"Correlation vectors loaded with shape: {self.correlation_vectors.shape}")
            else:
                self.correlation_vectors = None

        # 5. LOAD METADATA (bf_sos, pix2time, reg_param, MaskSoS)
        meta = load_metadata(str(self.data_path))
        self.bf_sos = meta.get('bf_sos', None)
        self.pix2time = meta.get('pix2time', None)
        self.reg_param = meta.get('reg_param', None)
        self.mask_sos = meta.get('MaskSoS', None)

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
    
        sample = {
            'coords': self.coords_norm,
            's_gt_raw': s_vec,
            's_gt_normalized': s_normalized,
            's_stats': (s_mean, s_std),
            'd_meas': d_vec,
            'mask': mask_vec,
            'L_matrix': self.L_matrix,
            'idx': idx
        }

        # Optional benchmark / correlation fields (additive only)
        if self.benchmarks_l1 is not None:
            sample['s_l1_recon'] = torch.tensor(
                self.benchmarks_l1[idx].flatten(), dtype=torch.float32
            ).unsqueeze(1)
        if self.benchmarks_l2 is not None:
            sample['s_l2_recon'] = torch.tensor(
                self.benchmarks_l2[idx].flatten(), dtype=torch.float32
            ).unsqueeze(1)
        if self.correlation_vectors is not None:
            sample['correlation'] = torch.tensor(
                self.correlation_vectors[idx].flatten(), dtype=torch.float32
            ).unsqueeze(1)

        return sample


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
