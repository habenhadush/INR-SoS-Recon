import scipy.io
import h5py
import numpy as np
from scipy.sparse import csc_matrix
from pathlib import Path
from joblib import Memory
from inr_sos import DATA_CACHE

memory = Memory(DATA_CACHE, verbose=0)

@memory.cache
def load_mat(filepath):
    """
    Robust MATLAB loader with disk caching.
    Fixes the KeyError by preserving structure and auto-building sparse matrices.
    Cached results are stored on disk to avoid reloading large files.
    """
    filepath = Path(filepath)
    
    try:
        return scipy.io.loadmat(str(filepath))
    except NotImplementedError:
        pass  #  HDF5
    
    # 2. HDF5 Handler (v7.3)
    print(f"Detected v7.3 MATLAB file: {filepath.name}")
    
    def parse_h5_node(node):
        # A. Handle Groups (Potential Sparse Matrices)
        if isinstance(node, h5py.Group):
            keys = list(node.keys())
            
            # Check: Is this a Sparse Matrix? (MATLAB saves them as Groups with these 3 keys)
            if 'data' in keys and 'ir' in keys and 'jc' in keys:
                print(f"  -> Reconstructing sparse matrix: {node.name}")
                data = node['data'][:]
                ir = node['ir'][:]
                jc = node['jc'][:]
                
                # Auto-Infer Shape: Columns = len(jc) - 1
                n_cols = len(jc) - 1
                n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
                
                # Create the matrix
                return csc_matrix((data, ir, jc), shape=(n_rows, n_cols))
            
            # If not sparse, just recurse deeper
            return {k: parse_h5_node(node[k]) for k in keys}
        
        # B. Handle Datasets (Dense Arrays)
        elif isinstance(node, h5py.Dataset):
            val = np.array(node)
            if val.ndim >= 2:
                return val.T
            return val
        
        return None
    
    with h5py.File(filepath, 'r') as f:
        # strip '#refs#'
        data = {k: parse_h5_node(f[k]) for k in f.keys() if k != '#refs#'}
    
    return data


def clear_cache():
    """Clear the disk cache if needed"""
    memory.clear(warn=False)

def inspect_mat_fileheader(filepath: str):
    filepath = Path(filepath)
    try:
        with h5py.File(filepath, 'r') as f:
            print("\n[HDF5 Format Detected]")
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Key: '{name}' | Shape: {obj.shape} | Type: Dataset")
                elif isinstance(obj, h5py.Group):
                    print(f"  Key: '{name}' | Type: Group")
            f.visititems(print_structure)
    except OSError:
        print("\n[Scipy Format Detected (v7.2 or older)]")
        try:
            data = scipy.io.loadmat(filepath)
            for key, val in data.items():
                if not key.startswith('__'):
                    if isinstance(val, np.ndarray):
                        print(f"  Key: '{key}' | Shape: {val.shape}")
                    else:
                        print(f"  Key: '{key}' | Type: {type(val)}")
        except Exception as e:
            print(f"Error loading file: {e}")


def load_ic_batch(filepath, idx):
    """
    Loads a single sample from 'train-VS-8pairs-IC-081225.mat'
    Handles the specific reshaping and transposing required for this file.
    """
    with h5py.File(filepath, 'r') as f:
        # 1. Load Ground Truth (Batch, 64, 64) -> Flatten to (4096,)
        # Note: We take sample 'idx'.
        s_raw = f['imgs_gt'][idx] # Shape (64, 64)
        s_vec = s_raw.flatten()   # Shape (4096,)
        
        # 2. Load Measurements (Batch, 131072) -> (131072,)
        d_vec = f['measmnts'][idx] 
        
        # 3. Load Mask (Batch, 131072)
        # 'nanidx': 1 = Invalid/NaN, 0 = Valid
        nan_vals = f['nanidx'][idx]
        mask_vec = 1.0 - nan_vals # Invert: 1 = Valid Data
        
    return s_vec, d_vec, mask_vec


def load_L_matrix(filepath):
    """
    Loads the A matrix just once (since it's the same for all patients).
    """
    with h5py.File(filepath, 'r') as f:
        # Shape is (4096, 131072) -> We need (131072, 4096)
        # We read it and then Transpose (.T)
        A_matrix = f['A'][:] 
        return A_matrix.T