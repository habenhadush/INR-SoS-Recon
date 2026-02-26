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


def load_sample(filepath, idx):
    """
    Loads a single sample from an HDF5 data file, returning all available fields.

    Always-present keys:
        's_gt'       : ndarray (4096,) — ground truth slowness
        'd_meas'     : ndarray (131072,) — measurements
        'mask'       : ndarray (131072,) — validity mask (1=valid, 0=invalid)

    Optional keys (present only when the corresponding dataset exists in the file):
        's_l1_recon'  : ndarray (4096,) — L1 benchmark reconstruction
        's_l2_recon'  : ndarray (4096,) — L2 benchmark reconstruction
        'correlation' : ndarray (131072,) — correlation vector
    """
    with h5py.File(filepath, 'r') as f:
        s_raw = f['imgs_gt'][idx]
        s_vec = s_raw.flatten()

        d_vec = f['measmnts'][idx]

        nan_vals = f['nanidx'][idx]
        mask_vec = 1.0 - nan_vals

        result = {
            's_gt': s_vec,
            'd_meas': d_vec,
            'mask': mask_vec,
        }

        if 'all_slowness_recons_l1' in f:
            result['s_l1_recon'] = f['all_slowness_recons_l1'][idx].flatten()
        if 'all_slowness_recons_l2' in f:
            result['s_l2_recon'] = f['all_slowness_recons_l2'][idx].flatten()
        if 'all_correlation_vector' in f:
            result['correlation'] = f['all_correlation_vector'][idx].flatten()

    return result


def load_ic_batch(filepath, idx):
    """
    Backward-compatible wrapper around load_sample().
    Returns the original 3-tuple (s_vec, d_vec, mask_vec).
    """
    sample = load_sample(filepath, idx)
    return sample['s_gt'], sample['d_meas'], sample['mask']


def load_L_matrix(filepath):
    """
    Loads the A matrix from an HDF5 file, handling three cases:
      1. Dense Dataset 'A' → read and transpose to (n_measurements, n_pixels)
      2. Sparse Group 'A' (data/ir/jc) → reconstruct CSC matrix (already correct shape)
      3. Absent 'A' → return None
    """
    with h5py.File(filepath, 'r') as f:
        if 'A' not in f:
            return None

        node = f['A']

        # Sparse Group: reconstruct CSC matrix
        if isinstance(node, h5py.Group):
            keys = list(node.keys())
            if 'data' in keys and 'ir' in keys and 'jc' in keys:
                data = node['data'][:]
                ir = node['ir'][:]
                jc = node['jc'][:]
                n_cols = len(jc) - 1
                n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
                sparse_A = csc_matrix((data, ir, jc), shape=(n_rows, n_cols))
                return sparse_A.toarray()
            else:
                raise ValueError(f"'A' is a Group but missing sparse keys (data/ir/jc). Found: {keys}")

        # Dense Dataset: read and transpose if needed
        A_matrix = np.array(node)
        if A_matrix.shape[0] < A_matrix.shape[1]:
            A_matrix = A_matrix.T
        return A_matrix


def load_metadata(filepath):
    """
    Extract scalar/small-array metadata from an HDF5 data file.

    Returns a dict with available keys from: 'bf_sos', 'pix2time', 'reg_param', 'MaskSoS'.
    Missing keys are omitted (caller checks with .get()).
    """
    metadata = {}
    scalar_keys = ['bf_sos', 'pix2time', 'reg_param']
    array_keys = ['MaskSoS']

    with h5py.File(filepath, 'r') as f:
        for key in scalar_keys:
            if key in f:
                val = np.array(f[key])
                metadata[key] = float(val.flat[0])
        for key in array_keys:
            if key in f:
                val = np.array(f[key])
                if val.ndim >= 2:
                    val = val.T
                metadata[key] = val

    return metadata