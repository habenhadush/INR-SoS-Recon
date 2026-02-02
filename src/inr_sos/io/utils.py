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
    
    # 1. Try generic Scipy Load (Fastest for v7.2 and older)
    try:
        return scipy.io.loadmat(str(filepath))
    except NotImplementedError:
        pass  # It's a v7.3 file, switch to HDF5
    
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
            # CAUTION: MATLAB saves dense arrays Transposed. We fix this here.
            if val.ndim >= 2:
                return val.T
            return val
        
        return None
    
    with h5py.File(filepath, 'r') as f:
        # We strip the strange '#refs#' group that MATLAB adds
        data = {k: parse_h5_node(f[k]) for k in f.keys() if k != '#refs#'}
    
    return data


def clear_cache():
    """Clear the disk cache if needed"""
    memory.clear(warn=False)