"""Diagnostic: empirically determine the column ordering L expects.

For the inverse_crime dataset, `d = L @ s_GT` holds exactly (no physics mismatch).
So whatever ordering of s_GT makes `||L @ s − d|| ≈ 0` is the one L expects.
We try several orderings and report the residual norms. The winner is the truth.
"""
import numpy as np
import h5py
import sys

from inr_sos.io.paths import DATA_DIR
from inr_sos.io.utils import load_mat

DATA_FILE = DATA_DIR + "/DL-based-SoS/train-VS-8pairs-IC-081225.mat"
IDX = 860


def report(label, residual):
    n = np.linalg.norm(residual)
    return f"  {label:40s} ||L@s - d||_2 = {n:.6e}"


with h5py.File(DATA_FILE, "r") as f:
    # 1. Read ground-truth image and measurement for sample IDX.
    s_raw = np.array(f["imgs_gt"][IDX])      # numpy shape after h5py: (64, 64)
    d     = np.array(f["measmnts"][IDX])     # (131072,)
    nan   = np.array(f["nanidx"][IDX]).astype(bool)  # invalid-ray mask

    # 2. Read L. Same logic as USDataset.
    A_node = f["A"]
    if isinstance(A_node, h5py.Group):
        from scipy.sparse import csc_matrix
        data = A_node["data"][:]
        ir   = A_node["ir"][:]
        jc   = A_node["jc"][:]
        n_cols = len(jc) - 1
        n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
        L = csc_matrix((data, ir, jc), shape=(n_rows, n_cols)).toarray()
    else:
        L = np.array(A_node)
        if L.shape[0] < L.shape[1]:
            L = L.T

print(f"sample idx = {IDX}")
print(f"  s_raw shape:  {s_raw.shape}")
print(f"  d shape:      {d.shape}")
print(f"  L shape:      {L.shape}")
print(f"  nan mask:     {nan.shape}, invalid count: {int(nan.sum())}")
print()

# Apply mask so the residual ignores invalid rays (where d is NaN).
valid = ~nan
d_clean = np.nan_to_num(d, nan=0.0)


def masked_residual(L_s):
    """Residual on valid rays only."""
    return (L_s - d_clean) * valid


# Build a few candidate slowness vectors via different flatten orderings.
candidates = {
    "s_raw.flatten() [C-order]":            s_raw.flatten(),
    "s_raw.flatten('F')":                   s_raw.flatten(order="F"),
    "s_raw.T.flatten() [C-order]":          s_raw.T.flatten(),
    "s_raw.T.flatten('F')":                 s_raw.T.flatten(order="F"),
    "np.flipud(s_raw).flatten()":           np.flipud(s_raw).flatten(),
    "np.fliplr(s_raw).flatten()":           np.fliplr(s_raw).flatten(),
    "np.rot90(s_raw, 1).flatten()":         np.rot90(s_raw, 1).flatten(),
    "np.rot90(s_raw, 2).flatten()":         np.rot90(s_raw, 2).flatten(),
    "np.rot90(s_raw, 3).flatten()":         np.rot90(s_raw, 3).flatten(),
    "np.rot90(s_raw, 1).flatten('F')":      np.rot90(s_raw, 1).flatten(order="F"),
    "np.rot90(s_raw, 3).flatten('F')":      np.rot90(s_raw, 3).flatten(order="F"),
}

print("=" * 75)
print("Residual norm under different slowness vector orderings")
print("(Lower is better; the ordering that L expects is the one with residual ~0.)")
print("=" * 75)
for label, s in candidates.items():
    if s.shape[0] != L.shape[1]:
        print(f"  {label:40s} SHAPE MISMATCH (s has {s.shape[0]}, L expects {L.shape[1]})")
        continue
    res = masked_residual(L @ s.astype(L.dtype))
    print(report(label, res))

print()
print("=" * 75)
print("Also sanity: norm of d itself (for context)")
print("=" * 75)
print(f"  ||d||_2 (valid rays only) = {np.linalg.norm(d_clean * valid):.6e}")
