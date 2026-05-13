"""Diagnostic: trace the orientation of GT vs external baselines from .mat to display.

Run on the server. Prints raw shapes from h5py, scipy, and load_mat; then runs the
C-flatten → F-reshape round-trip on a single sample so we can see exactly what
the plotting code receives.

No fix is applied here — pure inspection.
"""
import numpy as np
import h5py
import sys
from inr_sos.io.paths import DATA_DIR
from inr_sos.io.utils import load_mat

DATA_FILE      = DATA_DIR + "/DL-based-SoS/train-VS-8pairs-IC-081225.mat"
BASELINES_FILE = DATA_DIR + "/DL-based-SoS/train_IC_10k_l2rec_l1rec_imcon.mat"

IDX = 1   # same sample you've been inspecting
GRID = (64, 64)


def corners(arr2d, name):
    """Print the four corners of a 2D image to fingerprint its orientation."""
    return (f"{name}: shape={arr2d.shape}, "
            f"[0,0]={arr2d[0,0]:.6f}  [0,-1]={arr2d[0,-1]:.6f}  "
            f"[-1,0]={arr2d[-1,0]:.6f}  [-1,-1]={arr2d[-1,-1]:.6f}")


def roundtrip(arr2d, label):
    """Apply the C-flatten -> F-reshape round-trip the plotting code uses."""
    flat = arr2d.flatten()                            # order='C' (numpy default)
    back = flat.reshape(GRID, order="F")
    return corners(back, f"{label}.flatten().reshape(F)")


print("=" * 72)
print("1. GT from data file (h5py direct read — same as embedded path)")
print("=" * 72)
with h5py.File(DATA_FILE, "r") as f:
    print("  Keys present:", list(f.keys())[:20])
    arr = np.array(f["imgs_gt"])
    print(f"  imgs_gt raw shape: {arr.shape}")
    gt_img = arr[IDX]
    print(f"  imgs_gt[{IDX}] shape: {gt_img.shape}")
    print(" ", corners(gt_img, "gt_img"))
    print(" ", roundtrip(gt_img, "gt_img"))

print()
print("=" * 72)
print("2. External L1 via h5py direct read (what the EMBEDDED path would do)")
print("=" * 72)
try:
    with h5py.File(BASELINES_FILE, "r") as f:
        print("  Keys present:", list(f.keys())[:20])
        arr = np.array(f["all_slowness_recons_l1"])
        print(f"  raw shape: {arr.shape}")
        img = arr[IDX]
        print(f"  [{IDX}] shape: {img.shape}")
        print(" ", corners(img, "l1_img_h5py"))
        print(" ", roundtrip(img, "l1_img_h5py"))
except Exception as e:
    print(f"  h5py.File failed (file may be v7-classic): {type(e).__name__}: {e}")

print()
print("=" * 72)
print("3. External L1 via load_mat (the CURRENT external path)")
print("=" * 72)
mat = load_mat(BASELINES_FILE)
print("  Keys present:", list(mat.keys())[:20])
arr = np.asarray(mat["all_slowness_recons_l1"])
print(f"  raw shape from load_mat: {arr.shape}")

# Apply the CURRENT _normalize_baseline_array logic (with fix as of 13:23)
N = 10000
if arr.ndim == 3:
    if arr.shape[2] == N:
        norm = arr.transpose(2, 1, 0)
        print(f"  shape[2]==N → transpose(2, 1, 0) → {norm.shape}")
    elif arr.shape[0] == N:
        norm = arr.transpose(0, 2, 1)
        print(f"  shape[0]==N → transpose(0, 2, 1) → {norm.shape}")
    else:
        norm = arr
        print(f"  no axis matches N={N}; used as-is")
elif arr.ndim == 2:
    norm = arr.T if arr.shape[1] == N else arr
    print(f"  2D → shape after normalize: {norm.shape}")
else:
    norm = arr

img = norm[IDX]
print(f"  normalized[{IDX}] shape: {img.shape}")
print(" ", corners(img, "l1_img_loadmat"))
print(" ", roundtrip(img, "l1_img_loadmat"))

print()
print("=" * 72)
print("4. Comparison — do GT and L1 (loadmat path) have the same orientation?")
print("=" * 72)
print("If GT [-1,-1] is in the BOTTOM-RIGHT of the GT display,")
print("then L1 corners should look 'similar geometrically' for the same sample.")
print("If they DON'T match orientation, the L1 image is being displayed transposed/rotated.")
