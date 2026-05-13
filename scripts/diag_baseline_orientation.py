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

IDX = 860   # sample with a visible inclusion (curved blob, lower-left in display)
GRID = (64, 64)


def fingerprint(arr2d, name):
    """Print pixels along an asymmetric line so any transpose/flip becomes visible."""
    a = arr2d
    return (f"{name}: shape={a.shape}, dtype={a.dtype}\n"
            f"    argmax_idx={np.unravel_index(np.argmax(a), a.shape)}  "
            f"argmin_idx={np.unravel_index(np.argmin(a), a.shape)}\n"
            f"    diag5      = {[float(a[i, i]) for i in (0, 16, 32, 48, 63)]}\n"
            f"    row30_cols = {[float(a[30, c]) for c in (0, 16, 32, 48, 63)]}\n"
            f"    col30_rows = {[float(a[r, 30]) for r in (0, 16, 32, 48, 63)]}")


def roundtrip(arr2d, label):
    """Apply the C-flatten -> F-reshape round-trip the plotting code uses."""
    flat = arr2d.flatten()
    back = flat.reshape(GRID, order="F")
    return fingerprint(back, f"{label}.flatten().reshape(F)")


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
print("4. Direct equality — does the load_mat+normalize path produce the same")
print("   per-sample array as raw h5py read? If YES, my fix is correct.")
print("=" * 72)

# Recompute both for the same IDX
with h5py.File(BASELINES_FILE, "r") as f:
    h5_img = np.array(f["all_slowness_recons_l1"])[IDX]
loadmat_img = norm[IDX]

print(f"  h5py shape:    {h5_img.shape}")
print(f"  loadmat shape: {loadmat_img.shape}")
print(f"  np.allclose(h5py, loadmat)        = {np.allclose(h5_img, loadmat_img)}")
print(f"  np.allclose(h5py, loadmat.T)      = {np.allclose(h5_img, loadmat_img.T)}")
print(f"  np.allclose(h5py, np.flipud(lm))  = {np.allclose(h5_img, np.flipud(loadmat_img))}")
print(f"  np.allclose(h5py, np.fliplr(lm))  = {np.allclose(h5_img, np.fliplr(loadmat_img))}")
print(f"  np.allclose(h5py, rot90(lm,k=1))  = {np.allclose(h5_img, np.rot90(loadmat_img, 1))}")
print(f"  np.allclose(h5py, rot90(lm,k=2))  = {np.allclose(h5_img, np.rot90(loadmat_img, 2))}")
print(f"  np.allclose(h5py, rot90(lm,k=3))  = {np.allclose(h5_img, np.rot90(loadmat_img, 3))}")
print()
print("  Which of the above is True tells us exactly how load_mat's path relates")
print("  to the h5py-direct path — and therefore exactly what transform to apply")
print("  to make the external baseline orientation match the embedded one.")
