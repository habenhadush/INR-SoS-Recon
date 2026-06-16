#!/usr/bin/env python3
"""
analyze_mismatch_residual.py
----------------------------
Generates the §5.2.3 forward-model-mismatch figures from the residual
    e = L @ s_GT - d_kwave
ported from notebooks/exploration-real-data.ipynb (cells 18/19/21/26).

Produces three thesis-quality SVGs (+ PNG fallback):
  1. mismatch_residual_heatmap.svg   — signed e per firing pair, per dataset
  2. mismatch_residual_spectrum.svg  — 2D log-magnitude spectrum per pair
  3. mismatch_residual_per_pair.svg  — fraction of ||e||^2 by firing pair

Must run on the server (needs DATA_DIR / HDF5 access).

Usage:
    python scripts/analyze_mismatch_residual.py
    python scripts/analyze_mismatch_residual.py --sample_idx 3 --n_samples 32
"""

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix

from inr_sos import DATA_DIR

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR / "data" / "residual_analysis"

KWAVE_GEOM_PATH = DATA_DIR + "/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat"
KWAVE_BLOB_PATH = DATA_DIR + "/DL-based-SoS/test_kWaveBlob_final.mat"

N_PAIRS = 8
PAIR_SIZE = 131072 // N_PAIRS  # 16384
DT = 128  # beamformed channel grid is 128 x 128

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "svg.fonttype": "none",
})


# ─── Data loading ────────────────────────────────────────────────────────────

def load_L_from_h5(path):
    """Load L-matrix handling dense Dataset or sparse CSC Group."""
    with h5py.File(path, "r") as f:
        if "A" not in f:
            return None
        node = f["A"]
        if isinstance(node, h5py.Group):
            data, ir, jc = node["data"][:], node["ir"][:], node["jc"][:]
            n_cols = len(jc) - 1
            n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
            return csc_matrix((data, ir, jc), shape=(n_rows, n_cols)).toarray()
        A = np.array(node)
        if A.shape[0] < A.shape[1]:
            A = A.T
        return A


def compute_mismatch(path, L, n_samples=None):
    """Compute e = d_meas - L @ s_GT per sample (valid rays only)."""
    with h5py.File(path, "r") as f:
        total = f["imgs_gt"].shape[0]
        n = min(n_samples or total, total)
        results = []
        for i in range(n):
            s_gt = f["imgs_gt"][i].flatten()
            d_meas = np.nan_to_num(f["measmnts"][i].flatten(), nan=0.0)
            mask = 1.0 - f["nanidx"][i].flatten()
            epsilon = (d_meas - L @ s_gt) * mask
            valid = mask > 0.5
            results.append({
                "epsilon": epsilon,
                "mask": mask,
                "eps_energy_pct": (np.sum(epsilon[valid] ** 2)
                                   / (np.sum(d_meas[valid] ** 2) + 1e-30) * 100),
            })
    return results


# ─── Figure 1: residual heatmap ──────────────────────────────────────────────

def fig_heatmap(mismatch_by_ds, sample_idx, save_path):
    """Signed residual e per firing pair, one row per dataset.

    Single shared colorbar across all panels: vmax = max 95th-percentile of
    |e| across both datasets, so geom and blob are on the same scale.
    """
    n_ds = len(mismatch_by_ds)
    fig, axes = plt.subplots(n_ds, N_PAIRS, figsize=(2.0 * N_PAIRS, 2.2 * n_ds),
                             squeeze=False)
    # One shared vmax across all rows for a single colorbar.
    vmax = 0.0
    for _, mm in mismatch_by_ds.items():
        eps = mm[sample_idx]["epsilon"]
        valid = mm[sample_idx]["mask"] > 0.5
        if valid.sum():
            vmax = max(vmax, float(np.percentile(np.abs(eps[valid]), 95)))
    if vmax == 0.0:
        vmax = 1e-7
    im = None
    for row, (label, mm) in enumerate(mismatch_by_ds.items()):
        eps = mm[sample_idx]["epsilon"]
        for p in range(N_PAIRS):
            sl = slice(p * PAIR_SIZE, (p + 1) * PAIR_SIZE)
            eps_img = eps[sl].reshape(DT, DT, order="F")
            ax = axes[row, p]
            im = ax.imshow(eps_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Pair {p}")
            if p == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.012, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, save_path)


# ─── Figure 2: residual spectrum ─────────────────────────────────────────────

def fig_spectrum(mismatch_by_ds, sample_idx, save_path):
    """2D log-magnitude spatial-frequency spectrum of e per firing pair.

    Single shared colorbar across all panels (max log-magnitude across both
    datasets); per-cell LF fraction annotation retained.
    """
    n_ds = len(mismatch_by_ds)
    fig, axes = plt.subplots(n_ds, N_PAIRS, figsize=(2.0 * N_PAIRS, 2.2 * n_ds),
                             squeeze=False)
    cy = cx = DT // 2
    Y, X = np.ogrid[:DT, :DT]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    low_mask = r <= 16

    # First pass: compute all log-magnitudes + LF fractions.
    panels = []  # (row, col, log_mag, low_frac)
    vmax = 0.0
    for row, (_, mm) in enumerate(mismatch_by_ds.items()):
        eps = mm[sample_idx]["epsilon"]
        mask = mm[sample_idx]["mask"]
        for p in range(N_PAIRS):
            sl = slice(p * PAIR_SIZE, (p + 1) * PAIR_SIZE)
            eps_img = eps[sl].reshape(DT, DT, order="F")
            mask_img = mask[sl].reshape(DT, DT, order="F")
            mag = np.abs(np.fft.fftshift(np.fft.fft2(eps_img * mask_img)))
            log_mag = np.log1p(mag)
            low_frac = (float(np.sum(mag[low_mask] ** 2))
                        / (float(np.sum(mag ** 2)) + 1e-30))
            vmax = max(vmax, float(log_mag.max()))
            panels.append((row, p, log_mag, low_frac))

    # Second pass: draw with shared vmin=0 / vmax = max across all panels.
    im = None
    for row, p, log_mag, low_frac in panels:
        ax = axes[row, p]
        im = ax.imshow(log_mag, cmap="viridis", vmin=0.0, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"Pair {p}")
        if p == 0:
            label = list(mismatch_by_ds.keys())[row]
            ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.text(0.5, -0.13, f"LF {low_frac:.0%}", transform=ax.transAxes,
                ha="center", va="top", fontsize=7)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.012, pad=0.01)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, save_path)


# ─── Figure 3: per-firing-pair energy localization ───────────────────────────

def fig_per_pair(mismatch_by_ds, save_path):
    """Fraction of total ||e||^2 contributed by each firing pair."""
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(mismatch_by_ds))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    x = np.arange(N_PAIRS)
    for di, (label, mm) in enumerate(mismatch_by_ds.items()):
        # Per-pair energy fraction, averaged over samples
        fracs = np.zeros((len(mm), N_PAIRS))
        for i, m in enumerate(mm):
            eps, mask = m["epsilon"], m["mask"]
            valid = mask > 0.5
            pe = np.array([
                np.sum(eps[p * PAIR_SIZE:(p + 1) * PAIR_SIZE]
                       [valid[p * PAIR_SIZE:(p + 1) * PAIR_SIZE]] ** 2)
                for p in range(N_PAIRS)
            ])
            fracs[i] = pe / (pe.sum() + 1e-30)
        mean_frac = fracs.mean(axis=0) * 100
        std_frac = fracs.std(axis=0) * 100
        ax.bar(x + di * width, mean_frac, width, yerr=std_frac, capsize=3,
               color=colors[di % len(colors)], edgecolor="black",
               linewidth=0.5, label=f"{label} (n={len(mm)})")
    ax.axhline(100.0 / N_PAIRS, color="gray", ls="--", lw=1,
               label=f"uniform ({100.0/N_PAIRS:.1f}%)")
    ax.set_xticks(x + width * (len(mismatch_by_ds) - 1) / 2)
    ax.set_xticklabels([f"P{p + 1}" for p in range(N_PAIRS)])
    ax.set_xlabel("Firing pair", fontsize=12)
    ax.set_ylabel("Per-pair residual energy share (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    # Legend outside the plot area (right side) so it never overlaps the bars.
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)


def _save(fig, save_path):
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {p.with_suffix('.svg')}")


def main():
    ap = argparse.ArgumentParser(description="§5.2.3 mismatch-residual figures")
    ap.add_argument("--sample_idx", type=int, default=0,
                    help="Representative sample for heatmap/spectrum figures.")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="Samples to average for the per-pair chart (default: all).")
    ap.add_argument("--tag", default=None,
                    help="Optional tag appended to the run dir as <timestamp>_<tag>.")
    ap.add_argument("--figs_dir", default=None,
                    help="Override output dir (default: "
                         "scripts/data/residual_analysis/<timestamp>[_<tag>]/).")
    args = ap.parse_args()

    if args.figs_dir:
        figs_dir = Path(args.figs_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{args.tag}" if args.tag else timestamp
        figs_dir = OUTPUT_DIR / dir_name
    print("Loading L-matrix from kwave_geom (sparse CSC) ...")
    L = load_L_from_h5(KWAVE_GEOM_PATH)
    print(f"  L shape: {L.shape}")

    print("Computing mismatch — kwave_geom ...")
    geom = compute_mismatch(KWAVE_GEOM_PATH, L, n_samples=args.n_samples)
    print("Computing mismatch — kwave_blob ...")
    blob = compute_mismatch(KWAVE_BLOB_PATH, L, n_samples=args.n_samples)

    for label, mm in [("GeomSet", geom), ("BlobSet", blob)]:
        epct = np.mean([m["eps_energy_pct"] for m in mm])
        print(f"  {label}: residual energy {epct:.3f}% of "
              f"||d||^2  (n={len(mm)})")

    mismatch_by_ds = {"GeomSet": geom, "BlobSet": blob}
    print("\nRendering figures ...")
    fig_heatmap(mismatch_by_ds, args.sample_idx,
                figs_dir / "mismatch_residual_heatmap")
    fig_spectrum(mismatch_by_ds, args.sample_idx,
                 figs_dir / "mismatch_residual_spectrum")
    fig_per_pair(mismatch_by_ds, figs_dir / "mismatch_residual_per_pair")
    print(f"\nDone. Figures in {figs_dir}")


if __name__ == "__main__":
    main()
