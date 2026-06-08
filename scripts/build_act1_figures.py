#!/usr/bin/env python3
"""
build_act1_figures.py
─────────────────────
Generate F5, F6, F7 for Act 1 of the oral presentation.

  F5  Ray-integral schematic    (custom; no data needed)
  F6  L singular-value decay    (needs L matrix — server only)
  F7  Dataset GT mosaic         (uses existing recons.npz files)

Outputs to:  scripts/data/slides_figures/{F5,F6,F7}.svg + .png

Usage:
    source .venv/bin/activate
    uv run python scripts/build_act1_figures.py --figure all
    uv run python scripts/build_act1_figures.py --figure F5
    uv run python scripts/build_act1_figures.py --figure F6 --skip-svd-recompute
    uv run python scripts/build_act1_figures.py --figure F7

SCP back to local for the deck:
    scp 'thor:/mnt/asgard0/users/haben/INR-SoS-Recon/scripts/data/slides_figures/F*.svg' \
        thesis_reports/slides/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPTS_DIR / "data" / "slides_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── F5 ── Ray-integral schematic ─────────────────────────────────────────────

def build_F5(out_path: Path) -> None:
    """Custom matplotlib diagram showing how time-of-flight is a linear sum
    of voxel slowness × ray-segment-length. No project data needed."""
    print(f"[F5] building → {out_path}")

    # Synthesise a small 8x8 SoS field with a contrasting inclusion
    np.random.seed(0)
    sos = 1500 + 30 * np.random.randn(8, 8)
    # background + an inclusion in the middle
    yy, xx = np.mgrid[:8, :8]
    incl = ((xx - 4) ** 2 + (yy - 4) ** 2) < 2.0
    sos[incl] = 1580
    sos = np.clip(sos, 1450, 1600)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
    ax_field = fig.add_subplot(gs[0, 0])
    ax_eq = fig.add_subplot(gs[0, 1])

    # ── Left panel: SoS field with two rays
    im = ax_field.imshow(sos, cmap="RdBu_r", vmin=1450, vmax=1600,
                          origin="upper", extent=(0, 8, 8, 0))
    cbar = fig.colorbar(im, ax=ax_field, fraction=0.046, pad=0.04)
    cbar.set_label("SoS (m/s)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Probe at top
    ax_field.add_patch(mpatches.Rectangle((2.5, -0.7), 3, 0.4,
                                           fc="#1F3A5F", ec="black", lw=0.6,
                                           clip_on=False))
    ax_field.text(4.0, -0.85, "linear probe",
                  ha="center", va="bottom", fontsize=9,
                  color="#1F3A5F", fontweight="bold")

    # Ray 1: vertical through column ~3
    ax_field.plot([3.5, 3.5], [0, 8], color="darkorange", lw=2.0, alpha=0.85)
    ax_field.text(3.6, 4.0, "ray 1", fontsize=8, color="darkorange",
                  fontweight="bold", rotation=90, va="center")

    # Ray 2: diagonal through inclusion
    ax_field.plot([4.0, 6.5], [0, 8], color="darkgreen", lw=2.0, alpha=0.85)
    ax_field.text(5.7, 4.8, "ray 2", fontsize=8, color="darkgreen",
                  fontweight="bold", rotation=-58, va="center")

    # Highlight voxels Ray 1 crosses (column 3)
    for r in range(8):
        ax_field.add_patch(mpatches.Rectangle((3, r), 1, 1,
                                               fill=False, ec="darkorange",
                                               lw=1.2, ls="--", alpha=0.55))

    ax_field.set_xticks([]); ax_field.set_yticks([])
    ax_field.set_title("Each ray traverses a sequence of voxels",
                       fontsize=10, pad=4)
    ax_field.set_aspect("equal")

    # ── Right panel: equation block
    ax_eq.axis("off")
    eq_lines = [
        r"$\mathbf{Per\ ray:}$",
        r"$t_i = \int_{\mathrm{ray}\,i} s(\mathbf{r})\,d\ell$",
        "",
        r"$\mathbf{Discretised:}$",
        r"$t_i = \sum_j L_{ij}\,s_j$",
        r"$L_{ij}\,=\,$length of ray $i$ inside voxel $j$",
        "",
        r"$\mathbf{Stack\ all\ rays:}$",
        r"$\mathbf{d} = \mathbf{L}\,\mathbf{s}$",
        "",
        r"$\mathbf{d}\in\mathbb{R}^{131{,}072},\,$"
        r"$\mathbf{s}\in\mathbb{R}^{4{,}096}$",
        r"$\mathbf{L}\in\mathbb{R}^{131{,}072\times 4{,}096}$",
    ]
    y = 0.95
    for line in eq_lines:
        ax_eq.text(0.05, y, line, fontsize=12, va="top",
                    transform=ax_eq.transAxes, color="#222222")
        y -= 0.08

    fig.suptitle("Forward model: time-of-flight is linear in slowness",
                 fontsize=11, fontweight="bold", y=0.99)
    _save(fig, out_path)


# ─── F6 ── L singular-value decay + cumulative energy ─────────────────────────

def build_F6(out_path: Path, skip_svd: bool = False) -> None:
    """Re-render the SV-decay analysis from notebook cell 33 with clean
    publication styling. Loads L from kwave_geom, runs SVD on valid rays."""
    print(f"[F6] building → {out_path}")

    cache_path = OUT_DIR / "_F6_svd_cache.npz"

    if skip_svd and cache_path.exists():
        print(f"[F6] loading SVD from cache: {cache_path}")
        cached = np.load(cache_path)
        S = cached["S"]
    else:
        # ── Load L from kwave_geom HDF5 (sparse CSC)
        import h5py
        from scipy.sparse import csc_matrix
        from inr_sos import DATA_DIR

        KWAVE_GEOM_PATH = DATA_DIR + "/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat"

        print("[F6] loading L from kwave_geom HDF5 ...")
        with h5py.File(KWAVE_GEOM_PATH, "r") as f:
            node = f["A"]
            data, ir, jc = node["data"][:], node["ir"][:], node["jc"][:]
            n_cols = len(jc) - 1
            n_rows = int(ir.max()) + 1 if len(ir) > 0 else 0
            L_full = csc_matrix((data, ir, jc), shape=(n_rows, n_cols)).toarray()

            # Common valid-ray mask across all geom samples
            common_mask = np.ones(L_full.shape[0])
            for i in range(f["nanidx"].shape[0]):
                m = 1.0 - f["nanidx"][i].flatten()
                common_mask *= m

        valid = common_mask > 0.5
        L_valid = L_full[valid, :]
        print(f"[F6] L_valid shape: {L_valid.shape}")
        print("[F6] computing SVD ...")
        _, S, _ = np.linalg.svd(L_valid, full_matrices=False)
        np.savez(cache_path, S=S)
        print(f"[F6] SVD cached → {cache_path}")

    # ── Plot
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    threshold = S.max() * 1e-6
    n_eff = (S > threshold).sum()
    k_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    k_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1

    fig, (ax_sv, ax_cum) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: SV decay
    ax_sv.semilogy(np.arange(1, len(S) + 1), S, color="#1F3A5F",
                    linewidth=1.2)
    ax_sv.axhline(threshold, color="#C0392B", linestyle="--",
                   linewidth=0.9, alpha=0.7,
                   label=f"threshold = σ_max × 10⁻⁶")
    ax_sv.set_xlabel("Singular value index $i$", fontsize=10)
    ax_sv.set_ylabel(r"$\sigma_i$", fontsize=11)
    ax_sv.set_title("Singular-value decay of $L$", fontsize=11, pad=4)
    ax_sv.legend(fontsize=8, loc="upper right", frameon=False)
    ax_sv.grid(True, alpha=0.3, linewidth=0.5)
    ax_sv.tick_params(labelsize=9)

    # Annotate condition number
    kappa = S.max() / S.min()
    ax_sv.text(0.05, 0.05, f"$\\kappa(L) \\approx {kappa:.1e}$\n"
                            f"effective rank ≈ {n_eff} / {len(S)}",
               transform=ax_sv.transAxes,
               fontsize=9, va="bottom",
               bbox=dict(boxstyle="round,pad=0.4",
                         fc="white", ec="#888888", alpha=0.95))

    # Right: cumulative energy
    ax_cum.plot(np.arange(1, len(S) + 1), cumulative_energy * 100,
                 color="#1F3A5F", linewidth=1.4)
    ax_cum.axhline(90, color="#C0392B", linestyle="--",
                    linewidth=0.9, alpha=0.7)
    ax_cum.axvline(k_90, color="#C0392B", linestyle=":",
                    linewidth=0.9, alpha=0.7)
    ax_cum.plot(k_90, 90, "o", color="#C0392B", markersize=7,
                  markeredgecolor="black", markeredgewidth=0.6)
    ax_cum.annotate(f"  {k_90} SVs → 90 %",
                     xy=(k_90, 90), xytext=(k_90 + 350, 70),
                     fontsize=10, color="#C0392B",
                     arrowprops=dict(arrowstyle="-", color="#C0392B",
                                      linewidth=0.7, alpha=0.7))
    ax_cum.set_xlabel("Number of singular values", fontsize=10)
    ax_cum.set_ylabel("Cumulative energy (%)", fontsize=10)
    ax_cum.set_title("Cumulative spectral energy", fontsize=11, pad=4)
    ax_cum.set_ylim(0, 102)
    ax_cum.set_xlim(0, len(S))
    ax_cum.grid(True, alpha=0.3, linewidth=0.5)
    ax_cum.tick_params(labelsize=9)

    fig.suptitle(
        f"$L$ is ill-conditioned — top {k_90} of {len(S)} singular components carry 90 % of the signal",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_path)


# ─── F7 ── Dataset GT mosaic ─────────────────────────────────────────────────

# Local SCP'd recons.npz paths. Adjust on server if the layout differs.
DEFAULT_F7_SOURCES = [
    ("InverseCrime", "inr/hqt6bwmp/inverse_crime/20260527_103124_npz/recons.npz"),
    ("kwave_geom",   "inr/hqt6bwmp/kwave_geom/20260523_095618_npz/recons.npz"),
    ("kwave_blob",   "inr/hqt6bwmp/kwave_blob/20260523_095624_npz/recons.npz"),
    ("Phantom",      "inr/ti60qmx3/phantom/20260523_101853_npz/recons.npz"),
    ("Breast",       "inr/ti60qmx3/breast_data/20260523_101856_npz/recons.npz"),
]


def _slowness_to_sos(s_flat):
    return np.clip(1.0 / (s_flat + 1e-10), 1400, 1600)


def build_F7(out_path: Path, sources=None, data_root: Path | None = None) -> None:
    """1x4 mosaic of GT SoS fields, one per dataset family. Shared colorbar."""
    print(f"[F7] building → {out_path}")
    sources = sources or DEFAULT_F7_SOURCES
    data_root = data_root or (SCRIPTS_DIR / "data")

    fields = []
    labels = []
    for label, rel in sources:
        path = data_root / rel
        if not path.exists():
            print(f"[F7] WARN: missing {path} — skipping")
            continue
        from inr_sos.evaluation.recon_export import load_recons_npz
        d = load_recons_npz(path)
        # First sample's GT, converted to SoS, reshaped (F-order, transducer at top)
        gt_flat = d["gt"][0]
        img = _slowness_to_sos(gt_flat).reshape((64, 64), order="F")
        fields.append(img)
        labels.append(label)
        print(f"[F7]   loaded {label} from {rel}")

    if not fields:
        print("[F7] ERROR: no datasets loaded. Check paths.")
        return

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0))
    if n == 1:
        axes = [axes]
    vmin = min(img.min() for img in fields)
    vmax = max(img.max() for img in fields)
    for ax, img, label in zip(axes, fields, labels):
        im = ax.imshow(img, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])

    # Shared colorbar to the right
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("SoS (m/s)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Ground truth SoS across domain datasets",
                 fontsize=11, fontweight="bold", y=1.02)
    _save(fig, out_path)


# ─── Save helper ─────────────────────────────────────────────────────────────

def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), format="png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.with_suffix('.svg')}  (+ .png)")


# ─── CLI ─────────────────────────────────────────────────────────────────────

_BUILDERS = {
    "F5": lambda args: build_F5(OUT_DIR / "F5"),
    "F6": lambda args: build_F6(OUT_DIR / "F6", skip_svd=args.skip_svd_recompute),
    "F7": lambda args: build_F7(OUT_DIR / "F7"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Build Act 1 figures (F5, F6, F7)."
    )
    ap.add_argument("--figure", nargs="+", default=["all"],
                    help=f"Which to build: {', '.join(_BUILDERS)} or all")
    ap.add_argument("--skip-svd-recompute", action="store_true",
                    help="(F6) reuse cached SVD if present in slides_figures/")
    args = ap.parse_args()

    targets = list(_BUILDERS) if "all" in args.figure else args.figure
    bad = [t for t in targets if t not in _BUILDERS]
    if bad:
        ap.error(f"Unknown figure(s): {bad}. Choose from {list(_BUILDERS)}")

    for tag in targets:
        try:
            _BUILDERS[tag](args)
        except Exception as exc:
            print(f"[{tag}] ERROR: {exc!r}")

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
