"""
Drop-in replacement for the plot() function in engines.py.
Provides a publication-quality reconstruction visualisation with:

  1. Diverging colormap centred at background SoS (~1540 m/s)
     - Blue  = slower than background (e.g. fatty inclusions ~1450 m/s)
     - Red   = faster than background (e.g. dense tissue ~1580 m/s)
     - White = background tissue (clinically neutral)
     This is the standard convention in the SoS reconstruction literature
     (Rau et al. 2021, Bezek et al. 2025).

  2. Shared colorbar axis across GT and reconstruction so the two are
     directly comparable by eye.

  3. Absolute error map with saturating yellow at 50 m/s.

  4. Optional: save to file for paper figures.

Usage in notebook / engines.py:
    from inr_sos.visualization.plot_reconstruction import plot
    plot(result_dict, sample, title="...")
    plot(result_dict, sample, save_path="figures/sample_3133.pdf")
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SoS constants
# ─────────────────────────────────────────────────────────────────────────────
SOS_BG  = 1540.0    # background tissue (m/s)  — diverging centre
SOS_MIN = 1380.0    # slightly below 1400 for margin
SOS_MAX = 1620.0    # slightly above 1600 for margin


def _make_diverging_norm(vmin=SOS_MIN, vcenter=SOS_BG, vmax=SOS_MAX):
    """
    TwoSlopeNorm centres the diverging colourmap at background SoS.
    This means equal colour distance for equal SoS deviation from background,
    regardless of asymmetry in the data range.
    """
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


def plot(result_dict: dict,
         sample: dict,
         grid_shape: tuple = (64, 64),
         title: str = "Reconstruction",
         save_path: str = None,
         cmap_sos: str = "RdBu_r",
         show: bool = True) -> plt.Figure:
    """
    Publication-quality 4-panel reconstruction figure.

    Parameters
    ----------
    result_dict : output from any engine (must have 's_phys', 'loss_history')
    sample      : dataset sample dict (must have 's_gt_raw')
    grid_shape  : spatial grid (default 64×64)
    title       : figure suptitle (auto-includes MAE)
    save_path   : if set, saves figure to this path (PDF recommended for papers)
    cmap_sos    : colormap for SoS panels. 'RdBu_r' is the literature standard.
                  Alternatives: 'coolwarm', 'seismic', 'bwr'
    show        : call plt.show() (set False for server/batch use)

    Returns
    -------
    matplotlib Figure
    """
    import torch

    # ── Extract and convert to SoS (m/s) ─────────────────────────────────
    s_gt_raw = sample["s_gt_raw"]
    s_phys   = result_dict["s_phys"]

    # Handle torch tensors or numpy arrays
    if hasattr(s_gt_raw, "detach"):
        s_gt_raw = s_gt_raw.detach().cpu().numpy()
    if hasattr(s_phys, "detach"):
        s_phys = s_phys.detach().cpu().numpy()

    s_gt_raw = np.asarray(s_gt_raw, dtype=np.float32).flatten()
    s_phys   = np.asarray(s_phys,   dtype=np.float32).flatten()

    # Slowness → SoS (m/s), clamped to physiological range
    v_gt  = np.clip(1.0 / (s_gt_raw + 1e-8), SOS_MIN, SOS_MAX)
    v_rec = np.clip(1.0 / (s_phys   + 1e-8), SOS_MIN, SOS_MAX)

    v_gt  = v_gt.reshape(grid_shape)
    v_rec = v_rec.reshape(grid_shape)

    error_map = np.abs(v_gt - v_rec)
    mae       = float(np.mean(error_map))
    loss_hist = result_dict.get("loss_history", [])

    # ── Shared normalisation for GT and reconstruction panels ─────────────
    # Use the GT's actual range so the reconstruction is compared fairly
    v_min    = max(SOS_MIN, float(v_gt.min()) - 10)
    v_max    = min(SOS_MAX, float(v_gt.max()) + 10)
    bg_sos   = float(np.median(v_gt))   # estimate background from GT median
    norm_sos = _make_diverging_norm(vmin=v_min, vcenter=bg_sos, vmax=v_max)

    # ── Layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.1]})
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{title}  |  MAE: {mae:.2f} m/s  |  BG: {bg_sos:.0f} m/s",
                 fontsize=13, fontweight="bold", y=1.01)

    # ── Panel 0: Ground Truth ─────────────────────────────────────────────
    im0 = axes[0].imshow(v_gt, cmap=cmap_sos, norm=norm_sos,
                          interpolation="nearest", origin="upper")
    axes[0].set_title("Ground Truth (m/s)", fontsize=11)
    axes[0].axis("off")
    cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("SoS (m/s)", fontsize=9)
    _add_bg_line_to_colorbar(cb0, bg_sos, v_min, v_max)

    # ── Panel 1: Reconstruction ───────────────────────────────────────────
    im1 = axes[1].imshow(v_rec, cmap=cmap_sos, norm=norm_sos,
                          interpolation="nearest", origin="upper")
    axes[1].set_title("Reconstruction (m/s)", fontsize=11)
    axes[1].axis("off")
    cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("SoS (m/s)", fontsize=9)
    _add_bg_line_to_colorbar(cb1, bg_sos, v_min, v_max)

    # ── Panel 2: Absolute Error ───────────────────────────────────────────
    im2 = axes[2].imshow(error_map, cmap="hot", vmin=0, vmax=50,
                          interpolation="nearest", origin="upper")
    axes[2].set_title("Abs. Error (m/s)", fontsize=11)
    axes[2].axis("off")
    cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("|GT − Rec| (m/s)", fontsize=9)
    cb2.ax.axhline(y=mae, color="cyan", linewidth=1.5, linestyle="--")
    cb2.ax.text(1.15, mae / 50, f"MAE={mae:.1f}", transform=cb2.ax.transAxes,
                fontsize=7, color="cyan", va="center")

    # ── Panel 3: Optimisation Loss ────────────────────────────────────────
    if loss_hist:
        axes[3].plot(loss_hist, color="#1f77b4", linewidth=1.2, label="Loss")
        axes[3].set_yscale("log")
        axes[3].set_title("Optimisation Loss", fontsize=11)
        axes[3].set_xlabel("Iteration", fontsize=9)
        axes[3].set_ylabel("Loss (log scale)", fontsize=9)
        axes[3].grid(True, which="both", linestyle="--", alpha=0.4)
        axes[3].spines["top"].set_visible(False)
        axes[3].spines["right"].set_visible(False)
    else:
        axes[3].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white")
        print(f"  Figure saved → {save_path}")

    if show:
        plt.show()

    return fig


def plot_method_comparison(results: dict,
                            sample: dict,
                            grid_shape: tuple = (64, 64),
                            title: str = "Method Comparison",
                            save_path: str = None,
                            cmap_sos: str = "RdBu_r",
                            show: bool = True) -> plt.Figure:
    """
    Multi-method comparison figure — one row per method, 3 columns:
    GT | Reconstruction | Abs. Error.

    Parameters
    ----------
    results : dict of {method_label: result_dict}
              result_dict must have 's_phys'
    sample  : dataset sample dict (must have 's_gt_raw')
    """
    import torch

    s_gt_raw = sample["s_gt_raw"]
    if hasattr(s_gt_raw, "detach"):
        s_gt_raw = s_gt_raw.detach().cpu().numpy()
    s_gt_raw = np.asarray(s_gt_raw, dtype=np.float32).flatten()
    v_gt     = np.clip(1.0 / (s_gt_raw + 1e-8), SOS_MIN, SOS_MAX).reshape(grid_shape)

    v_min  = max(SOS_MIN, float(v_gt.min()) - 10)
    v_max  = min(SOS_MAX, float(v_gt.max()) + 10)
    bg_sos = float(np.median(v_gt))
    norm_sos = _make_diverging_norm(vmin=v_min, vcenter=bg_sos, vmax=v_max)

    n_methods = len(results)
    fig, axes = plt.subplots(n_methods + 1, 3,
                              figsize=(15, 4.5 * (n_methods + 1)))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # First row: Ground Truth only
    for col in range(3):
        ax = axes[0, col]
        if col == 0:
            im = ax.imshow(v_gt, cmap=cmap_sos, norm=norm_sos,
                           interpolation="nearest", origin="upper")
            ax.set_title("Ground Truth (m/s)", fontsize=11)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.set_visible(False)
        ax.axis("off")

    # Subsequent rows: one per method
    for row, (label, result_dict) in enumerate(results.items(), start=1):
        s_phys = result_dict["s_phys"]
        if hasattr(s_phys, "detach"):
            s_phys = s_phys.detach().cpu().numpy()
        s_phys  = np.asarray(s_phys, dtype=np.float32).flatten()
        v_rec   = np.clip(1.0 / (s_phys + 1e-8), SOS_MIN, SOS_MAX).reshape(grid_shape)
        err_map = np.abs(v_gt - v_rec)
        mae     = float(np.mean(err_map))

        # Reconstruction
        im1 = axes[row, 0].imshow(v_rec, cmap=cmap_sos, norm=norm_sos,
                                    interpolation="nearest", origin="upper")
        axes[row, 0].set_title(f"{label}  (MAE={mae:.1f} m/s)", fontsize=10)
        axes[row, 0].axis("off")
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Error
        im2 = axes[row, 1].imshow(err_map, cmap="hot", vmin=0, vmax=50,
                                    interpolation="nearest", origin="upper")
        axes[row, 1].set_title("Abs. Error (m/s)", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Difference from GT (signed)
        diff = v_rec - v_gt
        diff_max = max(abs(diff.min()), abs(diff.max()), 20)
        norm_diff = mcolors.TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
        im3 = axes[row, 2].imshow(diff, cmap="RdBu_r", norm=norm_diff,
                                    interpolation="nearest", origin="upper")
        axes[row, 2].set_title("Signed Difference (m/s)", fontsize=10)
        axes[row, 2].axis("off")
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Figure saved → {save_path}")
    if show:
        plt.show()
    return fig


def _add_bg_line_to_colorbar(cb, bg_sos, v_min, v_max):
    """Add a tick mark on the colorbar at the background SoS value."""
    # Normalised position in [0, 1]
    pos = (bg_sos - v_min) / (v_max - v_min)
    cb.ax.axhline(y=pos, color="white", linewidth=1.5, linestyle="--")
    cb.ax.text(1.1, pos, f"{bg_sos:.0f}", transform=cb.ax.transAxes,
               fontsize=7, va="center", color="gray")