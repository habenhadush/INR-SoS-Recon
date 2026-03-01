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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
matplotlib.use("Agg")

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
    

# ── Colour palette (matches reference image pastels) ─────────────────────────
# Order: baselines first, then INR ranks in blue gradient
_PALETTE = {
    # Baseline colours (matching the reference: gold, cyan, green, red, purple, brown)
    0: "#f5c842",   # gold / amber   → L2
    1: "#6dd6e8",   # cyan           → L1
    2: "#6dc96d",   # green          → Uncorrected (if present)
    3: "#e87575",   # salmon/red     → Artifact Corrected
    4: "#b57be8",   # purple         → Data Corrected
    5: "#a05050",   # brown/dark red → Dual Corrected
}
_INR_BLUE_START = 0.35   # starting shade in Blues colormap for INR ranks


def _short_label(method_str: str) -> str:
    """Short x-axis tick label."""
    if method_str == "L2_regularization":
        return "L2"
    if method_str == "L1_regularization":
        return "L1"
    parts = method_str.replace("INR_", "").split("_")
    return parts[0]


def _legend_label(method_str: str) -> str:
    """Full descriptive label for legend."""
    if method_str == "L2_regularization":
        return "L2 regularization"
    if method_str == "L1_regularization":
        return "L1 regularization"
    parts = method_str.replace("INR_", "").split("_")
    rank  = parts[0]
    model = parts[-1]
    meth  = "_".join(parts[1:-1])
    return f"{rank}  {meth} / {model}"


def _build_color_lists(baselines, inr_ranks):
    """
    Assign one face-colour per method.
    Baselines get palette entries; INR ranks get a blue gradient.
    Returns (face_colors, edge_colors) lists aligned with (baselines + inr_ranks).
    """
    face_colors = []
    edge_colors = []

    for i, _ in enumerate(baselines):
        fc = _PALETTE.get(i, "#aaaaaa")
        face_colors.append(fc)
        edge_colors.append(_darken(fc, 0.65))

    n_inr = len(inr_ranks)
    for i in range(n_inr):
        shade = _INR_BLUE_START + 0.15 * i
        fc = plt.cm.Blues(min(shade, 0.92))
        face_colors.append(fc)
        ec = plt.cm.Blues(min(shade + 0.25, 0.99))
        edge_colors.append(ec)

    return face_colors, edge_colors


def _darken(hex_color, factor=0.7):
    """Return a darker version of a hex colour string."""
    import matplotlib.colors as mc
    try:
        rgb = mc.to_rgb(hex_color)
    except ValueError:
        return hex_color
    return tuple(c * factor for c in rgb)


def _draw_violin_panel(ax, values_list, face_colors, edge_colors,
                        labels, ylabel, higher_better, n_samples):
    """
    Draw one panel with the reference-style violin + inner box + whiskers.

    Parameters
    ----------
    ax           : matplotlib Axes
    values_list  : list of 1D arrays, one per method (x-position order)
    face_colors  : list of colours matching values_list
    edge_colors  : list of edge colours matching values_list
    labels       : x-tick labels
    ylabel       : y-axis label string
    higher_better: bool — controls direction annotation
    n_samples    : int — used to decide strip vs violin
    """
    n = len(values_list)
    positions = np.arange(1, n + 1)

    # ── Choose plot type ──────────────────────────────────────────────────
    use_violin = n_samples >= 5

    if use_violin:
        # ── Violin bodies ─────────────────────────────────────────────────
        parts = ax.violinplot(
            values_list,
            positions=positions,
            widths=0.55,
            showmeans=False,
            showmedians=False,
            showextrema=False,        # we draw our own whiskers
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(face_colors[i])
            pc.set_edgecolor(edge_colors[i])
            pc.set_alpha(0.55)
            pc.set_linewidth(1.2)

        # ── Inner box + whiskers (drawn on top of violin) ─────────────────
        # Use a thin, pale-blue box identical to the reference image
        bp = ax.boxplot(
            values_list,
            positions=positions,
            widths=0.10,              # very narrow inner box
            patch_artist=True,
            notch=False,
            showfliers=False,         # outliers hidden inside violin
            medianprops=dict(color="black", linewidth=2.5, solid_capstyle="round"),
            boxprops=dict(facecolor="white", edgecolor="#4d9ed8",
                          linewidth=1.4, alpha=0.90),
            whiskerprops=dict(color="#4d9ed8", linewidth=1.3,
                              linestyle="-"),
            capprops=dict(color="#4d9ed8", linewidth=1.3),
        )

    else:
        # ── Strip plot for small n (< 5) ──────────────────────────────────
        rng = np.random.default_rng(seed=0)
        for col_i, (vals, fc, ec) in enumerate(
                zip(values_list, face_colors, edge_colors), start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(col_i + jitter, vals,
                       color=fc, edgecolors=ec, s=60,
                       zorder=3, alpha=0.85, linewidths=0.7)
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            ax.plot([col_i - 0.28, col_i + 0.28],
                    [mean_v, mean_v], color="black",
                    linewidth=2.5, zorder=4)
            ax.plot([col_i, col_i],
                    [mean_v - std_v, mean_v + std_v],
                    color="black", linewidth=1.3, zorder=4)

    # ── Axes styling ──────────────────────────────────────────────────────
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.tick_params(axis="y", labelsize=8)

    # Horizontal grid lines (matches reference)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5,
                  color="#cccccc", alpha=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Direction annotation
    direction = "Higher ↑ better" if higher_better else "Lower ↓ better"
    ax.set_title(direction, fontsize=8.5, color="#666666", pad=4)

    # ── x-tick styling — small black ticks like reference ─────────────────
    ax.tick_params(axis="x", length=4, width=1.2, color="black")

    # Light blue shading over INR region (subtle)
    n_baseline = len(face_colors) - sum(
        1 for i, fc in enumerate(face_colors) if fc not in list(_PALETTE.values())
    )
    # Simpler: count methods before the blue gradient begins
    inr_start = None
    for i, fc in enumerate(face_colors):
        if fc not in list(_PALETTE.values()):
            inr_start = i + 1   # 1-based position
            break
    if inr_start is not None and inr_start <= n:
        ax.axvspan(inr_start - 0.5, n + 0.5,
                   alpha=0.04, color="#1f77b4", zorder=0)

    return ax


def make_boxplots(all_results: list,
                  left_metric: tuple = ("RMSE", "RMSE (m/s)", False),
                  right_metric: tuple = ("SSIM", "SSIM", True),
                  title: str = "Method Comparison",
                  figsize: tuple = (13, 6)) -> plt.Figure:
    """
    Two-panel violin+box plot matching the reference image style.

    Parameters
    ----------
    all_results   : list of result dicts (same format as compare_topk.py)
    left_metric   : (key, ylabel, higher_better) for left panel  — default RMSE
    right_metric  : (key, ylabel, higher_better) for right panel — default SSIM
    title         : figure suptitle
    figsize       : figure size in inches

    Returns
    -------
    matplotlib Figure
    """
    # ── Order: baselines then INR sorted by rank ──────────────────────────
    baselines = [r for r in all_results if "INR" not in r["method"]]
    inr_ranks = sorted(
        [r for r in all_results if "INR" in r["method"]],
        key=lambda x: int(x["method"].split("rank")[1].split("_")[0])
        if "rank" in x["method"] else 999,
    )
    ordered = baselines + inr_ranks
    labels  = [_short_label(r["method"]) for r in ordered]
    n       = all_results[0]["n_samples"]

    face_colors, edge_colors = _build_color_lists(baselines, inr_ranks)

    # ── Build per-method value arrays ─────────────────────────────────────
    def _get_values(metric_key):
        out = []
        for r in ordered:
            vals = [s.get(metric_key) for s in r["per_sample"]
                    if s.get(metric_key) is not None]
            out.append(np.array(vals) if vals else np.array([0.0]))
        return out

    # ── Figure layout — two panels side by side ───────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(
        1, 2,
        figure=fig,
        wspace=0.35,
        left=0.08, right=0.72,   # leave room for legend on right
    )

    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    for ax, (metric_key, ylabel, higher) in [
        (ax_l, left_metric),
        (ax_r, right_metric),
    ]:
        _draw_violin_panel(
            ax=ax,
            values_list=_get_values(metric_key),
            face_colors=face_colors,
            edge_colors=edge_colors,
            labels=labels,
            ylabel=ylabel,
            higher_better=higher,
            n_samples=n,
        )

    # ── Suptitle ──────────────────────────────────────────────────────────
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    # ── Legend — positioned to the right of both panels ──────────────────
    legend_handles = [
        mpatches.Patch(facecolor=fc, edgecolor=ec, linewidth=0.8,
                       label=_legend_label(r["method"]))
        for r, fc, ec in zip(ordered, face_colors, edge_colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.73, 0.50),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        title="Methods",
        title_fontsize=8.5,
    )

    return fig


def make_boxplots_4panel(all_results: list,
                          title: str = "Method Comparison",
                          figsize: tuple = (16, 10)) -> plt.Figure:
    """
    4-panel violin+box plot: MAE | RMSE (top row) and SSIM | CNR (bottom row).
    Retains the same visual style as make_boxplots() (2-panel).
    """
    baselines  = [r for r in all_results if "INR" not in r["method"]]
    inr_ranks  = sorted(
        [r for r in all_results if "INR" in r["method"]],
        key=lambda x: int(x["method"].split("rank")[1].split("_")[0])
        if "rank" in x["method"] else 999,
    )
    ordered = baselines + inr_ranks
    labels  = [_short_label(r["method"]) for r in ordered]
    n       = all_results[0]["n_samples"]
    face_colors, edge_colors = _build_color_lists(baselines, inr_ranks)

    def _get_values(metric_key):
        out = []
        for r in ordered:
            vals = [s.get(metric_key) for s in r["per_sample"]
                    if s.get(metric_key) is not None]
            out.append(np.array(vals) if vals else np.array([0.0]))
        return out

    panels = [
        ("MAE",  "MAE (m/s)",               False),
        ("RMSE", "RMSE (m/s)",              False),
        ("SSIM", "SSIM",                    True),
        ("CNR",  "Contrast-to-Noise Ratio", True),
    ]

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.07, right=0.72,
    )

    for idx, (metric_key, ylabel, higher) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        _draw_violin_panel(
            ax=ax,
            values_list=_get_values(metric_key),
            face_colors=face_colors,
            edge_colors=edge_colors,
            labels=labels,
            ylabel=ylabel,
            higher_better=higher,
            n_samples=n,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    legend_handles = [
        mpatches.Patch(facecolor=fc, edgecolor=ec, linewidth=0.8,
                       label=_legend_label(r["method"]))
        for r, fc, ec in zip(ordered, face_colors, edge_colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.73, 0.50),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        title="Methods",
        title_fontsize=8.5,
    )

    return fig