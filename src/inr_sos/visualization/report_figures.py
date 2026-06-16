"""
report_figures.py
-----------------
Publication-quality figure generation for thesis / IEEE TMI-style reports.

Two main entry points:

    plot_method_grid(results, samples, save_path)
        Grid layout: rows = methods, columns = samples.
        Each cell shows a SoS reconstruction with a shared diverging colormap.
        Per-cell annotations: MAE and CNR.
        First row is always Ground Truth.
        Row labels on the left, Roman-numeral column headers at the top.

    plot_metrics_comparison(results, save_path)
        Box plots of MAE / RMSE / SSIM / CNR across methods.
        Baselines (L1, L2) are drawn in grey; INR methods in blue/green.

Style conventions:
    - Serif font family to match LaTeX body text
    - ~10-11 pt equivalent sizes throughout
    - RdBu_r colormap centred at background SoS via TwoSlopeNorm
    - Saved as SVG (vector, Overleaf-compatible) + PNG fallback at 300 DPI
    - No unnecessary grid lines; spines cleaned up
    - Works in black-and-white print (line styles vary, not just colours)

Data conventions assumed:
    - s_phys / s_gt_raw : float array of slowness values (1/m/s), flat (4096,)
    - SoS = 1 / slowness, clamped to [1200, 1800] m/s
    - Reshape with order="F" for correct transducer-at-top orientation

Usage:
    from inr_sos.visualization.report_figures import plot_method_grid, plot_metrics_comparison

    # results: {method_label: [per_sample_result_dict, ...]}
    # each result_dict: {"s_phys": ..., "metrics": {"MAE": ..., "CNR": ...}}
    # samples: list of sample dicts, each with "s_gt_raw"

    plot_method_grid(results, samples, save_path="figures/grid.svg")
    plot_metrics_comparison(results, save_path="figures/metrics.svg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Physical / colormap constants  (shared with plot_reconstruction.py)
# ─────────────────────────────────────────────────────────────────────────────
_SOS_BG   = 1540.0   # background tissue (m/s) — diverging centre
_SOS_MIN  = 1380.0
_SOS_MAX  = 1620.0
_GRID     = (64, 64)

# ─────────────────────────────────────────────────────────────────────────────
# Global style (IEEE TMI / thesis style)
# ─────────────────────────────────────────────────────────────────────────────
_SERIF_RCPARAMS: dict = {
    "font.family":        "serif",
    "mathtext.fontset":   "dejavuserif",
    "font.size":          11,   # body-text size (thesis is 11 pt)
    "axes.titlesize":     11,
    "axes.labelsize":     11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.titlesize":   12,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
}

# ─────────────────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────────────────
# Wong (2011) colour-blind-safe palette (8 colours)
_WONG = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Baseline methods drawn grey; INR methods drawn blue/teal.
# "Plain INR" is the bare ReluMLP fit (no denoiser / no staged curriculum);
# "Raw INR" kept for back-compat with results.json files written pre-rename.
_BASELINE_LABELS = {"L1", "L2", "PI", "Plain INR", "Raw INR"}

# Roman numerals for column headers (up to 20 samples)
_ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy array to a flat float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).flatten()


def _slowness_to_sos(s_flat: np.ndarray,
                     sos_min: float = 1200.0,
                     sos_max: float = 1800.0) -> np.ndarray:
    """Convert flat slowness vector to SoS (m/s), clamped."""
    s = _to_numpy(s_flat)
    return np.clip(1.0 / (s + 1e-10), sos_min, sos_max)


def _reshape(v_flat: np.ndarray, grid: tuple[int, int] = _GRID) -> np.ndarray:
    """Reshape flat SoS vector to 2-D image (Fortran order = transducer at top)."""
    return v_flat.reshape(grid, order="F")


def _diverging_norm(v_gt_2d: np.ndarray,
                    margin: float = 10.0) -> mcolors.TwoSlopeNorm:
    """Build a TwoSlopeNorm from GT image, centred at GT median."""
    bg = float(np.median(v_gt_2d))
    vmin = max(_SOS_MIN, float(v_gt_2d.min()) - margin)
    vmax = min(_SOS_MAX, float(v_gt_2d.max()) + margin)
    # Ensure vmin < vcenter < vmax (guard against flat GT)
    if vmin >= bg:
        vmin = bg - 1.0
    if vmax <= bg:
        vmax = bg + 1.0
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=bg, vmax=vmax)


def _annotate_cell(ax: plt.Axes, text: str, fontsize: int = 5) -> None:
    """Overlay metric text in the bottom-left corner of an image cell."""
    ax.text(
        0.03, 0.03, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color="white",
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                  alpha=0.45, linewidth=0),
    )


def _subfig_label(ax: plt.Axes, label: str, fontsize: int = 9) -> None:
    """Add a bold (a)/(b)/... label to the bottom-left of an axes."""
    ax.text(
        -0.04, -0.04, label,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="top", ha="right",
    )


def _save(fig: plt.Figure, save_path: str | Path, png_fallback: bool = True) -> None:
    """Save figure as SVG (vector) and optionally PNG at 300 DPI."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Primary: SVG (vector, Overleaf-compatible)
    svg_path = p.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"  Figure saved (SVG) -> {svg_path}")

    if png_fallback:
        png_path = p.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight",
                    facecolor="white")
        print(f"  Figure saved (PNG) -> {png_path}")


def _method_color(label: str, idx: int, n_methods: int) -> str:
    """Return a colour for a method bar/box. Baselines grey, others from Wong palette."""
    if label in _BASELINE_LABELS:
        return "#888888"
    # Use Wong colours (skip black at index 0) for INR methods
    non_baseline_palette = _WONG[1:]  # orange, sky blue, bluish green, ...
    return non_baseline_palette[idx % len(non_baseline_palette)]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_grid(
    results: dict[str, list[dict]],
    samples: list[dict],
    save_path: str | Path = "report_comparison.svg",
    *,
    grid_shape: tuple[int, int] = _GRID,
    cmap_sos: str = "jet",
    figwidth: float = 6.5,
    dataset_title: str | None = None,
    show: bool = False,
    png_fallback: bool = True,
) -> plt.Figure:
    """
    Publication-quality grid comparing SoS reconstructions.

    Layout
    ------
    Rows  : Ground Truth (first) + one row per method in ``results``
    Cols  : one column per sample in ``samples``
    Right : single shared colorbar for all SoS panels
    Left  : row labels (method names)
    Top   : Roman-numeral column headers ("Sample I", "Sample II", ...)

    Parameters
    ----------
    results     : {method_label: [per_sample_result_dict]}
                  Each result_dict must contain ``"s_phys"`` (slowness array)
                  and ``"metrics"`` (dict with at least ``"MAE"`` and ``"CNR"``).
    samples     : List of sample dicts; each must contain ``"s_gt_raw"``.
    save_path   : Output file path.  Extension is overridden: SVG is always
                  written; PNG fallback is optional.
    grid_shape  : Spatial grid (default 64x64).
    cmap_sos    : Colormap for SoS images (default ``"jet"``).
    figwidth    : Figure width in inches (6.5 = single thesis column).
    show        : Call plt.show() — set False for server/batch use.
    png_fallback: Also save a 300-DPI PNG alongside the SVG.

    Returns
    -------
    matplotlib.figure.Figure

    Suggested caption
    -----------------
    "SoS reconstruction comparison across methods (rows) and samples (columns).
    All images share the same diverging colorbar (blue: slower than background,
    red: faster, white: ~1540 m/s background). Per-cell annotations show MAE
    (m/s) and CNR."
    """
    with matplotlib.rc_context(_SERIF_RCPARAMS):
        n_cols    = len(samples)
        n_methods = len(results)
        n_rows    = 1 + n_methods      # GT row + method rows

        # ── Build shared range from all GT images ────────────────────────
        v_gts = []
        for sample in samples:
            v_flat = _slowness_to_sos(_to_numpy(sample["s_gt_raw"]))
            v_gts.append(_reshape(v_flat, grid_shape))
        v_stack = np.stack(v_gts)
        bg_sos = float(np.median(v_stack))
        # Fixed range matching the existing comparison plots
        global_vmin = 1400.0
        global_vmax = 1600.0
        norm_sos = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

        # ── Build short labels for the legend ────────────────────────────
        # Map each method to a compact letter/number tag for the row label
        _ALL_LABELS = ["GT"] + list(results.keys())
        _SHORT = {}
        _SHORT["GT"] = "GT"
        inr_letter = 0
        for label in results.keys():
            # Check if label is a baseline or contains a baseline tag
            is_baseline = any(b == label or (b in label and b in {"PI", "Plain INR", "Raw INR"})
                              for b in _BASELINE_LABELS)
            if is_baseline:
                if "PI" in label or "Plain INR" in label:
                    # If it has a rank, keep it unique in the short label
                    if "rank#" in label:
                        import re
                        match = re.search(r'rank#(\d+)', label)
                        rnum = match.group(1) if match else "?"
                        _SHORT[label] = f"PI-{rnum}"
                    else:
                        _SHORT[label] = "PI"
                elif "L1" in label:
                    _SHORT[label] = "L1"
                elif "L2" in label:
                    _SHORT[label] = "L2"
                else:
                    _SHORT[label] = label
            else:
                _SHORT[label] = chr(ord("A") + inr_letter)  # A, B, C, ...
                inr_letter += 1

        # ── Figure layout ────────────────────────────────────────────────
        # Extra column on the right for the colorbar
        cell_h  = figwidth / n_cols          # keep cells roughly square
        fig_h   = cell_h * n_rows + 0.6      # +0.6 for column header row
        cb_frac = 0.03                       # colorbar width fraction of figure

        # Reserve space at the bottom for the legend.
        # The legend box height is roughly 0.45 in for one legend row and
        # grows by ~0.18 in per additional row.  Estimate rows from ncol.
        _ncol_legend  = min(len(results), 4)
        _nrow_legend  = int(np.ceil(len(results) / _ncol_legend))
        _legend_h_in  = 0.30 + 0.18 * _nrow_legend   # empirical
        # Add a small gap between the legend top and the last image row.
        _gap_in       = 0.10
        _bottom_frac  = (_legend_h_in + _gap_in) / fig_h
        # Clamp: never eat more than 14% of the figure height
        _bottom_frac  = max(0.06, min(0.14, _bottom_frac))

        fig = plt.figure(figsize=(figwidth, fig_h))
        fig.patch.set_facecolor("white")

        # GridSpec: n_rows image rows + 1 colorbar column
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(
            n_rows, n_cols + 1,
            figure=fig,
            width_ratios=[1.0] * n_cols + [cb_frac * n_cols],
            hspace=0.04,
            wspace=0.03,
            left=0.06,    # tighter left margin — short labels only
            right=0.95,
            top=0.93,
            bottom=_bottom_frac,  # reserved for the legend below the last row
        )

        image_axes: list[list[plt.Axes]] = []

        # ── Row 0: Ground Truth ──────────────────────────────────────────
        gt_row_axes = []
        for col_idx, v_gt in enumerate(v_gts):
            ax = fig.add_subplot(gs[0, col_idx])
            im = ax.imshow(v_gt, cmap=cmap_sos, norm=norm_sos,
                           interpolation="nearest", origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            # Column headers on top row — Roman numerals only
            roman = _ROMAN[col_idx] if col_idx < len(_ROMAN) else str(col_idx + 1)
            ax.set_title(roman, fontsize=8, pad=3)
            gt_row_axes.append(ax)

        image_axes.append(gt_row_axes)

        # Dataset title above the column headers
        if dataset_title:
            fig.suptitle(dataset_title, fontsize=10, fontweight="bold", y=0.98)

        # Row label: "GT"
        gt_row_axes[0].set_ylabel("GT", fontsize=9, rotation=0,
                                   labelpad=14, va="center")

        # ── Method rows ──────────────────────────────────────────────────
        for row_offset, (method_label, per_sample) in enumerate(results.items()):
            row_idx = row_offset + 1
            method_row_axes = []
            for col_idx, (sample, result) in enumerate(zip(samples, per_sample)):
                v_gt   = v_gts[col_idx]
                s_phys = _to_numpy(result["s_phys"])
                v_rec  = _reshape(_slowness_to_sos(s_phys), grid_shape)

                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(v_rec, cmap=cmap_sos, norm=norm_sos,
                          interpolation="nearest", origin="upper")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)

                # Per-cell metric annotation
                m = result.get("metrics", {})
                mae_val = m.get("MAE", float("nan"))
                cnr_val = m.get("CNR", float("nan"))
                _annotate_cell(
                    ax,
                    f"MAE: {mae_val:.1f}\nCNR: {cnr_val:.2f}",
                    fontsize=5,
                )
                method_row_axes.append(ax)

            image_axes.append(method_row_axes)

            # Row label on the leftmost cell — short tag only
            method_row_axes[0].set_ylabel(
                _SHORT[method_label], fontsize=9, rotation=0,
                labelpad=14, va="center",
            )

        # ── Legend mapping short tags to full method names ────────────────
        from matplotlib.patches import Patch
        inr_idx = 0
        legend_handles = []
        for label in results.keys():
            is_baseline = any(b == label or (b in label and b in {"PI", "Plain INR", "Raw INR"})
                              for b in _BASELINE_LABELS)
            if is_baseline:
                c = "#888888"
                legend_label = _SHORT[label]
            else:
                c = _WONG[1 + inr_idx % (len(_WONG) - 1)]
                inr_idx += 1
                legend_label = f"{_SHORT[label]}: {label}"

            legend_handles.append(
                Patch(facecolor=c, alpha=0.65, edgecolor="grey",
                      linewidth=0.4, label=legend_label)
            )
        # Place the legend in the reserved bottom strip.
        # bbox_to_anchor uses figure-fraction coordinates.
        # loc="lower center" anchors the legend's bottom-centre to the point,
        # so setting y to a small positive value keeps the box off the figure
        # edge while remaining entirely below the GridSpec (at _bottom_frac).
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            bbox_to_anchor=(0.5, 0.005),
            fontsize=7,
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgrey",
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        # ── Shared colorbar ──────────────────────────────────────────────
        # Span all image rows in the last GridSpec column
        cb_ax = fig.add_subplot(gs[:, n_cols])
        sm = plt.cm.ScalarMappable(cmap=cmap_sos, norm=norm_sos)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label("SoS (m/s)", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7)

        if save_path:
            _save(fig, save_path, png_fallback=png_fallback)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig


def plot_metrics_comparison(
    results: dict[str, list[dict]],
    save_path: str | Path = "report_metrics.svg",
    *,
    metrics: Sequence[str] = ("MAE", "RMSE", "SSIM", "CNR"),
    metric_labels: dict[str, str] | None = None,
    metric_units: dict[str, str] | None = None,
    figwidth: float = 6.5,
    show: bool = False,
    png_fallback: bool = True,
) -> plt.Figure:
    """
    Box-plot comparison of reconstruction metrics across methods.

    Layout
    ------
    One subplot per metric (default: MAE, RMSE, SSIM, CNR), arranged in a
    single row.  Each subplot shows one box per method.
    Baselines (L1, L2) are drawn in grey; INR methods use the Wong palette.

    Parameters
    ----------
    results      : {method_label: [per_sample_result_dict]}
                   Each result_dict must have a ``"metrics"`` dict.
    save_path    : Output file path (SVG written; PNG optional fallback).
    metrics      : Metric keys to plot (must exist in ``result["metrics"]``).
    metric_labels: Display names for each metric key.
    metric_units : Units for each metric key (appended to y-axis label).
    figwidth     : Figure width in inches.
    show         : Call plt.show().
    png_fallback : Also save 300-DPI PNG.

    Returns
    -------
    matplotlib.figure.Figure

    Suggested caption
    -----------------
    "Distribution of reconstruction metrics across methods and samples.
    Boxes show interquartile range; whiskers extend to 1.5x IQR;
    outliers shown as dots.  Grey boxes: classical baselines (L1, L2).
    Colour boxes: INR-based methods."
    """
    _labels: dict[str, str] = {
        "MAE":  "MAE",
        "RMSE": "RMSE",
        "SSIM": "SSIM",
        "CNR":  "CNR",
    }
    _units: dict[str, str] = {
        "MAE":  "(m/s)",
        "RMSE": "(m/s)",
        "SSIM": "",
        "CNR":  "",
    }
    if metric_labels:
        _labels.update(metric_labels)
    if metric_units:
        _units.update(metric_units)

    method_names  = list(results.keys())
    n_methods     = len(method_names)
    n_metrics     = len(metrics)

    # Assign colours and line styles (for B&W compatibility)
    _linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    method_colors     = {}
    method_linestyles = {}
    inr_idx = 0
    for label in method_names:
        is_baseline = any(b == label or (b in label and b in {"PI", "Plain INR", "Raw INR"})
                          for b in _BASELINE_LABELS)
        if is_baseline:
            method_colors[label]     = "#888888"
            method_linestyles[label] = "--"
        else:
            method_colors[label]     = _WONG[1 + inr_idx % (len(_WONG) - 1)]
            method_linestyles[label] = _linestyles[inr_idx % len(_linestyles)]
            inr_idx += 1

    with matplotlib.rc_context(_SERIF_RCPARAMS):
        fig_h = 2.6 + 0.4            # fixed height: clean single-row layout
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(figwidth, fig_h),
            sharey=False,
        )
        if n_metrics == 1:
            axes = [axes]

        x_pos = np.arange(n_methods)

        for ax_idx, metric_key in enumerate(metrics):
            ax = axes[ax_idx]

            data_per_method = []
            for label in method_names:
                vals = [
                    r["metrics"][metric_key]
                    for r in results[label]
                    if metric_key in r.get("metrics", {})
                ]
                data_per_method.append(vals if vals else [float("nan")])

            bp = ax.boxplot(
                data_per_method,
                positions=x_pos,
                widths=0.55,
                patch_artist=True,
                notch=False,
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(
                    marker="o", markersize=3,
                    markerfacecolor="none", markeredgewidth=0.6,
                ),
                boxprops=dict(linewidth=0.6),
            )

            # Apply colours
            for patch, label in zip(bp["boxes"], method_names):
                patch.set_facecolor(method_colors[label])
                patch.set_alpha(0.65)

            # Axis styling — letter tags for INR methods (A, B, …) to match
            # the grid plot. Baselines keep their full label.
            short_labels = []
            inr_letter = 0
            for label in method_names:
                is_baseline = any(b == label or (b in label and b in {"PI", "Plain INR", "Raw INR"})
                                  for b in _BASELINE_LABELS)
                if is_baseline:
                    if "PI" in label or "Plain INR" in label:
                        if "rank#" in label:
                            import re
                            match = re.search(r'rank#(\d+)', label)
                            rnum = match.group(1) if match else "?"
                            short_labels.append(f"PI-{rnum}")
                        else:
                            short_labels.append("PI")
                    elif "L1" in label:
                        short_labels.append("L1")
                    elif "L2" in label:
                        short_labels.append("L2")
                    else:
                        short_labels.append(label)
                else:
                    short_labels.append(chr(ord("A") + inr_letter))
                    inr_letter += 1
            ax.set_xticks(x_pos)
            ax.set_xticklabels(short_labels, rotation=0, ha="center",
                               fontsize=7)
            # Title carries metric name + unit; no y-axis label
            unit = _units.get(metric_key, "")
            title = _labels.get(metric_key, metric_key)
            if unit:
                title = f"{title} {unit}"
            ax.set_title(title, fontsize=9, pad=3)
            ax.set_ylabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_linewidth(0.6)
            ax.spines["left"].set_linewidth(0.6)
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
            ax.tick_params(which="both", direction="in", width=0.5,
                           labelsize=7)
            ax.grid(axis="y", linewidth=0.4, linestyle="--", alpha=0.5,
                    zorder=0)

        # ── Legend with letter→name mapping (matches grid plot) ─────────
        from matplotlib.patches import Patch
        legend_handles = []
        letter_idx = 0
        for lbl in method_names:
            is_baseline = any(b == lbl or (b in lbl and b in {"PI", "Plain INR", "Raw INR"})
                              for b in _BASELINE_LABELS)
            if is_baseline:
                c = method_colors[lbl]
                tag = short_labels[method_names.index(lbl)]
            else:
                c = method_colors[lbl]
                tag = f"{chr(ord('A') + letter_idx)}: {lbl}"
                letter_idx += 1
            legend_handles.append(
                Patch(facecolor=c, alpha=0.65,
                      edgecolor="grey", linewidth=0.4, label=tag)
            )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(n_methods, 4),
            bbox_to_anchor=(0.5, -0.08),
            fontsize=6.5,
            frameon=True,
            framealpha=0.9,
            edgecolor="lightgrey",
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        fig.tight_layout(rect=[0, 0.08, 1, 1])

        if save_path:
            _save(fig, save_path, png_fallback=png_fallback)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig
