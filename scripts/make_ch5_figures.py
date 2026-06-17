"""make_ch5_figures.py — Chapter 5 thesis figure builder.

Reads pre-generated recons.npz + results.json artifacts from server run dirs
and produces publication-quality SVG + PNG figures for Chapter 5.

Usage
-----
# Build all 6 figures:
    python scripts/make_ch5_figures.py --figure all

# Build a single figure:
    python scripts/make_ch5_figures.py --figure J3

# Build multiple specific figures:
    python scripts/make_ch5_figures.py --figure J4 J5 J6

Before running: fill in the SOURCES dict below with the actual run-dir paths
once the server jobs finish.

SOURCES keys
------------
  hqt6bwmp_blob_recon      — run_reconstruction.py  output for sweep hqt6bwmp on kwave_blob
  ti60qmx3_geom_recon      — run_reconstruction.py  output for sweep ti60qmx3 on kwave_geom
  ti60qmx3_blob_recon      — run_reconstruction.py  output for sweep ti60qmx3 on kwave_blob
  hqt6bwmp_geom_recon      — run_reconstruction.py  output for sweep hqt6bwmp on kwave_geom
  ti60qmx3_geom_denoised   — run_denoised_reconstruction.py output for ti60qmx3 on kwave_geom
  ti60qmx3_blob_denoised   — run_denoised_reconstruction.py output for ti60qmx3 on kwave_blob
  z7bs7iy5_geom_honest      — run_joint_denoiser_recon.py (honest checkpoint) on kwave_geom
  z7bs7iy5_blob_honest      — run_joint_denoiser_recon.py (honest checkpoint) on kwave_blob
  ydma0yxl_geom_joint       — run_joint_denoiser_recon.py (CNR-selected, --tag honest) on kwave_geom
  ydma0yxl_blob_joint       — run_joint_denoiser_recon.py (CNR-selected, --tag honest) on kwave_blob
  q2p4zv3e_geom_joint       — run_joint_denoiser_recon.py on kwave_geom
  q2p4zv3e_blob_joint       — run_joint_denoiser_recon.py on kwave_blob
  edj3mqou_geom_joint       — run_joint_denoiser_recon.py on kwave_geom
  edj3mqou_blob_joint       — run_joint_denoiser_recon.py on kwave_blob
  ti60qmx3_phantom_recon    — run_reconstruction.py on phantom dataset
  ti60qmx3_breast_recon     — run_reconstruction.py on breast_data dataset
  z7bs7iy5_phantom_honest   — run_joint_denoiser_recon.py (honest) on phantom
  z7bs7iy5_breast_honest    — run_joint_denoiser_recon.py (honest) on breast_data
  ti60qmx3_phantom_denoised — run_denoised_reconstruction.py on phantom
  ti60qmx3_breast_denoised  — run_denoised_reconstruction.py on breast_data

Output directory
----------------
  thesis_reports/report/chapters/chapter5/sections/figs/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Project root — scripts/ lives one level below the repo root.
# ──────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ──────────────────────────────────────────────────────────────────────────────
# Lazy project-library accessor  (avoids import-at-module-load when the venv
# is not active; gives a clean error on first use instead)
# ──────────────────────────────────────────────────────────────────────────────

class _ProjectHelpers:
    """Namespace that holds lazily-imported project helpers."""
    _loaded: bool = False

    # Populated on first call to _get()
    load_recons_npz: Any = None
    SERIF_RCPARAMS: dict = {}
    WONG: list = []
    annotate_cell: Any = None
    diverging_norm: Any = None
    reshape: Any = None
    save: Any = None
    slowness_to_sos: Any = None
    plot_metrics_comparison: Any = None


_P = _ProjectHelpers()


def _get_p() -> _ProjectHelpers:
    """Return populated _P; import project helpers on first call."""
    if _P._loaded:
        return _P
    try:
        from inr_sos.evaluation.recon_export import load_recons_npz
        from inr_sos.visualization.report_figures import (
            _SERIF_RCPARAMS,
            _WONG,
            _annotate_cell,
            _diverging_norm,
            _reshape,
            _save,
            _slowness_to_sos,
            plot_metrics_comparison,
        )
    except ImportError as exc:
        raise ImportError(
            f"Could not import project helpers ({exc}).\n"
            "Activate the venv: source .venv/bin/activate\n"
            "Then run: uv run python scripts/make_ch5_figures.py ..."
        ) from exc

    _P.load_recons_npz = load_recons_npz
    _P.SERIF_RCPARAMS = _SERIF_RCPARAMS
    _P.WONG = _WONG
    _P.annotate_cell = _annotate_cell
    _P.diverging_norm = _diverging_norm
    _P.reshape = _reshape
    _P.save = _save
    _P.slowness_to_sos = _slowness_to_sos
    _P.plot_metrics_comparison = plot_metrics_comparison
    _P._loaded = True
    return _P


# ──────────────────────────────────────────────────────────────────────────────
# SOURCES — fill these in once server runs complete.
# Each value should be the ABSOLUTE path to the run directory that contains
# recons.npz (and results.json).  Example:
#   "ti60qmx3_geom_recon": "/mnt/asgard0/users/haben/INR-SoS-Recon/scripts/data/inr/ti60qmx3/kwave_geom/20260520_143210",
# ──────────────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_DAT  = _REPO / "scripts" / "data"

SOURCES: dict[str, str | None] = {
    # §5.1 — IC PoC: self-supervised (uses L) vs oracle direct supervision (no L)
    "hqt6bwmp_ic_self":          str(_DAT / "inr/hqt6bwmp/inverse_crime/20260527_103124_npz"),
    "hqt6bwmp_ic_oracle":        str(_DAT / "inr/hqt6bwmp/inverse_crime/20260527_103336_npz-oracle"),
    # J3 / J4 — standalone INR reconstruction, IC-tuned vs blob-tuned
    "hqt6bwmp_blob_recon":       str(_DAT / "inr/hqt6bwmp/kwave_blob/20260523_095624_npz"),
    "ti60qmx3_geom_recon":       str(_DAT / "inr/ti60qmx3/kwave_geom/20260523_095636_npz"),
    "ti60qmx3_blob_recon":       str(_DAT / "inr/ti60qmx3/kwave_blob/20260523_095705_npz"),
    "hqt6bwmp_geom_recon":       str(_DAT / "inr/hqt6bwmp/kwave_geom/20260523_095618_npz"),
    # J5 — measurement-regularised (denoised) reconstruction
    "ti60qmx3_geom_denoised":    str(_DAT / "denoised_reconstruction/ti60qmx3/kwave_geom/20260523_101906_npz"),
    "ti60qmx3_blob_denoised":    str(_DAT / "denoised_reconstruction/ti60qmx3/kwave_blob/20260523_101945_npz"),
    # J5b / J6 / J7 — joint staged pipeline (honest checkpoint)
    "z7bs7iy5_geom_honest":      str(_DAT / "joint_denoiser_recon/z7bs7iy5/kwave_geom/20260523_115443_honest"),
    "z7bs7iy5_blob_honest":      str(_DAT / "joint_denoiser_recon/z7bs7iy5/kwave_blob/20260523_115451_honest"),
    # J6 — ranking strategy comparison
    # ydma0yxl CNR-selected  (--selection_metric cnr --tag honest)  labels: cnr1_..., cnr2_...
    "ydma0yxl_geom_joint":       str(_DAT / "joint_denoiser_recon/ydma0yxl/kwave_geom/20260522_202717_honest"),
    "ydma0yxl_blob_joint":       str(_DAT / "joint_denoiser_recon/ydma0yxl/kwave_blob/20260522_202721_honest"),
    # ydma0yxl mean-MAE-selected  (--selection_metric loss --tag honest_meanmae)  labels: rank1_..., rank2_...
    "ydma0yxl_geom_meanmae":     str(_DAT / "joint_denoiser_recon/ydma0yxl/kwave_geom/20260522_202735_honest_meanmae"),
    "ydma0yxl_blob_meanmae":     str(_DAT / "joint_denoiser_recon/ydma0yxl/kwave_blob/20260522_202800_honest_meanmae"),
    "q2p4zv3e_geom_joint":       str(_DAT / "joint_denoiser_recon/q2p4zv3e/kwave_geom/20260523_002153_honest"),
    "q2p4zv3e_blob_joint":       str(_DAT / "joint_denoiser_recon/q2p4zv3e/kwave_blob/20260523_002159_honest"),
    "edj3mqou_geom_joint":       str(_DAT / "joint_denoiser_recon/edj3mqou/kwave_geom/20260523_003750_honest"),
    "edj3mqou_blob_joint":       str(_DAT / "joint_denoiser_recon/edj3mqou/kwave_blob/20260523_003755_honest"),
    # J7 — phantom and breast datasets
    "ti60qmx3_phantom_recon":    str(_DAT / "inr/ti60qmx3/phantom/20260523_101853_npz"),
    "ti60qmx3_breast_recon":     str(_DAT / "inr/ti60qmx3/breast_data/20260523_101856_npz"),
    "z7bs7iy5_phantom_honest":   str(_DAT / "joint_denoiser_recon/z7bs7iy5/phantom/20260523_005358_honest"),
    "z7bs7iy5_breast_honest":    str(_DAT / "joint_denoiser_recon/z7bs7iy5/breast_data/20260523_005405_honest"),
    "ti60qmx3_phantom_denoised": str(_DAT / "denoised_reconstruction/ti60qmx3/phantom/20260523_102958_npz"),
    "ti60qmx3_breast_denoised":  str(_DAT / "denoised_reconstruction/ti60qmx3/breast_data/20260523_105218_npz"),
}

# Output directory — the LaTeX project lives in a SIBLING `thesis_reports/`
# directory next to the INR-SoS-Recon code repo, not inside it.
_FIG_DIR = (
    _REPO_ROOT.parent
    / "thesis_reports"
    / "report"
    / "chapters"
    / "chapter5"
    / "sections"
    / "figs"
)

# Roman numerals used as column headers (up to 12 cols)
_ROMAN = [
    "I", "II", "III", "IV", "V", "VI",
    "VII", "VIII", "IX", "X", "XI", "XII",
]


# ──────────────────────────────────────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────────────────────────────────────

def _require_sources(*keys: str) -> dict[str, Path]:
    """Resolve SOURCES entries to Paths; abort with a clear message if missing."""
    missing = [k for k in keys if not SOURCES.get(k)]
    if missing:
        print(
            "\n[ERROR] The following SOURCES entries are not set:\n"
            + "\n".join(f"  {k}" for k in missing)
            + "\nFill them in at the top of make_ch5_figures.py and re-run.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    resolved: dict[str, Path] = {}
    for k in keys:
        p = Path(str(SOURCES[k]))
        if not p.exists():
            print(
                f"\n[ERROR] Source directory does not exist: {p}\n"
                f"  (SOURCES key: {k!r})\n",
                file=sys.stderr,
            )
            sys.exit(1)
        resolved[k] = p
    return resolved


def _load_npz(run_dir: Path, source_key: str) -> dict:
    """Load recons.npz from run_dir; abort clearly if the file is missing."""
    p = _get_p()
    npz = run_dir / "recons.npz"
    if not npz.exists():
        print(
            f"\n[ERROR] recons.npz not found in {run_dir}\n"
            f"  (SOURCES key: {source_key!r})\n"
            "  The pipeline must be run with recon-export support.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return p.load_recons_npz(npz)


def _try_load_metrics_from_json(run_dir: Path, method_key: str) -> list[dict] | None:
    """Load per-sample metrics from results.json if available.

    Returns a list of metric dicts (one per sample) or None if unavailable.
    Handles both pipeline JSON formats:
      - run_reconstruction.py : {"methods": {"rank#1 Full_Matrix": {..., "per_sample": [...]}}, ...}
      - run_joint_denoiser_recon.py : {"methods": {label: [metric_dict, ...]}, ...}
      - baselines in either format : {"baselines": {"L1": {..., "per_sample": [...]}}, ...}
    """
    import json
    rj = run_dir / "results.json"
    if not rj.exists():
        return None
    with open(rj) as f:
        data = json.load(f)

    methods  = data.get("methods", {})
    baseline = data.get("baselines", {})

    def _extract(entry) -> list[dict] | None:
        """Extract per-sample metric list from a results.json entry."""
        if isinstance(entry, list):
            # Joint pipeline: list of metric dicts
            out = []
            for e in entry:
                if isinstance(e, dict):
                    m = e if "MAE" in e else e.get("metrics", e)
                    if "MAE" in m:
                        out.append(m)
            return out if out else None
        if isinstance(entry, dict):
            # Reconstruction pipeline: dict with "per_sample" list
            per = entry.get("per_sample", [])
            if per:
                out = []
                for e in per:
                    m = e if "MAE" in e else e.get("metrics", e)
                    if "MAE" in m:
                        out.append(m)
                return out if out else None
        return None

    # Exact key match first
    if method_key in methods:
        return _extract(methods[method_key])
    if method_key in baseline:
        return _extract(baseline[method_key])

    # Fuzzy match: method_key is a substring of a stored key (rank-aware labels)
    for key, entry in methods.items():
        if method_key in key or key in method_key:
            result = _extract(entry)
            if result:
                return result

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Method-selection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_rank(recons_dict: dict[str, np.ndarray], rank: int = 1) -> tuple[str, np.ndarray]:
    """Return (label, array) for the method whose rank index is `rank`.

    Matching strategies tried in order:
    1. Exact token "rank#<rank>" in label (run_reconstruction.py format).
    2. Bare "rank<rank>" token (run_denoised_reconstruction.py format: "rank1_Full_Matrix_...").
    3. Insertion-order fallback: rank-th non-baseline INR label (1-indexed).

    Raises KeyError if nothing is found.
    """
    pat_hash = re.compile(rf"(?<!\d)rank#{rank}(?!\d)")
    pat_bare = re.compile(rf"(?<!\d)rank{rank}(?!\d)")

    for lbl in recons_dict:
        if pat_hash.search(lbl) or pat_bare.search(lbl):
            return lbl, recons_dict[lbl]

    # Fallback: count non-baseline labels in insertion order
    inr_keys = [k for k in recons_dict if k not in {"L1", "L2", "PI"}]
    if 1 <= rank <= len(inr_keys):
        lbl = inr_keys[rank - 1]
        return lbl, recons_dict[lbl]

    raise KeyError(
        f"Rank {rank} not found. Available labels: {list(recons_dict.keys())}"
    )


def _pick_by_prefix(recons_dict: dict[str, np.ndarray],
                    prefix: str) -> tuple[str, np.ndarray] | None:
    """Return the rank-1 (numerically lowest suffix) label with the given prefix.

    Labels carry a source-tag prefix from fetch_topk: "rank" (loss/mean-MAE
    selection), "cnr" (CNR selection), "roi" (mae_roi selection).
    """
    candidates = [(k, v) for k, v in recons_dict.items() if k.startswith(prefix)]
    if not candidates:
        return None

    def _suffix_rank(lbl: str) -> int:
        m = re.search(r"(\d+)", lbl[len(prefix):])
        return int(m.group(1)) if m else 999

    candidates.sort(key=lambda kv: _suffix_rank(kv[0]))
    return candidates[0]


def _first_match_by_prefixes(
    recons_dict: dict[str, np.ndarray],
    prefixes: list[str],
) -> tuple[str, np.ndarray] | None:
    """Return rank-1 label matching the first prefix that has candidates.

    Falls back to _pick_rank(1) if no prefix matches.
    """
    for pf in prefixes:
        r = _pick_by_prefix(recons_dict, pf)
        if r is not None:
            return r
    try:
        return _pick_rank(recons_dict, 1)
    except KeyError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Conversion + metric helpers  (pure numpy — no project import required)
# ──────────────────────────────────────────────────────────────────────────────

def _to_sos_img(flat_slowness: np.ndarray) -> np.ndarray:
    """Convert flat slowness (4096,) -> SoS image (64, 64) via project helpers."""
    p = _get_p()
    return p.reshape(p.slowness_to_sos(flat_slowness))


def _shared_norm(gt_images: list[np.ndarray]) -> mcolors.TwoSlopeNorm:
    """Build a TwoSlopeNorm centred at GT median, shared across panels."""
    stack = np.stack(gt_images)
    bg    = float(np.median(stack))
    vmin  = max(1380.0, float(stack.min()) - 10.0)
    vmax  = min(1620.0, float(stack.max()) + 10.0)
    if vmin >= bg:
        vmin = bg - 1.0
    if vmax <= bg:
        vmax = bg + 1.0
    return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=bg, vmax=vmax)


def _mae_cnr(rec_img: np.ndarray, gt_img: np.ndarray) -> tuple[float, float]:
    """Return (MAE in m/s, CNR) from SoS images."""
    mae = float(np.mean(np.abs(rec_img - gt_img)))
    bg  = float(np.median(gt_img))
    roi_mask = np.abs(gt_img - bg) > 5.0
    bg_mask  = np.abs(gt_img - bg) < 2.0
    if roi_mask.sum() < 4 or bg_mask.sum() < 4:
        return mae, float("nan")
    roi_mean = float(np.mean(rec_img[roi_mask]))
    bg_mean  = float(np.mean(rec_img[bg_mask]))
    bg_std   = float(np.std(rec_img[bg_mask]))
    cnr = abs(roi_mean - bg_mean) / (bg_std + 1e-8)
    return mae, cnr


def _ssim_approx(rec_img: np.ndarray, gt_img: np.ndarray) -> float:
    """Luminance-only SSIM approximation (used when results.json unavailable)."""
    mu_gt  = float(np.mean(gt_img))
    mu_rec = float(np.mean(rec_img))
    s_gt   = float(np.std(gt_img))
    s_rec  = float(np.std(rec_img))
    s_cross = float(np.mean((gt_img - mu_gt) * (rec_img - mu_rec)))
    c1, c2  = (0.01 * 200.0) ** 2, (0.03 * 200.0) ** 2
    return float(
        (2 * mu_gt * mu_rec + c1) * (2 * s_cross + c2)
        / ((mu_gt ** 2 + mu_rec ** 2 + c1) * (s_gt ** 2 + s_rec ** 2 + c2))
    )


def _compute_per_sample_metrics(npz: dict, method_key: str) -> list[dict]:
    """Compute per-sample MAE, RMSE, SSIM (approx), CNR from npz arrays."""
    out = []
    for s_gt, s_rec in zip(npz["gt"], npz["recons"][method_key]):
        gt_img  = _to_sos_img(s_gt)
        rec_img = _to_sos_img(s_rec)
        mae, cnr = _mae_cnr(rec_img, gt_img)
        rmse = float(np.sqrt(np.mean((rec_img - gt_img) ** 2)))
        ssim = _ssim_approx(rec_img, gt_img)
        out.append({"MAE": mae, "RMSE": rmse, "SSIM": ssim, "CNR": cnr})
    return out


def _build_results_entry(
    display_label: str,
    npz: dict,
    method_key: str,
    run_dir: Path | None = None,
) -> tuple[str, list[dict]]:
    """Build a (display_label, per_sample_list) pair for plot_metrics_comparison.

    Prefers metrics from results.json; falls back to computing from arrays.
    Each element of per_sample_list: {"metrics": {...}, "s_phys": flat_array}.
    """
    json_metrics = None
    if run_dir is not None:
        json_metrics = _try_load_metrics_from_json(run_dir, method_key)

    n = len(npz["gt"])
    if json_metrics is not None and len(json_metrics) == n:
        per_sample = [
            {"metrics": m, "s_phys": npz["recons"][method_key][i]}
            for i, m in enumerate(json_metrics)
        ]
    else:
        computed = _compute_per_sample_metrics(npz, method_key)
        per_sample = [
            {"metrics": m, "s_phys": npz["recons"][method_key][i]}
            for i, m in enumerate(computed)
        ]
    return display_label, per_sample


def _col_headers(n: int, prefix: str = "") -> list[str]:
    """Roman-numeral column headers, optionally prefixed."""
    out = []
    for i in range(n):
        r = _ROMAN[i] if i < len(_ROMAN) else str(i + 1)
        out.append(f"{prefix}{r}" if prefix else r)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core grid builder
# ──────────────────────────────────────────────────────────────────────────────

def _draw_recon_grid(
    row_specs: list[tuple[str, list[np.ndarray]]],
    col_labels: list[str],
    save_path: Path,
    *,
    figwidth: float = 7.0,
    cmap: str = "RdBu_r",
    shared_norm: mcolors.TwoSlopeNorm | None = None,
    gt_images: list[np.ndarray] | None = None,
    annotate_mae_cnr: bool = True,
    dataset_title: str | None = None,
) -> None:
    """Generic reconstruction grid figure.

    Parameters
    ----------
    row_specs      : [(row_label, [img, ...]), ...] — one entry per row.
    col_labels     : column header strings (same length as images per row).
    save_path      : output path (SVG + PNG written by _save).
    shared_norm    : colour normalisation used for all cells.
                     Built from row_specs[0] (GT row) if not supplied.
    gt_images      : GT SoS images used for MAE/CNR annotation; must match
                     column order. No annotation if None.
    annotate_mae_cnr: draw per-cell MAE/CNR text overlay (skipped for GT row).
    """
    p = _get_p()
    n_rows = len(row_specs)
    n_cols = len(col_labels)
    cb_frac = 0.03

    if shared_norm is None:
        shared_norm = _shared_norm(row_specs[0][1])

    cell_h = figwidth / n_cols
    fig_h  = cell_h * n_rows + 0.6

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(figwidth, fig_h))
        fig.patch.set_facecolor("white")

        # Reserve a bit more headroom when a dataset title is drawn.
        _top = 0.91 if dataset_title else 0.93
        gs = GridSpec(
            n_rows, n_cols + 1,
            figure=fig,
            width_ratios=[1.0] * n_cols + [cb_frac * n_cols],
            hspace=0.04,
            wspace=0.03,
            left=0.07,
            right=0.95,
            top=_top,
            bottom=0.04,
        )
        if dataset_title:
            fig.suptitle(dataset_title, fontsize=10, fontweight="bold", y=0.98)

        for row_idx, (row_label, images) in enumerate(row_specs):
            for col_idx, img in enumerate(images):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(
                    img, cmap=cmap, norm=shared_norm,
                    interpolation="nearest", origin="upper",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.4)

                # Column headers on the first row only
                if row_idx == 0:
                    ax.set_title(col_labels[col_idx], fontsize=8, pad=3)

                # Row label on the leftmost cell
                if col_idx == 0:
                    ax.set_ylabel(
                        row_label, fontsize=9, rotation=0,
                        labelpad=20, va="center",
                    )

                # Per-cell annotation (skip GT row, skip if no GT provided)
                if annotate_mae_cnr and gt_images is not None and row_idx > 0:
                    gt_img = gt_images[col_idx]
                    mae, cnr = _mae_cnr(img, gt_img)
                    p.annotate_cell(
                        ax,
                        f"MAE: {mae:.1f}\nCNR: {cnr:.2f}",
                        fontsize=5,
                    )

        # Shared colorbar spanning all rows
        cb_ax = fig.add_subplot(gs[:, n_cols])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=shared_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label("SoS (m/s)", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7)

        p.save(fig, save_path, png_fallback=False)
        plt.close(fig)

    print(f"  [grid] saved -> {save_path.with_suffix('.svg')}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: select column indices evenly from a dataset
# ──────────────────────────────────────────────────────────────────────────────

def _draw_combined_dataset_grid(
    panels,
    row_labels,
    save_path: Path,
    *,
    figwidth: float = 11.0,
    cmap: str = "RdBu_r",
    annotate_mae_cnr: bool = True,
    highlight_cells: list[tuple[int, int, int]] | tuple[int, int, int] | None = None,
) -> None:
    """One reconstruction grid with multiple column-groups SIDE-BY-SIDE.

    Each column-group ("panel") has its own header title (e.g. "GeomSet",
    "BlobSet") and is separated from the next by a vertical dashed line.
    All groups share the row dimension and a single colorbar.

    panels: list of (dataset_title, col_labels, rows_imgs)
        rows_imgs[r] is the list of images for row r in this panel; row 0 is
        treated as GT (used for shared diverging norm + MAE/CNR baseline).
    row_labels: ["GT", "Standalone", ...]  (must equal len(rows_imgs))
    highlight_cells: ``(panel_idx, row_idx, col_idx)`` or a list of such
        tuples — optional. Draws a dotted circle around the inclusion on
        every listed cell to teach the reader what "inclusion" refers to.
        The inclusion mask is derived from the GT for that column; the
        circle is the same regardless of which row is highlighted.
    """
    # Normalise highlight_cells to a list of (panel, row, col) tuples.
    if highlight_cells is None:
        highlight_list: list[tuple[int, int, int]] = []
    elif (isinstance(highlight_cells, tuple)
          and len(highlight_cells) == 3
          and all(isinstance(v, int) for v in highlight_cells)):
        highlight_list = [highlight_cells]
    else:
        highlight_list = list(highlight_cells)
    highlight_set = set(highlight_list)
    from matplotlib.lines import Line2D

    p = _get_p()
    n_panels = len(panels)
    n_rows = len(row_labels)
    panel_widths = [len(cl) for _, cl, _ in panels]
    sep_count = max(0, n_panels - 1)

    # Shared diverging norm across all GT images in all panels.
    all_gt = [img for _, _, rows in panels for img in rows[0]]
    shared_norm = _shared_norm(all_gt) if all_gt else None

    # Build the column width-ratio sequence: data cols (1.0) + thin spacers
    # (0.18) between panels + colorbar (0.4) at the end.
    width_ratios: list[float] = []
    col_map: dict[tuple[int, int], int] = {}
    cur = 0
    for pi, w in enumerate(panel_widths):
        if pi > 0:
            width_ratios.append(0.18)
            cur += 1
        for ci in range(w):
            width_ratios.append(1.0)
            col_map[(pi, ci)] = cur
            cur += 1
    cb_col = cur
    width_ratios.append(0.45)

    total_units = sum(width_ratios)
    cell_h = figwidth / total_units
    fig_h = cell_h * n_rows + 0.7

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        fig = plt.figure(figsize=(figwidth, fig_h))
        fig.patch.set_facecolor("white")

        gs = fig.add_gridspec(
            n_rows, len(width_ratios),
            width_ratios=width_ratios,
            hspace=0.04, wspace=0.03,
            left=0.05, right=0.97, top=0.91, bottom=0.04,
        )

        for pi, (_, col_labels, rows) in enumerate(panels):
            gt_imgs = rows[0]
            for r in range(n_rows):
                for c in range(panel_widths[pi]):
                    ax = fig.add_subplot(gs[r, col_map[(pi, c)]])
                    img = rows[r][c]
                    ax.imshow(img, cmap=cmap, norm=shared_norm,
                              interpolation="nearest", origin="upper")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_linewidth(0.4)
                    if r == 0:
                        ax.set_title(col_labels[c], fontsize=7, pad=2)
                    if c == 0 and pi == 0:
                        ax.set_ylabel(row_labels[r], fontsize=8, rotation=0,
                                      labelpad=20, va="center")
                    if annotate_mae_cnr and r > 0 and c < len(gt_imgs):
                        mae, cnr = _mae_cnr(img, gt_imgs[c])
                        p.annotate_cell(
                            ax, f"MAE: {mae:.1f}\nCNR: {cnr:.2f}", fontsize=5,
                        )
                    if (pi, r, c) in highlight_set and c < len(gt_imgs):
                        _overlay_inclusion_outline(ax, gt_imgs[c])

        # Panel header titles, centred above each group
        for pi, (title, _, _) in enumerate(panels):
            left_ss  = gs[0, col_map[(pi, 0)]].get_position(fig)
            right_ss = gs[0, col_map[(pi, panel_widths[pi] - 1)]].get_position(fig)
            x_center = (left_ss.x0 + right_ss.x1) / 2
            fig.text(
                x_center, left_ss.y1 + 0.025, title,
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        # Dashed vertical separators between panels
        for pi in range(1, n_panels):
            # spacer column index = right boundary of previous panel + 1
            sep_gs_col = col_map[(pi, 0)] - 1
            sep_ss = gs[0, sep_gs_col].get_position(fig)
            x_sep = (sep_ss.x0 + sep_ss.x1) / 2
            top_ss = gs[0, col_map[(pi, 0)]].get_position(fig)
            bot_ss = gs[n_rows - 1, col_map[(pi, 0)]].get_position(fig)
            line = Line2D(
                [x_sep, x_sep], [bot_ss.y0, top_ss.y1],
                transform=fig.transFigure, color="black",
                linestyle="--", linewidth=0.8, alpha=0.55,
            )
            fig.add_artist(line)

        # Shared colorbar
        cb_ax = fig.add_subplot(gs[:, cb_col])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=shared_norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cb_ax)
        cb.set_label("SoS (m/s)", fontsize=8, labelpad=4)
        cb.ax.tick_params(labelsize=7)

        p.save(fig, save_path, png_fallback=False)
        plt.close(fig)
    print(f"  [combined-grid] saved -> {save_path.with_suffix('.svg')}")


def _draw_combined_metrics(
    geom_results: dict[str, list[dict]],
    blob_results: dict[str, list[dict]] | None,
    save_path: Path,
    *,
    metrics: tuple[str, ...] = ("MAE", "RMSE", "SSIM", "CNR"),
    figwidth: float = 11.0,
    fig_height: float = 3.2,
    short_labels: list[str] | None = None,
    group_labels: tuple[str, str] = ("GeomSet", "BlobSet"),
) -> None:
    """4-metric box-plot figure.

    If ``blob_results`` is provided, the figure is split left | right by a
    dashed line and the two panels share the metric axes. If ``blob_results``
    is ``None``, a single-panel (geom-only) figure is drawn — used by §5.4
    after the breast metrics panel was dropped (no valid in-vivo GT).

    When two panels are given, method ordering MUST be identical between them.
    """
    p = _get_p()
    method_names = list(geom_results.keys())
    single_panel = blob_results is None
    if not single_panel and list(blob_results.keys()) != method_names:
        raise ValueError("geom and blob results must have identical methods/order")

    metric_titles = {"MAE": "MAE (m/s)", "RMSE": "RMSE (m/s)",
                     "SSIM": "SSIM", "CNR": "CNR"}

    # Resolve x-axis short labels and per-method colors.
    if short_labels is None:
        short_labels = []
        inr_idx = 0
        for lbl in method_names:
            if lbl in ("L1", "L2", "PI"):
                short_labels.append(lbl)
            else:
                inr_idx += 1
                short_labels.append(str(inr_idx))
    if len(short_labels) != len(method_names):
        raise ValueError("short_labels length must match method_names")

    method_colors: dict[str, str] = {}
    inr_color_idx = 0
    for lbl in method_names:
        if lbl in ("L1", "L2", "PI"):
            method_colors[lbl] = "#888888"
        else:
            method_colors[lbl] = p.WONG[1 + inr_color_idx % (len(p.WONG) - 1)]
            inr_color_idx += 1

    n = len(method_names)
    sep = 1.2                                    # spacer between geom and blob
    x_geom = np.arange(1, n + 1, dtype=float)
    x_blob = x_geom + n + sep
    x_sep  = (x_geom[-1] + x_blob[0]) / 2.0

    def _vals(res, lbl, mk):
        v = [r["metrics"][mk] for r in res[lbl] if mk in r.get("metrics", {})]
        return v if v else [float("nan")]

    common_box_kw = dict(
        widths=0.55, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor="none", markeredgewidth=0.5),
        boxprops=dict(linewidth=0.5),
    )

    # Single-panel mode uses a narrower figure since there's no second half.
    eff_width = (figwidth * 0.55) if single_panel else figwidth

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        fig, axes = plt.subplots(1, len(metrics),
                                 figsize=(eff_width, fig_height),
                                 sharey=False)
        if len(metrics) == 1:
            axes = [axes]

        for ax, mk in zip(axes, metrics):
            data_g = [_vals(geom_results, lbl, mk) for lbl in method_names]
            bp_g = ax.boxplot(data_g, positions=x_geom, **common_box_kw)
            for patch, lbl in zip(bp_g["boxes"], method_names):
                patch.set_facecolor(method_colors[lbl]); patch.set_alpha(0.65)

            if not single_panel:
                data_b = [_vals(blob_results, lbl, mk) for lbl in method_names]
                bp_b = ax.boxplot(data_b, positions=x_blob, **common_box_kw)
                for patch, lbl in zip(bp_b["boxes"], method_names):
                    patch.set_facecolor(method_colors[lbl]); patch.set_alpha(0.65)

                # Dashed vertical separator between panels
                ax.axvline(x_sep, color="black", linestyle="--",
                           linewidth=0.8, alpha=0.55)

                ax.set_xticks(list(x_geom) + list(x_blob))
                ax.set_xticklabels(short_labels * 2, fontsize=6.5)
            else:
                ax.set_xticks(list(x_geom))
                ax.set_xticklabels(short_labels, fontsize=6.5)

            ax.set_title(metric_titles.get(mk, mk), fontsize=11, pad=14)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(which="both", direction="in", width=0.5,
                           labelsize=6.5)
            ax.grid(axis="y", linewidth=0.4, linestyle="--", alpha=0.5,
                    zorder=0)

            # Group headers above each half (in axes coordinates so they sit
            # right under the metric title and don't depend on the y-range).
            mid_g = (x_geom[0] + x_geom[-1]) / 2
            if single_panel:
                xmin = x_geom[0] - 0.6
                xmax = x_geom[-1] + 0.6
                ax.set_xlim(xmin, xmax)
                ax.text((mid_g - xmin) / (xmax - xmin), 1.02, group_labels[0],
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
            else:
                mid_b = (x_blob[0] + x_blob[-1]) / 2
                xmin = x_geom[0] - 0.6
                xmax = x_blob[-1] + 0.6
                ax.set_xlim(xmin, xmax)
                ax.text((mid_g - xmin) / (xmax - xmin), 1.02, group_labels[0],
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
                ax.text((mid_b - xmin) / (xmax - xmin), 1.02, group_labels[1],
                        transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        fig.tight_layout(pad=0.6)
        p.save(fig, save_path, png_fallback=False)
        plt.close(fig)
    print(f"  [combined-metrics] saved -> {save_path.with_suffix('.svg')}")


def _col_indices(n_total: int, n_want: int) -> list[int]:
    """Return n_want indices evenly spread over [0, n_total)."""
    n = min(n_want, n_total)
    return np.linspace(0, n_total - 1, n, dtype=int).tolist()


def _overlay_inclusion_outline(
    ax,
    img: np.ndarray,
    *,
    edgecolor: str = "#ffd60a",   # high-visibility gold; reads clearly on RdBu_r
    linewidth: float = 1.3,
    linestyle=(0, (1.5, 1.5)),    # dotted
    threshold: float = 5.0,
    fallback_percentile: float = 95.0,
    margin_px: float = 2.0,
    forced_center: tuple[float, float] | None = None,
    forced_radius: float | None = None,
) -> bool:
    """Draw a dotted circle around the inclusion on a single axes.

    Primary inclusion mask uses the same |img - median(img)| > ``threshold``
    rule as the CNR metric (`_mae_cnr`). When that produces too few pixels
    (smooth reconstructions, e.g. the staged-joint breast result), the rule
    is relaxed to the top ``100 - fallback_percentile`` percent most-deviant
    pixels, so a meaningful region is always identified. The circle is then
    the bounding circle of that mask (centroid + max-radius + ``margin_px``
    padding), which guarantees it covers every inclusion pixel and reads as
    a clean shape at thesis scale. Returns ``True`` on success.

    The reference image (``img``) is typically the column's GT, but for the
    breast qualitative figure — which has no valid GT — the reconstruction
    itself is passed in. The circle is then placed around whichever region
    that reconstruction sets apart from its own median.
    """
    from matplotlib.patches import Circle

    if forced_center is not None and forced_radius is not None:
        # Brute-force placement (used for the breast figure, where the recon
        # is too smooth for any threshold/percentile rule to find the inclusion).
        cx, cy = float(forced_center[0]), float(forced_center[1])
        radius = float(forced_radius)
    else:
        bg = float(np.median(img))
        deviation = np.abs(img - bg)
        mask = deviation > threshold
        if mask.sum() < 4:
            cutoff = float(np.percentile(deviation, fallback_percentile))
            mask = deviation >= cutoff
            if mask.sum() < 4:
                return False
        rows, cols = np.where(mask)
        cy = float(rows.mean())
        cx = float(cols.mean())
        radius = float(np.max(np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2))) + margin_px

    ax.add_patch(Circle(
        (cx, cy), radius,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=5,
    ))
    return True


def _draw_qualitative_row(images, col_labels, save_path: Path,
                          title: str = "", cmap: str = "RdBu_r",
                          highlight_idx: int | None = None,
                          highlight_center: tuple[float, float] | None = None,
                          highlight_radius: float | None = None) -> None:
    """1-row × N-col qualitative grid for a single sample across methods.

    No GT, no per-cell metric annotation. Used by §5.4 for the breast sample
    where there is no valid in-vivo ground truth and only a qualitative
    comparison across reconstruction methods is meaningful.

    ``highlight_idx`` (optional): column index to overlay with a dotted
    inclusion circle. Since there is no GT, the inclusion mask is derived
    from that column's own reconstruction (``images[highlight_idx]``).
    """
    p = _get_p()
    n = len(images)
    norm = _shared_norm(images) if images else None

    cell_w = 1.8
    fig_w = cell_w * n + 1.0     # +1 for the right-side colorbar
    fig_h = cell_w + 0.6         # +0.6 for the column titles

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes[0]
        fig.patch.set_facecolor("white")

        for col, (ax, img, lbl) in enumerate(zip(axes, images, col_labels)):
            ax.imshow(img, cmap=cmap, norm=norm,
                      interpolation="nearest", origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            ax.set_title(lbl, fontsize=8, pad=3)
            if highlight_idx is not None and col == highlight_idx:
                _overlay_inclusion_outline(
                    ax, img,
                    forced_center=highlight_center,
                    forced_radius=highlight_radius,
                )

        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)

        fig.subplots_adjust(left=0.02, right=0.88,
                            top=0.78 if title else 0.85,
                            bottom=0.04, wspace=0.05)

        # Right-side colorbar
        if norm is not None:
            cb_ax = fig.add_axes([0.90, 0.05, 0.025, 0.72])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cb_ax)
            cb.set_label("SoS (m/s)", fontsize=8, labelpad=4)
            cb.ax.tick_params(labelsize=7)

        p.save(fig, save_path, png_fallback=False)
        plt.close(fig)
    print(f"  [qual-row] saved -> {save_path.with_suffix('.svg')}")


def _load_dataset_canonical_gt(mat_filename: str) -> list[np.ndarray] | None:
    """Return the ground-truth slowness slabs from ``DATA_DIR/DL-based-SoS/<mat>``,
    in the order stored by the source ``.mat`` file (sample I = row 0, etc.).

    Used by figure_J7 so the PhantomSet column labelled "I" in the figure
    truly corresponds to dataset sample 0, regardless of how each pipeline
    run permuted its npz output.

    Returns ``None`` if the file cannot be opened (e.g. running off the
    server) so callers can fall back to the recon-npz GT order.
    """
    try:
        import h5py
        from inr_sos.io.paths import DATA_DIR
    except Exception as exc:
        print(f"  [canonical-gt] cannot import DATA_DIR/h5py: {exc}; falling back.")
        return None

    path = Path(DATA_DIR) / "DL-based-SoS" / mat_filename
    if not path.exists():
        print(f"  [canonical-gt] {path} not found; falling back to recon-npz order.")
        return None
    try:
        with h5py.File(path, "r") as f:
            if "imgs_gt" not in f:
                print(f"  [canonical-gt] {path} has no 'imgs_gt' key; falling back.")
                return None
            imgs = np.array(f["imgs_gt"])
        return [imgs[i] for i in range(imgs.shape[0])]
    except Exception as exc:
        print(f"  [canonical-gt] {path} read failed ({exc}); falling back.")
        return None


def _match_permutation(src_gt, ref_gt) -> list[int]:
    """Find indices into ``src_gt`` that align it to ``ref_gt`` order.

    ``ref_gt`` and ``src_gt`` are arrays of the same physical GT samples but
    possibly stored in different orders by two pipeline runs. For each
    ``ref_gt[k]`` we pick the unused ``src_gt[i]`` that minimises L2 distance
    (after flattening). Returned ``perm`` is such that
    ``src_gt[perm[k]] ≈ ref_gt[k]``.

    If shapes disagree or no candidate matches, falls back to identity for
    that slot.
    """
    n_ref = len(ref_gt)
    n_src = len(src_gt)
    perm: list[int] = []
    used: set[int] = set()
    for k in range(n_ref):
        rk = np.asarray(ref_gt[k]).reshape(-1).astype(np.float64, copy=False)
        best_i = -1
        best_d = float("inf")
        for i in range(n_src):
            if i in used:
                continue
            si = np.asarray(src_gt[i]).reshape(-1).astype(np.float64, copy=False)
            if si.shape != rk.shape:
                continue
            d = float(np.linalg.norm(si - rk))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i == -1:
            best_i = k if k < n_src else 0
        used.add(best_i)
        perm.append(best_i)
    return perm


# ──────────────────────────────────────────────────────────────────────────────
# Figure J3 — IC-trained transfer to blob: 8×8 grid + metrics box
# ──────────────────────────────────────────────────────────────────────────────

def figure_J3() -> None:
    """5.2_v1_ictrans_blob_comparison.svg + 5.2_v1_ictrans_blob_metrics.svg

    Grid  : 8 rows × 8 cols — GT, rank#1..rank#5, L2, L1; 8 samples
    Box   : MAE/RMSE/SSIM/CNR — 7 methods (rank#1-5, L2, L1)
    Source: hqt6bwmp on kwave_blob (IC-tuned INR applied to blob)
    """
    print("\n[J3] Building IC-transfer combined figures ...")
    dirs = _require_sources("hqt6bwmp_geom_recon", "hqt6bwmp_blob_recon")
    g_run = dirs["hqt6bwmp_geom_recon"]
    b_run = dirs["hqt6bwmp_blob_recon"]
    g_data = _load_npz(g_run, "hqt6bwmp_geom_recon")
    b_data = _load_npz(b_run, "hqt6bwmp_blob_recon")

    def _build_panel(title, data):
        n = len(data["gt"])
        col_labels = _col_headers(n)
        gt_imgs = [_to_sos_img(data["gt"][i]) for i in range(n)]
        rows_imgs = [gt_imgs]                # row 0 = GT
        for r in range(1, 4):                # top-3 (chapter-wide standard)
            try:
                _, arr = _pick_rank(data["recons"], r)
                rows_imgs.append([_to_sos_img(arr[i]) for i in range(n)])
            except KeyError:
                rows_imgs.append([np.full_like(gt_imgs[0], np.nan)
                                  for _ in gt_imgs])
        for baseline in ("L2", "L1"):
            if baseline in data["recons"]:
                arr = data["recons"][baseline]
                rows_imgs.append([_to_sos_img(arr[i]) for i in range(n)])
            else:
                rows_imgs.append([np.full_like(gt_imgs[0], np.nan)
                                  for _ in gt_imgs])
        return (title, col_labels, rows_imgs)

    panels = [
        _build_panel("GeomSet", g_data),
        _build_panel("BlobSet", b_data),
    ]
    _draw_combined_dataset_grid(
        panels,
        row_labels=["GT", "1", "2", "3", "L2", "L1"],
        save_path=_FIG_DIR / "5.2_v1_ictrans_grid.svg",
    )

    # Combined metrics figure
    def _build_results(data, run_dir):
        entries: list[tuple[str, list[dict]]] = []
        for r in range(1, 4):                # top-3 (chapter-wide standard)
            try:
                lbl, _ = _pick_rank(data["recons"], r)
                entries.append(_build_results_entry(f"R{r}", data, lbl, run_dir))
            except KeyError:
                pass
        for baseline in ("L2", "L1"):
            if baseline in data["recons"]:
                entries.append(
                    _build_results_entry(baseline, data, baseline, run_dir)
                )
        return dict(entries)

    geom_res = _build_results(g_data, g_run)
    blob_res = _build_results(b_data, b_run)
    _draw_combined_metrics(
        geom_res, blob_res,
        save_path=_FIG_DIR / "5.2_v1_ictrans_metrics.svg",
    )
    print("  [J3] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J4 — IC-tuned vs blob-tuned, per dataset
# ──────────────────────────────────────────────────────────────────────────────

def figure_J4() -> None:
    """5.2_retune_grid_{geom,blob}.svg + 5.2_retune_metrics_{geom,blob}.svg

    Grid : 5 rows × 8 cols — GT, ti60qmx3 rank#1, hqt6bwmp rank#1, L2, L1
    Box  : 8 boxes — ti60qmx3 rank#1-5, hqt6bwmp rank#1, L2, L1
    """
    print("\n[J4] Building re-tune transfer combined figures ...")
    dirs = _require_sources(
        "ti60qmx3_geom_recon", "hqt6bwmp_geom_recon",
        "ti60qmx3_blob_recon", "hqt6bwmp_blob_recon",
    )
    self_g = _load_npz(dirs["ti60qmx3_geom_recon"], "ti60qmx3_geom_recon")
    self_b = _load_npz(dirs["ti60qmx3_blob_recon"], "ti60qmx3_blob_recon")
    ic_g   = _load_npz(dirs["hqt6bwmp_geom_recon"], "hqt6bwmp_geom_recon")
    ic_b   = _load_npz(dirs["hqt6bwmp_blob_recon"], "hqt6bwmp_blob_recon")

    def _grid_panel(title, self_data, ic_data):
        n = len(self_data["gt"])
        col_labels = _col_headers(n)
        gt_imgs = [_to_sos_img(self_data["gt"][i]) for i in range(n)]

        def _row_from(arr):
            return [_to_sos_img(arr[i]) for i in range(min(n, len(arr)))]

        try:
            _, arr_s = _pick_rank(self_data["recons"], 1)
            row_self = _row_from(arr_s)
        except KeyError:
            row_self = [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        try:
            _, arr_ic = _pick_rank(ic_data["recons"], 1)
            row_ic = _row_from(arr_ic)
        except KeyError:
            row_ic = [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        row_L2 = (_row_from(self_data["recons"]["L2"])
                  if "L2" in self_data["recons"]
                  else [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs])
        row_L1 = (_row_from(self_data["recons"]["L1"])
                  if "L1" in self_data["recons"]
                  else [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs])
        return (title, col_labels, [gt_imgs, row_self, row_ic, row_L2, row_L1])

    panels = [
        _grid_panel("GeomSet", self_g, ic_g),
        _grid_panel("BlobSet", self_b, ic_b),
    ]
    _draw_combined_dataset_grid(
        panels,
        row_labels=["GT", "k-Wave\nR1", "IC\nR1", "L2", "L1"],
        save_path=_FIG_DIR / "5.2_retune_grid.svg",
    )

    # Combined metrics: ti60qmx3 rank#1-5 + hqt6bwmp rank#1 + L2 + L1
    def _build_results(self_data, ic_data, self_run, ic_run):
        entries: list[tuple[str, list[dict]]] = []
        for r in range(1, 6):
            try:
                lbl_r, _ = _pick_rank(self_data["recons"], r)
                entries.append(
                    _build_results_entry(f"k-Wave R{r}", self_data, lbl_r, self_run)
                )
            except KeyError:
                pass
        try:
            lbl_ic, _ = _pick_rank(ic_data["recons"], 1)
            entries.append(_build_results_entry("IC R1", ic_data, lbl_ic, ic_run))
        except KeyError:
            pass
        for baseline in ("L2", "L1"):
            if baseline in self_data["recons"]:
                entries.append(
                    _build_results_entry(baseline, self_data, baseline, self_run)
                )
        return dict(entries)

    geom_res = _build_results(self_g, ic_g, dirs["ti60qmx3_geom_recon"],
                              dirs["hqt6bwmp_geom_recon"])
    blob_res = _build_results(self_b, ic_b, dirs["ti60qmx3_blob_recon"],
                              dirs["hqt6bwmp_blob_recon"])
    # Method order: k-Wave R1-5, IC R1, L2, L1 → short labels: 1..5, IC, L2, L1
    short = ["1", "2", "3", "4", "5", "IC", "L2", "L1"]
    short = short[:len(geom_res)]                 # trim if a rank missing
    _draw_combined_metrics(
        geom_res, blob_res,
        save_path=_FIG_DIR / "5.2_retune_metrics.svg",
        short_labels=short,
    )
    print("  [J4] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J5 — Blind measurement regularisation: metrics box plots only
# ──────────────────────────────────────────────────────────────────────────────

def figure_J5() -> None:
    """5.3_blind_metrics.svg — combined GeomSet | BlobSet box-plot.

    4 panels (MAE/RMSE/SSIM/CNR); each panel split by a dashed line.
    Methods (boxes per side): Standalone INR R1 | Meas. Reg. R1 | L2 | L1.
    Short labels on x-axis: "Std", "MR", "L2", "L1".
    """
    print("\n[J5] Building combined blind-regularisation metrics ...")
    dirs = _require_sources(
        "ti60qmx3_geom_recon", "ti60qmx3_geom_denoised",
        "ti60qmx3_blob_recon", "ti60qmx3_blob_denoised",
    )

    def _build(recon_key, dn_key):
        recon_data = _load_npz(dirs[recon_key], recon_key)
        dn_data    = _load_npz(dirs[dn_key],    dn_key)
        entries: list[tuple[str, list[dict]]] = []
        try:
            lbl_std, _ = _pick_rank(recon_data["recons"], 1)
            entries.append(
                _build_results_entry("Standalone INR", recon_data, lbl_std,
                                     dirs[recon_key])
            )
        except KeyError:
            pass
        try:
            lbl_dn, _ = _pick_rank(dn_data["recons"], 1)
            entries.append(
                _build_results_entry("Meas. Reg.", dn_data, lbl_dn,
                                     dirs[dn_key])
            )
        except KeyError:
            pass
        for baseline in ("L2", "L1"):
            if baseline in recon_data["recons"]:
                entries.append(
                    _build_results_entry(baseline, recon_data, baseline,
                                         dirs[recon_key])
                )
            elif baseline in dn_data["recons"]:
                entries.append(
                    _build_results_entry(baseline, dn_data, baseline,
                                         dirs[dn_key])
                )
        return dict(entries)

    geom_res = _build("ti60qmx3_geom_recon", "ti60qmx3_geom_denoised")
    blob_res = _build("ti60qmx3_blob_recon", "ti60qmx3_blob_denoised")
    _draw_combined_metrics(
        geom_res, blob_res,
        save_path=_FIG_DIR / "5.3_blind_metrics.svg",
        short_labels=["Std", "MR", "L2", "L1"],
        group_labels=("GeomSet", "BlobSet"),
    )
    print("  [J5] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J5g — Blind measurement regularisation: reconstruction grid
# ──────────────────────────────────────────────────────────────────────────────

def figure_J5g() -> None:
    """5.3_blind_grid.svg — side-by-side GeomSet | BlobSet recon grid for the
    measurement-regularised (MR) rung. Companion to J5's metrics box-plot.

    Rows: GT | Standalone | Meas. Reg. | L2 | L1.
    Cols: all geom samples + dashed separator + all blob samples.
    Shared diverging colormap centred on background sound speed.
    """
    print("\n[J5g] Building MR reconstruction grid figure ...")
    dirs = _require_sources(
        "ti60qmx3_geom_recon",    "ti60qmx3_blob_recon",
        "ti60qmx3_geom_denoised", "ti60qmx3_blob_denoised",
    )

    g_recon = _load_npz(dirs["ti60qmx3_geom_recon"],     "ti60qmx3_geom_recon")
    b_recon = _load_npz(dirs["ti60qmx3_blob_recon"],     "ti60qmx3_blob_recon")
    g_mr    = _load_npz(dirs["ti60qmx3_geom_denoised"],  "ti60qmx3_geom_denoised")
    b_mr    = _load_npz(dirs["ti60qmx3_blob_denoised"],  "ti60qmx3_blob_denoised")

    try:
        _, arr_gr = _pick_rank(g_recon["recons"], 1)
        _, arr_br = _pick_rank(b_recon["recons"], 1)
        _, arr_gm = _pick_rank(g_mr["recons"],    1)
        _, arr_bm = _pick_rank(b_mr["recons"],    1)
    except KeyError as exc:
        print(f"  [J5g] rank#1 missing: {exc}")
        return

    def _panel(title, data_gt, arr_std, arr_mr):
        n = len(data_gt["gt"])
        gt_imgs = [_to_sos_img(data_gt["gt"][i]) for i in range(n)]
        def _baseline_row(key):
            if key in data_gt["recons"]:
                arr = data_gt["recons"][key]
                return [_to_sos_img(arr[i]) for i in range(n)]
            return [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        return (
            title,
            _col_headers(n),
            [
                gt_imgs,
                [_to_sos_img(arr_std[i]) for i in range(n)],
                [_to_sos_img(arr_mr[i])  for i in range(n)],
                _baseline_row("L2"),
                _baseline_row("L1"),
            ],
        )

    panels = [
        _panel("GeomSet", g_recon, arr_gr, arr_gm),
        _panel("BlobSet", b_recon, arr_br, arr_bm),
    ]
    _draw_combined_dataset_grid(
        panels,
        row_labels=["GT", "Standalone", "Meas.\nReg.", "L2", "L1"],
        save_path=_FIG_DIR / "5.3_blind_grid.svg",
    )
    print("  [J5g] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J5b — Staged joint grid (geom + blob combined)
# ──────────────────────────────────────────────────────────────────────────────

def figure_J5b() -> None:
    """5.3_staged_grid.svg — side-by-side GeomSet | BlobSet, one row dim.

    Rows: GT | Standalone | Staged Joint.  Cols: all geom samples + dashed
    separator + all blob samples.  Single shared colorbar.
    """
    print("\n[J5b] Building staged joint grid figure ...")
    dirs = _require_sources(
        "ti60qmx3_geom_recon", "ti60qmx3_blob_recon",
        "z7bs7iy5_geom_honest", "z7bs7iy5_blob_honest",
    )

    g_recon = _load_npz(dirs["ti60qmx3_geom_recon"],  "ti60qmx3_geom_recon")
    b_recon = _load_npz(dirs["ti60qmx3_blob_recon"],  "ti60qmx3_blob_recon")
    g_joint = _load_npz(dirs["z7bs7iy5_geom_honest"], "z7bs7iy5_geom_honest")
    b_joint = _load_npz(dirs["z7bs7iy5_blob_honest"], "z7bs7iy5_blob_honest")

    try:
        _, arr_gr = _pick_rank(g_recon["recons"], 1)
        _, arr_br = _pick_rank(b_recon["recons"], 1)
        _, arr_gj = _pick_rank(g_joint["recons"], 1)
        _, arr_bj = _pick_rank(b_joint["recons"], 1)
    except KeyError as exc:
        print(f"  [J5b] rank#1 missing: {exc}")
        return

    def _panel(title, data_gt, arr_std, arr_jt):
        n = len(data_gt["gt"])
        gt_imgs = [_to_sos_img(data_gt["gt"][i]) for i in range(n)]
        def _baseline_row(key):
            if key in data_gt["recons"]:
                arr = data_gt["recons"][key]
                return [_to_sos_img(arr[i]) for i in range(n)]
            return [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        return (
            title,
            _col_headers(n),
            [
                gt_imgs,
                [_to_sos_img(arr_std[i]) for i in range(n)],
                [_to_sos_img(arr_jt[i])  for i in range(n)],
                _baseline_row("L2"),
                _baseline_row("L1"),
            ],
        )

    panels = [
        _panel("GeomSet", g_recon, arr_gr, arr_gj),
        _panel("BlobSet", b_recon, arr_br, arr_bj),
    ]
    _draw_combined_dataset_grid(
        panels,
        row_labels=["GT", "Standalone", "Staged\nJoint", "L2", "L1"],
        save_path=_FIG_DIR / "5.3_staged_grid.svg",
        # Fig 5.12: dotted inclusion outline on the Staged-Joint row,
        # GeomSet sample III (col 2) and BlobSet sample IV (col 3).
        highlight_cells=[(0, 2, 2), (1, 2, 3)],
    )
    print("  [J5b] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J6 — Ranking strategy comparison: grid + MAE-vs-CNR scatter
# ──────────────────────────────────────────────────────────────────────────────

def figure_J6() -> None:
    """5.3_ranking_grid.svg (6×6) + 5.3_mae_cnr_scatter.svg.

    Grid rows (6): GT | ydma-MAE-R1 | ydma-CNR-R1 | q2p4-R1 | edj3-R1 | z7bs-R1
    Grid cols (6): 3 geom + 3 blob  (column headers G-I…B-III)
    Scatter     : x=mean MAE (m/s), y=mean CNR; 10 points (5 rows × geom/blob);
                  geom=circle, blob=square; per-point text labels;
                  faded reference markers for standalone INR + L1 + L2.
    """
    print("\n[J6] Building ranking strategy figures ...")
    p    = _get_p()
    dirs = _require_sources(
        "ydma0yxl_geom_joint",   "ydma0yxl_blob_joint",    # CNR-selected
        "ydma0yxl_geom_meanmae", "ydma0yxl_blob_meanmae",  # mean-MAE-selected
        "q2p4zv3e_geom_joint",   "q2p4zv3e_blob_joint",
        "edj3mqou_geom_joint",   "edj3mqou_blob_joint",
        "z7bs7iy5_geom_honest",  "z7bs7iy5_blob_honest",
        "ti60qmx3_geom_recon",   "ti60qmx3_blob_recon",
    )

    # CNR-selected replay  (labels: cnr1_..., cnr2_...)
    ydma_cnr_g = _load_npz(dirs["ydma0yxl_geom_joint"],   "ydma0yxl_geom_joint")
    ydma_cnr_b = _load_npz(dirs["ydma0yxl_blob_joint"],   "ydma0yxl_blob_joint")
    # mean-MAE-selected replay  (labels: rank1_..., rank2_...)
    ydma_mae_g = _load_npz(dirs["ydma0yxl_geom_meanmae"], "ydma0yxl_geom_meanmae")
    ydma_mae_b = _load_npz(dirs["ydma0yxl_blob_meanmae"], "ydma0yxl_blob_meanmae")
    q2p4_g = _load_npz(dirs["q2p4zv3e_geom_joint"],  "q2p4zv3e_geom_joint")
    q2p4_b = _load_npz(dirs["q2p4zv3e_blob_joint"],  "q2p4zv3e_blob_joint")
    edj3_g = _load_npz(dirs["edj3mqou_geom_joint"],  "edj3mqou_geom_joint")
    edj3_b = _load_npz(dirs["edj3mqou_blob_joint"],  "edj3mqou_blob_joint")
    z7bs_g = _load_npz(dirs["z7bs7iy5_geom_honest"], "z7bs7iy5_geom_honest")
    z7bs_b = _load_npz(dirs["z7bs7iy5_blob_honest"], "z7bs7iy5_blob_honest")
    std_g  = _load_npz(dirs["ti60qmx3_geom_recon"],  "ti60qmx3_geom_recon")
    std_b  = _load_npz(dirs["ti60qmx3_blob_recon"],  "ti60qmx3_blob_recon")

    # ── Build grid ────────────────────────────────────────────────────────
    # New layout (peer-review F6): two stacked panels (GeomSet, BlobSet), each
    # with 3 rows (GT, flat-winner ydma mean-MAE rank-1, inclusion-aware
    # z7bs7iy5 rank-1) over all dataset samples. Drops the 5-row objective
    # comparison from the grid — the scatter carries that.
    def _pick(d, prefixes):
        m = _first_match_by_prefixes(d["recons"], prefixes)
        if m is None:
            raise KeyError(f"no match for prefixes {prefixes}")
        return m[1]

    try:
        arr_g_flat = _pick(ydma_mae_g, ["rank"])
        arr_b_flat = _pick(ydma_mae_b, ["rank"])
        arr_g_inc  = _pick(z7bs_g,     ["cnr", "rank"])
        arr_b_inc  = _pick(z7bs_b,     ["cnr", "rank"])
    except KeyError as exc:
        print(f"  [J6/grid] required row missing: {exc}")
        arr_g_flat = arr_b_flat = arr_g_inc = arr_b_inc = None

    if arr_g_flat is not None:
        def _panel(title, data_gt, arr_flat, arr_inc, baseline_source):
            n = len(data_gt["gt"])
            gt_imgs = [_to_sos_img(data_gt["gt"][i]) for i in range(n)]
            def _baseline_row(key):
                if key in baseline_source["recons"]:
                    arr = baseline_source["recons"][key]
                    m = min(n, len(arr))
                    row = [_to_sos_img(arr[i]) for i in range(m)]
                    while len(row) < n:
                        row.append(np.full_like(gt_imgs[0], np.nan))
                    return row
                return [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
            return (
                title,
                _col_headers(n),
                [
                    gt_imgs,
                    [_to_sos_img(arr_flat[i]) for i in range(n)],
                    [_to_sos_img(arr_inc[i])  for i in range(n)],
                    _baseline_row("L2"),
                    _baseline_row("L1"),
                ],
            )

        panels = [
            _panel("GeomSet", ydma_mae_g, arr_g_flat, arr_g_inc, std_g),
            _panel("BlobSet", ydma_mae_b, arr_b_flat, arr_b_inc, std_b),
        ]
        _draw_combined_dataset_grid(
            panels,
            row_labels=["GT", "Flat\nwinner", "Inclusion\naware", "L2", "L1"],
            save_path=_FIG_DIR / "5.3_ranking_grid.svg",
            # Fig 5.13: dotted inclusion outline on the Inclusion-aware row,
            # GeomSet sample III (col 2) and BlobSet sample IV (col 3).
            highlight_cells=[(0, 2, 2), (1, 2, 3)],
        )

    # The scatter still consumes all 5 objectives × 2 datasets; build a small
    # spec list to feed it.
    row_defs = [
        ("ydma MAE-R1", ydma_mae_g, ydma_mae_b, ["rank"],        ["rank"]),
        ("ydma CNR-R1", ydma_cnr_g, ydma_cnr_b, ["cnr"],         ["cnr"]),
        ("q2p4 R1",     q2p4_g,     q2p4_b,     ["cnr", "rank"], ["cnr", "rank"]),
        ("edj3 R1",     edj3_g,     edj3_b,     ["cnr", "rank"], ["cnr", "rank"]),
        ("z7bs R1",     z7bs_g,     z7bs_b,     ["cnr", "rank"], ["cnr", "rank"]),
    ]
    scatter_rows: list[tuple[str, np.ndarray, np.ndarray, dict, dict]] = []
    for row_label, gd, bd, gp, bp in row_defs:
        g_match = _first_match_by_prefixes(gd["recons"], gp)
        b_match = _first_match_by_prefixes(bd["recons"], bp)
        if g_match is None or b_match is None:
            print(f"  [J6/scatter] missing data for '{row_label}' — skipping")
            continue
        scatter_rows.append((row_label, g_match[1], b_match[1], gd, bd))

    # ── Scatter: mean MAE vs mean CNR ─────────────────────────────────────
    def _mean_mae_cnr(npz_data: dict, arr: np.ndarray) -> tuple[float, float]:
        maes, cnrs = [], []
        for s_gt, s_rec in zip(npz_data["gt"], arr):
            gt_img  = _to_sos_img(s_gt)
            rec_img = _to_sos_img(s_rec)
            mae, cnr = _mae_cnr(rec_img, gt_img)
            maes.append(mae)
            if not np.isnan(cnr):
                cnrs.append(cnr)
        return (
            float(np.mean(maes)) if maes else float("nan"),
            float(np.mean(cnrs)) if cnrs else float("nan"),
        )

    scatter_pts: list[dict] = []
    for row_label, arr_g, arr_b, gd, bd in scatter_rows:
        clean_label = row_label.replace("\n", " ")
        mae_g, cnr_g = _mean_mae_cnr(gd, arr_g)
        mae_b, cnr_b = _mean_mae_cnr(bd, arr_b)
        scatter_pts.append({"label": clean_label, "dataset": "geom", "MAE": mae_g, "CNR": cnr_g})
        scatter_pts.append({"label": clean_label, "dataset": "blob", "MAE": mae_b, "CNR": cnr_b})

    # Reference points: standalone INR, L1, L2
    ref_pts: list[dict] = []
    for ref_lbl_suffix, ref_npz, ds_tag in [
        ("std",  std_g, "geom"),
        ("std",  std_b, "blob"),
        ("L2",   std_g, "geom"),
        ("L2",   std_b, "blob"),
        ("L1",   std_g, "geom"),
        ("L1",   std_b, "blob"),
    ]:
        if ref_lbl_suffix in ("L1", "L2"):
            if ref_lbl_suffix not in ref_npz["recons"]:
                continue
            arr_ref = ref_npz["recons"][ref_lbl_suffix]
        else:
            try:
                _, arr_ref = _pick_rank(ref_npz["recons"], 1)
            except KeyError:
                continue
        mae_r, cnr_r = _mean_mae_cnr(ref_npz, arr_ref)
        ref_pts.append({
            "label": f"{ref_lbl_suffix} ({ds_tag})",
            "dataset": ds_tag,
            "MAE": mae_r,
            "CNR": cnr_r,
        })

    # Sweep → descriptive label (matching the §5.3.3 ranking-table names that
    # appear in the report text), marker style.
    # Color carries the DATASET (blue=geom, orange=blob); marker shape is a
    # secondary cue (o=geom, ^=blob) except the flat winner which uses ★.
    SHORT = {
        "ydma MAE-R1": "flat winner",          # mean-MAE-selected replay of mean-error sweep
        "ydma CNR-R1": "mean-error",           # CNR-selected replay of same sweep
        "q2p4 R1":     "ROI-composite",
        "edj3 R1":     "pure-ROI",
        "z7bs R1":     "full inclusion-aware",
    }
    DS_COLOR  = {"geom": "#1f77b4", "blob": "#d95f02"}
    DS_MARKER = {"geom": "o",        "blob": "^"}
    FLAT_MARKER = "*"   # used for ydma MAE-R1 in both datasets

    def _marker_for(sp):
        return FLAT_MARKER if sp["label"] == "ydma MAE-R1" else DS_MARKER[sp["dataset"]]

    def _short_for(lbl):
        return SHORT.get(lbl, lbl)

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        fig.patch.set_facecolor("white")

        # Reference points: faded grey, small markers.
        for rp in ref_pts:
            marker = DS_MARKER[rp["dataset"]]
            ax.scatter(
                rp["MAE"], rp["CNR"],
                marker=marker, s=38, alpha=0.45,
                color="lightgrey", linewidths=0.6,
                edgecolors="dimgrey", zorder=2,
            )
            ax.annotate(
                rp["label"], xy=(rp["MAE"], rp["CNR"]),
                fontsize=7, color="dimgrey",
                xytext=(5, 5), textcoords="offset points",
            )

        # Build geom↔blob pair index for connectors + per-method midpoint labels.
        by_method: dict[str, dict[str, tuple[float, float]]] = {}
        for sp in scatter_pts:
            by_method.setdefault(sp["label"], {})[sp["dataset"]] = (sp["MAE"], sp["CNR"])

        # Faint NEUTRAL connector linking each method's geom↔blob pair.
        for lbl, pts in by_method.items():
            if "geom" in pts and "blob" in pts:
                gx, gy = pts["geom"]
                bx, by = pts["blob"]
                ax.plot([gx, bx], [gy, by],
                        color="#666666", alpha=0.30, linewidth=0.7,
                        linestyle=(0, (3, 2)), zorder=2)

        # Method points — DOT-sized markers coloured by dataset; star for the
        # flat-winner. Color + shape carry the dataset; labels stay short.
        for sp in scatter_pts:
            marker = _marker_for(sp)
            color  = DS_COLOR[sp["dataset"]]
            size   = 140 if marker == FLAT_MARKER else 55
            ax.scatter(
                sp["MAE"], sp["CNR"],
                marker=marker, s=size, alpha=0.95,
                color=color, linewidths=0.7, edgecolors="black", zorder=4,
            )

        # ONE label per method placed at the midpoint of its geom↔blob pair,
        # with a clear leader line. Per-method radial offsets (data coords)
        # spread the labels around the cluster so they don't overlap.
        # Approximate cluster spans: geom MAE ≈3.5, blob MAE ≈6.0; midpoint ≈4.7.
        LABEL_OFFSETS = {
            "flat winner":          (-1.30, +0.50),   # upper-left
            "mean-error":           (+1.00, -0.50),   # lower-right
            "ROI-composite":        (-1.40, -0.50),   # lower-left
            "pure-ROI":             (+0.90, +0.50),   # upper-right
            "full inclusion-aware": (+0.10, +1.00),   # straight up
        }
        for lbl, pts in by_method.items():
            short = _short_for(lbl)
            if "geom" in pts and "blob" in pts:
                gx, gy = pts["geom"]
                bx, by = pts["blob"]
                mx, my = (gx + bx) / 2.0, (gy + by) / 2.0
            elif "geom" in pts:
                mx, my = pts["geom"]
            else:
                mx, my = pts["blob"]
            dx, dy = LABEL_OFFSETS.get(short, (0.6, 0.4))
            ax.annotate(
                short,
                xy=(mx, my),
                xytext=(mx + dx, my + dy),
                textcoords="data",
                fontsize=9, fontweight="bold",
                ha="center", va="center",
                color="black",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="black", lw=0.4, alpha=0.85),
                arrowprops=dict(arrowstyle="-", linewidth=0.5,
                                color="black", alpha=0.55,
                                shrinkA=2, shrinkB=2),
                zorder=6,
            )

        # Reference gridlines (MAE=4, 6; CNR=1, 2).
        for x_ref in (4.0, 6.0):
            ax.axvline(x_ref, color="#cccccc", linewidth=0.7,
                       linestyle=":", alpha=0.7, zorder=1)
        for y_ref in (1.0, 2.0):
            ax.axhline(y_ref, color="#cccccc", linewidth=0.7,
                       linestyle=":", alpha=0.7, zorder=1)

        # Legend OUTSIDE the plot.
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color=DS_COLOR["geom"], linestyle="None",
                   markersize=9, markeredgecolor="black", label="GeomSet"),
            Line2D([0], [0], marker="^", color=DS_COLOR["blob"], linestyle="None",
                   markersize=9, markeredgecolor="black", label="BlobSet"),
            Line2D([0], [0], marker=FLAT_MARKER, color="grey", linestyle="None",
                   markersize=11, markeredgecolor="black",
                   label="flat winner (mean-MAE\nreplay of mean-error sweep)"),
            Line2D([0], [0], marker="s", color="lightgrey", linestyle="None",
                   markersize=8, markeredgecolor="dimgrey",
                   label="reference (L1/L2/standalone)"),
        ]
        ax.legend(
            handles=legend_handles,
            fontsize=8, frameon=False,
            loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            handlelength=1.2, handletextpad=0.5,
            labelspacing=1.0,
        )

        ax.set_xlabel("Mean MAE (m/s)", fontsize=11)
        ax.set_ylabel("Mean CNR", fontsize=11)
        ax.set_title("MAE vs CNR — ranking strategy comparison",
                     fontsize=11, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, direction="in")
        fig.subplots_adjust(right=0.70, left=0.10,
                            top=0.93, bottom=0.10)

        p.save(fig, _FIG_DIR / "5.3_mae_cnr_scatter.svg", png_fallback=False)
        plt.close(fig)

    print("  [J6] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J7 — Phantom and breast dataset grids
# ──────────────────────────────────────────────────────────────────────────────

def figure_J7() -> None:
    """§5.4 real-data figures — splits into THREE files:

      * ``5.4_realdata_grid.svg``     — PhantomSet grid (GT + 5 methods × 4 samples).
      * ``5.4_realdata_breast.svg``   — single-row qualitative BreastSet figure
                                        (no GT row, no metric annotations, one sample
                                        across methods).
      * ``5.4_realdata_metrics.svg``  — PhantomSet-only 4-metric box-plot
                                        (BreastSet metrics dropped: no in-vivo GT,
                                        and the 2-sample box plot was meaningless).

    Comment-B fixes folded in:
      * Req 1 — Staged-Joint sample order in ``z7bs7iy5`` differs from the
        ``ti60qmx3`` (dataset) order; per-method permutations against the GT
        re-align the SJ (and the MR, as a sanity check) row to dataset order.
      * Req 2 — breast GT is a supervisor analytical proxy, not a real in-vivo
        truth; the two stored slots are bit-identical (= one sample). Breast
        becomes a qualitative single-row figure with no GT.
      * Req 3 — no breast metrics panel.
    """
    print("\n[J7] Building §5.4 real-data figures ...")
    dirs = _require_sources(
        "ti60qmx3_phantom_recon",    "ti60qmx3_breast_recon",
        "z7bs7iy5_phantom_honest",   "z7bs7iy5_breast_honest",
        "ti60qmx3_phantom_denoised", "ti60qmx3_breast_denoised",
    )

    def _gt_aligned_indices(src_data, ref_gt_imgs):
        """Recover the indices into ``src_data`` that match dataset/GT order.

        ``src_data["gt"]`` and ``ref_gt_imgs`` are the same physical GTs in
        possibly-different orders. Returns the permutation (list of int) that
        re-orders ``src_data`` to match ``ref_gt_imgs``.
        """
        return _match_permutation(src_data["gt"], ref_gt_imgs)

    # ── PhantomSet grid (Req 1: every row re-aligned to dataset order) ────
    def _build_phantom_panel():
        recon = _load_npz(dirs["ti60qmx3_phantom_recon"],    "ti60qmx3_phantom_recon")
        dn    = _load_npz(dirs["ti60qmx3_phantom_denoised"], "ti60qmx3_phantom_denoised")
        jt    = _load_npz(dirs["z7bs7iy5_phantom_honest"],   "z7bs7iy5_phantom_honest")

        # ANCHOR: the source .mat file's `imgs_gt` order — that is the
        # "natural" dataset order (sample I, II, ... in the figure column
        # headers). Every pipeline run may shuffle samples on save; we use
        # the .mat as the canonical reference and re-align all rows.
        canonical = _load_dataset_canonical_gt("test_PhantomData.mat")
        if canonical is not None:
            n_cols = min(4, len(canonical))
            ref_gt = canonical[:n_cols]
            recon_perm = _match_permutation(recon["gt"], ref_gt)
            print(f"  [J7/PhantomSet] recon -> dataset perm: {recon_perm}")
        else:
            # Fallback when the .mat is not reachable (off-server runs):
            # the recon's own gt order becomes the reference.
            n_samples = len(recon["gt"])
            n_cols    = min(4, n_samples)
            recon_perm = _col_indices(n_samples, n_cols)
            ref_gt    = [recon["gt"][i] for i in recon_perm]

        # Permutations of MR and SJ against the canonical reference.
        dn_perm = _match_permutation(dn["gt"], ref_gt)
        jt_perm = _match_permutation(jt["gt"], ref_gt)
        print(f"  [J7/PhantomSet] MR    -> dataset perm: {dn_perm}")
        print(f"  [J7/PhantomSet] SJ    -> dataset perm: {jt_perm}")

        gt_imgs = [_to_sos_img(recon["gt"][i]) for i in recon_perm]
        has_gt = not np.all(recon["gt"] == 0.0)
        if not has_gt:
            print("  [J7/PhantomSet] WARNING: GT absent (placeholder zeros).")

        def _row(arr, idx_list):
            return [_to_sos_img(arr[i]) for i in idx_list]

        try:
            _, arr_r = _pick_rank(recon["recons"], 1)
            row_std = _row(arr_r, recon_perm)
        except KeyError:
            row_std = [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        try:
            _, arr_dn = _pick_rank(dn["recons"], 1)
            row_dn = _row(arr_dn, dn_perm)
        except KeyError:
            row_dn = [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        try:
            _, arr_jt = _pick_rank(jt["recons"], 1)
            row_jt = _row(arr_jt, jt_perm)
        except KeyError:
            row_jt = [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs]
        row_L2 = (_row(recon["recons"]["L2"], recon_perm)
                  if "L2" in recon["recons"]
                  else [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs])
        row_L1 = (_row(recon["recons"]["L1"], recon_perm)
                  if "L1" in recon["recons"]
                  else [np.full_like(gt_imgs[0], np.nan) for _ in gt_imgs])

        return (
            "PhantomSet",
            _col_headers(n_cols),
            [gt_imgs, row_std, row_dn, row_jt, row_L2, row_L1],
            has_gt,
        )

    p_title, p_cols, p_rows, p_has_gt = _build_phantom_panel()
    _draw_combined_dataset_grid(
        [(p_title, p_cols, p_rows)],
        row_labels=["GT", "Standalone", "Meas.\nReg.",
                    "Staged\nJoint", "L2", "L1"],
        save_path=_FIG_DIR / "5.4_realdata_grid.svg",
        annotate_mae_cnr=p_has_gt,
        # Fig 5.15: dotted inclusion circle on the Staged-Joint reconstruction
        # at column I (panel 0, row 3, col 0). Mask derived from that column's GT.
        highlight_cells=[(0, 3, 0)],
    )

    # ── BreastSet qualitative — Req 2: no GT row, one sample, no metrics ──
    breast_recon = _load_npz(dirs["ti60qmx3_breast_recon"],    "ti60qmx3_breast_recon")
    breast_dn    = _load_npz(dirs["ti60qmx3_breast_denoised"], "ti60qmx3_breast_denoised")
    breast_jt    = _load_npz(dirs["z7bs7iy5_breast_honest"],   "z7bs7iy5_breast_honest")

    breast_imgs: list[np.ndarray] = []
    breast_lbls: list[str] = []
    sample_i = 0   # the two stored breast slots are bit-identical (= 1 sample)
    try:
        _, arr_r = _pick_rank(breast_recon["recons"], 1)
        breast_imgs.append(_to_sos_img(arr_r[sample_i]))
        breast_lbls.append("Standalone")
    except (KeyError, IndexError):
        pass
    try:
        _, arr_dn = _pick_rank(breast_dn["recons"], 1)
        breast_imgs.append(_to_sos_img(arr_dn[sample_i]))
        breast_lbls.append("Meas. Reg.")
    except (KeyError, IndexError):
        pass
    try:
        _, arr_jt = _pick_rank(breast_jt["recons"], 1)
        breast_imgs.append(_to_sos_img(arr_jt[sample_i]))
        breast_lbls.append("Staged Joint")
    except (KeyError, IndexError):
        pass
    for baseline in ("L2", "L1"):
        if baseline in breast_recon["recons"]:
            breast_imgs.append(_to_sos_img(breast_recon["recons"][baseline][sample_i]))
            breast_lbls.append(baseline)

    if breast_imgs:
        # Fig 5.17: dotted inclusion circle on the Staged-Joint column of the
        # qualitative breast figure. With no GT the mask is derived from the
        # reconstruction shown in that column.
        sj_col = next(
            (i for i, lbl in enumerate(breast_lbls) if lbl == "Staged Joint"),
            None,
        )
        # Fig 5.17: brute-force-placed dotted circle on the Staged-Joint cell.
        # Center horizontal: image middle (col ~32 for 64×64).
        # Center vertical:   a little up from the bottom (row ~42 for 64×64).
        # Radius:            ~half the distance from the center to the top.
        _draw_qualitative_row(
            breast_imgs, breast_lbls,
            save_path=_FIG_DIR / "5.4_realdata_breast.svg",
            title="BreastSet",
            highlight_idx=sj_col,
            highlight_center=(32.0, 42.0),
            highlight_radius=20.0,
        )
    else:
        print("  [J7/BreastSet] no reconstructions available; skipping qualitative figure.")

    # ── PhantomSet metrics (Req 3: breast panel dropped) ──────────────────
    def _build_metrics(recon_key, dn_key, jt_key):
        recon_data = _load_npz(dirs[recon_key], recon_key)
        dn_data    = _load_npz(dirs[dn_key],    dn_key)
        jt_data    = _load_npz(dirs[jt_key],    jt_key)
        entries: list[tuple[str, list[dict]]] = []
        try:
            lbl_std, _ = _pick_rank(recon_data["recons"], 1)
            entries.append(_build_results_entry("Standalone INR", recon_data,
                                                lbl_std, dirs[recon_key]))
        except KeyError:
            pass
        try:
            lbl_dn, _ = _pick_rank(dn_data["recons"], 1)
            entries.append(_build_results_entry("Meas. Reg.", dn_data,
                                                lbl_dn, dirs[dn_key]))
        except KeyError:
            pass
        try:
            lbl_jt, _ = _pick_rank(jt_data["recons"], 1)
            entries.append(_build_results_entry("Staged Joint", jt_data,
                                                lbl_jt, dirs[jt_key]))
        except KeyError:
            pass
        for baseline in ("L2", "L1"):
            if baseline in recon_data["recons"]:
                entries.append(_build_results_entry(baseline, recon_data,
                                                    baseline, dirs[recon_key]))
        return dict(entries)

    phantom_res = _build_metrics("ti60qmx3_phantom_recon",
                                 "ti60qmx3_phantom_denoised",
                                 "z7bs7iy5_phantom_honest")
    _draw_combined_metrics(
        phantom_res, None,
        save_path=_FIG_DIR / "5.4_realdata_metrics.svg",
        short_labels=["Std", "MR", "SJ", "L2", "L1"],
        group_labels=("PhantomSet", ""),
    )
    print("  [J7] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────

def figure_5_1_grid() -> None:
    """5.1_grid.svg — combined IC | Oracle reconstruction grid on IC data.

    Rows : GT | R1 | R2 | R3 | L2 | L1  (same on both panels; L1/L2 share)
    Left  panel header : "IC"      (self-supervised, uses L)
    Right panel header : "Oracle"  (direct supervision, no L)
    Single shared diverging colormap + colorbar.
    """
    print("\n[F5.1/grid] Building combined IC | Oracle grid ...")
    dirs = _require_sources("hqt6bwmp_ic_self", "hqt6bwmp_ic_oracle")
    self_data   = _load_npz(dirs["hqt6bwmp_ic_self"],   "hqt6bwmp_ic_self")
    oracle_data = _load_npz(dirs["hqt6bwmp_ic_oracle"], "hqt6bwmp_ic_oracle")

    def _panel(title, data):
        n = len(data["gt"])
        col_labels = _col_headers(n)
        gt_imgs = [_to_sos_img(data["gt"][i]) for i in range(n)]
        rows_imgs = [gt_imgs]
        for r in range(1, 4):
            try:
                _, arr = _pick_rank(data["recons"], r)
                rows_imgs.append([_to_sos_img(arr[i]) for i in range(n)])
            except KeyError:
                rows_imgs.append([np.full_like(gt_imgs[0], np.nan)
                                  for _ in gt_imgs])
        for baseline in ("L2", "L1"):
            if baseline in data["recons"]:
                arr = data["recons"][baseline]
                rows_imgs.append([_to_sos_img(arr[i]) for i in range(n)])
            else:
                rows_imgs.append([np.full_like(gt_imgs[0], np.nan)
                                  for _ in gt_imgs])
        return (title, col_labels, rows_imgs)

    panels = [
        _panel("IC",     self_data),
        _panel("Oracle", oracle_data),
    ]
    _draw_combined_dataset_grid(
        panels,
        row_labels=["GT", "1", "2", "3", "L2", "L1"],
        save_path=_FIG_DIR / "5.1_grid.svg",
    )
    print("  [F5.1/grid] done.")


def figure_5_1_metrics() -> None:
    """5.1_metrics.svg — combined IC | Oracle metrics box plots."""
    print("\n[F5.1/metrics] Building combined IC | Oracle metrics ...")
    dirs = _require_sources("hqt6bwmp_ic_self", "hqt6bwmp_ic_oracle")
    self_data   = _load_npz(dirs["hqt6bwmp_ic_self"],   "hqt6bwmp_ic_self")
    oracle_data = _load_npz(dirs["hqt6bwmp_ic_oracle"], "hqt6bwmp_ic_oracle")

    def _build(data, run_dir):
        entries: list[tuple[str, list[dict]]] = []
        for r in range(1, 4):
            try:
                lbl, _ = _pick_rank(data["recons"], r)
                entries.append(_build_results_entry(f"R{r}", data, lbl, run_dir))
            except KeyError:
                pass
        for baseline in ("L2", "L1"):
            if baseline in data["recons"]:
                entries.append(
                    _build_results_entry(baseline, data, baseline, run_dir)
                )
        return dict(entries)

    ic_res     = _build(self_data,   dirs["hqt6bwmp_ic_self"])
    oracle_res = _build(oracle_data, dirs["hqt6bwmp_ic_oracle"])
    _draw_combined_metrics(
        ic_res, oracle_res,
        save_path=_FIG_DIR / "5.1_metrics.svg",
        group_labels=("IC", "Oracle"),
    )
    print("  [F5.1/metrics] done.")


_FIGURE_MAP: dict[str, Any] = {
    "F5.1g": figure_5_1_grid,
    "F5.1m": figure_5_1_metrics,
    "J3":  figure_J3,
    "J4":  figure_J4,
    "J5":  figure_J5,
    "J5g": figure_J5g,
    "J5b": figure_J5b,
    "J6":  figure_J6,
    "J7":  figure_J7,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Chapter 5 thesis figures from pre-generated recons.npz artifacts.\n\n"
            "Fill in the SOURCES dict at the top of this script with the actual run-dir\n"
            "paths, then run:  python scripts/make_ch5_figures.py --figure all"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--figure",
        nargs="+",
        default=["all"],
        metavar="FIG",
        help=(
            f"Which figure(s) to build. One or more of: "
            f"{', '.join(_FIGURE_MAP)}, or 'all'. "
            "Multiple values accepted: --figure J3 J4"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Override output directory (default: thesis_reports/.../figs/).",
    )
    args = parser.parse_args()

    # Validate --figure values
    valid = set(_FIGURE_MAP.keys()) | {"all"}
    bad = [f for f in args.figure if f not in valid]
    if bad:
        parser.error(
            f"Unknown figure(s): {bad}. "
            f"Choose from: {sorted(valid)}"
        )

    global _FIG_DIR
    if args.outdir:
        _FIG_DIR = Path(args.outdir).resolve()

    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {_FIG_DIR}")

    figures = list(_FIGURE_MAP.keys()) if "all" in args.figure else args.figure

    for fig_name in figures:
        fn = _FIGURE_MAP[fig_name]
        try:
            fn()
        except SystemExit:
            raise  # propagate _require_sources / _load_npz aborts
        except Exception:
            import traceback
            print(
                f"\n[ERROR] Figure {fig_name} raised an exception:\n"
                + traceback.format_exc(),
                file=sys.stderr,
            )
            print("  Continuing with remaining figures ...\n")

    print("\nAll requested figures complete.")


if __name__ == "__main__":
    main()
