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

        p.save(fig, save_path, png_fallback=True)
        plt.close(fig)

    print(f"  [grid] saved -> {save_path.with_suffix('.svg')}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: select column indices evenly from a dataset
# ──────────────────────────────────────────────────────────────────────────────

def _col_indices(n_total: int, n_want: int) -> list[int]:
    """Return n_want indices evenly spread over [0, n_total)."""
    n = min(n_want, n_total)
    return np.linspace(0, n_total - 1, n, dtype=int).tolist()


# ──────────────────────────────────────────────────────────────────────────────
# Figure J3 — IC-trained transfer to blob: 8×8 grid + metrics box
# ──────────────────────────────────────────────────────────────────────────────

def figure_J3() -> None:
    """5.2_v1_ictrans_blob_comparison.svg + 5.2_v1_ictrans_blob_metrics.svg

    Grid  : 8 rows × 8 cols — GT, rank#1..rank#5, L2, L1; 8 samples
    Box   : MAE/RMSE/SSIM/CNR — 7 methods (rank#1-5, L2, L1)
    Source: hqt6bwmp on kwave_blob (IC-tuned INR applied to blob)
    """
    print("\n[J3] Building IC-transfer blob figures ...")
    p    = _get_p()
    dirs = _require_sources("hqt6bwmp_blob_recon")
    run_dir = dirs["hqt6bwmp_blob_recon"]
    data = _load_npz(run_dir, "hqt6bwmp_blob_recon")

    n_cols  = min(8, len(data["gt"]))
    c_idx   = _col_indices(len(data["gt"]), n_cols)
    gt_imgs = [_to_sos_img(data["gt"][i]) for i in c_idx]
    norm    = _shared_norm(gt_imgs)

    # Build rows: GT, rank#1..5, L2, L1
    row_specs: list[tuple[str, list[np.ndarray]]] = [("GT", gt_imgs)]
    metrics_entries: list[tuple[str, list[dict]]] = []

    for r in range(1, 6):
        try:
            lbl, arr = _pick_rank(data["recons"], r)
        except KeyError:
            print(f"  [J3] rank#{r} not found — skipping")
            continue
        row_specs.append((f"R{r}", [_to_sos_img(arr[i]) for i in c_idx]))
        metrics_entries.append(
            _build_results_entry(f"R{r}", data, lbl, run_dir)
        )

    for baseline in ("L2", "L1"):
        if baseline in data["recons"]:
            arr = data["recons"][baseline]
            row_specs.append((baseline, [_to_sos_img(arr[i]) for i in c_idx]))
            metrics_entries.append(
                _build_results_entry(baseline, data, baseline, run_dir)
            )

    _draw_recon_grid(
        row_specs, _col_headers(n_cols),
        save_path=_FIG_DIR / "5.2_v1_ictrans_blob_comparison.svg",
        shared_norm=norm,
        gt_images=gt_imgs,
        dataset_title="BlobSet",
    )

    results_dict = dict(metrics_entries)
    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        p.plot_metrics_comparison(
            results=results_dict,
            save_path=_FIG_DIR / "5.2_v1_ictrans_blob_metrics.svg",
            show=False,
            png_fallback=True,
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
    print("\n[J4] Building re-tune transfer figures ...")
    p    = _get_p()
    dirs = _require_sources(
        "ti60qmx3_geom_recon", "hqt6bwmp_geom_recon",
        "ti60qmx3_blob_recon", "hqt6bwmp_blob_recon",
    )

    for ds_tag, self_key, ic_key in [
        ("geom", "ti60qmx3_geom_recon", "hqt6bwmp_geom_recon"),
        ("blob", "ti60qmx3_blob_recon", "hqt6bwmp_blob_recon"),
    ]:
        print(f"  [J4/{ds_tag}] loading ...")
        self_data = _load_npz(dirs[self_key], self_key)
        ic_data   = _load_npz(dirs[ic_key],  ic_key)

        n_cols  = min(8, len(self_data["gt"]))
        c_idx   = _col_indices(len(self_data["gt"]), n_cols)
        gt_imgs = [_to_sos_img(self_data["gt"][i]) for i in c_idx]
        norm    = _shared_norm(gt_imgs)

        row_specs: list[tuple[str, list[np.ndarray]]] = [("GT", gt_imgs)]
        metrics_entries: list[tuple[str, list[dict]]] = []

        # blob/geom-tuned (self): rank#1
        try:
            lbl_s, arr_s = _pick_rank(self_data["recons"], 1)
            row_specs.append(("ti60 R1", [_to_sos_img(arr_s[i]) for i in c_idx]))
        except KeyError:
            print(f"  [J4/{ds_tag}] ti60qmx3 rank#1 not found")
            lbl_s = None

        # IC-tuned: rank#1 (applied to this dataset)
        ic_c_idx = _col_indices(len(ic_data["gt"]), n_cols)
        try:
            lbl_ic, arr_ic = _pick_rank(ic_data["recons"], 1)
            row_specs.append(("hqt6 R1", [_to_sos_img(arr_ic[i]) for i in ic_c_idx]))
        except KeyError:
            print(f"  [J4/{ds_tag}] hqt6bwmp rank#1 not found")
            lbl_ic = None

        for baseline in ("L2", "L1"):
            if baseline in self_data["recons"]:
                arr = self_data["recons"][baseline]
                row_specs.append((baseline, [_to_sos_img(arr[i]) for i in c_idx]))

        _draw_recon_grid(
            row_specs, _col_headers(n_cols),
            save_path=_FIG_DIR / f"5.2_retune_grid_{ds_tag}.svg",
            shared_norm=norm,
            gt_images=gt_imgs,
            dataset_title=("GeomSet" if ds_tag == "geom" else "BlobSet"),
        )

        # Metrics: ti60qmx3 rank#1-5 + hqt6bwmp rank#1 + L2 + L1
        for r in range(1, 6):
            try:
                lbl_r, _ = _pick_rank(self_data["recons"], r)
                metrics_entries.append(
                    _build_results_entry(f"ti60 R{r}", self_data, lbl_r, dirs[self_key])
                )
            except KeyError:
                pass

        if lbl_ic is not None:
            metrics_entries.append(
                _build_results_entry("hqt6 R1", ic_data, lbl_ic, dirs[ic_key])
            )

        for baseline in ("L2", "L1"):
            if baseline in self_data["recons"]:
                metrics_entries.append(
                    _build_results_entry(baseline, self_data, baseline, dirs[self_key])
                )

        results_dict = dict(metrics_entries)
        with matplotlib.rc_context(p.SERIF_RCPARAMS):
            p.plot_metrics_comparison(
                results=results_dict,
                save_path=_FIG_DIR / f"5.2_retune_metrics_{ds_tag}.svg",
                show=False,
                png_fallback=True,
            )
        print(f"  [J4/{ds_tag}] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J5 — Blind measurement regularisation: metrics box plots only
# ──────────────────────────────────────────────────────────────────────────────

def figure_J5() -> None:
    """5.3_blind_metrics_{geom,blob}.svg — box-plot only (no grid).

    4 panels (MAE/RMSE/SSIM/CNR), 4 boxes:
      standalone INR rank-1  |  measurement-regularised rank-1  |  L2  |  L1
    """
    print("\n[J5] Building blind regularisation metrics figures ...")
    p    = _get_p()
    dirs = _require_sources(
        "ti60qmx3_geom_recon",    "ti60qmx3_geom_denoised",
        "ti60qmx3_blob_recon",    "ti60qmx3_blob_denoised",
    )

    for ds_tag, recon_key, dn_key in [
        ("geom", "ti60qmx3_geom_recon",  "ti60qmx3_geom_denoised"),
        ("blob", "ti60qmx3_blob_recon",   "ti60qmx3_blob_denoised"),
    ]:
        print(f"  [J5/{ds_tag}] loading ...")
        recon_data = _load_npz(dirs[recon_key], recon_key)
        dn_data    = _load_npz(dirs[dn_key],    dn_key)

        metrics_entries: list[tuple[str, list[dict]]] = []

        # Standalone INR rank#1
        try:
            lbl_std, _ = _pick_rank(recon_data["recons"], 1)
            metrics_entries.append(
                _build_results_entry("Standalone INR", recon_data, lbl_std, dirs[recon_key])
            )
        except KeyError:
            print(f"  [J5/{ds_tag}] standalone rank#1 not found")

        # Measurement-regularised rank#1
        try:
            lbl_dn, _ = _pick_rank(dn_data["recons"], 1)
            metrics_entries.append(
                _build_results_entry("Meas. Reg.", dn_data, lbl_dn, dirs[dn_key])
            )
        except KeyError:
            print(f"  [J5/{ds_tag}] denoised rank#1 not found")

        # Baselines (prefer standalone source; fall back to denoised)
        for baseline in ("L2", "L1"):
            if baseline in recon_data["recons"]:
                metrics_entries.append(
                    _build_results_entry(baseline, recon_data, baseline, dirs[recon_key])
                )
            elif baseline in dn_data["recons"]:
                metrics_entries.append(
                    _build_results_entry(baseline, dn_data, baseline, dirs[dn_key])
                )

        results_dict = dict(metrics_entries)
        with matplotlib.rc_context(p.SERIF_RCPARAMS):
            p.plot_metrics_comparison(
                results=results_dict,
                save_path=_FIG_DIR / f"5.3_blind_metrics_{ds_tag}.svg",
                show=False,
                png_fallback=True,
            )
        print(f"  [J5/{ds_tag}] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J5b — Staged joint grid (geom + blob combined)
# ──────────────────────────────────────────────────────────────────────────────

def figure_J5b() -> None:
    """5.3_staged_grid.svg — 3 rows × 6 cols.

    Rows : GT  |  standalone INR (ti60qmx3 rank#1)  |  staged joint (z7bs7iy5 honest rank#1)
    Cols : 3 geom samples + 3 blob samples (column headers show G-I/B-I etc.)
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

    n_geom = min(3, len(g_recon["gt"]))
    n_blob = min(3, len(b_recon["gt"]))
    g_c    = _col_indices(len(g_recon["gt"]), n_geom)
    b_c    = _col_indices(len(b_recon["gt"]), n_blob)
    gj_c   = _col_indices(len(g_joint["gt"]), n_geom)
    bj_c   = _col_indices(len(b_joint["gt"]), n_blob)

    gt_g_imgs = [_to_sos_img(g_recon["gt"][i]) for i in g_c]
    gt_b_imgs = [_to_sos_img(b_recon["gt"][i]) for i in b_c]
    gt_imgs   = gt_g_imgs + gt_b_imgs
    norm      = _shared_norm(gt_imgs)

    try:
        lbl_gr, arr_gr = _pick_rank(g_recon["recons"], 1)
        lbl_br, arr_br = _pick_rank(b_recon["recons"], 1)
    except KeyError as exc:
        print(f"  [J5b] standalone rank#1 missing: {exc}")
        return

    try:
        lbl_gj, arr_gj = _pick_rank(g_joint["recons"], 1)
        lbl_bj, arr_bj = _pick_rank(b_joint["recons"], 1)
    except KeyError as exc:
        print(f"  [J5b] joint rank#1 missing: {exc}")
        return

    row_standalone = (
        [_to_sos_img(arr_gr[i]) for i in g_c]
        + [_to_sos_img(arr_br[i]) for i in b_c]
    )
    row_joint = (
        [_to_sos_img(arr_gj[i]) for i in gj_c]
        + [_to_sos_img(arr_bj[i]) for i in bj_c]
    )

    col_labels = (
        [f"G-{_ROMAN[i]}" for i in range(n_geom)]
        + [f"B-{_ROMAN[i]}" for i in range(n_blob)]
    )

    _draw_recon_grid(
        [
            ("GT",            gt_imgs),
            ("Standalone",    row_standalone),
            ("Staged\nJoint", row_joint),
        ],
        col_labels,
        save_path=_FIG_DIR / "5.3_staged_grid.svg",
        shared_norm=norm,
        gt_images=gt_imgs,
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
    # Use the CNR-selected ydma npz as the GT source (same dataset, any run will do)
    n_g = 3
    n_b = 3
    g_c = _col_indices(len(ydma_cnr_g["gt"]), n_g)
    b_c = _col_indices(len(ydma_cnr_b["gt"]), n_b)

    gt_g_imgs = [_to_sos_img(ydma_cnr_g["gt"][i]) for i in g_c]
    gt_b_imgs = [_to_sos_img(ydma_cnr_b["gt"][i]) for i in b_c]
    gt_imgs   = gt_g_imgs + gt_b_imgs
    norm      = _shared_norm(gt_imgs)

    col_labels = (
        [f"G-{_ROMAN[i]}" for i in range(n_g)]
        + [f"B-{_ROMAN[i]}" for i in range(n_b)]
    )

    # Row definitions: (display, geom_npz, blob_npz, geom_prefixes, blob_prefixes)
    # ydma MAE-R1: separate mean-MAE-selected run dirs; labels rank1_..., rank2_...
    # ydma CNR-R1: CNR-selected run dirs; labels cnr1_..., cnr2_...
    # q2p4/edj3/z7bs: replayed with --selection_metric cnr; labels cnr1_...
    row_defs = [
        ("ydma\nMAE-R1", ydma_mae_g, ydma_mae_b, ["rank"],        ["rank"]),
        ("ydma\nCNR-R1", ydma_cnr_g, ydma_cnr_b, ["cnr"],         ["cnr"]),
        ("q2p4 R1",      q2p4_g,     q2p4_b,     ["cnr", "rank"], ["cnr", "rank"]),
        ("edj3 R1",      edj3_g,     edj3_b,     ["cnr", "rank"], ["cnr", "rank"]),
        ("z7bs R1",      z7bs_g,     z7bs_b,     ["cnr", "rank"], ["cnr", "rank"]),
    ]

    row_specs: list[tuple[str, list[np.ndarray]]] = [("GT", gt_imgs)]

    # Store (display_label, arr_g, arr_b, gd, bd) for scatter re-use
    scatter_rows: list[tuple[str, np.ndarray, np.ndarray, dict, dict]] = []

    for row_label, gd, bd, gp, bp in row_defs:
        g_match = _first_match_by_prefixes(gd["recons"], gp)
        b_match = _first_match_by_prefixes(bd["recons"], bp)
        if g_match is None or b_match is None:
            print(f"  [J6] missing data for row '{row_label}' — skipping")
            continue
        lbl_g, arr_g = g_match
        lbl_b, arr_b = b_match

        gc_local = _col_indices(len(gd["gt"]), n_g)
        bc_local = _col_indices(len(bd["gt"]), n_b)
        imgs = (
            [_to_sos_img(arr_g[i]) for i in gc_local]
            + [_to_sos_img(arr_b[i]) for i in bc_local]
        )
        row_specs.append((row_label, imgs))
        scatter_rows.append((row_label, arr_g, arr_b, gd, bd))

    _draw_recon_grid(
        row_specs, col_labels,
        save_path=_FIG_DIR / "5.3_ranking_grid.svg",
        shared_norm=norm,
        gt_images=gt_imgs,
    )

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

    palette = p.WONG[1:]  # skip black

    with matplotlib.rc_context(p.SERIF_RCPARAMS):
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor("white")

        # Faded reference points
        for rp in ref_pts:
            marker = "o" if rp["dataset"] == "geom" else "s"
            ax.scatter(
                rp["MAE"], rp["CNR"],
                marker=marker, s=28, alpha=0.25,
                color="grey", linewidths=0.4, edgecolors="grey", zorder=2,
            )
            ax.annotate(
                rp["label"], xy=(rp["MAE"], rp["CNR"]),
                fontsize=4.5, color="grey", alpha=0.45,
                xytext=(2, 2), textcoords="offset points",
            )

        # Method points
        seen_labels: dict[str, int] = {}
        for sp in scatter_pts:
            lbl = sp["label"]
            if lbl not in seen_labels:
                seen_labels[lbl] = len(seen_labels)
            color = palette[seen_labels[lbl] % len(palette)]
            marker = "o" if sp["dataset"] == "geom" else "s"
            ax.scatter(
                sp["MAE"], sp["CNR"],
                marker=marker, s=55, alpha=0.85,
                color=color, linewidths=0.6, edgecolors="black", zorder=3,
            )
            ax.annotate(
                f"{lbl} ({sp['dataset']})",
                xy=(sp["MAE"], sp["CNR"]),
                fontsize=4.5,
                xytext=(4, 3), textcoords="offset points",
            )

        # Legend: shapes for dataset, colours for method
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        shape_handles = [
            Line2D([0], [0], marker="o", color="grey", linestyle="None",
                   markersize=5, label="GeomSet"),
            Line2D([0], [0], marker="s", color="grey", linestyle="None",
                   markersize=5, label="BlobSet"),
        ]
        color_handles = [
            Patch(
                facecolor=palette[idx % len(palette)], alpha=0.8,
                edgecolor="black", linewidth=0.4, label=lbl,
            )
            for lbl, idx in seen_labels.items()
        ]
        ax.legend(
            handles=shape_handles + color_handles,
            fontsize=6, frameon=True, framealpha=0.9,
            edgecolor="lightgrey", loc="best",
            handlelength=1.0, handletextpad=0.4, columnspacing=0.8,
        )

        ax.set_xlabel("Mean MAE (m/s)", fontsize=9)
        ax.set_ylabel("Mean CNR", fontsize=9)
        ax.set_title("MAE vs CNR — ranking strategy comparison", fontsize=9, pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7, direction="in")

        p.save(fig, _FIG_DIR / "5.3_mae_cnr_scatter.svg", png_fallback=True)
        plt.close(fig)

    print("  [J6] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Figure J7 — Phantom and breast dataset grids
# ──────────────────────────────────────────────────────────────────────────────

def figure_J7() -> None:
    """5.4_phantom_grid.svg (6 rows × 4 cols) + 5.4_breast_grid.svg (6 rows × 2 cols).

    Rows : GT  |  standalone INR  |  meas-reg  |  staged joint  |  L2  |  L1
    """
    print("\n[J7] Building phantom / breast figures ...")
    dirs = _require_sources(
        "ti60qmx3_phantom_recon",    "ti60qmx3_breast_recon",
        "z7bs7iy5_phantom_honest",   "z7bs7iy5_breast_honest",
        "ti60qmx3_phantom_denoised", "ti60qmx3_breast_denoised",
    )

    for ds_tag, n_target_cols, src in [
        ("phantom", 4, {
            "recon":    "ti60qmx3_phantom_recon",
            "denoised": "ti60qmx3_phantom_denoised",
            "joint":    "z7bs7iy5_phantom_honest",
        }),
        ("breast", 2, {
            "recon":    "ti60qmx3_breast_recon",
            "denoised": "ti60qmx3_breast_denoised",
            "joint":    "z7bs7iy5_breast_honest",
        }),
    ]:
        print(f"  [J7/{ds_tag}] loading ...")
        recon_data = _load_npz(dirs[src["recon"]],    src["recon"])
        dn_data    = _load_npz(dirs[src["denoised"]], src["denoised"])
        jt_data    = _load_npz(dirs[src["joint"]],    src["joint"])

        n_samples = len(recon_data["gt"])
        n_cols    = min(n_target_cols, n_samples)
        c_idx     = _col_indices(n_samples, n_cols)
        dn_c_idx  = _col_indices(len(dn_data["gt"]), n_cols)
        jt_c_idx  = _col_indices(len(jt_data["gt"]), n_cols)

        gt_imgs = [_to_sos_img(recon_data["gt"][i]) for i in c_idx]

        # Detect absent GT (all-zero array written as placeholder)
        has_gt = not np.all(recon_data["gt"] == 0.0)
        if not has_gt:
            print(
                f"  [J7/{ds_tag}] WARNING: GT appears absent (all-zero array). "
                "GT row will show blank — verify source."
            )

        if has_gt:
            norm = _shared_norm(gt_imgs)
        else:
            norm = mcolors.Normalize(vmin=1400.0, vmax=1600.0)

        row_specs: list[tuple[str, list[np.ndarray]]] = [("GT", gt_imgs)]

        # Standalone INR rank#1
        try:
            lbl_r, arr_r = _pick_rank(recon_data["recons"], 1)
            row_specs.append(
                ("Standalone", [_to_sos_img(arr_r[i]) for i in c_idx])
            )
        except KeyError:
            print(f"  [J7/{ds_tag}] standalone rank#1 not found")

        # Measurement-regularised rank#1
        try:
            lbl_dn, arr_dn = _pick_rank(dn_data["recons"], 1)
            row_specs.append(
                ("Meas. Reg.", [_to_sos_img(arr_dn[i]) for i in dn_c_idx])
            )
        except KeyError:
            print(f"  [J7/{ds_tag}] denoised rank#1 not found")

        # Staged joint rank#1
        try:
            lbl_jt, arr_jt = _pick_rank(jt_data["recons"], 1)
            row_specs.append(
                ("Staged\nJoint", [_to_sos_img(arr_jt[i]) for i in jt_c_idx])
            )
        except KeyError:
            print(f"  [J7/{ds_tag}] joint rank#1 not found")

        # Baselines
        for baseline in ("L2", "L1"):
            if baseline in recon_data["recons"]:
                arr = recon_data["recons"][baseline]
                row_specs.append(
                    (baseline, [_to_sos_img(arr[i]) for i in c_idx])
                )

        _draw_recon_grid(
            row_specs, _col_headers(n_cols),
            save_path=_FIG_DIR / f"5.4_{ds_tag}_grid.svg",
            shared_norm=norm,
            gt_images=gt_imgs if has_gt else None,
            annotate_mae_cnr=has_gt,
        )
        print(f"  [J7/{ds_tag}] done.")


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch
# ──────────────────────────────────────────────────────────────────────────────

_FIGURE_MAP: dict[str, Any] = {
    "J3":  figure_J3,
    "J4":  figure_J4,
    "J5":  figure_J5,
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
