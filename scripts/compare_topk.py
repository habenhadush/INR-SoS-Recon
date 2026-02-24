#!/usr/bin/env python3
"""
compare_topk.py
---------------
Runs ALL top-K INR configs from the sweep on fresh disjoint samples and
compares against L1 / L2 baselines.

Outputs:
  1. Terminal + log: full comparison table with reconstruction time
  2. W&B per-config runs: per-sample metrics streamed live
  3. W&B summary run:
       - comparison_table  (sortable)
       - boxplots          (2×2 metrics + 1 full-width time panel)
       - reconstruction_grid (GT vs all methods on 2 random samples)
  4. PDF box plots saved locally for thesis
  5. Registry updated

Usage:
    # 2-sample bug check (~8 min)
    python scripts/compare_topk.py --sweep_id hqt6bwmp --n_samples 2

    # overnight job
    python scripts/compare_topk.py --sweep_id hqt6bwmp --n_samples 20
"""

import argparse
import copy
import io
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import wandb

import inr_sos
from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics

SCRIPTS_DIR   = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
LOG_DIR       = SCRIPTS_DIR / "logs"
PLOT_DIR      = SCRIPTS_DIR / "plots"
LOG_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

SOS_BG  = 1540.0
SOS_MIN = 1380.0
SOS_MAX = 1620.0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_run_tag(description: str, sweep_id: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{ts}_{description}_{sweep_id[:8]}"


def _fig_to_wandb_image(fig, caption: str = "") -> wandb.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return wandb.Image(Image.open(buf), caption=caption)


def _short_label(method_str: str) -> str:
    """x-axis tick label — rank only for INR, short name for baselines."""
    if method_str == "L2_regularization":
        return "L2"
    if method_str == "L1_regularization":
        return "L1"
    parts = method_str.replace("INR_", "").split("_")
    return parts[0]   # rank1, rank2, rank4 ...


def _legend_label(method_str: str) -> str:
    """Full descriptive label used only in the figure legend."""
    if method_str == "L2_regularization":
        return "L2 regularization"
    if method_str == "L1_regularization":
        return "L1 regularization"
    parts = method_str.replace("INR_", "").split("_")
    rank  = parts[0]
    model = parts[-1]
    meth  = "_".join(parts[1:-1])
    return f"{rank}  {meth} / {model}"


def setup_logging(run_tag: str):
    log_path = LOG_DIR / f"{run_tag}.log"
    logger   = logging.getLogger("compare_topk")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh  = logging.FileHandler(log_path)
        sh  = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return log_path, logger


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(sweep_id: str):
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    entry = next((e for e in registry if e["sweep_id"].startswith(sweep_id)), None)
    if entry is None:
        raise ValueError(f"Sweep {sweep_id} not found in registry.")
    return registry, entry


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_used_indices(entry: dict) -> set:
    used = set(entry.get("indices", []))
    used |= set(entry.get("validation", {}).get("holdout_indices", []))
    for key in entry:
        if key.startswith("comparison") or key.startswith("topk"):
            used |= set(entry[key].get("indices", []))
    return used


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(label, mae, rmse, ssim, cnr, recon_times, per_sample) -> dict:
    return {
        "method":          label,
        "MAE_mean":        float(np.mean(mae)),
        "MAE_std":         float(np.std(mae)),
        "RMSE_mean":       float(np.mean(rmse)),
        "RMSE_std":        float(np.std(rmse)),
        "SSIM_mean":       float(np.mean(ssim)),
        "SSIM_std":        float(np.std(ssim)),
        "CNR_mean":        float(np.mean(cnr)),
        "CNR_std":         float(np.std(cnr)),
        "recon_time_mean": float(np.mean(recon_times)) if recon_times else None,
        "recon_time_std":  float(np.std(recon_times))  if recon_times else None,
        "n_samples":       len(mae),
        "per_sample":      per_sample,
    }


def _log_agg(log, r):
    t = (f"  ReconTime={r['recon_time_mean']:.0f}±{r['recon_time_std']:.0f}s"
         if r["recon_time_mean"] is not None else "  precomputed")
    log.info(f"  {r['method']:<42} "
             f"MAE={r['MAE_mean']:.3f}±{r['MAE_std']:.3f}  "
             f"RMSE={r['RMSE_mean']:.3f}±{r['RMSE_std']:.3f}  "
             f"SSIM={r['SSIM_mean']:.3f}±{r['SSIM_std']:.3f}  "
             f"CNR={r['CNR_mean']:.3f}±{r['CNR_std']:.3f}{t}")


def compute_baseline_metrics(recons, gt, indices, label, log) -> dict:
    """
    recons / gt: (64, 64, N) MATLAB axis order.
    Stores s_phys_np and s_gt_np in per_sample for reconstruction grid.
    """
    mae, rmse, ssim, cnr, per_sample = [], [], [], [], []
    for idx in indices:
        # MATLAB stores (x, z, N) column-major → transpose to (z, x) to match
        # the INR coordinate convention from np.meshgrid(x_sos, z_sos)
        s_pred = np.asarray(recons[:, :, idx].T.flatten(), dtype=np.float32)
        s_gt_v = np.asarray(gt[:, :, idx].T.flatten(),    dtype=np.float32)
        m = calculate_metrics(s_phys_pred=s_pred, s_gt_raw=s_gt_v,
                              grid_shape=(64, 64))
        mae.append(m["MAE"]); rmse.append(m["RMSE"])
        ssim.append(m["SSIM"]); cnr.append(m["CNR"])
        per_sample.append({
            "idx":       idx,
            "s_phys_np": s_pred,    # store for reconstruction grid
            "s_gt_np":   s_gt_v,
            **m
        })
    result = _aggregate(label, mae, rmse, ssim, cnr, [], per_sample)
    _log_agg(log, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# INR runner
# ─────────────────────────────────────────────────────────────────────────────

def run_inr_config(sweep_cfg, dataset, indices, base_config, run_tag, log) -> dict:
    from inr_sos.models.mlp import FourierMLP, ReluMLP
    from inr_sos.models.siren import SirenMLP
    from inr_sos.training.engines import (
        optimize_full_forward_operator,
        optimize_sequential_views,
        optimize_stochastic_ray_batching,
    )

    engine_map = {
        "Full_Matrix":    optimize_full_forward_operator,
        "Sequential_SGD": optimize_sequential_views,
        "Ray_Batching":   optimize_stochastic_ray_batching,
    }
    model_map = {"FourierMLP": FourierMLP, "ReluMLP": ReluMLP, "SirenMLP": SirenMLP}

    rank    = sweep_cfg["rank"]
    method  = sweep_cfg["method"]
    mtype   = sweep_cfg["model_type"]
    hparams = sweep_cfg["hyperparams"]
    label   = f"INR_rank{rank}_{method}_{mtype}"
    wb_name = f"{run_tag}_rank{rank}_{method}_{mtype}"

    log.info(f"\n  ── rank#{rank}: {method} / {mtype} ──")
    log.info(f"  W&B run: {wb_name}")

    cfg = copy.deepcopy(base_config)
    cfg.model_type      = mtype
    cfg.hidden_features = hparams.get("hidden_features", cfg.hidden_features)
    cfg.hidden_layers   = hparams.get("hidden_layers",   cfg.hidden_layers)
    cfg.lr              = hparams.get("lr",              cfg.lr)
    cfg.steps           = hparams.get("steps",           cfg.steps)
    cfg.mapping_size    = hparams.get("mapping_size",    cfg.mapping_size)
    cfg.scale           = hparams.get("scale",           cfg.scale)
    cfg.omega           = hparams.get("omega",           cfg.omega)
    cfg.tv_weight       = hparams.get("tv_weight",       cfg.tv_weight)
    cfg.reg_weight      = hparams.get("reg_weight",      cfg.reg_weight)
    if method == "Ray_Batching":
        cfg.epochs     = hparams.get("epochs",     cfg.epochs)
        cfg.batch_size = hparams.get("batch_size", cfg.batch_size)

    engine_fn = engine_map[method]
    model_cls = model_map[mtype]
    mae, rmse, ssim, cnr, recon_times, per_sample = [], [], [], [], [], []

    wandb.init(
        project=base_config.project_name,
        name=wb_name,
        group=f"{run_tag}_topk",
        tags=["topk_comparison", f"rank{rank}", method, mtype, f"n{len(indices)}"],
        notes=(f"rank#{rank} | {method}/{mtype} | steps={cfg.steps} | "
               f"lr={cfg.lr:.2e} | tv={cfg.tv_weight:.2e} | n={len(indices)}"),
        config={"rank": rank, "method": method, "model_type": mtype,
                "n_samples": len(indices), **hparams},
        reinit=True,
    )

    for i, idx in enumerate(indices):
        log.info(f"    [{i+1:>3}/{len(indices)}]  idx={idx}")
        sample = dataset[idx]

        kwargs = dict(in_features=cfg.in_features,
                      hidden_features=cfg.hidden_features,
                      hidden_layers=cfg.hidden_layers,
                      mapping_size=cfg.mapping_size)
        if mtype == "FourierMLP": kwargs["scale"] = cfg.scale
        elif mtype == "SirenMLP": kwargs["omega"]  = cfg.omega
        model = model_cls(**kwargs)

        t0 = time.perf_counter()
        result = engine_fn(sample=sample, L_matrix=dataset.L_matrix,
                           model=model, label=mtype, config=cfg, use_wandb=False)
        recon_s = time.perf_counter() - t0
        recon_times.append(recon_s)

        s_phys_np = (result["s_phys"].detach().cpu().numpy()
                     if hasattr(result["s_phys"], "detach")
                     else np.asarray(result["s_phys"]))
        s_gt_np   = (sample["s_gt_raw"].detach().cpu().numpy()
                     if hasattr(sample["s_gt_raw"], "detach")
                     else np.asarray(sample["s_gt_raw"]))

        m = calculate_metrics(s_phys_pred=s_phys_np,
                              s_gt_raw=s_gt_np, grid_shape=(64, 64))
        mae.append(m["MAE"]); rmse.append(m["RMSE"])
        ssim.append(m["SSIM"]); cnr.append(m["CNR"])
        per_sample.append({
            "idx":         idx,
            "recon_time_s": recon_s,
            "s_phys_np":   s_phys_np.flatten().astype(np.float32),
            "s_gt_np":     s_gt_np.flatten().astype(np.float32),
            **m
        })

        log.info(f"           MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
                 f"SSIM={m['SSIM']:.3f}  CNR={m['CNR']:.3f}  "
                 f"time={recon_s:.0f}s")

        wandb.log({
            "sample/MAE":           m["MAE"],
            "sample/RMSE":          m["RMSE"],
            "sample/SSIM":          m["SSIM"],
            "sample/CNR":           m["CNR"],
            "sample/recon_time_s":  recon_s,
            "running/MAE_mean":     float(np.mean(mae)),
            "running/RMSE_mean":    float(np.mean(rmse)),
            "running/SSIM_mean":    float(np.mean(ssim)),
            "running/recon_time_s": float(np.mean(recon_times)),
        }, step=i)

    agg = _aggregate(label, mae, rmse, ssim, cnr, recon_times, per_sample)
    wandb.log({
        "final/MAE_mean":        agg["MAE_mean"],
        "final/RMSE_mean":       agg["RMSE_mean"],
        "final/SSIM_mean":       agg["SSIM_mean"],
        "final/CNR_mean":        agg["CNR_mean"],
        "final/recon_time_mean": agg["recon_time_mean"],
        "final/recon_time_std":  agg["recon_time_std"],
    })
    wandb.finish()
    _log_agg(log, agg)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_results, log):
    n = all_results[0]["n_samples"]
    log.info(f"\n{'═'*108}")
    log.info(f"  FINAL TOP-K COMPARISON — {n} fresh samples  (sorted by RMSE)")
    log.info(f"{'═'*108}")
    log.info(f"  {'Method':<42} {'MAE±std':>13} {'RMSE±std':>13} "
             f"{'SSIM±std':>13} {'CNR±std':>11} {'ReconTime(s)':>14}")
    log.info(f"  {'─'*106}")
    for r in sorted(all_results, key=lambda x: x["RMSE_mean"]):
        t = (f"{r['recon_time_mean']:>6.0f}±{r['recon_time_std']:<5.0f}"
             if r["recon_time_mean"] is not None else "    precomputed")
        log.info(
            f"  {r['method']:<42}"
            f"  {r['MAE_mean']:>5.3f}±{r['MAE_std']:<5.3f}"
            f"  {r['RMSE_mean']:>5.3f}±{r['RMSE_std']:<5.3f}"
            f"  {r['SSIM_mean']:>5.3f}±{r['SSIM_std']:<5.3f}"
            f"  {r['CNR_mean']:>5.3f}±{r['CNR_std']:<5.3f}"
            f"  {t}"
        )
    log.info(f"{'═'*108}")


# ─────────────────────────────────────────────────────────────────────────────
# Box plots — redesigned layout
#   Row 1: MAE | RMSE
#   Row 2: SSIM | CNR
#   Row 3: Reconstruction Time (full width)
# ─────────────────────────────────────────────────────────────────────────────

def make_boxplots(all_results: list) -> plt.Figure:
    """
    Adaptive distribution plot.
      n < 5  → strip plot: individual dots + mean line (test runs with 2 samples)
      n >= 5 → proper box plot with whiskers and outliers
    Layout: Row 1: MAE | RMSE
            Row 2: SSIM | CNR
            Row 3: Reconstruction Time (full width)
    """
    # Order: baselines then INR sorted by rank
    baselines = [r for r in all_results if "INR" not in r["method"]]
    inr_ranks = sorted(
        [r for r in all_results if "INR" in r["method"]],
        key=lambda x: int(x["method"].split("rank")[1].split("_")[0])
    )
    ordered = baselines + inr_ranks
    labels  = [_short_label(r["method"]) for r in ordered]
    n       = all_results[0]["n_samples"]
    use_box = n >= 5

    # Colors: grey for baselines, blue gradient for INR
    colors = ["#aaaaaa", "#888888"] + [
        plt.cm.Blues(0.35 + 0.13 * i) for i in range(len(inr_ranks))
    ]
    # Darker versions for strip plot dots
    dot_colors = ["#666666", "#444444"] + [
        plt.cm.Blues(0.55 + 0.12 * i) for i in range(len(inr_ranks))
    ]

    fig = plt.figure(figsize=(14, 13))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.52, wspace=0.35,
                            height_ratios=[1, 1, 0.85])

    panels = [
        (gs[0, 0], "MAE",          "MAE (m/s)",               False),
        (gs[0, 1], "RMSE",         "RMSE (m/s)",              False),
        (gs[1, 0], "SSIM",         "SSIM",                    True),
        (gs[1, 1], "CNR",          "Contrast-to-Noise Ratio", True),
    ]

    def _draw_panel(ax, metric_key, ylabel, higher_better, show_time=False):
        values = []
        for r in ordered:
            v = [s.get(metric_key) for s in r["per_sample"]
                 if s.get(metric_key) is not None]
            values.append(v if v else [0.0])

        if use_box:
            # ── Box plot (n >= 5) ─────────────────────────────────────
            bp = ax.boxplot(
                values,
                patch_artist=True,
                notch=False,
                showfliers=True,
                medianprops=dict(color="black", linewidth=2.5),
                flierprops=dict(marker="o", markersize=4, alpha=0.5,
                                markerfacecolor="gray", linestyle="none"),
                whiskerprops=dict(linewidth=1.4),
                capprops=dict(linewidth=1.4),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
        else:
            # ── Strip plot (n < 5) — dots + mean line ─────────────────
            # Each method gets jittered x positions so dots don't overlap
            rng_jitter = np.random.default_rng(seed=0)
            for col_i, (vals, color, dcolor) in enumerate(
                    zip(values, colors, dot_colors), start=1):
                jitter = rng_jitter.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(col_i + jitter, vals,
                           color=dcolor, s=60, zorder=3, alpha=0.85,
                           edgecolors="white", linewidths=0.5)
                # Mean line
                mean_v = np.mean(vals)
                ax.plot([col_i - 0.3, col_i + 0.3], [mean_v, mean_v],
                        color="black", linewidth=2.5, zorder=4)
                # Std range bar
                std_v = np.std(vals)
                ax.plot([col_i, col_i],
                        [mean_v - std_v, mean_v + std_v],
                        color="black", linewidth=1.2, zorder=4)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        direction = "Higher ↑" if higher_better else "Lower ↓"
        plot_type = "box plot" if use_box else f"strip plot  (n={n}, mean±std)"
        ax.set_title(f"{direction} is better   [{plot_type}]",
                     fontsize=9, color="#555555", pad=5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Shade INR region
        if len(baselines) < len(ordered):
            ax.axvspan(len(baselines) + 0.5, len(ordered) + 0.5,
                       alpha=0.06, color="#1f77b4", zorder=0)

        # (clinical threshold line removed)

    for spec, key, ylabel, higher in panels:
        ax = fig.add_subplot(spec)
        _draw_panel(ax, key, ylabel, higher)

    ax_time = fig.add_subplot(gs[2, :])
    _draw_panel(ax_time, "recon_time_s", "Reconstruction Time (s)",
                False, show_time=True)
    ax_time.set_title(
        "Reconstruction Time per Sample  |  Lower ↓ is better",
        fontsize=9, color="#555555", pad=5
    )

    fig.suptitle("Method Comparison — Per-Sample Metric Distributions",
                 fontsize=13, fontweight="bold", y=1.01)

    legend_handles = [
        mpatches.Patch(facecolor=color, edgecolor="#555555", linewidth=0.5,
                       label=_legend_label(r["method"]))
        for r, color in zip(ordered, colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.06),
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        title="Methods  (x-axis shows rank only)",
        title_fontsize=8,
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction grid
#   Layout: 4 rows × (1 + N_methods) cols
#     row 0: SoS maps for sample A     (GT | L2 | L1 | rank1 | ...)
#     row 1: Error maps for sample A
#     row 2: SoS maps for sample B
#     row 3: Error maps for sample B
# ─────────────────────────────────────────────────────────────────────────────

def make_reconstruction_grid(all_results: list,
                              n_visual_samples: int = 2,
                              rng_seed: int = 42) -> plt.Figure:
    """
    Grid: 4 rows × (1 + N_methods) columns
      row 0: SoS maps sample A  — GT | L2 | L1 | rank1 | ...
      row 1: Abs error sample A
      row 2: SoS maps sample B
      row 3: Abs error sample B

    Colormap: jet with vmin=1400, vmax=1600 m/s — matches original engine
    plot design. Blue = slow inclusion, orange/red = fast, clear contrast.
    """
    # Common indices across all methods
    common_indices = set(s["idx"] for s in all_results[0]["per_sample"])
    for r in all_results[1:]:
        common_indices &= {s["idx"] for s in r["per_sample"]}
    common_indices = sorted(common_indices)

    rng      = np.random.default_rng(seed=rng_seed)
    n_pick   = min(n_visual_samples, len(common_indices))
    vis_idxs = rng.choice(common_indices, size=n_pick, replace=False).tolist()

    n_cols = 1 + len(all_results)     # GT + one col per method
    n_rows = n_pick * 2               # SoS row + error row per sample

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.9 * n_cols, 3.0 * n_rows),
        squeeze=False
    )
    fig.suptitle("Reconstruction Grid — GT vs All Methods",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_pair, vis_idx in enumerate(vis_idxs):
        row_sos = row_pair * 2
        row_err = row_pair * 2 + 1

        # GT from first result's per_sample
        gt_entry = next(s for s in all_results[0]["per_sample"]
                        if s["idx"] == vis_idx)
        s_gt_np  = gt_entry["s_gt_np"].flatten().astype(np.float32)
        v_gt     = np.clip(1.0 / (s_gt_np + 1e-8), 1400, 1600).reshape(64, 64)
        bg_sos   = float(np.median(v_gt))

        # ── Column 0: Ground Truth ─────────────────────────────────────
        ax_gt = axes[row_sos, 0]
        im_gt = ax_gt.imshow(v_gt, cmap="jet", vmin=1400, vmax=1600,
                              interpolation="nearest", origin="upper")
        ax_gt.axis("off")
        if row_pair == 0:
            ax_gt.set_title("Ground\nTruth", fontsize=9, fontweight="bold")
        cb = plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        cb.set_label("m/s", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        # Info cell below GT
        axes[row_err, 0].axis("off")
        axes[row_err, 0].text(
            0.5, 0.5,
            f"sample idx\n{vis_idx}\n\nbg ≈ {bg_sos:.0f} m/s",
            ha="center", va="center", fontsize=8, color="#444444",
            transform=axes[row_err, 0].transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                      edgecolor="#cccccc", linewidth=0.8)
        )

        # ── Columns 1..N: each method ──────────────────────────────────
        for col, result in enumerate(all_results, start=1):
            entry   = next(s for s in result["per_sample"]
                           if s["idx"] == vis_idx)
            s_phys  = entry["s_phys_np"].flatten().astype(np.float32)
            v_rec   = np.clip(1.0 / (s_phys + 1e-8), 1400, 1600).reshape(64, 64)
            err_map = np.abs(v_gt - v_rec)
            mae_val = float(np.mean(err_map))

            # SoS reconstruction
            ax_sos = axes[row_sos, col]
            im_sos = ax_sos.imshow(v_rec, cmap="jet", vmin=1400, vmax=1600,
                                    interpolation="nearest", origin="upper")
            ax_sos.axis("off")
            if row_pair == 0:
                ax_sos.set_title(_short_label(result["method"]),
                                  fontsize=9, fontweight="bold")
            cb2 = plt.colorbar(im_sos, ax=ax_sos, fraction=0.046, pad=0.04)
            cb2.set_label("m/s", fontsize=7)
            cb2.ax.tick_params(labelsize=7)

            # Abs error
            ax_err = axes[row_err, col]
            im_err = ax_err.imshow(err_map, cmap="hot", vmin=0, vmax=50,
                                    interpolation="nearest", origin="upper")
            ax_err.axis("off")
            ax_err.set_title(f"MAE = {mae_val:.1f} m/s",
                              fontsize=8, color="#444444", pad=3)
            cb3 = plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
            cb3.set_label("m/s", fontsize=7)
            cb3.ax.tick_params(labelsize=7)

    # Row labels on left margin
    for rp in range(n_pick):
        for row_offset, txt in [(0, "SoS (m/s)"), (1, "Abs. Error")]:
            axes[rp * 2 + row_offset, 0].annotate(
                txt,
                xy=(0, 0.5), xycoords="axes fraction",
                xytext=(-0.38, 0.5), textcoords="axes fraction",
                fontsize=8, color="#333333", rotation=90, va="center",
                annotation_clip=False
            )

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# W&B summary run
# ─────────────────────────────────────────────────────────────────────────────

def log_summary_to_wandb(all_results, entry, indices, run_tag, sweep_id, log):
    wb_name = f"{run_tag}_SUMMARY"
    log.info(f"\n  Logging summary → W&B '{wb_name}'")

    wandb.init(
        project=entry["project"],
        entity=entry["entity"],
        name=wb_name,
        group=f"{run_tag}_topk",
        tags=["topk_comparison", "summary", f"n{len(indices)}"],
        notes=(f"Top-K comparison summary. sweep={sweep_id} | "
               f"n_samples={len(indices)} | indices={indices}"),
        config={"sweep_id": sweep_id, "n_samples": len(indices),
                "indices": indices},
    )

    # Sortable comparison table
    table = wandb.Table(
        columns=["Method", "MAE_mean", "MAE_std",
                 "RMSE_mean", "RMSE_std",
                 "SSIM_mean", "SSIM_std",
                 "CNR_mean",  "CNR_std",
                 "ReconTime_mean_s", "ReconTime_std_s", "n_samples"],
        data=[
            [r["method"],
             r["MAE_mean"],  r["MAE_std"],
             r["RMSE_mean"], r["RMSE_std"],
             r["SSIM_mean"], r["SSIM_std"],
             r["CNR_mean"],  r["CNR_std"],
             r["recon_time_mean"], r["recon_time_std"],
             r["n_samples"]]
            for r in sorted(all_results, key=lambda x: x["RMSE_mean"])
        ]
    )
    wandb.log({"comparison_table": table})

    # Box plots — in-memory to W&B
    log.info("  Rendering box plots ...")
    fig_bp = make_boxplots(all_results)
    wandb.log({"boxplots": _fig_to_wandb_image(fig_bp, "Metric distributions")})
    # Also save PDF for thesis
    pdf_path = PLOT_DIR / f"{run_tag}_boxplots.pdf"
    fig_bp.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bp)
    log.info(f"  Box plot PDF → {pdf_path}")

    # Reconstruction grid — 2 random samples
    log.info("  Rendering reconstruction grid ...")
    fig_grid = make_reconstruction_grid(all_results, n_visual_samples=2)
    wandb.log({"reconstruction_grid": _fig_to_wandb_image(
        fig_grid, "GT vs all methods — 2 random samples")})
    pdf_grid = PLOT_DIR / f"{run_tag}_recon_grid.pdf"
    fig_grid.savefig(pdf_grid, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    log.info(f"  Reconstruction grid PDF → {pdf_grid}")

    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id",  required=True)
    parser.add_argument("--top_k",     default=5,  type=int)
    parser.add_argument("--n_samples", default=20, type=int,
                        help="Use 2 for a quick bug check before the overnight run")
    parser.add_argument("--no_wandb",  action="store_true")
    args = parser.parse_args()

    run_tag      = make_run_tag(f"topk{args.top_k}_fresh{args.n_samples}",
                                args.sweep_id)
    log_path, log = setup_logging(run_tag)

    log.info("=" * 70)
    log.info(f"  Top-K comparison  |  {run_tag}")
    log.info(f"  Sweep    : {args.sweep_id}")
    log.info(f"  Top-K    : {args.top_k}  |  Samples: {args.n_samples}")
    est = args.top_k * args.n_samples * 20
    log.info(f"  Est. time: ~{est//60}h{est%60:02d}m  (20 min/sample estimate)")
    log.info("=" * 70)

    registry, entry = load_registry(args.sweep_id)

    # ── Fresh indices ─────────────────────────────────────────────────────
    used    = get_used_indices(entry)
    log.info(f"Previously used: {len(used)} indices — excluded")
    rng     = np.random.default_rng(seed=3141)
    pool    = [i for i in range(10000) if i not in used]
    indices = rng.choice(pool, size=args.n_samples, replace=False).tolist()
    log.info(f"Fresh indices ({len(indices)}): {indices}")

    # ── Baselines ─────────────────────────────────────────────────────────
    log.info("\nLoading analytical baselines ...")
    analytical = inr_sos.load_mat(
        DATA_DIR + "/DL-based-SoS/train_IC_10k_l2rec_l1rec_imcon.mat"
    )
    l2_result = compute_baseline_metrics(
        analytical["all_slowness_recons_l2"], analytical["imgs_gt"],
        indices, "L2_regularization", log)
    l1_result = compute_baseline_metrics(
        analytical["all_slowness_recons_l1"], analytical["imgs_gt"],
        indices, "L1_regularization", log)

    # ── Dataset ───────────────────────────────────────────────────────────
    log.info("\nLoading dataset ...")
    dataset     = USDataset(
        DATA_DIR + "/DL-based-SoS/train-VS-8pairs-IC-081225.mat",
        DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    )
    base_config = ExperimentConfig(project_name=entry["project"])

    # ── Fetch top-K configs from W&B ──────────────────────────────────────
    api   = wandb.Api()
    sweep = api.sweep(
        f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}"
    )
    runs = sorted(
        [r for r in sweep.runs if "MAE_mean" in r.summary],
        key=lambda r: r.summary["MAE_mean"]
    )[:args.top_k]

    top_k_cfgs = []
    log.info(f"\nTop-{args.top_k} sweep configs:")
    for rank, run in enumerate(runs, 1):
        cfg = {
            "rank":       rank,
            "method":     run.config.get("method"),
            "model_type": run.config.get("model_type"),
            "hyperparams": {k: v for k, v in run.config.items()
                            if k not in {"method", "model_type", "_wandb"}},
        }
        top_k_cfgs.append(cfg)
        log.info(f"  rank#{rank}: {cfg['method']:<20} / {cfg['model_type']:<12} "
                 f"sweep_MAE={run.summary['MAE_mean']:.3f}")

    # ── Run each INR config ───────────────────────────────────────────────
    inr_results = []
    t_start = time.time()
    for cfg in top_k_cfgs:
        t0     = time.time()
        result = run_inr_config(cfg, dataset, indices, base_config, run_tag, log)
        log.info(f"  rank#{cfg['rank']} wall time: {(time.time()-t0)/60:.1f} min")
        inr_results.append(result)

    total_hrs = (time.time() - t_start) / 3600
    log.info(f"\nAll configs done in {total_hrs:.2f} hours")

    # ── Final table ───────────────────────────────────────────────────────
    all_results = [l2_result, l1_result] + inr_results
    print_table(all_results, log)

    # ── W&B summary ───────────────────────────────────────────────────────
    if not args.no_wandb:
        log_summary_to_wandb(all_results, entry, indices, run_tag,
                             args.sweep_id, log)

    # ── Registry ──────────────────────────────────────────────────────────
    slim_results = [{k: v for k, v in r.items() if k != "per_sample"}
                    for r in all_results]
    for e in registry:
        if e["sweep_id"].startswith(args.sweep_id):
            e["topk_comparison"] = {
                "run_tag":     run_tag,
                "ran_at":      datetime.now().isoformat(),
                "top_k":       args.top_k,
                "n_samples":   len(indices),
                "indices":     indices,
                "elapsed_hrs": round(total_hrs, 2),
                "results":     slim_results,
            }
    save_registry(registry)
    log.info(f"\n  Registry updated  |  Log: {log_path}")


if __name__ == "__main__":
    main()