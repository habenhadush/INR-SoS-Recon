#!/usr/bin/env python3
"""
run_joint_denoiser_recon.py
---------------------------
Experiment 11: Joint denoiser + reconstructor (staged training).

Compares: Joint (staged, with adaptive λ strategies) | L1 | L2

Usage:
    # Fixed λ (default)
    python scripts/run_joint_denoiser_recon.py --n_samples 5 --lambda_fit 0.07

    # Cosine decay
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy cosine --lambda_max 1.0 --lambda_min 0.01

    # Loss-ratio balanced
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy balanced --target_ratio 1.0

    # Residual-normalized (sweep α)
    python scripts/run_joint_denoiser_recon.py --n_samples 5 \
        --lambda_strategy residual --alpha 0.1 0.5 1.0

--------------------------------------------------------------------------
SWEEP → REPLAY MAPPING (kwave_blob joint sweeps, Plan B ablation)
--------------------------------------------------------------------------
Each row lists the create/run commands that produced the sweep and the
replay command we use to evaluate its top-K on held-out samples.

[ydma0yxl] Pre-Plan-B baseline — ranked by MAE_mean (flat-winner regime)
  Create: uv run create_joint_sweep.py --dataset kwave_blob --n_runs 200
  Run:    python run_joint_sweep.py --sweep_id ydma0yxl --n_runs 200 \
            --dataset kwave_blob --n_samples 10
  Replay: python run_joint_denoiser_recon.py --dataset kwave_blob \
            --sweep_id ydma0yxl --top_k 7 --n_samples 10 \
            --selection_metric mae_roi --report_plots

[q2p4zv3e] B1-B3 only — ROI composite (5 sweep samples via yaml default)
  Create: uv run create_joint_sweep.py --dataset kwave_blob --n_runs 100 \
            --metric MAE_composite_mean
  Run:    python run_joint_sweep.py --sweep_id q2p4zv3e --n_runs 100 \
            --dataset kwave_blob --roi_weight 0.7
          # --n_samples not passed → falls back to joint_sweep.n_eval_samples
          # (= 5) in scripts/datasets.yaml, producing indices [22,0,49,4,54].
  Replay: python run_joint_denoiser_recon.py --dataset kwave_blob \
            --sweep_id q2p4zv3e --top_k 7 --n_samples 10 \
            --selection_metric mae_roi --report_plots

[edj3mqou] Pure ROI — roi_weight = 1.0 (no contrast penalty)
  Create: uv run create_joint_sweep.py --dataset kwave_blob --n_runs 100 \
            --metric MAE_composite_mean
  Run:    python run_joint_sweep.py --sweep_id edj3mqou --n_runs 100 \
            --dataset kwave_blob --n_samples 10 --roi_weight 1.0
  Replay: python run_joint_denoiser_recon.py --dataset kwave_blob \
            --sweep_id edj3mqou --top_k 7 --n_samples 10 \
            --selection_metric mae_roi --report_plots

[z7bs7iy5] Full Plan B — inclusion-aware (composite + contrast + oracle)
  Create: uv run create_joint_sweep.py --dataset kwave_blob --n_runs 150 \
            --metric MAE_composite_mean
  Run:    python run_joint_sweep.py --sweep_id z7bs7iy5 --n_runs 150 \
            --dataset kwave_blob --n_samples 10 \
            --roi_weight 0.7 --contrast_weight 1.0 --selection_metric mae_roi
  Replay: python run_joint_denoiser_recon.py --dataset kwave_blob \
            --sweep_id z7bs7iy5 --top_k 7 --n_samples 10 \
            --selection_metric mae_roi --report_plots

Replay convention: --roi_weight / --contrast_weight belong to the sweep
(they configure the ranking objective inside run_joint_sweep.py). They are
NOT replay-side flags and do nothing here. Only --selection_metric is a
replay-side knob: it sets the Stage 3c checkpoint criterion during replay
training. For a fair comparison across sweeps, use --selection_metric mae_roi
on all replays.
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.evaluation.sweep_indices import load_sweep_indices
from inr_sos.models.mlp import ReluMLP, FourierMLP, GeluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import optimize_full_forward_operator
from inr_sos.training.denoise_engine import DEFAULT_DENOISE_CFG
from inr_sos.training.joint_engine import optimize_joint

_RECON_MODEL_MAP = {
    "ReluMLP": ReluMLP,
    "FourierMLP": FourierMLP,
    "GeluMLP": GeluMLP,
    "SirenMLP": SirenMLP,
}
from inr_sos.visualization.report_figures import (
    plot_method_grid,
    plot_metrics_comparison,
)

SCRIPTS_DIR = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
OUTPUT_DIR = SCRIPTS_DIR / "data" / "joint_denoiser_recon"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("joint_exp")


def make_recon_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Joint-Denoiser-Recon",
        model_type="ReluMLP",
        hidden_features=256,
        hidden_layers=3,
        mapping_size=64,
        lr=1e-4,
        steps=2000,
        early_stopping=True,
        patience=100,
        clamp_slowness=True,
        loss_type="mse",
    )


def build_recon_model(cfg):
    return ReluMLP(
        in_features=cfg.in_features,
        hidden_features=cfg.hidden_features,
        hidden_layers=cfg.hidden_layers,
        mapping_size=cfg.mapping_size,
    )


def fetch_topk_joint_configs(sweep_id, top_k, logger,
                             selection_metric="loss", mixed_split=None):
    """Fetch top-k configs from a joint W&B sweep.

    selection_metric controls sweep-level ranking:
      - "loss"    : sort by MAE_mean ascending (legacy behaviour).
      - "mae_roi" : sort by MAE_roi_mean ascending.
      - "cnr"     : sort by CNR_mean descending.
      - "mixed"   : n by MAE_roi + m by CNR, deduped with fill-up.
                    mixed_split must be a (n, m) tuple with n+m == top_k.

    Labels are prefixed with the selection source: rank, roi, or cnr.
    """
    import wandb as wb

    # Find entry in registry
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    entry = None
    for e in registry:
        if e["sweep_id"].startswith(sweep_id):
            entry = e
            break
    if entry is None:
        raise ValueError(f"Sweep ID '{sweep_id}' not found in {REGISTRY_FILE}")

    logger.info(f"  Fetching top-{top_k} from joint sweep {entry['sweep_id']} "
                f"(selection_metric={selection_metric})")

    api = wb.Api()
    sweep = api.sweep(f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}")
    all_runs = list(sweep.runs)

    def _by_mae_roi(rs):
        have = [r for r in rs if "MAE_roi_mean" in r.summary]
        if not have:
            have = [r for r in rs if "MAE_mean" in r.summary]
            logger.warning("  MAE_roi_mean not in sweep summary — falling back to MAE_mean")
            return sorted(have, key=lambda r: r.summary["MAE_mean"])
        return sorted(have, key=lambda r: r.summary["MAE_roi_mean"])

    def _by_cnr(rs):
        have = [r for r in rs if "CNR_mean" in r.summary]
        return sorted(have, key=lambda r: -r.summary["CNR_mean"])

    def _by_mae_mean(rs):
        have = [r for r in rs if "MAE_mean" in r.summary]
        return sorted(have, key=lambda r: r.summary["MAE_mean"])

    # Build (run, source_tag) list honouring the requested selection_metric
    pairs: list[tuple] = []
    if selection_metric == "mae_roi":
        pairs = [(r, "roi") for r in _by_mae_roi(all_runs)[:top_k]]
    elif selection_metric == "cnr":
        pairs = [(r, "cnr") for r in _by_cnr(all_runs)[:top_k]]
    elif selection_metric == "mixed":
        if mixed_split is None:
            raise ValueError("--selection_metric mixed requires --mixed_split n,m")
        n, m = mixed_split
        if n + m != top_k:
            raise ValueError(f"mixed_split {n}+{m}={n+m} must equal top_k={top_k}")
        roi_ranked = _by_mae_roi(all_runs)
        cnr_ranked = _by_cnr(all_runs)
        picked_ids: set = set()
        pairs_roi: list[tuple] = []
        pairs_cnr: list[tuple] = []
        # First pass: take top-n from MAE_roi, top-m from CNR (deduped)
        for r in roi_ranked:
            if len(pairs_roi) >= n:
                break
            if r.id not in picked_ids:
                pairs_roi.append((r, "roi"))
                picked_ids.add(r.id)
        for r in cnr_ranked:
            if len(pairs_cnr) >= m:
                break
            if r.id not in picked_ids:
                pairs_cnr.append((r, "cnr"))
                picked_ids.add(r.id)
        # Fill-up pass: if dedupe shrank either list, pull next-best from each
        roi_iter = iter(roi_ranked)
        cnr_iter = iter(cnr_ranked)
        while len(pairs_roi) < n:
            try:
                r = next(roi_iter)
            except StopIteration:
                break
            if r.id not in picked_ids:
                pairs_roi.append((r, "roi"))
                picked_ids.add(r.id)
        while len(pairs_cnr) < m:
            try:
                r = next(cnr_iter)
            except StopIteration:
                break
            if r.id not in picked_ids:
                pairs_cnr.append((r, "cnr"))
                picked_ids.add(r.id)
        pairs = pairs_roi + pairs_cnr
        logger.info(f"  mixed({n},{m}): {len(pairs_roi)} from MAE_roi, "
                    f"{len(pairs_cnr)} from CNR (total {len(pairs)})")
    else:  # "loss" — legacy
        pairs = [(r, "rank") for r in _by_mae_mean(all_runs)[:top_k]]

    if not pairs:
        raise RuntimeError(f"No completed runs found in sweep {entry['sweep_id']}")

    def _g(key, default, cast=float):
        """Safely read a value from W&B config — handles param spec dicts."""
        val = sc.get(key, default)
        if isinstance(val, dict):
            val = default
        return cast(val)

    configs = []
    # Per-source counters so labels read "roi1", "roi2", "cnr1", "cnr2" or "rank1"...
    src_counter: dict = {}
    for abs_rank, (run, src) in enumerate(pairs, 1):
        sc = run.config
        src_counter[src] = src_counter.get(src, 0) + 1
        local_rank = src_counter[src]

        # Denoiser config
        dn_cfg = dict(DEFAULT_DENOISE_CFG)
        dn_cfg["model_type"] = _g("dn_model_type", dn_cfg.get("model_type", "FourierMLP"), str)
        dn_cfg["scale"] = _g("dn_scale", dn_cfg["scale"], float)
        dn_cfg["omega"] = _g("dn_omega", dn_cfg.get("omega", 15.0), float)
        dn_cfg["hidden_features"] = _g("dn_hidden_features", dn_cfg["hidden_features"], int)
        dn_cfg["hidden_layers"] = _g("dn_hidden_layers", dn_cfg["hidden_layers"], int)

        # Reconstructor overrides
        rc_model_type = _g("rc_model_type", "ReluMLP", str)
        rc_overrides = {
            "model_type": rc_model_type,
            "hidden_features": _g("rc_hidden_features", 256, int),
            "hidden_layers": _g("rc_hidden_layers", 3, int),
            "mapping_size": _g("rc_mapping_size", 64, int),
            "lr": _g("rc_lr", 1e-4, float),
            "tv_weight": _g("rc_tv_weight", 0.0, float),
            "reg_weight": _g("rc_reg_weight", 0.0, float),
        }
        if rc_model_type == "FourierMLP":
            rc_overrides["scale"] = _g("rc_scale", 10.0, float)
        elif rc_model_type == "SirenMLP":
            rc_overrides["omega"] = _g("rc_omega", 30.0, float)

        # Lambda
        lam_strategy = _g("lambda_strategy", "fixed", str)
        lam_fit = _g("lambda_fit", 0.1, float)
        lam_cfg = {}
        if lam_strategy == "cosine":
            lam_cfg["lambda_max"] = _g("lambda_max", 1.0, float)
            lam_cfg["lambda_min"] = _g("lambda_min", 0.01, float)
        elif lam_strategy == "balanced":
            lam_cfg["target_ratio"] = _g("target_ratio", 1.0, float)
            lam_cfg["lambda_min"] = _g("lambda_min", 0.001, float)
            lam_cfg["lambda_max"] = _g("lambda_max", 10.0, float)
        elif lam_strategy == "residual":
            lam_cfg["alpha"] = _g("alpha", 0.5, float)

        # Schedule
        dn_steps = _g("pretrain_dn_steps", 300, int)
        rc_steps = _g("pretrain_rc_steps", 500, int)
        j_steps = _g("joint_steps", 2000, int)
        j_lr_factor = _g("joint_lr_factor", 0.1, float)

        sweep_mae = run.summary.get("MAE_mean", float("inf"))
        sweep_mae_roi = run.summary.get("MAE_roi_mean", float("inf"))
        sweep_cnr = run.summary.get("CNR_mean", 0.0)

        short = f"{src}{local_rank}_{rc_model_type}_{lam_strategy}"
        logger.info(f"  #{abs_rank} [{src}]: {short}  "
                    f"sweep_MAE={sweep_mae:.2f}  "
                    f"MAE_roi={sweep_mae_roi:.2f}  "
                    f"CNR={sweep_cnr:.3f}")

        configs.append({
            "rank": abs_rank,
            "source": src,
            "local_rank": local_rank,
            "label": short,
            "denoise_cfg": dn_cfg,
            "rc_overrides": rc_overrides,
            "lambda_strategy": lam_strategy,
            "lambda_fit": lam_fit,
            "lambda_cfg": lam_cfg,
            "pretrain_dn_steps": dn_steps,
            "pretrain_rc_steps": rc_steps,
            "joint_steps": j_steps,
            "joint_lr_factor": j_lr_factor,
            "sweep_mae": sweep_mae,
            "sweep_mae_roi": sweep_mae_roi,
            "sweep_cnr": sweep_cnr,
            "raw_config": {k: v for k, v in sc.items() if k != "_wandb"},
        })

    return configs


def _build_recon_model_from_overrides(rc_overrides, base_cfg):
    """Build a reconstructor model from sweep overrides."""
    mtype = rc_overrides["model_type"]
    model_cls = _RECON_MODEL_MAP[mtype]
    kwargs = dict(
        in_features=base_cfg.in_features,
        hidden_features=rc_overrides["hidden_features"],
        hidden_layers=rc_overrides["hidden_layers"],
        mapping_size=rc_overrides["mapping_size"],
    )
    if mtype == "FourierMLP":
        kwargs["scale"] = rc_overrides.get("scale", 10.0)
    elif mtype == "SirenMLP":
        kwargs["omega"] = rc_overrides.get("omega", 30.0)
    return model_cls(**kwargs)


def _to_sos(s_flat):
    if hasattr(s_flat, "detach"):
        s_flat = s_flat.detach().cpu().numpy()
    s_flat = np.asarray(s_flat).flatten()
    s_clamped = np.clip(s_flat, 1.0 / 1800.0, 1.0 / 1200.0)
    return (1.0 / s_clamped).reshape(64, 64, order="F")


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_sample_comparison(sample, all_methods, idx, out_dir):
    """Multi-row comparison: GT + each method with SoS, error, convergence."""
    v_gt = _to_sos(sample["s_gt_raw"])

    n_rows = 1 + len(all_methods)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    fig.suptitle(f"Sample {idx} — Joint Denoiser+Recon Comparison", fontsize=13)

    # Row 0: GT
    im = axes[0, 0].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
    axes[0, 0].set_title("Ground Truth (m/s)", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for row, (label, res) in enumerate(all_methods, 1):
        v_rec = _to_sos(res["s_phys"])
        err = np.abs(v_gt - v_rec)
        m = res["metrics"]

        im = axes[row, 0].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
        axes[row, 0].set_title(
            f"{label}  MAE={m['MAE']:.1f}  CNR={m['CNR']:.2f}", fontsize=10
        )
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im_e = axes[row, 1].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[row, 1].set_title("Abs. Error", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im_e, ax=axes[row, 1], fraction=0.046, pad=0.04)

        loss_hist = res.get("loss_history")
        if loss_hist:
            axes[row, 2].plot(loss_hist, color="#1f77b4", linewidth=1)
            axes[row, 2].set_yscale("log")
            axes[row, 2].set_title("Convergence", fontsize=10)
            axes[row, 2].set_xlabel("Iteration", fontsize=9)
            axes[row, 2].grid(True, which="both", ls="--", alpha=0.4)
            axes[row, 2].spines["top"].set_visible(False)
            axes[row, 2].spines["right"].set_visible(False)

            # If we have separate recon/fit losses, overlay them
            if "recon_losses" in res and "fit_losses" in res:
                ax2 = axes[row, 2]
                ax2.clear()
                ax2.plot(res["recon_losses"], color="#1f77b4", linewidth=1,
                         label="L_recon", alpha=0.8)
                ax2.plot(res["fit_losses"], color="#ff7f0e", linewidth=1,
                         label="L_fit", alpha=0.8)
                ax2.set_yscale("log")
                ax2.set_title("Convergence", fontsize=10)
                ax2.set_xlabel("Iteration", fontsize=9)
                ax2.legend(fontsize=7)
                ax2.grid(True, which="both", ls="--", alpha=0.4)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)

                # Overlay λ trajectory on secondary y-axis
                lam_traj = res.get("lambda_trajectory")
                if lam_traj and len(set(lam_traj)) > 1:
                    ax_lam = ax2.twinx()
                    ax_lam.plot(lam_traj, color="#2ca02c", linewidth=1,
                                label="λ(t)", alpha=0.6, linestyle="--")
                    ax_lam.set_ylabel("λ", fontsize=8, color="#2ca02c")
                    ax_lam.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)
        else:
            axes[row, 2].axis("off")

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_comparison.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_summary_bars(all_results, out_dir):
    """Bar chart comparing all methods across samples."""
    methods = list(all_results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(methods) * 2), 6))
    fig.suptitle("Joint Denoiser+Recon — Method Comparison", fontsize=13)

    mae_means, mae_stds, cnr_means, cnr_stds = [], [], [], []
    for method in methods:
        maes = [r["metrics"]["MAE"] for r in all_results[method]]
        cnrs = [r["metrics"]["CNR"] for r in all_results[method]]
        mae_means.append(np.mean(maes))
        mae_stds.append(np.std(maes))
        cnr_means.append(np.mean(cnrs))
        cnr_stds.append(np.std(cnrs))

    x = np.arange(len(methods))
    colors = ["#999999" if m in ("L1", "L2") else "#1f77b4" for m in methods]

    ax1.bar(x, mae_means, yerr=mae_stds, color=colors, capsize=4,
            edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("MAE (m/s)")
    ax1.set_title("MAE (lower is better)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, cnr_means, yerr=cnr_stds, color=colors, capsize=4,
            edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("CNR")
    ax2.set_title("CNR (higher is better)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = out_dir / "summary_bars.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Joint denoiser+reconstructor (Exp 11)")
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--lambda_fit", nargs="+", type=float, default=[0.1])
    parser.add_argument("--mode", default="staged",
                        choices=["staged", "end_to_end", "alternating"],
                        help="Training mode: staged (default), end_to_end, or alternating")
    parser.add_argument("--joint_steps", type=int, default=2000)
    parser.add_argument("--pretrain_dn_steps", type=int, default=300)
    parser.add_argument("--pretrain_rc_steps", type=int, default=500)
    # Lambda strategy
    parser.add_argument("--lambda_strategy", default="fixed",
                        choices=["fixed", "cosine", "balanced", "residual"],
                        help="Adaptive λ strategy for Stage 3")
    parser.add_argument("--lambda_max", type=float, default=1.0,
                        help="Cosine/balanced: starting λ (cosine) or upper bound (balanced)")
    parser.add_argument("--lambda_min", type=float, default=0.01,
                        help="Cosine/balanced: ending λ (cosine) or lower bound (balanced)")
    parser.add_argument("--target_ratio", type=float, default=1.0,
                        help="Balanced: target L_recon / L_fit ratio")
    parser.add_argument("--alpha", nargs="+", type=float, default=[0.5],
                        help="Residual: scaling factor(s) for per-sample λ")
    # Denoiser config
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--report_plots",
        action="store_true",
        help="Generate thesis-quality comparison figures (SVG + PNG) after the run.",
    )
    parser.add_argument(
        "--no_exclude_sweep_samples",
        action="store_true",
        help="Do NOT exclude sweep indices from the evaluation pool "
             "(default: sweep + validation indices are excluded).",
    )
    parser.add_argument("--tag", default=None,
                        help="Optional tag to append to the result directory name.")
    parser.add_argument(
        "--sweep_id",
        default=None,
        help="Joint sweep ID. When combined with --top_k, fetches top-k configs "
             "from this sweep and runs each one. Also used for sample exclusion.",
    )
    parser.add_argument("--top_k", type=int, default=None,
                        help="Fetch top-k configs from --sweep_id and run each one. "
                             "Overrides manual lambda/denoiser/recon config.")
    parser.add_argument(
        "--selection_metric", default="loss",
        choices=["loss", "mae_roi", "cnr", "mixed"],
        help="Sweep-level ranking criterion for --top_k fetch. "
             "'loss' sorts by MAE_mean (legacy), 'mae_roi' by MAE_roi_mean asc, "
             "'cnr' by CNR_mean desc, 'mixed' combines n MAE_roi + m CNR picks "
             "(requires --mixed_split). Also controls Stage 3c checkpoint criterion: "
             "'mae_roi' (or 'cnr'/'mixed') use oracle MAE_roi selection; 'loss' uses training loss.",
    )
    parser.add_argument(
        "--mixed_split", default=None,
        help="For --selection_metric mixed: 'n,m' where n+m==top_k. "
             "n configs ranked by MAE_roi, m by CNR (deduped, filled).",
    )
    args = parser.parse_args()

    # Parse --mixed_split if present
    mixed_split = None
    if args.selection_metric == "mixed":
        if not args.mixed_split:
            parser.error("--selection_metric mixed requires --mixed_split n,m")
        try:
            n_str, m_str = args.mixed_split.split(",")
            mixed_split = (int(n_str), int(m_str))
        except ValueError:
            parser.error(f"--mixed_split must be 'n,m' (got '{args.mixed_split}')")
        if args.top_k is None or sum(mixed_split) != args.top_k:
            parser.error(f"--mixed_split {mixed_split} must sum to --top_k ({args.top_k})")

    # Stage 3c checkpoint criterion: any GT-aware sweep ranking uses mae_roi oracle
    stage3c_metric = "mae_roi" if args.selection_metric in ("mae_roi", "cnr", "mixed") else "loss"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = timestamp
    if args.sweep_id:
        folder_name += f"_{args.sweep_id}"
    if args.tag:
        folder_name += f"_{args.tag}"
    run_dir = OUTPUT_DIR / args.dataset / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)

    denoise_cfg = dict(DEFAULT_DENOISE_CFG)
    denoise_cfg.update({
        "scale": args.scale,
        "hidden_features": args.hidden_features,
    })

    log.info("=" * 70)
    log.info(f"  Experiment 11: Joint Denoiser + Reconstructor ({args.mode})")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  λ strategy: {args.lambda_strategy}")
    if args.lambda_strategy == "fixed":
        log.info(f"  lambda_fit: {args.lambda_fit}")
    elif args.lambda_strategy == "cosine":
        log.info(f"  λ_max={args.lambda_max}, λ_min={args.lambda_min}")
    elif args.lambda_strategy == "balanced":
        log.info(f"  target_ratio={args.target_ratio}, λ_init={args.lambda_fit}")
    elif args.lambda_strategy == "residual":
        log.info(f"  α={args.alpha}")
    log.info(f"  Denoiser: scale={denoise_cfg['scale']}, h={denoise_cfg['hidden_features']}")
    log.info(f"  Output: {run_dir}")
    log.info("=" * 70)

    # ── Load dataset ─────────────────────────────────────────────────────
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        ds_cfg = yaml.safe_load(f)["datasets"][args.dataset]

    data_path = DATA_DIR + ds_cfg["data_file"]
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"

    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True

    dataset = USDataset(data_path, grid_path, **ds_kwargs)
    grid = dataset.grid

    if args.indices:
        indices = args.indices
    else:
        if args.no_exclude_sweep_samples:
            sweep_used: set[int] = set()
            log.info("  Sweep-index exclusion disabled (--no_exclude_sweep_samples)")
        else:
            sweep_used = load_sweep_indices(
                dataset_key=args.dataset,
                sweep_id=args.sweep_id,
                registry_path=SCRIPTS_DIR / "sweep_registry.json",
            )
            if sweep_used:
                log.info(f"  Excluding {len(sweep_used)} sweep indices from pool")
        n = args.n_samples if args.n_samples is not None else len(dataset)
        pool = [i for i in range(len(dataset)) if i not in sweep_used]
        indices = pool[: min(n, len(pool))]
    log.info(f"  Samples: {indices}")

    recon_cfg = make_recon_config()
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time
    elif ds_cfg.get("pix2time") is not None:
        recon_cfg.time_scale = 1.0 / float(ds_cfg["pix2time"])
        log.info(f"  time_scale from yaml pix2time: {recon_cfg.time_scale:.4e}")

    samples = [dataset[idx] for idx in indices]

    # ── All results: {method_name: [per_sample_results]} ─────────────────
    # Joint methods are inserted first; L1/L2 baselines are appended at the
    # end so they render last in plots and summary tables.
    all_results = {}

    # ── Build joint configs ─────────────────────────────────────────────
    if args.top_k and args.sweep_id:
        # ── Top-K from sweep: fetch configs from W&B ────────────────────
        log.info(f"\n── Top-{args.top_k} configs from joint sweep {args.sweep_id} ──")
        topk_configs = fetch_topk_joint_configs(
            args.sweep_id, args.top_k, log,
            selection_metric=args.selection_metric,
            mixed_split=mixed_split,
        )

        for cfg_entry in topk_configs:
            method_label = cfg_entry["label"]
            log.info(f"\n── {method_label} ──")
            all_results[method_label] = []

            # Build per-config recon_cfg
            rc_ov = cfg_entry["rc_overrides"]
            cfg_rc = copy.deepcopy(recon_cfg)
            cfg_rc.model_type = rc_ov["model_type"]
            cfg_rc.hidden_features = rc_ov["hidden_features"]
            cfg_rc.hidden_layers = rc_ov["hidden_layers"]
            cfg_rc.mapping_size = rc_ov["mapping_size"]
            cfg_rc.lr = rc_ov["lr"]
            cfg_rc.tv_weight = rc_ov["tv_weight"]
            cfg_rc.reg_weight = rc_ov["reg_weight"]
            cfg_rc.steps = cfg_entry["joint_steps"]

            for i, (idx, sample) in enumerate(zip(indices, samples)):
                log.info(f"  {method_label}: sample {i+1}/{len(indices)} (idx={idx})")
                model = _build_recon_model_from_overrides(rc_ov, cfg_rc)
                t0 = time.perf_counter()
                res = optimize_joint(
                    sample=sample,
                    L_matrix=dataset.L_matrix,
                    grid=grid,
                    recon_model=model,
                    recon_config=copy.deepcopy(cfg_rc),
                    denoise_cfg=cfg_entry["denoise_cfg"],
                    lambda_fit=cfg_entry["lambda_fit"],
                    mode=args.mode,
                    lambda_strategy=cfg_entry["lambda_strategy"],
                    lambda_cfg=cfg_entry["lambda_cfg"],
                    pretrain_denoiser_steps=cfg_entry["pretrain_dn_steps"],
                    pretrain_recon_steps=cfg_entry["pretrain_rc_steps"],
                    joint_steps=cfg_entry["joint_steps"],
                    joint_lr_factor=cfg_entry["joint_lr_factor"],
                    use_wandb=False,
                    label=method_label,
                    selection_metric=stage3c_metric,
                    gt_for_selection=sample["s_gt_raw"] if stage3c_metric == "mae_roi" else None,
                )
                elapsed = time.perf_counter() - t0
                m = calculate_metrics(res["s_phys"], sample["s_gt_raw"])
                m["time_s"] = elapsed

                log.info(f"    Joint MAE={m['MAE']:.1f}")

                all_results[method_label].append({
                    "metrics": m, "s_phys": res["s_phys"],
                    "loss_history": res["loss_history"],
                    "recon_losses": res.get("recon_losses"),
                    "fit_losses": res.get("fit_losses"),
                    "lambda_trajectory": res.get("lambda_trajectory"),
                })

    else:
        # ── Manual configs from CLI args ────────────────────────────────
        joint_configs = []  # list of (method_label, lambda_fit, lambda_cfg)

        if args.lambda_strategy == "fixed":
            for lf in args.lambda_fit:
                joint_configs.append((f"Joint(fix,λ={lf})", lf, {}))

        elif args.lambda_strategy == "cosine":
            lam_cfg = {"lambda_max": args.lambda_max, "lambda_min": args.lambda_min}
            label = f"Joint(cos,{args.lambda_max}→{args.lambda_min})"
            joint_configs.append((label, args.lambda_max, lam_cfg))

        elif args.lambda_strategy == "balanced":
            for lf in args.lambda_fit:
                lam_cfg = {
                    "target_ratio": args.target_ratio,
                    "lambda_min": args.lambda_min,
                    "lambda_max": args.lambda_max,
                }
                label = f"Joint(bal,r={args.target_ratio},λ₀={lf})"
                joint_configs.append((label, lf, lam_cfg))

        elif args.lambda_strategy == "residual":
            for a in args.alpha:
                lam_cfg = {"alpha": a}
                label = f"Joint(res,α={a})"
                joint_configs.append((label, 0.1, lam_cfg))  # base λ unused for residual

        for method_label, lf, lam_cfg in joint_configs:
            log.info(f"\n── {method_label} ──")
            all_results[method_label] = []

            for i, (idx, sample) in enumerate(zip(indices, samples)):
                log.info(f"  {method_label}: sample {i+1}/{len(indices)} (idx={idx})")
                model = build_recon_model(recon_cfg)
                t0 = time.perf_counter()
                res = optimize_joint(
                    sample=sample,
                    L_matrix=dataset.L_matrix,
                    grid=grid,
                    recon_model=model,
                    recon_config=copy.deepcopy(recon_cfg),
                    denoise_cfg=denoise_cfg,
                    lambda_fit=lf,
                    mode=args.mode,
                    lambda_strategy=args.lambda_strategy,
                    lambda_cfg=lam_cfg,
                    pretrain_denoiser_steps=args.pretrain_dn_steps,
                    pretrain_recon_steps=args.pretrain_rc_steps,
                    joint_steps=args.joint_steps,
                    use_wandb=False,
                    label=method_label,
                    selection_metric=stage3c_metric,
                    gt_for_selection=sample["s_gt_raw"] if stage3c_metric == "mae_roi" else None,
                )
                elapsed = time.perf_counter() - t0
                m = calculate_metrics(res["s_phys"], sample["s_gt_raw"])
                m["time_s"] = elapsed

                log.info(f"    Joint MAE={m['MAE']:.1f}")

                all_results[method_label].append({
                    "metrics": m, "s_phys": res["s_phys"],
                    "loss_history": res["loss_history"],
                    "recon_losses": res.get("recon_losses"),
                    "fit_losses": res.get("fit_losses"),
                    "lambda_trajectory": res.get("lambda_trajectory"),
                })

    # ── L1/L2 baselines (appended last so they render at the end) ────────
    for key, label in [("s_l1_recon", "L1"), ("s_l2_recon", "L2")]:
        if key in samples[0]:
            all_results[label] = []
            for sample in samples:
                m = calculate_metrics(sample[key], sample["s_gt_raw"])
                all_results[label].append({"metrics": m, "s_phys": sample[key]})

    # ── Summary table ────────────────────────────────────────────────────
    log.info(f"\n{'='*90}")
    log.info(f"  JOINT DENOISER+RECON RESULTS — {len(indices)} samples")
    log.info(f"{'='*90}")
    log.info(f"  {'Method':<30} {'MAE±std':>14} {'RMSE±std':>14} "
             f"{'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*88}")

    for method, per_sample in all_results.items():
        maes = [r["metrics"]["MAE"] for r in per_sample]
        rmses = [r["metrics"]["RMSE"] for r in per_sample]
        ssims = [r["metrics"]["SSIM"] for r in per_sample]
        cnrs = [r["metrics"]["CNR"] for r in per_sample]
        log.info(
            f"  {method:<30}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )
    log.info(f"{'='*90}")

    # ── Thesis-quality report figures (optional, staged results only) ────
    if args.report_plots:
        _baselines = {"L1", "L2", "PI"}
        _topk_prefixes = ("rank", "roi", "cnr")
        report_results = {
            method: per_sample
            for method, per_sample in all_results.items()
            if method in _baselines
            or method.startswith("Joint")
            or method.startswith(_topk_prefixes)
        }
        if not any(m not in _baselines for m in report_results):
            log.warning("  No joint results to plot — skipping report figures")
        else:
            log.info("\n  Generating report figures (staged only) ...")
            grid_path_svg = run_dir / "report_comparison.svg"
            metrics_path_svg = run_dir / "report_metrics.svg"
            try:
                # Dataset title for the grid plot
                ds_titles = {
                    "kwave_geom": "GeomSet",
                    "kwave_blob": "BlobSet",
                    "inverse_crime": "InverseCrime",
                }
                plot_method_grid(
                    results=report_results,
                    samples=samples,
                    save_path=grid_path_svg,
                    dataset_title=ds_titles.get(args.dataset, args.dataset),
                    show=False,
                    png_fallback=True,
                )
                plot_metrics_comparison(
                    results=report_results,
                    save_path=metrics_path_svg,
                    show=False,
                    png_fallback=True,
                )
                log.info(f"  Report grid    -> {grid_path_svg}")
                log.info(f"  Report metrics -> {metrics_path_svg}")
            except Exception as exc:
                log.warning(f"  Report figure generation failed: {exc}")

    # ── Per-sample comparison plots ──────────────────────────────────────
    for i, idx in enumerate(indices):
        methods_for_plot = [
            (method, per_sample[i])
            for method, per_sample in all_results.items()
        ]
        plot_sample_comparison(samples[i], methods_for_plot, idx, run_dir)

    plot_summary_bars(all_results, run_dir)

    # ── Save results JSON ────────────────────────────────────────────────
    results_json = {
        "timestamp": timestamp,
        "tag": args.tag,
        "dataset": args.dataset,
        "mode": args.mode,
        "n_samples": len(indices),
        "indices": indices,
        "methods": {},
        "baselines": {},
    }
    if args.top_k and args.sweep_id:
        results_json["config_source"] = "sweep"
        results_json["sweep_id"] = args.sweep_id
        results_json["top_k"] = args.top_k
        results_json["configs"] = {
            c["label"]: c["raw_config"] for c in topk_configs
        }
    else:
        results_json["config_source"] = "default"
        results_json["lambda_strategy"] = args.lambda_strategy
        results_json["lambda_fit_values"] = args.lambda_fit
        results_json["denoise_cfg"] = denoise_cfg

    _BASELINE_KEYS = {"L1", "L2"}
    for method, per_sample in all_results.items():
        per_sample_metrics = [r["metrics"] for r in per_sample]
        if method in _BASELINE_KEYS:
            entries = [{"idx": int(idx), **m}
                       for idx, m in zip(indices, per_sample_metrics)]
            agg = {"method": method, "n_samples": len(per_sample_metrics),
                   "per_sample": entries}
            for key in ("MAE", "RMSE", "SSIM", "CNR"):
                vals = [m[key] for m in per_sample_metrics
                        if key in m and m[key] is not None]
                if vals:
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"]  = float(np.std(vals))
            results_json["baselines"][method] = agg
        else:
            results_json["methods"][method] = per_sample_metrics

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")
    log.info(f"  Experiment complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
