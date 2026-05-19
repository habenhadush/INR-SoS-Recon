#!/usr/bin/env python3
"""
run_denoised_reconstruction.py
------------------------------
Two-stage pipeline: (1) denoise measurements with per-pair INR,
(2) reconstruct SoS using existing engine on denoised measurements.

Compares raw vs denoised vs oracle reconstruction quality.

Usage:
    # Quick test (2 samples, default denoiser)
    python scripts/run_denoised_reconstruction.py --n_samples 2

    # Sweep denoiser scale
    python scripts/run_denoised_reconstruction.py --scale 0.5

    # Full run
    python scripts/run_denoised_reconstruction.py --dataset kwave_geom
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
import numpy as np
import torch
import yaml

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.models.mlp import FourierMLP, ReluMLP, GeluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import (
    optimize_full_forward_operator,
    optimize_sequential_views,
    optimize_stochastic_ray_batching,
)
from inr_sos.training.denoise_engine import (
    denoise_measurements,
    DEFAULT_DENOISE_CFG,
)
from inr_sos.visualization.report_figures import (
    plot_method_grid,
    plot_metrics_comparison,
)

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR / "data" / "denoised_reconstruction"
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"

ENGINE_MAP = {
    "Full_Matrix":    optimize_full_forward_operator,
    "Sequential_SGD": optimize_sequential_views,
    "Ray_Batching":   optimize_stochastic_ray_batching,
}
MODEL_MAP = {
    "ReluMLP": ReluMLP, "FourierMLP": FourierMLP,
    "SirenMLP": SirenMLP, "GeluMLP": GeluMLP,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("denoiser_exp")


# ─── Reconstruction config (fixed, known good) ───────────────────────────────

def make_recon_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Denoiser-Experiment",
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


# ─── Sweep config fetchers (compose mode) ────────────────────────────────────

def _load_registry_entry(sweep_id):
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    for e in registry:
        if e["sweep_id"].startswith(sweep_id):
            return e
    raise ValueError(f"Sweep ID '{sweep_id}' not found in {REGISTRY_FILE}")


def fetch_regularizer_config(sweep_id, rank, steps_override, logger):
    """Pull a single denoise_cfg + pretrain_dn_steps from a joint sweep at given rank.

    The joint sweep's Stage-3a denoiser config defines the measurement-space
    regularizer with no joint coupling — exactly the §5.3 middle-rung object.
    """
    import wandb as wb
    entry = _load_registry_entry(sweep_id)
    logger.info(f"  Regularizer source: joint sweep {entry['sweep_id']}, rank={rank}")
    api = wb.Api()
    sweep = api.sweep(f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}")
    runs = list(sweep.runs)
    runs = [r for r in runs if "MAE_mean" in r.summary]
    if not runs:
        raise RuntimeError(f"No completed runs in joint sweep {entry['sweep_id']}")
    runs = sorted(runs, key=lambda r: r.summary["MAE_mean"])
    if rank < 1 or rank > len(runs):
        raise ValueError(f"--regularizer_rank={rank} out of range (sweep has {len(runs)})")
    run = runs[rank - 1]
    sc = run.config

    def _g(key, default, cast=float):
        val = sc.get(key, default)
        if isinstance(val, dict):
            val = default
        return cast(val)

    dn_cfg = dict(DEFAULT_DENOISE_CFG)
    dn_cfg["model_type"]     = _g("dn_model_type", dn_cfg["model_type"], str)
    dn_cfg["scale"]          = _g("dn_scale", dn_cfg["scale"], float)
    dn_cfg["omega"]          = _g("dn_omega", dn_cfg.get("omega", 15.0), float)
    dn_cfg["hidden_features"]= _g("dn_hidden_features", dn_cfg["hidden_features"], int)
    dn_cfg["hidden_layers"]  = _g("dn_hidden_layers", dn_cfg["hidden_layers"], int)
    dn_cfg["mapping_size"]   = _g("dn_mapping_size", dn_cfg.get("mapping_size", 32), int)
    dn_cfg["lr"]             = _g("dn_lr", dn_cfg["lr"], float)
    sweep_steps              = _g("pretrain_dn_steps", dn_cfg["steps"], int)
    dn_cfg["steps"]          = int(steps_override) if steps_override else sweep_steps

    logger.info(f"  Denoiser cfg: {dn_cfg['model_type']} "
                f"{dn_cfg['hidden_features']}x{dn_cfg['hidden_layers']} "
                f"steps={dn_cfg['steps']} "
                f"(sweep_default={sweep_steps}"
                f"{', overridden' if steps_override else ''})")
    return dn_cfg, {"sweep_id": entry["sweep_id"], "rank": rank,
                    "sweep_steps_default": sweep_steps,
                    "steps_used": dn_cfg["steps"],
                    "run_id": run.id}


def fetch_topk_reconstructor_configs(sweep_id, top_k, logger):
    """Pull top-K plain-INR configs from a reconstruction sweep (sorted by MAE_mean)."""
    import wandb as wb
    entry = _load_registry_entry(sweep_id)
    logger.info(f"  Reconstructor source: plain-INR sweep {entry['sweep_id']}, "
                f"top_k={top_k}")
    api = wb.Api()
    sweep = api.sweep(f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}")
    runs = [r for r in sweep.runs if "MAE_mean" in r.summary]
    runs = sorted(runs, key=lambda r: r.summary["MAE_mean"])
    if not runs:
        raise RuntimeError(f"No completed runs in reconstruction sweep {entry['sweep_id']}")
    selected = runs[:top_k]

    configs = []
    for rank, run in enumerate(selected, 1):
        method = run.config.get("method", "Full_Matrix")
        mtype = run.config.get("model_type", "ReluMLP")
        hp = {k: v for k, v in run.config.items()
              if k not in {"method", "model_type", "_wandb"}}
        sweep_mae = run.summary.get("MAE_mean", float("inf"))
        label = f"rank{rank}_{method}_{mtype}"
        logger.info(f"  #{rank}: {label}  sweep_MAE={sweep_mae:.2f}")
        configs.append({
            "rank": rank, "label": label, "method": method,
            "model_type": mtype, "hyperparams": hp,
            "sweep_mae": float(sweep_mae), "run_id": run.id,
        })
    return configs


def _build_recon_model(model_type, base_cfg, hp):
    cls = MODEL_MAP[model_type]
    kwargs = dict(
        in_features=base_cfg.in_features,
        hidden_features=hp.get("hidden_features", base_cfg.hidden_features),
        hidden_layers=hp.get("hidden_layers", base_cfg.hidden_layers),
        mapping_size=hp.get("mapping_size", base_cfg.mapping_size),
    )
    if model_type == "FourierMLP":
        kwargs["scale"] = hp.get("scale", 10.0)
    elif model_type == "SirenMLP":
        kwargs["omega"] = hp.get("omega", 30.0)
    return cls(**kwargs)


def run_compose_sample(sample, dataset, grid, denoise_cfg, recon_top_k,
                       base_recon_cfg, idx, logger):
    """Compose-mode per-sample run.

    Stage A: train the regularizer (denoiser) ONCE on raw measurements →
             d_clean. No feedback from the reconstructor.
    Stage B: fit each of the K reconstruction configs against d_clean.

    Returns dict: {method_label: {"metrics": ..., "s_phys": ..., ...}, "L1": ..., "L2": ...,
                   "oracle_comparison": {...}, "d_denoised": Tensor}
    """
    L_matrix = dataset.L_matrix
    s_gt = sample["s_gt_raw"]

    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        base_recon_cfg.time_scale = 1.0 / dataset.pix2time

    out = {}

    # L1 / L2 analytical baselines
    for key, label in [("s_l1_recon", "L1"), ("s_l2_recon", "L2")]:
        if key in sample:
            m = calculate_metrics(sample[key], s_gt)
            out[label] = {"metrics": m, "s_phys": sample[key]}

    # Stage A — regularize (no feedback)
    logger.info(f"  [idx={idx}] Regularizing measurements (denoise)...")
    t0 = time.perf_counter()
    d_clean, denoise_info = denoise_measurements(sample, grid, denoise_cfg)
    t_denoise = time.perf_counter() - t0
    out["_d_denoised"] = d_clean

    # Oracle diagnostic against ideal L @ s_GT
    d_raw_np = sample["d_meas"].numpy().flatten()
    d_dn_np = d_clean.numpy().flatten()
    mask_np = sample["mask"].numpy().flatten()
    s_gt_np = (s_gt.detach().cpu().numpy() if hasattr(s_gt, "detach")
               else np.asarray(s_gt)).flatten()
    if isinstance(L_matrix, torch.Tensor):
        L_np = (L_matrix.to_dense().cpu().numpy() if L_matrix.is_sparse
                else L_matrix.cpu().numpy())
    else:
        L_np = np.asarray(L_matrix)
    d_oracle = (L_np @ s_gt_np).flatten()
    valid = mask_np > 0.5
    raw_residual = np.abs(d_raw_np[valid] - d_oracle[valid]).mean()
    dn_residual = np.abs(d_dn_np[valid] - d_oracle[valid]).mean()
    out["oracle_comparison"] = {
        "raw_vs_oracle_mae": float(raw_residual),
        "denoised_vs_oracle_mae": float(dn_residual),
        "improvement_pct": float((raw_residual - dn_residual) / raw_residual * 100)
        if raw_residual > 0 else 0.0,
        "denoise_time_s": float(t_denoise),
    }
    logger.info(f"  [idx={idx}] d_residual: raw={raw_residual:.2e} "
                f"dn={dn_residual:.2e}  ({t_denoise:.1f}s)")

    # Stage B — reconstruct, looping over K configs against d_clean
    sample_clean = copy.copy(sample)
    sample_clean["d_meas"] = d_clean
    sample_clean["mask"] = sample["mask"]

    for cfg_entry in recon_top_k:
        label = cfg_entry["label"]
        method = cfg_entry["method"]
        mtype = cfg_entry["model_type"]
        hp = cfg_entry["hyperparams"]
        engine_fn = ENGINE_MAP.get(method)
        if engine_fn is None:
            logger.warning(f"  [idx={idx}] {label}: engine '{method}' not supported, skip")
            continue

        cfg = copy.deepcopy(base_recon_cfg)
        cfg.model_type      = mtype
        cfg.hidden_features = hp.get("hidden_features", cfg.hidden_features)
        cfg.hidden_layers   = hp.get("hidden_layers",   cfg.hidden_layers)
        cfg.mapping_size    = hp.get("mapping_size",    cfg.mapping_size)
        cfg.lr              = hp.get("lr",              cfg.lr)
        cfg.steps           = hp.get("steps",           cfg.steps)
        cfg.tv_weight       = hp.get("tv_weight",       getattr(cfg, "tv_weight", 0.0))
        cfg.reg_weight      = hp.get("reg_weight",      getattr(cfg, "reg_weight", 0.0))
        cfg.clamp_slowness  = hp.get("clamp_slowness",  cfg.clamp_slowness)
        cfg.early_stopping  = hp.get("early_stopping",  cfg.early_stopping)
        cfg.patience        = hp.get("patience",        cfg.patience)
        if method == "Ray_Batching":
            cfg.epochs     = hp.get("epochs",     getattr(cfg, "epochs", 50))
            cfg.batch_size = hp.get("batch_size", getattr(cfg, "batch_size", 1024))

        model = _build_recon_model(mtype, cfg, hp)
        t0 = time.perf_counter()
        res = engine_fn(sample=sample_clean, L_matrix=L_matrix, model=model,
                        label=label, config=cfg, use_wandb=False)
        elapsed = time.perf_counter() - t0
        m = calculate_metrics(res["s_phys"], s_gt)
        m["time_s"] = elapsed
        out[label] = {"metrics": m, "s_phys": res["s_phys"],
                      "loss_history": res["loss_history"]}
        logger.info(f"  [idx={idx}] {label}: MAE={m['MAE']:.2f} "
                    f"CNR={m['CNR']:.3f}  ({elapsed:.0f}s)")
    return out


# ─── Run one sample through the two-stage pipeline ───────────────────────────

def run_single_sample(sample, dataset, grid, denoise_cfg, recon_cfg, idx):
    """Run raw, denoised, and oracle reconstruction for one sample.

    Returns dict with metrics for each condition.
    """
    L_matrix = dataset.L_matrix
    s_gt = sample["s_gt_raw"]

    # Set time_scale from dataset
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time

    results = {}

    # ── L1/L2 baselines (no optimization, just metrics) ─────────────────
    for key, label in [("s_l1_recon", "l1"), ("s_l2_recon", "l2")]:
        if key in sample:
            m = calculate_metrics(sample[key], s_gt)
            results[label] = {"metrics": m, "s_phys": sample[key]}

    # ── Condition 1: Raw measurements (baseline) ─────────────────────────
    log.info(f"  [idx={idx}] Raw reconstruction...")
    model_raw = build_recon_model(recon_cfg)
    t0 = time.perf_counter()
    res_raw = optimize_full_forward_operator(
        sample=sample, L_matrix=L_matrix, model=model_raw,
        label="ReluMLP_raw", config=copy.deepcopy(recon_cfg), use_wandb=False,
    )
    t_raw = time.perf_counter() - t0
    m_raw = calculate_metrics(res_raw["s_phys"], s_gt)
    m_raw["time_s"] = t_raw
    results["raw"] = {"metrics": m_raw, "s_phys": res_raw["s_phys"],
                      "loss_history": res_raw["loss_history"]}

    # ── Stage 1: Denoise measurements ────────────────────────────────────
    log.info(f"  [idx={idx}] Denoising measurements...")
    t0 = time.perf_counter()
    d_denoised, denoise_info = denoise_measurements(sample, grid, denoise_cfg)
    t_denoise = time.perf_counter() - t0

    # ── Stage 2: Reconstruct from denoised ───────────────────────────────
    log.info(f"  [idx={idx}] Denoised reconstruction...")
    sample_denoised = copy.copy(sample)
    sample_denoised["d_meas"] = d_denoised
    sample_denoised["mask"] = sample["mask"]  # keep original valid-ray mask

    model_dn = build_recon_model(recon_cfg)
    t0 = time.perf_counter()
    res_dn = optimize_full_forward_operator(
        sample=sample_denoised, L_matrix=L_matrix, model=model_dn,
        label="ReluMLP_denoised", config=copy.deepcopy(recon_cfg), use_wandb=False,
    )
    t_recon = time.perf_counter() - t0
    m_dn = calculate_metrics(res_dn["s_phys"], s_gt)
    m_dn["time_s"] = t_denoise + t_recon
    m_dn["denoise_time_s"] = t_denoise
    results["denoised"] = {"metrics": m_dn, "s_phys": res_dn["s_phys"],
                           "loss_history": res_dn["loss_history"],
                           "denoise_info": denoise_info,
                           "d_denoised": d_denoised}

    # ── Oracle comparison (diagnostic) ───────────────────────────────────
    d_raw_np = sample["d_meas"].numpy().flatten()
    d_dn_np = d_denoised.numpy().flatten()
    mask_np = sample["mask"].numpy().flatten()

    if hasattr(s_gt, "detach"):
        s_gt_np = s_gt.detach().cpu().numpy().flatten()
    else:
        s_gt_np = np.asarray(s_gt).flatten()

    if isinstance(L_matrix, torch.Tensor):
        L_np = L_matrix.cpu().numpy() if not L_matrix.is_sparse else L_matrix.to_dense().cpu().numpy()
    else:
        L_np = np.asarray(L_matrix)

    d_oracle = (L_np @ s_gt_np).flatten()

    valid = mask_np > 0.5
    raw_residual = np.abs(d_raw_np[valid] - d_oracle[valid]).mean()
    dn_residual = np.abs(d_dn_np[valid] - d_oracle[valid]).mean()
    results["oracle_comparison"] = {
        "raw_vs_oracle_mae": float(raw_residual),
        "denoised_vs_oracle_mae": float(dn_residual),
        "improvement_pct": float((raw_residual - dn_residual) / raw_residual * 100)
        if raw_residual > 0 else 0.0,
    }

    log.info(
        f"  [idx={idx}] Raw: MAE={m_raw['MAE']:.2f} CNR={m_raw['CNR']:.3f} | "
        f"Denoised: MAE={m_dn['MAE']:.2f} CNR={m_dn['CNR']:.3f} | "
        f"d_residual: raw={raw_residual:.2e} dn={dn_residual:.2e}"
    )

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _to_sos(s_flat):
    if hasattr(s_flat, "detach"):
        s_flat = s_flat.detach().cpu().numpy()
    s_flat = np.asarray(s_flat).flatten()
    s_clamped = np.clip(s_flat, 1.0 / 1800.0, 1.0 / 1200.0)
    return (1.0 / s_clamped).reshape(64, 64, order="F")


def plot_sample_comparison(sample, results, idx, out_dir):
    """Comparison plot: GT | L1 | L2 | Raw INR | Denoised INR.

    Each method gets: SoS map + Abs Error side by side.
    INR methods also get a convergence column.
    """
    from inr_sos.evaluation.metrics import calculate_metrics

    v_gt = _to_sos(sample["s_gt_raw"])

    # Build list of (label, v_rec, metrics, loss_history)
    rows = []

    # L1/L2 baselines from dataset
    for key, label in [("s_l1_recon", "L1 Baseline"), ("s_l2_recon", "L2 Baseline")]:
        if key in sample:
            v = _to_sos(sample[key])
            m = calculate_metrics(sample[key], sample["s_gt_raw"])
            rows.append((label, v, m, None))

    # Raw and Denoised INR
    for label, res_key in [("Raw INR", "raw"), ("Denoised INR", "denoised")]:
        res = results[res_key]
        v = _to_sos(res["s_phys"])
        rows.append((label, v, res["metrics"], res.get("loss_history")))

    n_rows = 1 + len(rows)  # GT row + method rows
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    fig.suptitle(f"Sample {idx} — Reconstruction Comparison", fontsize=13)

    # Row 0: Ground Truth
    im = axes[0, 0].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
    axes[0, 0].set_title("Ground Truth (m/s)", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    for row, (label, v_rec, m, loss_hist) in enumerate(rows, 1):
        err = np.abs(v_gt - v_rec)

        # SoS map
        im = axes[row, 0].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
        axes[row, 0].set_title(
            f"{label}  MAE={m['MAE']:.1f}  CNR={m['CNR']:.2f}", fontsize=10
        )
        axes[row, 0].axis("off")
        plt.colorbar(im, ax=axes[row, 0], fraction=0.046, pad=0.04)

        # Error map
        im_e = axes[row, 1].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[row, 1].set_title("Abs. Error", fontsize=10)
        axes[row, 1].axis("off")
        plt.colorbar(im_e, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Convergence (only for INR methods)
        if loss_hist:
            axes[row, 2].plot(loss_hist, color="#1f77b4", linewidth=1)
            axes[row, 2].set_yscale("log")
            axes[row, 2].set_title("Convergence", fontsize=10)
            axes[row, 2].set_xlabel("Iteration", fontsize=9)
            axes[row, 2].grid(True, which="both", ls="--", alpha=0.4)
            axes[row, 2].spines["top"].set_visible(False)
            axes[row, 2].spines["right"].set_visible(False)
        else:
            axes[row, 2].axis("off")

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_comparison.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_displacement_fields(sample, d_denoised, idx, out_dir):
    """Visualize raw vs denoised displacement for all 8 firing pairs.

    Layout: 8 rows (one per pair) × 4 columns:
      Col 0: Raw (valid only)
      Col 1: Denoised (ALL pixels — shows hole-filling + cone extrapolation)
      Col 2: Difference on valid pixels (denoised - raw)
      Col 3: Filled holes only (pixels that were NaN in raw, now predicted)
    Orientation: transducer at top, depth downward (order="F").
    """
    d_raw = sample["d_meas"].numpy().flatten()
    d_dn = d_denoised.numpy().flatten()
    mask = sample["mask"].numpy().flatten()

    n_pairs = 8
    fig, axes = plt.subplots(n_pairs, 4, figsize=(20, 4 * n_pairs))
    fig.suptitle(f"Sample {idx} — Displacement Fields (all pairs)", fontsize=14, y=1.0)

    col_titles = ["Raw (valid only)", "Denoised (all pixels)",
                   "Diff on valid", "Filled holes"]

    for k in range(n_pairs):
        start = k * 16384
        end = (k + 1) * 16384

        # Reshape with order="F": depth (z) as rows, lateral (x) as cols
        raw_2d = d_raw[start:end].reshape(128, 128, order="F")
        dn_2d = d_dn[start:end].reshape(128, 128, order="F")
        mask_2d = mask[start:end].reshape(128, 128, order="F")

        # Masked versions
        raw_masked = np.where(mask_2d > 0.5, raw_2d, np.nan)
        holes_only = np.where(mask_2d < 0.5, dn_2d, np.nan)  # NaN pixels filled by denoiser

        vmin = np.nanmin(raw_masked)
        vmax = np.nanmax(raw_masked)

        # Col 0: Raw valid
        im0 = axes[k, 0].imshow(raw_masked, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 0].set_ylabel(f"Pair {k}", fontsize=10, fontweight="bold")
        axes[k, 0].set_xticks([]); axes[k, 0].set_yticks([])
        if k == 0:
            axes[k, 0].set_title(col_titles[0], fontsize=11)
        plt.colorbar(im0, ax=axes[k, 0], fraction=0.046, pad=0.04)

        # Col 1: Denoised ALL pixels (unmasked — shows the full INR prediction)
        im1 = axes[k, 1].imshow(dn_2d, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 1].axis("off")
        if k == 0:
            axes[k, 1].set_title(col_titles[1], fontsize=11)
        plt.colorbar(im1, ax=axes[k, 1], fraction=0.046, pad=0.04)

        # Col 2: Difference on valid pixels
        diff = np.where(mask_2d > 0.5, dn_2d - raw_2d, np.nan)
        abs_max = np.nanmax(np.abs(diff)) if not np.all(np.isnan(diff)) else 1.0
        im2 = axes[k, 2].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                                  origin="upper")
        axes[k, 2].axis("off")
        if k == 0:
            axes[k, 2].set_title(col_titles[2], fontsize=11)
        plt.colorbar(im2, ax=axes[k, 2], fraction=0.046, pad=0.04)

        # Col 3: Filled holes (only the pixels that were NaN)
        im3 = axes[k, 3].imshow(holes_only, cmap="viridis", vmin=vmin, vmax=vmax,
                                  origin="upper")
        axes[k, 3].axis("off")
        n_holes = int((mask_2d < 0.5).sum())
        if k == 0:
            axes[k, 3].set_title(col_titles[3], fontsize=11)
        axes[k, 3].text(0.02, 0.98, f"{n_holes} px", transform=axes[k, 3].transAxes,
                         fontsize=8, va="top", color="white",
                         bbox=dict(boxstyle="round", fc="black", alpha=0.6))
        plt.colorbar(im3, ax=axes[k, 3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fp = out_dir / f"sample_{idx}_displacement_all_pairs.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage denoised reconstruction"
    )
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    # Denoiser config overrides
    parser.add_argument("--model_type", default="FourierMLP")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--mapping_size", type=int, default=32)
    parser.add_argument("--denoise_steps", type=int, default=500)
    parser.add_argument("--denoise_lr", type=float, default=1e-3)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--report_plots", action="store_true",
                        help="Generate thesis-quality comparison figures (SVG + PNG).")
    parser.add_argument("--tag", default=None,
                        help="Optional tag appended to result dir as <timestamp>_<tag>.")
    parser.add_argument("--comment", default=None,
                        help="Free-text comment logged into results.json.")
    # Compose mode: pull regularizer from a joint sweep, reconstructor top-K
    # from a plain-INR sweep, run the §5.3 middle rung.
    parser.add_argument("--regularizer_sweep_id", default=None,
                        help="Joint sweep ID whose Stage-3a denoiser config feeds "
                             "the measurement-space regularizer.")
    parser.add_argument("--regularizer_rank", type=int, default=1,
                        help="Which rank from the regularizer sweep (default: 1).")
    parser.add_argument("--regularizer_steps", type=int, default=None,
                        help="Override the regularizer's training-step count "
                             "(default: use the joint sweep's pretrain_dn_steps).")
    parser.add_argument("--reconstructor_sweep_id", default=None,
                        help="Plain-INR sweep ID whose top-K configs reconstruct "
                             "from the regularized measurements.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-K reconstructor configs (default: 5).")
    args = parser.parse_args()

    compose_mode = bool(args.regularizer_sweep_id and args.reconstructor_sweep_id)

    # ── Setup ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{args.tag}" if args.tag else timestamp
    if compose_mode:
        # Layout: data/denoised_reconstruction/<reconstructor_sweep_id>/<dataset>/<timestamp>[_<tag>]/
        run_dir = OUTPUT_DIR / args.reconstructor_sweep_id / args.dataset / dir_name
    else:
        run_dir = OUTPUT_DIR / args.dataset / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    regularizer_source = None
    if compose_mode:
        denoise_cfg, regularizer_source = fetch_regularizer_config(
            args.regularizer_sweep_id, args.regularizer_rank,
            args.regularizer_steps, log,
        )
    else:
        denoise_cfg = dict(DEFAULT_DENOISE_CFG)
        denoise_cfg.update({
            "model_type": args.model_type,
            "scale": args.scale,
            "hidden_features": args.hidden_features,
            "hidden_layers": args.hidden_layers,
            "mapping_size": args.mapping_size,
            "steps": args.denoise_steps,
            "lr": args.denoise_lr,
        })

    log.info("=" * 70)
    log.info("  Denoised Reconstruction — "
             + ("compose mode" if compose_mode else "single-config"))
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  Denoiser: {denoise_cfg['model_type']}, scale={denoise_cfg['scale']}, "
             f"hidden={denoise_cfg['hidden_features']}x{denoise_cfg['hidden_layers']}, "
             f"steps={denoise_cfg['steps']}")
    log.info(f"  Output: {run_dir}")
    log.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────
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
    log.info(f"  Loaded {len(dataset)} samples")

    # ── Sample indices ────────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        n = args.n_samples if args.n_samples is not None else len(dataset)
        indices = list(range(min(n, len(dataset))))
    log.info(f"  Using {len(indices)} samples: {indices}")

    # ── Run experiment ────────────────────────────────────────────────────
    recon_cfg = make_recon_config()
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        recon_cfg.time_scale = 1.0 / dataset.pix2time
    elif ds_cfg.get("pix2time") is not None:
        recon_cfg.time_scale = 1.0 / float(ds_cfg["pix2time"])
        log.info(f"  time_scale from yaml pix2time: {recon_cfg.time_scale:.4e}")

    # ════════════════════════════════════════════════════════════════════════
    # Compose mode: sweep-driven middle-rung pipeline for §5.3.
    # Decoupled two-stage: regularize once per sample, then run top-K plain
    # INR reconstructions against the cleaned measurements. No feedback loop.
    # ════════════════════════════════════════════════════════════════════════
    if compose_mode:
        recon_top_k = fetch_topk_reconstructor_configs(
            args.reconstructor_sweep_id, args.top_k, log,
        )

        # all_results: {method_label: [per_sample_dict, ...]} matching the
        # joint pipeline's results.json schema; plus L1/L2 appended at end.
        all_results: dict[str, list] = {}
        per_sample_meta = []  # idx + oracle_comparison + denoise info

        for i, idx in enumerate(indices):
            log.info(f"\n── Sample {i+1}/{len(indices)} (idx={idx}) ──")
            sample = dataset[idx]
            out = run_compose_sample(
                sample, dataset, grid, denoise_cfg, recon_top_k,
                copy.deepcopy(recon_cfg), idx, log,
            )
            # Collect per-method
            for cfg_entry in recon_top_k:
                lbl = cfg_entry["label"]
                if lbl not in out:
                    continue
                all_results.setdefault(lbl, []).append({
                    "metrics": out[lbl]["metrics"],
                    "s_phys": out[lbl]["s_phys"],
                    "loss_history": out[lbl].get("loss_history"),
                })
            per_sample_meta.append({
                "idx": idx,
                "oracle_comparison": out["oracle_comparison"],
            })

            # Displacement-field plot (one per sample), if requested
            if args.report_plots and "_d_denoised" in out:
                try:
                    plot_displacement_fields(sample, out["_d_denoised"], idx, run_dir)
                except Exception as exc:
                    log.warning(f"  displacement plot failed (idx={idx}): {exc}")

        # Append L1/L2 baselines AT END so they render last in plots/tables
        sample_zero = dataset[indices[0]]
        for key, lbl in [("s_l1_recon", "L1"), ("s_l2_recon", "L2")]:
            if key in sample_zero:
                all_results[lbl] = []
                for idx in indices:
                    s = dataset[idx]
                    m = calculate_metrics(s[key], s["s_gt_raw"])
                    all_results[lbl].append({"metrics": m, "s_phys": s[key]})

        # Summary table
        log.info(f"\n{'='*92}")
        log.info(f"  DENOISED RECONSTRUCTION (compose) — {len(indices)} samples")
        log.info(f"{'='*92}")
        log.info(f"  {'Method':<32} {'MAE±std':>14} {'RMSE±std':>14} "
                 f"{'SSIM±std':>14} {'CNR±std':>14}")
        log.info(f"  {'─'*90}")
        for method, per in all_results.items():
            maes  = [r["metrics"]["MAE"]  for r in per]
            rmses = [r["metrics"]["RMSE"] for r in per]
            ssims = [r["metrics"]["SSIM"] for r in per]
            cnrs  = [r["metrics"]["CNR"]  for r in per]
            log.info(
                f"  {method:<32}"
                f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
                f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
                f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
                f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
            )
        log.info(f"{'='*92}")

        # Report-quality figures (SVG + PNG fallback)
        if args.report_plots:
            log.info("\n  Generating report figures ...")
            samples = [dataset[idx] for idx in indices]
            ds_titles = {"kwave_geom": "GeomSet", "kwave_blob": "BlobSet",
                         "inverse_crime": "InverseCrime"}
            try:
                grid_path_svg = run_dir / "report_comparison.svg"
                metrics_path_svg = run_dir / "report_metrics.svg"
                plot_method_grid(
                    results=all_results, samples=samples, save_path=grid_path_svg,
                    dataset_title=ds_titles.get(args.dataset, args.dataset),
                    show=False, png_fallback=True,
                )
                plot_metrics_comparison(
                    results=all_results, save_path=metrics_path_svg,
                    show=False, png_fallback=True,
                )
                log.info(f"  Report grid    -> {grid_path_svg}")
                log.info(f"  Report metrics -> {metrics_path_svg}")
            except Exception as exc:
                log.warning(f"  Report figure generation failed: {exc}")

        # Save results.json with joint-pipeline schema
        results_json = {
            "timestamp": timestamp,
            "comment": args.comment,
            "dataset": args.dataset,
            "mode": "compose",
            "n_samples": len(indices),
            "indices": indices,
            "regularizer_source": regularizer_source,
            "regularizer_cfg": denoise_cfg,
            "reconstructor_source": {
                "sweep_id": args.reconstructor_sweep_id, "top_k": args.top_k,
            },
            "reconstructor_configs": {
                c["label"]: {"rank": c["rank"], "method": c["method"],
                             "model_type": c["model_type"],
                             "hyperparams": c["hyperparams"],
                             "sweep_mae": c["sweep_mae"], "run_id": c["run_id"]}
                for c in recon_top_k
            },
            "per_sample_meta": per_sample_meta,
            "methods": {},
            "baselines": {},
        }
        _BASELINE_KEYS = {"L1", "L2"}
        for method, per in all_results.items():
            per_metrics = [r["metrics"] for r in per]
            if method in _BASELINE_KEYS:
                entries = [{"idx": int(idx), **m} for idx, m in zip(indices, per_metrics)]
                agg = {"method": method, "n_samples": len(per_metrics),
                       "per_sample": entries}
                for key in ("MAE", "RMSE", "SSIM", "CNR"):
                    vals = [m[key] for m in per_metrics
                            if key in m and m[key] is not None]
                    if vals:
                        agg[f"{key}_mean"] = float(np.mean(vals))
                        agg[f"{key}_std"]  = float(np.std(vals))
                results_json["baselines"][method] = agg
            else:
                results_json["methods"][method] = per_metrics

        results_path = run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        log.info(f"\n  Results saved → {results_path}")
        log.info(f"  Experiment complete. All outputs in {run_dir}")
        return

    # ── Legacy single-config path (kept for backward compatibility) ──────
    all_results = []
    for i, idx in enumerate(indices):
        log.info(f"\n── Sample {i+1}/{len(indices)} (idx={idx}) ──")
        sample = dataset[idx]
        results = run_single_sample(
            sample, dataset, grid, denoise_cfg, recon_cfg, idx
        )
        results["idx"] = idx
        all_results.append(results)

        # Save plots
        plot_sample_comparison(sample, results, idx, run_dir)
        # Displacement fields for all 8 pairs
        d_dn_full = results["denoised"].get("d_denoised")
        if d_dn_full is not None:
            plot_displacement_fields(sample, d_dn_full, idx, run_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info(f"  DENOISER EXPERIMENT RESULTS — {len(indices)} samples")
    log.info(f"{'='*80}")
    log.info(f"  {'Condition':<15} {'MAE±std':>14} {'RMSE±std':>14} "
             f"{'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*75}")

    conditions = ["l1", "l2", "raw", "denoised"]
    for cond in conditions:
        if cond not in all_results[0]:
            continue
        maes = [r[cond]["metrics"]["MAE"] for r in all_results]
        rmses = [r[cond]["metrics"]["RMSE"] for r in all_results]
        ssims = [r[cond]["metrics"]["SSIM"] for r in all_results]
        cnrs = [r[cond]["metrics"]["CNR"] for r in all_results]
        label = {"l1": "L1", "l2": "L2",
                 "raw": "PI", "denoised": "Denoised INR"}[cond]
        log.info(
            f"  {label:<15}"
            f"  {np.mean(maes):>6.2f}±{np.std(maes):<5.2f}"
            f"  {np.mean(rmses):>6.2f}±{np.std(rmses):<5.2f}"
            f"  {np.mean(ssims):>6.3f}±{np.std(ssims):<5.3f}"
            f"  {np.mean(cnrs):>6.3f}±{np.std(cnrs):<5.3f}"
        )

    # Oracle comparison
    raw_oracle = [r["oracle_comparison"]["raw_vs_oracle_mae"] for r in all_results]
    dn_oracle = [r["oracle_comparison"]["denoised_vs_oracle_mae"] for r in all_results]
    log.info(f"\n  Oracle displacement MAE (d vs L@s_true):")
    log.info(f"    Raw:      {np.mean(raw_oracle):.2e} ± {np.std(raw_oracle):.2e}")
    log.info(f"    Denoised: {np.mean(dn_oracle):.2e} ± {np.std(dn_oracle):.2e}")
    log.info(f"{'='*80}")

    # ── Thesis-quality report figures (optional) ─────────────────────────
    if args.report_plots:
        log.info("\n  Generating report figures ...")
        # Restructure: {method: [per_sample_results]} for report_figures API
        samples = [dataset[idx] for idx in indices]
        report_results = {}
        for cond, label in [("l1", "L1"), ("l2", "L2"),
                            ("raw", "PI"), ("denoised", "Denoised INR")]:
            if cond not in all_results[0]:
                continue
            report_results[label] = [
                {"metrics": r[cond]["metrics"], "s_phys": r[cond]["s_phys"]}
                for r in all_results
            ]

        ds_titles = {
            "kwave_geom": "GeomSet",
            "kwave_blob": "BlobSet",
            "inverse_crime": "InverseCrime",
        }
        try:
            grid_path_svg = run_dir / "report_comparison.svg"
            metrics_path_svg = run_dir / "report_metrics.svg"
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

    # ── Save results ──────────────────────────────────────────────────────
    results_json = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "denoise_cfg": denoise_cfg,
        "n_samples": len(indices),
        "indices": indices,
        "per_sample": [
            {
                "idx": r["idx"],
                **{k: r[k]["metrics"] for k in ["l1", "l2"] if k in r},
                "raw": r["raw"]["metrics"],
                "denoised": r["denoised"]["metrics"],
                "oracle_comparison": r["oracle_comparison"],
            }
            for r in all_results
        ],
    }
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")

    # ── W&B logging ───────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb

        wandb.init(
            project="INR-SoS-Recon",
            name=f"denoiser_{denoise_cfg['model_type']}_s{denoise_cfg['scale']}_{timestamp}",
            group="Denoiser-Experiment",
            tags=["denoiser", args.dataset, denoise_cfg["model_type"]],
            config={"denoise_cfg": denoise_cfg, "recon_cfg": recon_cfg.to_dict(),
                    "n_samples": len(indices)},
        )

        columns = ["Condition", "MAE_mean", "MAE_std", "CNR_mean", "CNR_std",
                    "SSIM_mean", "SSIM_std"]
        data = []
        for cond in ["l1", "l2", "raw", "denoised"]:
            if cond not in all_results[0]:
                continue
            maes = [r[cond]["metrics"]["MAE"] for r in all_results]
            cnrs = [r[cond]["metrics"]["CNR"] for r in all_results]
            ssims = [r[cond]["metrics"]["SSIM"] for r in all_results]
            data.append([cond, np.mean(maes), np.std(maes),
                         np.mean(cnrs), np.std(cnrs),
                         np.mean(ssims), np.std(ssims)])
        wandb.log({"comparison_table": wandb.Table(columns=columns, data=data)})

        # Log plots
        for fp in run_dir.glob("*.png"):
            wandb.log({fp.stem: wandb.Image(str(fp))})

        wandb.finish()

    log.info(f"\n  Experiment complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
