#!/usr/bin/env python3
"""
run_reconstruction.py
---------------------
Runs the top-K INR configs from a completed sweep on a dataset.
Same workflow as compare_topk but without baselines — works with
datasets that only have displacement fields (no GT, no baselines).

If ground truth is available  → full metrics (MAE, RMSE, SSIM, CNR)
If ground truth is missing    → reconstruction-only (plots + residual)

Outputs:
  1. Terminal + log: results table with reconstruction time
  2. Per-sample reconstruction plots saved to plots/<sweep_id>/
  3. W&B per-config runs + summary run (optional)
  4. Registry updated

Usage:
    # quick test
    python scripts/run_reconstruction.py --sweep_id <ID> --dataset kwave_arranged --n_samples 2

    # full run
    python scripts/run_reconstruction.py --sweep_id <ID> --dataset kwave_arranged --n_samples 20

    # per-model top-K
    python scripts/run_reconstruction.py --sweep_id <ID> --dataset kwave_arranged \\
        --top_k_per_model 2 --n_samples 10
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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import wandb
import yaml

from PIL import Image

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.models.mlp import FourierMLP, ReluMLP, GeluMLP, BiasMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import (
    optimize_full_forward_operator,
    optimize_sequential_views,
    optimize_stochastic_ray_batching,
    optimize_with_bias_absorption,
)

SCRIPTS_DIR   = Path(__file__).parent
REGISTRY_FILE = SCRIPTS_DIR / "sweep_registry.json"
LOG_DIR       = SCRIPTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

ENGINE_MAP = {
    "Full_Matrix":    optimize_full_forward_operator,
    "Sequential_SGD": optimize_sequential_views,
    "Ray_Batching":   optimize_stochastic_ray_batching,
    "Bias_Absorption": optimize_with_bias_absorption,
}
MODEL_MAP = {
    "FourierMLP": FourierMLP,
    "ReluMLP":    ReluMLP,
    "SirenMLP":   SirenMLP,
    "GeluMLP":    GeluMLP,
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_dataset_config(key=None):
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    key = key or cfg["active"]
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def make_run_tag(description, sweep_id):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{ts}_{description}_{sweep_id[:8]}"


def setup_logging(run_tag):
    log_path = LOG_DIR / f"{run_tag}.log"
    logger = logging.getLogger("run_reconstruction")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(sh)
    return log_path, logger


def _fig_to_wandb_image(fig, caption=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return wandb.Image(Image.open(buf), caption=caption)


# ─── Registry ────────────────────────────────────────────────────────────────

def load_registry(sweep_id):
    with open(REGISTRY_FILE) as f:
        registry = json.load(f)
    entry = next((e for e in registry if e["sweep_id"].startswith(sweep_id)), None)
    if entry is None:
        raise ValueError(f"Sweep {sweep_id} not found in registry.")
    return registry, entry


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def get_used_indices(entry):
    used = set(entry.get("indices", []))
    used |= set(entry.get("validation", {}).get("holdout_indices", []))
    for key in entry:
        if key.startswith("comparison") or key.startswith("topk") or key.startswith("reconstruction"):
            used |= set(entry[key].get("indices", []))
    return used


def warm_init_weights(model):
    import torch
    with torch.no_grad():
        if hasattr(model, "final"):
            model.final.bias.zero_()
        elif hasattr(model, "net"):
            model.net[-1].bias.zero_()


# ─── INR runner ───────────────────────────────────────────────────────────────

def run_inr_config(sweep_cfg, dataset, indices, base_config, run_tag, log,
                   warm_init=False, wb_group=None, plot_dir=None):
    """Run one INR config on all indices. Handles no-GT gracefully."""
    has_gt = dataset.has_ground_truth

    rank    = sweep_cfg["rank"]
    method  = sweep_cfg["method"]
    mtype   = sweep_cfg["model_type"]
    hparams = sweep_cfg["hyperparams"]
    label   = f"INR_rank{rank}_{method}_{mtype}"
    wb_name = f"{run_tag}_rank{rank}_{method}_{mtype}"

    log.info(f"\n  ── rank#{rank}: {method} / {mtype} ──")

    cfg = copy.deepcopy(base_config)
    if hasattr(dataset, 'pix2time') and dataset.pix2time is not None:
        cfg.time_scale = 1.0 / dataset.pix2time
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
    cfg.loss_type       = hparams.get("loss_type",       cfg.loss_type)
    cfg.huber_delta     = hparams.get("huber_delta",     cfg.huber_delta)
    cfg.clamp_slowness  = hparams.get("clamp_slowness",  cfg.clamp_slowness)
    cfg.early_stopping  = hparams.get("early_stopping",  cfg.early_stopping)
    cfg.patience        = hparams.get("patience",        cfg.patience)
    if method == "Ray_Batching":
        cfg.epochs     = hparams.get("epochs",     cfg.epochs)
        cfg.batch_size = hparams.get("batch_size", cfg.batch_size)

    engine_fn = ENGINE_MAP[method]
    model_cls = MODEL_MAP[mtype]
    mae, rmse, ssim, cnr, recon_times, per_sample = [], [], [], [], [], []

    wandb.init(
        project=base_config.project_name,
        name=wb_name,
        group=wb_group or f"{run_tag}_recon",
        tags=["reconstruction", f"rank{rank}", method, mtype, f"n{len(indices)}",
              "warm_init" if warm_init else "cold_init",
              "no_gt" if not has_gt else "has_gt"],
        config={"rank": rank, "method": method, "model_type": mtype,
                "n_samples": len(indices), "warm_init": warm_init,
                "has_gt": has_gt, **hparams},
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
        elif mtype == "SirenMLP": kwargs["omega"] = cfg.omega
        model = model_cls(**kwargs)
        if warm_init:
            warm_init_weights(model)

        t0 = time.perf_counter()
        if method == "Bias_Absorption":
            bias_model = BiasMLP(
                in_features=2, 
                hidden_features=cfg.bias_hidden_features, 
                mapping_size=32, 
                scale=cfg.bias_scale
            )
            result = engine_fn(sample=sample, L_matrix=dataset.L_matrix,
                               model=model, bias_model=bias_model, 
                               label=mtype, config=cfg, use_wandb=False)
        else:
            result = engine_fn(sample=sample, L_matrix=dataset.L_matrix,
                               model=model, label=mtype, config=cfg, use_wandb=False)
        recon_s = time.perf_counter() - t0
        recon_times.append(recon_s)

        s_phys_np = (result["s_phys"].detach().cpu().numpy()
                     if hasattr(result["s_phys"], "detach")
                     else np.asarray(result["s_phys"]))

        per_entry = {
            "idx":          idx,
            "recon_time_s": recon_s,
            "s_phys_np":    s_phys_np.flatten().astype(np.float32),
        }

        wb_log = {
            "sample/recon_time_s": recon_s,
            "sample/final_loss":   result["loss_history"][-1] if result["loss_history"] else 0,
        }

        if has_gt:
            s_gt_np = (sample["s_gt_raw"].detach().cpu().numpy()
                       if hasattr(sample["s_gt_raw"], "detach")
                       else np.asarray(sample["s_gt_raw"]))
            m = calculate_metrics(s_phys_pred=s_phys_np,
                                  s_gt_raw=s_gt_np, grid_shape=(64, 64))
            mae.append(m["MAE"]); rmse.append(m["RMSE"])
            ssim.append(m["SSIM"]); cnr.append(m["CNR"])
            per_entry.update({"s_gt_np": s_gt_np.flatten().astype(np.float32), **m})
            wb_log.update({
                "sample/MAE": m["MAE"], "sample/RMSE": m["RMSE"],
                "sample/SSIM": m["SSIM"], "sample/CNR": m["CNR"],
            })
            log.info(f"           MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  "
                     f"SSIM={m['SSIM']:.3f}  CNR={m['CNR']:.3f}  time={recon_s:.0f}s")
        else:
            per_entry["s_gt_np"] = None
            log.info(f"           time={recon_s:.0f}s  (no GT)")

        per_sample.append(per_entry)
        wandb.log(wb_log, step=i)

        # Save per-sample plot
        if plot_dir:
            fig = _make_sample_plot(s_phys_np, s_gt_np if has_gt else None,
                                    result["loss_history"], idx, method, mtype,
                                    per_entry if has_gt else None)
            
            # Additional plot for the absorbed bias if applicable
            if "epsilon_bias" in result:
                _plot_absorbed_bias(result["epsilon_bias"], idx, method, mtype, plot_dir)

            fp = plot_dir / f"rank{rank}_{method}_{mtype}_idx{idx}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Aggregate
    agg = {
        "method":          label,
        "recon_time_mean": float(np.mean(recon_times)),
        "recon_time_std":  float(np.std(recon_times)),
        "n_samples":       len(indices),
        "per_sample":      per_sample,
    }
    final_log = {
        "final/recon_time_mean": agg["recon_time_mean"],
        "final/recon_time_std":  agg["recon_time_std"],
    }
    if has_gt and mae:
        agg.update({
            "MAE_mean":  float(np.mean(mae)),  "MAE_std":  float(np.std(mae)),
            "RMSE_mean": float(np.mean(rmse)), "RMSE_std": float(np.std(rmse)),
            "SSIM_mean": float(np.mean(ssim)), "SSIM_std": float(np.std(ssim)),
            "CNR_mean":  float(np.mean(cnr)),  "CNR_std":  float(np.std(cnr)),
        })
        final_log.update({
            "final/MAE_mean":  agg["MAE_mean"],
            "final/RMSE_mean": agg["RMSE_mean"],
            "final/SSIM_mean": agg["SSIM_mean"],
            "final/CNR_mean":  agg["CNR_mean"],
        })
    else:
        for k in ("MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
                   "SSIM_mean", "SSIM_std", "CNR_mean", "CNR_std"):
            agg[k] = None

    wandb.log(final_log)
    wandb.finish()

    # Terminal summary
    if has_gt:
        log.info(f"  {label:<42} "
                 f"MAE={agg['MAE_mean']:.3f}±{agg['MAE_std']:.3f}  "
                 f"RMSE={agg['RMSE_mean']:.3f}±{agg['RMSE_std']:.3f}  "
                 f"SSIM={agg['SSIM_mean']:.3f}±{agg['SSIM_std']:.3f}  "
                 f"Time={agg['recon_time_mean']:.0f}±{agg['recon_time_std']:.0f}s")
    else:
        log.info(f"  {label:<42} "
                 f"Time={agg['recon_time_mean']:.0f}±{agg['recon_time_std']:.0f}s  (no GT)")
    return agg


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _make_sample_plot(s_phys_np, s_gt_np, loss_history, idx, method, mtype,
                      metrics=None):
    """Per-sample plot. With GT: 4 panels. Without GT: 2 panels."""
    v_rec = np.clip(1.0 / (s_phys_np.reshape(64, 64) + 1e-8), 1200, 1800)
    has_gt = s_gt_np is not None

    n_panels = 4 if has_gt else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))

    title = f"{method}/{mtype} — Sample {idx}"
    if metrics:
        title += (f"  |  MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
                  f"SSIM={metrics['SSIM']:.4f}  CNR={metrics['CNR']:.3f}")
    fig.suptitle(title, fontsize=12)

    col = 0
    if has_gt:
        v_gt = np.clip(1.0 / (s_gt_np.reshape(64, 64) + 1e-8), 1200, 1800)
        im0 = axes[col].imshow(v_gt, cmap="jet", vmin=1400, vmax=1600)
        axes[col].set_title("Ground Truth (m/s)"); axes[col].axis("off")
        plt.colorbar(im0, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    im1 = axes[col].imshow(v_rec, cmap="jet", vmin=1400, vmax=1600)
    axes[col].set_title("Reconstruction (m/s)"); axes[col].axis("off")
    plt.colorbar(im1, ax=axes[col], fraction=0.046, pad=0.04)
    col += 1

    if has_gt:
        err = np.abs(v_gt - v_rec)
        im2 = axes[col].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[col].set_title("Abs. Error (m/s)"); axes[col].axis("off")
        plt.colorbar(im2, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

    axes[col].plot(loss_history, color="blue")
    axes[col].set_title("Optimization Loss")
    axes[col].set_xlabel("Iterations"); axes[col].set_ylabel("Loss")
    axes[col].set_yscale("log")
    axes[col].grid(True, which="both", ls="-", alpha=0.5)

    plt.tight_layout()
    return fig


def _plot_absorbed_bias(epsilon_bias, idx, method, mtype, plot_dir):
    """Save a visualization of the bias field absorbed by the secondary INR."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    eps_np = epsilon_bias.detach().cpu().numpy()
    n_pairs = 8
    pair_size = len(eps_np) // n_pairs
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Absorbed Bias Field - {method}/{mtype} - Sample {idx}", fontsize=16)
    
    for p in range(n_pairs):
        ax = axes[p // 4, p % 4]
        pair_eps = eps_np[p * pair_size : (p + 1) * pair_size].reshape(128, 128, order='F')
        im = ax.imshow(pair_eps, cmap='RdBu_r')
        ax.set_title(f"Firing Pair {p}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"bias_absorbed_idx{idx}.png", dpi=150)
    plt.close(fig)


def _make_recon_grid(all_results, n_vis=2, rng_seed=42):
    """Grid of reconstructions across all configs. Works with or without GT."""
    has_gt = all_results[0]["per_sample"][0].get("s_gt_np") is not None

    common_indices = set(s["idx"] for s in all_results[0]["per_sample"])
    for r in all_results[1:]:
        common_indices &= {s["idx"] for s in r["per_sample"]}
    common_indices = sorted(common_indices)

    rng = np.random.default_rng(seed=rng_seed)
    n_pick = min(n_vis, len(common_indices))
    vis_idxs = rng.choice(common_indices, size=n_pick, replace=False).tolist()

    n_cols = (1 if has_gt else 0) + len(all_results)
    n_rows = n_pick * (2 if has_gt else 1)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.9 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    fig.suptitle("Reconstruction Grid", fontsize=13, fontweight="bold", y=1.01)

    for row_pair, vis_idx in enumerate(vis_idxs):
        if has_gt:
            row_sos = row_pair * 2
            row_err = row_pair * 2 + 1
            # GT column
            gt_entry = next(s for s in all_results[0]["per_sample"]
                            if s["idx"] == vis_idx)
            s_gt_np = gt_entry["s_gt_np"].flatten().astype(np.float32)
            v_gt = np.clip(1.0 / (s_gt_np + 1e-8), 1400, 1600).reshape(64, 64)

            ax_gt = axes[row_sos, 0]
            im_gt = ax_gt.imshow(v_gt, cmap="jet", vmin=1400, vmax=1600,
                                 interpolation="nearest")
            ax_gt.axis("off")
            if row_pair == 0:
                ax_gt.set_title("GT", fontsize=9, fontweight="bold")
            plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
            axes[row_err, 0].axis("off")
            axes[row_err, 0].text(0.5, 0.5, f"idx {vis_idx}",
                                  ha="center", va="center", fontsize=9,
                                  transform=axes[row_err, 0].transAxes)
        else:
            row_sos = row_pair

        # Method columns
        col_offset = 1 if has_gt else 0
        for col, result in enumerate(all_results, start=col_offset):
            entry = next(s for s in result["per_sample"] if s["idx"] == vis_idx)
            s_phys = entry["s_phys_np"].flatten().astype(np.float32)
            v_rec = np.clip(1.0 / (s_phys + 1e-8), 1400, 1600).reshape(64, 64)

            ax_sos = axes[row_sos, col]
            im = ax_sos.imshow(v_rec, cmap="jet", vmin=1400, vmax=1600,
                               interpolation="nearest")
            ax_sos.axis("off")
            if row_pair == 0:
                parts = result["method"].replace("INR_", "").split("_")
                ax_sos.set_title(parts[0], fontsize=9, fontweight="bold")
            plt.colorbar(im, ax=ax_sos, fraction=0.046, pad=0.04)

            if has_gt:
                err_map = np.abs(v_gt - v_rec)
                ax_err = axes[row_err, col]
                im_e = ax_err.imshow(err_map, cmap="hot", vmin=0, vmax=50,
                                     interpolation="nearest")
                ax_err.axis("off")
                mae_val = float(np.mean(err_map))
                ax_err.set_title(f"MAE={mae_val:.1f}", fontsize=8, color="#444")
                plt.colorbar(im_e, ax=ax_err, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ─── Table / Summary ─────────────────────────────────────────────────────────

def print_table(all_results, has_gt, log):
    n = all_results[0]["n_samples"]
    log.info(f"\n{'═'*90}")
    log.info(f"  RECONSTRUCTION RESULTS — {n} samples")
    log.info(f"{'═'*90}")
    if has_gt:
        log.info(f"  {'Method':<42} {'MAE±std':>13} {'RMSE±std':>13} "
                 f"{'SSIM±std':>13} {'Time(s)':>14}")
        log.info(f"  {'─'*88}")
        for r in sorted(all_results, key=lambda x: x.get("MAE_mean") or 999):
            t = f"{r['recon_time_mean']:>6.0f}±{r['recon_time_std']:<5.0f}"
            log.info(
                f"  {r['method']:<42}"
                f"  {r['MAE_mean']:>5.3f}±{r['MAE_std']:<5.3f}"
                f"  {r['RMSE_mean']:>5.3f}±{r['RMSE_std']:<5.3f}"
                f"  {r['SSIM_mean']:>5.3f}±{r['SSIM_std']:<5.3f}"
                f"  {t}"
            )
    else:
        log.info(f"  {'Method':<42} {'Time(s)':>14}")
        log.info(f"  {'─'*60}")
        for r in all_results:
            t = f"{r['recon_time_mean']:>6.0f}±{r['recon_time_std']:<5.0f}"
            log.info(f"  {r['method']:<42}  {t}")
    log.info(f"{'═'*90}")


def log_wandb_summary(all_results, entry, indices, run_tag, sweep_id, log,
                      has_gt, wb_group=None, plot_dir=None):
    wb_name = f"{run_tag}_SUMMARY"
    log.info(f"\n  Logging summary → W&B '{wb_name}'")

    wandb.init(
        project=entry["project"],
        entity=entry["entity"],
        name=wb_name,
        group=wb_group or f"{run_tag}_recon",
        tags=["reconstruction", "summary", f"n{len(indices)}",
              "no_gt" if not has_gt else "has_gt"],
        config={"sweep_id": sweep_id, "n_samples": len(indices),
                "indices": indices, "has_gt": has_gt},
    )

    # Comparison table
    if has_gt:
        columns = ["Method", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
                    "SSIM_mean", "SSIM_std", "CNR_mean", "CNR_std",
                    "Time_mean_s", "Time_std_s"]
        data = [[r["method"], r["MAE_mean"], r["MAE_std"],
                 r["RMSE_mean"], r["RMSE_std"], r["SSIM_mean"], r["SSIM_std"],
                 r["CNR_mean"], r["CNR_std"],
                 r["recon_time_mean"], r["recon_time_std"]]
                for r in sorted(all_results, key=lambda x: x.get("MAE_mean") or 999)]
    else:
        columns = ["Method", "Time_mean_s", "Time_std_s"]
        data = [[r["method"], r["recon_time_mean"], r["recon_time_std"]]
                for r in all_results]
    wandb.log({"comparison_table": wandb.Table(columns=columns, data=data)})

    # Reconstruction grid
    out_dir = plot_dir or Path("plots") / sweep_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("  Rendering reconstruction grid ...")
    fig_grid = _make_recon_grid(all_results, n_vis=2)
    wandb.log({"reconstruction_grid": _fig_to_wandb_image(
        fig_grid, "Reconstructions — 2 random samples")})
    pdf_grid = out_dir / f"{run_tag}_recon_grid.pdf"
    fig_grid.savefig(pdf_grid, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    log.info(f"  Grid PDF → {pdf_grid}")

    wandb.finish()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct SoS using top-K sweep configs"
    )
    parser.add_argument("--dataset", default=None,
                        help="Dataset key from datasets.yaml")
    parser.add_argument("--sweep_id", required=True)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--n_samples", default=None, type=int)
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--warm_init", action="store_true")
    parser.add_argument("--top_k_per_model", default=None, type=int)
    parser.add_argument("--job_name", default=None)

    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────
    ds_cfg    = load_dataset_config(args.dataset)
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True
    if ds_cfg.get("h5_keys"):
        ds_kwargs["h5_keys"] = ds_cfg["h5_keys"]
    dataset = USDataset(ds_cfg["data_path"], grid_path, **ds_kwargs)
    has_gt = dataset.has_ground_truth

    # ── Resolve n_samples ─────────────────────────────────────────────────
    n_samples = args.n_samples if args.n_samples is not None else len(dataset)

    init_tag = "warm" if args.warm_init else "cold"
    sel_tag  = (f"top{args.top_k_per_model}permodel"
                if args.top_k_per_model else f"topk{args.top_k}")
    run_tag  = make_run_tag(f"recon_{sel_tag}_fresh{n_samples}_{init_tag}",
                            args.sweep_id)
    wb_group = args.job_name or run_tag
    log_path, log = setup_logging(run_tag)

    log.info("=" * 70)
    log.info(f"  Reconstruction  |  {run_tag}")
    log.info(f"  Sweep    : {args.sweep_id}")
    log.info(f"  Dataset  : {ds_cfg['name']}  ({ds_cfg['key']})")
    log.info(f"  GT       : {'yes' if has_gt else 'NO — reconstruction only'}")
    log.info(f"  Samples  : {n_samples} / {len(dataset)} available")
    log.info("=" * 70)

    registry, entry = load_registry(args.sweep_id)

    # ── Plot dir per sweep ────────────────────────────────────────────────
    plot_dir = SCRIPTS_DIR / "plots" / args.sweep_id
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample indices ────────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        used = get_used_indices(entry)
        log.info(f"Previously used: {len(used)} indices — excluded")
        rng  = np.random.default_rng(seed=3141)
        pool = [i for i in range(len(dataset)) if i not in used]
        if n_samples > len(pool):
            log.warning(f"Requested {n_samples} but only {len(pool)} available. "
                        f"Using all.")
            n_samples = len(pool)
        indices = rng.choice(pool, size=n_samples, replace=False).tolist()
    log.info(f"Indices ({len(indices)}): {indices}")

    # ── Resolve time_scale ────────────────────────────────────────────────
    if hasattr(dataset, 'pix2time') and dataset.pix2time is not None:
        resolved_ts = 1.0 / dataset.pix2time
        log.info(f"time_scale from dataset pix2time: {resolved_ts:.4e}")
    elif ds_cfg.get("pix2time") is not None:
        resolved_ts = 1.0 / float(ds_cfg["pix2time"])
        log.info(f"time_scale from yaml pix2time: {resolved_ts:.4e}")
    else:
        resolved_ts = 1e6
        log.info(f"time_scale using default: {resolved_ts:.4e}")

    base_config = ExperimentConfig(
        project_name=entry["project"],
        time_scale=resolved_ts,
    )

    # ── Fetch configs from W&B ────────────────────────────────────────────
    api   = wandb.Api()
    sweep = api.sweep(
        f"{entry['entity']}/{entry['project']}/{entry['sweep_id']}"
    )
    # Prefer MAE_mean (GT datasets), fallback to loss_mean (no-GT sweeps)
    runs_with_mae = [r for r in sweep.runs if "MAE_mean" in r.summary]
    if runs_with_mae:
        completed = sorted(runs_with_mae,
                           key=lambda r: r.summary["MAE_mean"])
    else:
        runs_with_loss = [r for r in sweep.runs if "loss_mean" in r.summary]
        completed = sorted(runs_with_loss,
                           key=lambda r: r.summary["loss_mean"])
        log.info("  No MAE in sweep runs — sorting by loss_mean")

    sort_key = "MAE_mean" if runs_with_mae else "loss_mean"

    if args.top_k_per_model:
        k = args.top_k_per_model
        model_types = ["ReluMLP", "FourierMLP", "SirenMLP", "GeluMLP"]
        selected = []
        for mtype in model_types:
            mpool = [r for r in completed
                     if r.config.get("model_type") == mtype]
            if not mpool:
                log.warning(f"  No completed runs for {mtype}")
                continue
            selected.extend(mpool[:k])
            best_val = mpool[0].summary.get(sort_key, "?")
            log.info(f"  {mtype}: {len(mpool)} completed → top-{min(k, len(mpool))} "
                     f"(best {sort_key}={best_val})")
        selected = sorted(selected,
                          key=lambda r: r.summary.get(sort_key, 1e9))
    else:
        selected = completed[:args.top_k]
        log.info(f"\nGlobal top-{args.top_k} sweep configs:")

    top_k_cfgs = []
    for rank, run in enumerate(selected, 1):
        cfg = {
            "rank":       rank,
            "method":     run.config.get("method"),
            "model_type": run.config.get("model_type"),
            "hyperparams": {k: v for k, v in run.config.items()
                            if k not in {"method", "model_type", "_wandb"}},
        }
        top_k_cfgs.append(cfg)
        score = run.summary.get(sort_key, "?")
        log.info(f"  rank#{rank}: {cfg['method']:<20} / {cfg['model_type']:<12} "
                 f"sweep_{sort_key}={score}")

    # ── Run each INR config ───────────────────────────────────────────────
    all_results = []
    t_start = time.time()
    for cfg in top_k_cfgs:
        t0 = time.time()
        result = run_inr_config(cfg, dataset, indices, base_config, run_tag,
                                log, warm_init=args.warm_init,
                                wb_group=wb_group, plot_dir=plot_dir)
        log.info(f"  rank#{cfg['rank']} wall time: "
                 f"{(time.time() - t0) / 60:.1f} min")
        all_results.append(result)

    total_hrs = (time.time() - t_start) / 3600
    log.info(f"\nAll configs done in {total_hrs:.2f} hours")

    # ── Final table ───────────────────────────────────────────────────────
    print_table(all_results, has_gt, log)

    # ── W&B summary ───────────────────────────────────────────────────────
    if not args.no_wandb:
        log_wandb_summary(all_results, entry, indices, run_tag,
                          args.sweep_id, log, has_gt=has_gt,
                          wb_group=wb_group, plot_dir=plot_dir)

    # ── Registry ──────────────────────────────────────────────────────────
    slim_results = [{k: v for k, v in r.items() if k != "per_sample"}
                    for r in all_results]
    for e in registry:
        if e["sweep_id"].startswith(args.sweep_id):
            e["reconstruction"] = {
                "run_tag":     run_tag,
                "dataset":     ds_cfg["key"],
                "has_gt":      has_gt,
                "ran_at":      datetime.now().isoformat(),
                "top_k":       args.top_k,
                "n_samples":   len(indices),
                "indices":     indices,
                "elapsed_hrs": round(total_hrs, 2),
                "results":     slim_results,
            }
    save_registry(registry)
    log.info(f"\n  Registry updated  |  Log: {log_path}")
    log.info(f"  Plots: {plot_dir}")


if __name__ == "__main__":
    main()
