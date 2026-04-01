#!/usr/bin/env python3
"""
run_exp9_cnr_improvement.py
---------------------------
Experiment 9: CNR Improvement via TV Regularization.

Sweeps tv_weight on kwave_geom (32 samples) using a fixed good INR config.
Compares against L1/L2 analytical baselines and no-TV INR baseline.

Usage:
    # Quick test (2 samples)
    python scripts/run_exp9_cnr_improvement.py --n_samples 2

    # Full run (all 32 samples)
    python scripts/run_exp9_cnr_improvement.py

    # Custom TV weights
    python scripts/run_exp9_cnr_improvement.py --tv_weights 0 1e-4 1e-3 1e-2

    # No W&B logging
    python scripts/run_exp9_cnr_improvement.py --no_wandb
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

from inr_sos import DATA_DIR
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.models.mlp import ReluMLP
from inr_sos.training.engines import optimize_full_forward_operator

SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR / "data" / "experiment9_cnr"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("exp9")


# ─── Fixed INR config (known good for kwave_geom) ────────────────────────────

def make_base_config():
    return ExperimentConfig(
        project_name="INR-SoS-Recon",
        experiment_group="Exp9-CNR-Improvement",
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
        reg_weight=0.0,
        tv_weight=0.0,  # will be overridden per run
    )


def build_model(cfg):
    return ReluMLP(
        in_features=cfg.in_features,
        hidden_features=cfg.hidden_features,
        hidden_layers=cfg.hidden_layers,
        mapping_size=cfg.mapping_size,
    )


# ─── Evaluation helpers ──────────────────────────────────────────────────────

def eval_baselines(dataset, indices):
    """Compute metrics for L1 and L2 baselines on given indices."""
    results = {"L1": [], "L2": []}

    for idx in indices:
        sample = dataset[idx]
        s_gt = sample["s_gt_raw"]

        if "s_l1_recon" in sample:
            m = calculate_metrics(sample["s_l1_recon"], s_gt)
            results["L1"].append(m)
        if "s_l2_recon" in sample:
            m = calculate_metrics(sample["s_l2_recon"], s_gt)
            results["L2"].append(m)

    agg = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            continue
        agg[name] = {
            k: {
                "mean": float(np.mean([m[k] for m in metrics_list])),
                "std": float(np.std([m[k] for m in metrics_list])),
            }
            for k in ["MAE", "RMSE", "SSIM", "CNR"]
        }
    return agg


def run_tv_sweep(dataset, indices, tv_weight, base_config, use_wandb=False):
    """Run INR with a specific tv_weight on all indices. Return per-sample metrics."""
    cfg = copy.deepcopy(base_config)
    cfg.tv_weight = tv_weight

    # Set time_scale from dataset
    if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
        cfg.time_scale = 1.0 / dataset.pix2time

    all_metrics = []
    all_reconstructions = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        model = build_model(cfg)

        t0 = time.perf_counter()
        result = optimize_full_forward_operator(
            sample=sample,
            L_matrix=dataset.L_matrix,
            model=model,
            label=f"ReluMLP_tv{tv_weight:.0e}",
            config=cfg,
            use_wandb=False,
        )
        elapsed = time.perf_counter() - t0

        s_gt = sample["s_gt_raw"]
        m = calculate_metrics(result["s_phys"], s_gt)
        m["idx"] = idx
        m["time_s"] = elapsed
        all_metrics.append(m)
        all_reconstructions.append({
            "idx": idx,
            "s_phys": result["s_phys"].detach().cpu().numpy().flatten(),
            "s_gt": s_gt.detach().cpu().numpy().flatten() if hasattr(s_gt, "detach") else np.asarray(s_gt).flatten(),
            "loss_history": result["loss_history"],
        })

        log.info(
            f"  tv={tv_weight:.0e}  [{i+1:>2}/{len(indices)}]  idx={idx}  "
            f"MAE={m['MAE']:.2f}  CNR={m['CNR']:.3f}  SSIM={m['SSIM']:.3f}  "
            f"time={elapsed:.0f}s"
        )

    agg = {
        k: {
            "mean": float(np.mean([m[k] for m in all_metrics])),
            "std": float(np.std([m[k] for m in all_metrics])),
        }
        for k in ["MAE", "RMSE", "SSIM", "CNR"]
    }
    agg["tv_weight"] = tv_weight
    agg["per_sample"] = all_metrics
    agg["reconstructions"] = all_reconstructions

    return agg


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison_table(baseline_agg, tv_results, out_dir):
    """Bar chart comparing CNR, MAE, SSIM across methods."""
    methods = []
    cnr_means, cnr_stds = [], []
    mae_means, mae_stds = [], []
    ssim_means, ssim_stds = [], []

    # Baselines
    for name in ["L1", "L2"]:
        if name in baseline_agg:
            methods.append(name)
            cnr_means.append(baseline_agg[name]["CNR"]["mean"])
            cnr_stds.append(baseline_agg[name]["CNR"]["std"])
            mae_means.append(baseline_agg[name]["MAE"]["mean"])
            mae_stds.append(baseline_agg[name]["MAE"]["std"])
            ssim_means.append(baseline_agg[name]["SSIM"]["mean"])
            ssim_stds.append(baseline_agg[name]["SSIM"]["std"])

    # TV sweep results
    for r in tv_results:
        tv = r["tv_weight"]
        label = f"INR (tv=0)" if tv == 0 else f"INR (tv={tv:.0e})"
        methods.append(label)
        cnr_means.append(r["CNR"]["mean"])
        cnr_stds.append(r["CNR"]["std"])
        mae_means.append(r["MAE"]["mean"])
        mae_stds.append(r["MAE"]["std"])
        ssim_means.append(r["SSIM"]["mean"])
        ssim_stds.append(r["SSIM"]["std"])

    x = np.arange(len(methods))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Exp 9: CNR Improvement via TV Regularization — kwave_geom", fontsize=14)

    # CNR
    axes[0].bar(x, cnr_means, width, yerr=cnr_stds, capsize=4, color="steelblue")
    axes[0].set_ylabel("CNR")
    axes[0].set_title("Contrast-to-Noise Ratio (higher is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # MAE
    axes[1].bar(x, mae_means, width, yerr=mae_stds, capsize=4, color="coral")
    axes[1].set_ylabel("MAE (m/s)")
    axes[1].set_title("Mean Absolute Error (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # SSIM
    axes[2].bar(x, ssim_means, width, yerr=ssim_stds, capsize=4, color="mediumseagreen")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Structural Similarity (higher is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fp = out_dir / "exp9_comparison_bars.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Bar chart saved → {fp}")
    return fp


def _slowness_to_sos(s_flat):
    """Convert flat slowness array to 64x64 SoS image (m/s), clamped."""
    if hasattr(s_flat, "detach"):
        s_flat = s_flat.detach().cpu().numpy()
    s_flat = np.asarray(s_flat).flatten()
    slowness_min = 1.0 / 1800.0
    slowness_max = 1.0 / 1200.0
    s_clamped = np.clip(s_flat, slowness_min, slowness_max)
    return (1.0 / s_clamped).reshape(64, 64)


def plot_sample_comparison(tv_results, baseline_agg, dataset, sample_idx, out_dir):
    """Visual comparison of reconstructions for one sample across TV weights.
    Every panel shows both MAE and CNR."""
    from inr_sos.evaluation.metrics import calculate_metrics

    sample = dataset[sample_idx]
    s_gt_raw = sample["s_gt_raw"]
    v_gt = _slowness_to_sos(s_gt_raw)

    # Collect: (image, title_line1, mae, cnr)
    panels = []

    # GT
    panels.append({"img": v_gt, "label": "Ground Truth", "mae": 0.0, "cnr": None})

    # L1
    if "s_l1_recon" in sample:
        v_l1 = _slowness_to_sos(sample["s_l1_recon"])
        m = calculate_metrics(sample["s_l1_recon"], s_gt_raw)
        panels.append({"img": v_l1, "label": "L1 (LASSO)", "mae": m["MAE"], "cnr": m["CNR"]})

    # L2
    if "s_l2_recon" in sample:
        v_l2 = _slowness_to_sos(sample["s_l2_recon"])
        m = calculate_metrics(sample["s_l2_recon"], s_gt_raw)
        panels.append({"img": v_l2, "label": "L2 (Tikhonov)", "mae": m["MAE"], "cnr": m["CNR"]})

    # INR variants
    for r in tv_results:
        tv = r["tv_weight"]
        entry = next((e for e in r["reconstructions"] if e["idx"] == sample_idx), None)
        if entry is None:
            continue
        v_rec = _slowness_to_sos(entry["s_phys"])
        m = next(m for m in r["per_sample"] if m["idx"] == sample_idx)
        label = f"INR (no TV)" if tv == 0 else f"INR tv={tv:.0e}"
        panels.append({"img": v_rec, "label": label, "mae": m["MAE"], "cnr": m["CNR"]})

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    fig.suptitle(f"Sample {sample_idx} — Reconstruction Comparison", fontsize=13)

    for i, p in enumerate(panels):
        img = p["img"]
        # Title: label + MAE + CNR
        if p["cnr"] is not None:
            title = f"{p['label']}\nMAE={p['mae']:.1f}  CNR={p['cnr']:.2f}"
        else:
            title = p["label"]

        # SoS image
        im = axes[0, i].imshow(img, cmap="jet", vmin=1400, vmax=1600)
        axes[0, i].set_title(title, fontsize=9)
        axes[0, i].axis("off")
        plt.colorbar(im, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Error map
        err = np.abs(v_gt - img)
        im_e = axes[1, i].imshow(err, cmap="hot", vmin=0, vmax=50)
        axes[1, i].set_title(f"Error (MAE={np.mean(err):.1f})", fontsize=9)
        axes[1, i].axis("off")
        plt.colorbar(im_e, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fp = out_dir / f"exp9_sample_{sample_idx}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp 9: CNR improvement via TV")
    parser.add_argument("--dataset", default="kwave_geom")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--tv_weights", nargs="+", type=float,
                        default=[0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--n_vis", type=int, default=3,
                        help="Number of samples to visualize")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / args.dataset / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"  Experiment 9: CNR Improvement via TV Regularization")
    log.info(f"  Dataset: {args.dataset}")
    log.info(f"  TV weights: {args.tv_weights}")
    log.info(f"  Output: {run_dir}")
    log.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────
    import yaml
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
    log.info(f"  Loaded {len(dataset)} samples")

    # ── Sample indices ────────────────────────────────────────────────────
    if args.indices:
        indices = args.indices
    else:
        n = args.n_samples if args.n_samples is not None else len(dataset)
        indices = list(range(min(n, len(dataset))))
    log.info(f"  Using {len(indices)} samples: {indices}")

    # ── Baselines ─────────────────────────────────────────────────────────
    log.info("\n── Evaluating L1/L2 baselines ──")
    baseline_agg = eval_baselines(dataset, indices)
    for name, metrics in baseline_agg.items():
        log.info(
            f"  {name:<6}  MAE={metrics['MAE']['mean']:.2f}±{metrics['MAE']['std']:.2f}  "
            f"CNR={metrics['CNR']['mean']:.3f}±{metrics['CNR']['std']:.3f}  "
            f"SSIM={metrics['SSIM']['mean']:.3f}±{metrics['SSIM']['std']:.3f}"
        )

    # ── TV weight sweep ───────────────────────────────────────────────────
    base_config = make_base_config()
    tv_results = []

    for tv_w in args.tv_weights:
        log.info(f"\n── TV weight = {tv_w:.0e} ──")
        t0 = time.time()
        result = run_tv_sweep(dataset, indices, tv_w, base_config)
        elapsed_min = (time.time() - t0) / 60
        log.info(
            f"  AGGREGATE: MAE={result['MAE']['mean']:.2f}±{result['MAE']['std']:.2f}  "
            f"CNR={result['CNR']['mean']:.3f}±{result['CNR']['std']:.3f}  "
            f"SSIM={result['SSIM']['mean']:.3f}±{result['SSIM']['std']:.3f}  "
            f"({elapsed_min:.1f} min)"
        )
        tv_results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    log.info(f"\n{'='*90}")
    log.info(f"  EXPERIMENT 9 RESULTS — {len(indices)} samples on {args.dataset}")
    log.info(f"{'='*90}")
    log.info(f"  {'Method':<25} {'MAE±std':>14} {'RMSE±std':>14} {'SSIM±std':>14} {'CNR±std':>14}")
    log.info(f"  {'─'*85}")

    for name in ["L1", "L2"]:
        if name not in baseline_agg:
            continue
        m = baseline_agg[name]
        log.info(
            f"  {name:<25}"
            f"  {m['MAE']['mean']:>6.2f}±{m['MAE']['std']:<5.2f}"
            f"  {m['RMSE']['mean']:>6.2f}±{m['RMSE']['std']:<5.2f}"
            f"  {m['SSIM']['mean']:>6.3f}±{m['SSIM']['std']:<5.3f}"
            f"  {m['CNR']['mean']:>6.3f}±{m['CNR']['std']:<5.3f}"
        )

    for r in tv_results:
        tv = r["tv_weight"]
        label = f"INR (tv=0)" if tv == 0 else f"INR (tv={tv:.0e})"
        log.info(
            f"  {label:<25}"
            f"  {r['MAE']['mean']:>6.2f}±{r['MAE']['std']:<5.2f}"
            f"  {r['RMSE']['mean']:>6.2f}±{r['RMSE']['std']:<5.2f}"
            f"  {r['SSIM']['mean']:>6.3f}±{r['SSIM']['std']:<5.3f}"
            f"  {r['CNR']['mean']:>6.3f}±{r['CNR']['std']:<5.3f}"
        )
    log.info(f"{'='*90}")

    # ── Save results ──────────────────────────────────────────────────────
    # Strip large arrays before saving JSON
    results_json = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "n_samples": len(indices),
        "indices": indices,
        "baselines": baseline_agg,
        "tv_sweep": [
            {
                "tv_weight": r["tv_weight"],
                "MAE": r["MAE"],
                "RMSE": r["RMSE"],
                "SSIM": r["SSIM"],
                "CNR": r["CNR"],
                "per_sample": r["per_sample"],
            }
            for r in tv_results
        ],
    }
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"\n  Results saved → {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("\n── Generating plots ──")
    plot_comparison_table(baseline_agg, tv_results, run_dir)

    # Visual comparison for a few samples
    vis_indices = indices[:args.n_vis]
    for idx in vis_indices:
        fp = plot_sample_comparison(tv_results, baseline_agg, dataset, idx, run_dir)
        log.info(f"  Sample plot saved → {fp}")

    # ── W&B logging ───────────────────────────────────────────────────────
    if not args.no_wandb:
        import wandb

        wandb.init(
            project="INR-SoS-Recon",
            name=f"exp9_tv_sweep_{timestamp}",
            group="Exp9-CNR-Improvement",
            tags=["exp9", "tv_sweep", args.dataset],
            config={
                "experiment": "Exp9-CNR-Improvement",
                "dataset": args.dataset,
                "n_samples": len(indices),
                "tv_weights": args.tv_weights,
                "base_config": base_config.to_dict(),
            },
        )

        # Summary table
        columns = ["Method", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
                    "SSIM_mean", "SSIM_std", "CNR_mean", "CNR_std"]
        data = []
        for name in ["L1", "L2"]:
            if name in baseline_agg:
                m = baseline_agg[name]
                data.append([name, m["MAE"]["mean"], m["MAE"]["std"],
                             m["RMSE"]["mean"], m["RMSE"]["std"],
                             m["SSIM"]["mean"], m["SSIM"]["std"],
                             m["CNR"]["mean"], m["CNR"]["std"]])
        for r in tv_results:
            tv = r["tv_weight"]
            label = f"INR_tv={tv:.0e}"
            data.append([label, r["MAE"]["mean"], r["MAE"]["std"],
                         r["RMSE"]["mean"], r["RMSE"]["std"],
                         r["SSIM"]["mean"], r["SSIM"]["std"],
                         r["CNR"]["mean"], r["CNR"]["std"]])
        wandb.log({"comparison_table": wandb.Table(columns=columns, data=data)})

        # Log plots
        bar_path = run_dir / "exp9_comparison_bars.png"
        if bar_path.exists():
            wandb.log({"comparison_bars": wandb.Image(str(bar_path))})
        for idx in vis_indices:
            fp = run_dir / f"exp9_sample_{idx}.png"
            if fp.exists():
                wandb.log({f"sample_{idx}": wandb.Image(str(fp))})

        wandb.finish()

    log.info(f"\n  Experiment 9 complete. All outputs in {run_dir}")


if __name__ == "__main__":
    main()
