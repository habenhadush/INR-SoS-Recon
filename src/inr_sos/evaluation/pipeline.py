import numpy as np
import wandb
import copy
import logging
import sys

from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.utils.config import ExperimentConfig
from inr_sos.utils.tracker import save_artifacts, log_to_local_database

from inr_sos.models.mlp import FourierMLP, ReluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import (
    optimize_full_forward_operator,
    optimize_sequential_views,
    optimize_stochastic_ray_batching,
)

_DEFAULT_ENGINES = {
    "Full_Matrix":      optimize_full_forward_operator,
    "Sequential_SGD":   optimize_sequential_views,
    "Ray_Batching":     optimize_stochastic_ray_batching,
}

_DEFAULT_MODELS = {
    "ReluMLP":   ReluMLP,
    "FourierMLP": FourierMLP,
    "SirenMLP":  SirenMLP,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def _build_model(model_cls, model_name: str, cfg: ExperimentConfig):
    """Instantiate the correct model from its name and config."""
    if model_name == "FourierMLP":
        return model_cls(
            in_features=cfg.in_features,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            mapping_size=cfg.mapping_size,
            scale=cfg.scale,
        )
    elif model_name == "SirenMLP":
        return model_cls(
            in_features=cfg.in_features,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            mapping_size=cfg.mapping_size,
            omega=cfg.omega,
        )
    elif model_name == "ReluMLP":
        return model_cls(
            in_features=cfg.in_features,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            mapping_size=cfg.mapping_size,
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")


def run_grid_comparison(
    dataset,
    target_indices: list,
    base_config: ExperimentConfig,
    engines: dict = None,
    models: dict = None,
    use_wandb: bool = True,
):
    """
    Run every (method, model, sample) combination and log results to W&B.

    W&B structure
    -------------
    • One *per-sample* run per (method, model, sample):
        group  = "<method>/<model>"   ← lets you filter by either axis
        name   = "<method>_<model>_Sample_<idx>"
        Logs:  step-level loss + final metrics + reconstruction image

    • One *master summary* run at the end:
        group  = "MASTER_SUMMARY"
        Logs:  flat table  [Method | Model | Sample_Idx | MAE | RMSE | SSIM | CNR | Image]
               per-(method,model) aggregate bar charts via grouped keys

    Returns
    -------
    results : dict  {(method_name, model_name): {"MAE": [...], "RMSE": [...], ...}}
    """
    if engines is None:
        engines = _DEFAULT_ENGINES
    if models is None:
        models = _DEFAULT_MODELS

    # ── Storage ──────────────────────────────────────────────────────────────
    # results[(method, model)] = {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}
    results = {
        (m, n): {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}
        for m in engines for n in models
    }
    # Rows for the master W&B table — collected across all runs
    master_rows = []   # list of dicts

    total_combos  = len(engines) * len(models)
    total_samples = len(target_indices)
    logging.info(
        f"Grid: {len(engines)} methods × {len(models)} models × "
        f"{total_samples} samples = {total_combos * total_samples} runs"
    )

    # ── Main grid loop ───────────────────────────────────────────────────────
    for method_name, engine_fn in engines.items():
        for model_name, model_cls in models.items():

            combo_tag = f"{method_name}/{model_name}"
            logging.info(f"\n{'='*60}\n  {combo_tag}\n{'='*60}")

            combo_metrics = {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}

            for idx in target_indices:
                sample = dataset[idx]

                cfg = copy.deepcopy(base_config)
                cfg.experiment_group = method_name
                cfg.model_type       = model_name
                cfg.sample_idx       = idx

                run_name = f"{method_name}_{model_name}_Sample_{idx}"

                # ── Per-sample W&B run ────────────────────────────────────
                if use_wandb:
                    wandb.init(
                        project=cfg.project_name,
                        group=combo_tag,          # e.g. "Full_Matrix/FourierMLP"
                        job_type="sample_train",
                        name=run_name,
                        config={
                            **cfg.to_dict(),
                            "method": method_name,
                            "model":  model_name,
                        },
                        reinit=True,
                        tags=[method_name, model_name, f"sample_{idx}"],
                    )

                # ── Train ─────────────────────────────────────────────────
                model = _build_model(model_cls, model_name, cfg)
                result_dict = engine_fn(
                    sample=sample,
                    L_matrix=dataset.L_matrix,
                    model=model,
                    label=model_name,
                    config=cfg,
                    use_wandb=use_wandb,
                )

                # ── Metrics ───────────────────────────────────────────────
                metrics = calculate_metrics(
                    s_phys_pred=result_dict["s_phys"],
                    s_gt_raw=sample["s_gt_raw"],
                    grid_shape=(64, 64),
                )

                # ── Artifacts ─────────────────────────────────────────────
                plot_fp = save_artifacts(
                    result_dict=result_dict,
                    sample=sample,
                    config=cfg,
                    metrics=metrics,
                )

                # ── Log to per-sample run and close it ────────────────────
                if use_wandb:
                    wandb.log({
                        "Final/MAE":  metrics["MAE"],
                        "Final/RMSE": metrics["RMSE"],
                        "Final/SSIM": metrics["SSIM"],
                        "Final/CNR":  metrics["CNR"],
                        "Reconstruction": wandb.Image(plot_fp),
                    })
                    # Capture image ref BEFORE finish()
                    wb_image = wandb.Image(plot_fp)
                    wandb.finish()
                else:
                    wb_image = None

                # ── Accumulate ────────────────────────────────────────────
                for k in combo_metrics:
                    combo_metrics[k].append(metrics[k])
                    results[(method_name, model_name)][k].append(metrics[k])

                master_rows.append({
                    "Method":     method_name,
                    "Model":      model_name,
                    "Sample_Idx": idx,
                    "MAE":        metrics["MAE"],
                    "RMSE":       metrics["RMSE"],
                    "SSIM":       metrics["SSIM"],
                    "CNR":        metrics["CNR"],
                    "Image":      wb_image,
                })

                logging.info(
                    f"  [{run_name}] MAE={metrics['MAE']:.3f} "
                    f"RMSE={metrics['RMSE']:.3f} "
                    f"SSIM={metrics['SSIM']:.4f} "
                    f"CNR={metrics['CNR']:.3f}"
                )

            # ── Per-(method, model) aggregate logged as config for later query
            _log_combo_aggregate(combo_tag, method_name, model_name, combo_metrics)

    # ── Baseline metrics (if embedded in dataset) ───────────────────────────
    baseline_rows, baseline_results = _compute_baseline_rows(
        dataset, target_indices, use_wandb=use_wandb
    )
    if baseline_rows:
        master_rows.extend(baseline_rows)
        results.update(baseline_results)
        logging.info(f"Added {len(baseline_results)} baseline method(s) to results.")

    # ── Master summary run ───────────────────────────────────────────────────
    agg_stats = _compute_aggregate_stats(results)
    log_to_local_database(base_config, agg_stats["overall"], target_indices)

    if use_wandb:
        _log_master_summary(base_config, master_rows, agg_stats)

    return results


def _compute_baseline_rows(dataset, target_indices, use_wandb=True):
    """
    Compute metrics for embedded L1/L2 baselines if available.

    Returns (rows, results) where rows is a list of dicts for the master table
    and results is a dict keyed by ("L1_baseline","Classical") etc.
    """
    rows = []
    results = {}

    baselines = []
    if dataset.benchmarks_l1 is not None:
        baselines.append(("L1_baseline", "s_l1_recon", dataset.benchmarks_l1))
    if dataset.benchmarks_l2 is not None:
        baselines.append(("L2_baseline", "s_l2_recon", dataset.benchmarks_l2))

    for method_label, sample_key, _ in baselines:
        combo_key = (method_label, "Classical")
        results[combo_key] = {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}

        for idx in target_indices:
            sample = dataset[idx]
            if sample_key not in sample:
                continue

            metrics = calculate_metrics(
                s_phys_pred=sample[sample_key],
                s_gt_raw=sample["s_gt_raw"],
                grid_shape=(64, 64),
            )

            for k in results[combo_key]:
                results[combo_key][k].append(metrics[k])

            rows.append({
                "Method":     method_label,
                "Model":      "Classical",
                "Sample_Idx": idx,
                "idx":        idx,       # alias for run_evaluation table format
                "MAE":        metrics["MAE"],
                "RMSE":       metrics["RMSE"],
                "SSIM":       metrics["SSIM"],
                "CNR":        metrics["CNR"],
                "Image":      None,
                "image":      None,      # alias for run_evaluation table format
            })

            logging.info(
                f"  [{method_label}_Sample_{idx}] MAE={metrics['MAE']:.3f} "
                f"RMSE={metrics['RMSE']:.3f} "
                f"SSIM={metrics['SSIM']:.4f} "
                f"CNR={metrics['CNR']:.3f}"
            )

    return rows, results


def _log_combo_aggregate(combo_tag, method_name, model_name, combo_metrics):
    """Log a tiny summary run for each (method, model) pair."""
    pass   # intentionally empty — aggregate is handled in master summary


def _compute_aggregate_stats(results: dict) -> dict:
    """
    Compute mean/std for every (method, model) combo and overall.
    Returns {'overall': {...}, 'per_combo': {(m,n): {...}}}
    """
    per_combo = {}
    all_mae, all_rmse, all_ssim, all_cnr = [], [], [], []

    for (method, model), vals in results.items():
        per_combo[(method, model)] = {
            f"{k}_mean": float(np.mean(v))
            for k, v in vals.items()
        }
        per_combo[(method, model)].update({
            f"{k}_std": float(np.std(v))
            for k, v in vals.items()
        })
        all_mae.extend(vals["MAE"])
        all_rmse.extend(vals["RMSE"])
        all_ssim.extend(vals["SSIM"])
        all_cnr.extend(vals["CNR"])

    overall = {
        "MAE_mean":  float(np.mean(all_mae)),   "MAE_std":  float(np.std(all_mae)),
        "RMSE_mean": float(np.mean(all_rmse)),  "RMSE_std": float(np.std(all_rmse)),
        "SSIM_mean": float(np.mean(all_ssim)),  "SSIM_std": float(np.std(all_ssim)),
        "CNR_mean":  float(np.mean(all_cnr)),   "CNR_std":  float(np.std(all_cnr)),
    }
    return {"overall": overall, "per_combo": per_combo}


def _log_master_summary(base_config: ExperimentConfig, master_rows: list, agg_stats: dict):
    """
    Creates one W&B summary run with:
    - A flat per-sample table (Method | Model | Sample | metrics | Image)
    - Flat aggregated scalars keyed as  <method>/<model>/MAE_mean  etc.
      so W&B can render grouped bar charts automatically.
    """
    wandb.init(
        project=base_config.project_name,
        group="MASTER_SUMMARY",
        name=f"Grid_Summary_{base_config.experiment_group}",
        job_type="grid_summary",
        config=base_config.to_dict(),
        reinit=True,
    )

    # 1. Master flat table
    table = wandb.Table(
        columns=["Method", "Model", "Sample_Idx",
                 "MAE", "RMSE", "SSIM", "CNR", "Reconstruction"]
    )
    for row in master_rows:
        table.add_data(
            row["Method"], row["Model"], row["Sample_Idx"],
            row["MAE"], row["RMSE"], row["SSIM"], row["CNR"], row["Image"]
        )
    wandb.log({"Grid_Results": table})

    # 2. Aggregate scalars with namespaced keys → W&B renders grouped bar charts
    #    e.g.  "Full_Matrix/FourierMLP/MAE_mean" : 0.63
    flat_agg = {}
    for (method, model), stats in agg_stats["per_combo"].items():
        prefix = f"{method}/{model}"
        for k, v in stats.items():
            flat_agg[f"{prefix}/{k}"] = v

    # Also log overall
    flat_agg.update({f"overall/{k}": v for k, v in agg_stats["overall"].items()})
    wandb.log(flat_agg)

    wandb.finish()
    logging.info("Master summary logged to W&B.")


def run_evaluation(
    dataset,
    model_class,
    train_engine,
    config: "ExperimentConfig",
    target_indices=None,
    use_wandb=True
):
    if target_indices is None:
        target_indices = [1]

    num_samples = len(target_indices)

    logging.info(f"Starting Benchmark: {config.experiment_group} | Model: {config.model_type}")

    aggregated_metrics = {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}

    # We collect table rows here and build the W&B table only after the loop,
    # inside the summary run. This avoids the reinit=True killing the upfront run.
    table_rows = []   # list of dicts

    for i, idx in enumerate(target_indices):
        logging.info(f"--- Processing Sample {i+1}/{num_samples} (Index: {idx}) ---")

        sample = dataset[idx]
        current_config = copy.deepcopy(config)
        current_config.sample_idx = idx

        # ── Per-sample W&B run ──────────────────────────────────────────────
        if use_wandb:
            run = wandb.init(
                project=current_config.project_name,
                group=current_config.experiment_group,
                job_type=current_config.model_type,
                name=f"{current_config.model_type}_Sample_{idx}",
                config=current_config.to_dict(),
                reinit=True
            )

        # ── Build model ─────────────────────────────────────────────────────
        if current_config.model_type == "FourierMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size,
                scale=current_config.scale
            )
        elif current_config.model_type == "SirenMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size,
                omega=current_config.omega
            )
        elif current_config.model_type == "ReluMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size
            )
        else:
            raise ValueError(f"Unsupported model type: {current_config.model_type}")

        # ── Train ────────────────────────────────────────────────────────────
        result_dict = train_engine(
            sample=sample,
            L_matrix=dataset.L_matrix,
            model=model,
            label=current_config.model_type,
            config=current_config,
            use_wandb=use_wandb
        )

        # ── Metrics ──────────────────────────────────────────────────────────
        metrics = calculate_metrics(
            s_phys_pred=result_dict['s_phys'],
            s_gt_raw=sample['s_gt_raw'],
            grid_shape=(64, 64)
        )

        # ── Artifacts ────────────────────────────────────────────────────────
        plot_filepath = save_artifacts(
            result_dict=result_dict,
            sample=sample,
            config=current_config,
            metrics=metrics
        )

        # ── Log to per-sample run and close it ──────────────────────────────
        if use_wandb:
            wandb.log({
                "Final MAE":  metrics["MAE"],
                "Final RMSE": metrics["RMSE"],
                "Final SSIM": metrics["SSIM"],
                "Final CNR":  metrics["CNR"],
                "Reconstruction Plot": wandb.Image(plot_filepath)
            })
            # Stash the W&B image reference BEFORE finishing the run
            table_rows.append({
                "idx":   idx,
                "MAE":   metrics["MAE"],
                "RMSE":  metrics["RMSE"],
                "SSIM":  metrics["SSIM"],
                "CNR":   metrics["CNR"],
                "image": wandb.Image(plot_filepath)   # captured while run is alive
            })
            wandb.finish()   # close per-sample run
        else:
            table_rows.append({
                "idx":  idx,
                "MAE":  metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "SSIM": metrics["SSIM"],
                "CNR":  metrics["CNR"],
                "image": None
            })

        # ── Accumulate ───────────────────────────────────────────────────────
        for key in aggregated_metrics:
            aggregated_metrics[key].append(metrics[key])

        logging.info(f"Sample {idx} -> MAE: {metrics['MAE']:.2f} | SSIM: {metrics['SSIM']:.4f}")

    # ── Baseline metrics (if embedded in dataset) ───────────────────────────
    baseline_rows, baseline_results = _compute_baseline_rows(
        dataset, target_indices, use_wandb=use_wandb
    )
    if baseline_rows:
        table_rows.extend(baseline_rows)
        logging.info(f"Added {len(baseline_results)} baseline method(s) to evaluation.")

    # ══════════════════════════════════════════════════════════════════════════
    # AGGREGATION
    # ══════════════════════════════════════════════════════════════════════════
    final_stats = {}
    for key, values in aggregated_metrics.items():
        final_stats[f"{key}_mean"] = float(np.mean(values))
        final_stats[f"{key}_std"]  = float(np.std(values))

    # Add baseline aggregate stats
    for (method, model), vals in baseline_results.items():
        for k, v in vals.items():
            prefix = f"{method}"
            final_stats[f"{prefix}_{k}_mean"] = float(np.mean(v))
            final_stats[f"{prefix}_{k}_std"] = float(np.std(v))

    # Log to local CSV
    log_to_local_database(config, final_stats, target_indices)

    # ── Summary W&B run (created AFTER the loop so reinit can't kill it) ────
    if use_wandb:
        summary_run = wandb.init(
            project=config.project_name,
            group=f"{config.experiment_group}_SUMMARY",
            name=f"Benchmark_{config.model_type}",
            job_type="benchmark",
            config=config.to_dict(),
            reinit=True
        )

        # Build the per-sample table
        benchmark_table = wandb.Table(
            columns=["Sample_Idx", "MAE", "RMSE", "SSIM", "CNR", "Reconstruction"]
        )
        for row in table_rows:
            benchmark_table.add_data(
                row["idx"], row["MAE"], row["RMSE"],
                row["SSIM"], row["CNR"], row["image"]
            )

        # Log scalar summary + table
        summary_run.log(final_stats)
        summary_run.log({"Benchmark_Results": benchmark_table})
        wandb.finish()

    logging.info(
        f"Benchmark complete — MAE {final_stats['MAE_mean']:.2f} ± {final_stats['MAE_std']:.2f} | "
        f"SSIM {final_stats['SSIM_mean']:.4f} ± {final_stats['SSIM_std']:.4f}"
    )
    return final_stats