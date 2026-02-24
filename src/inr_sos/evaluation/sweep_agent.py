"""
sweep_agent.py  (unified)
-------------------------
One Bayesian sweep that jointly optimises over:
  - method      (Full_Matrix | Sequential_SGD | Ray_Batching)
  - model_type  (FourierMLP | ReluMLP | SirenMLP)
  - all shared and model/method-specific hyperparameters

This produces a single PCP in W&B with method and model_type as axes
alongside lr, scale, mapping_size etc., coloured by MAE_mean.

Design notes
------------
All model-specific params (scale, omega) and method-specific params
(epochs, batch_size) are included in the global parameters dict.
W&B passes every param to every trial regardless of which model/method
is sampled. _sweep_train_fn reads only the relevant ones based on the
sampled method and model_type, ignoring the rest. This is the standard
W&B pattern for conditional hyperparameter spaces.

Step-logging fix (from previous version)
-----------------------------------------
engine_fn is called with use_wandb=False so the engine's internal
step=0..N-1 logging does not conflict with the sweep run's step cursor.
Per-sample final metrics are logged at step=sample_num (monotonic).
"""

import wandb
import copy
import logging
import numpy as np

from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Unified sweep config
# ──────────────────────────────────────────────────────────────────────────────

def get_sweep_config(
    metric_goal: str = "MAE_mean",
    metric_direction: str = "minimize",
) -> dict:
    """
    Returns a single W&B Bayesian sweep config covering all 3 methods
    × 3 models × shared hyperparameters.

    The PCP in W&B will show:
      method | model_type | hidden_features | hidden_layers | lr |
      mapping_size | scale | omega | reg_weight | tv_weight | steps |
      epochs | batch_size | CNR_mean | MAE_mean | SSIM_mean | RMSE_mean

    Coloured by MAE_mean (set in the W&B UI: colour axis → MAE_mean).
    """
    return {
        "method": "bayes",
        "name":   "unified_sweep_all_methods_models",
        "metric": {
            "name": metric_goal,
            "goal": metric_direction,
        },
        "parameters": {
            # ── What to vary (the two new categorical axes) ──────────────
            "method": {
                "values": ["Full_Matrix", "Sequential_SGD", "Ray_Batching"]
            },
            "model_type": {
                "values": ["FourierMLP", "ReluMLP", "SirenMLP"]
            },

            # ── Shared architecture params ────────────────────────────────
            "hidden_features": {"values": [128, 256, 512]},
            "hidden_layers":   {"values": [2, 3, 4, 5]},

            # ── Shared optimisation params ────────────────────────────────
            "lr":    {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
            "steps": {"values": [1000, 2000, 3000]},

            # ── Shared regularisation ─────────────────────────────────────
            "tv_weight":  {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
            "reg_weight": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},

            # ── FourierMLP-specific (ignored when model_type != FourierMLP)
            "mapping_size": {"values": [32, 64, 128, 256]},
            "scale":        {"distribution": "uniform", "min": 0.1, "max": 20.0},

            # ── SirenMLP-specific (ignored when model_type != SirenMLP) ──
            "omega": {"distribution": "uniform", "min": 10.0, "max": 60.0},

            # ── Ray_Batching-specific (ignored for other methods) ─────────
            "epochs":     {"values": [100, 150, 200]},
            "batch_size": {"values": [2048, 4096, 8192]},
        },
        # Kill clearly bad trials early (saves ~40% GPU time over 30 runs)
        "early_terminate": {
            "type":     "hyperband",
            "min_iter": 3,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep_agent(
    sweep_id: str,
    dataset,
    target_indices: list,
    base_config: ExperimentConfig,
    n_runs: int = 60,
    entity: str = None,
    project: str = None,
):
    """
    Launch the unified sweep agent.

    Parameters
    ----------
    sweep_id       : returned by wandb.sweep(get_sweep_config(), project=...)
    dataset        : USDataset
    target_indices : samples to average metrics over (recommend n≥5)
    base_config    : ExperimentConfig used as baseline; sweep overrides it
    n_runs         : total Bayesian trials (recommend 60 for 9 combos × ~7 trials each)
    entity         : W&B entity (username or team). If None, uses wandb default.
    project        : W&B project name. If None, uses base_config.project_name.
    """
    project = project or base_config.project_name
    from inr_sos.models.mlp import FourierMLP, ReluMLP
    from inr_sos.models.siren import SirenMLP
    from inr_sos.training.engines import (
        optimize_full_forward_operator,
        optimize_sequential_views,
        optimize_stochastic_ray_batching,
    )

    _engine_map = {
        "Full_Matrix":    optimize_full_forward_operator,
        "Sequential_SGD": optimize_sequential_views,
        "Ray_Batching":   optimize_stochastic_ray_batching,
    }
    _model_map = {
        "FourierMLP": FourierMLP,
        "ReluMLP":    ReluMLP,
        "SirenMLP":   SirenMLP,
    }

    def _sweep_train_fn():
        wandb.init()
        sc = wandb.config   # W&B fills this from the sampled parameters

        # ── Which method and model did the sweep sample? ──────────────────
        method     = sc.get("method",     "Full_Matrix")
        model_type = sc.get("model_type", "FourierMLP")

        engine_fn = _engine_map[method]
        model_cls = _model_map[model_type]

        # ── Build config — merge sweep values over the base ───────────────
        cfg = copy.deepcopy(base_config)
        cfg.model_type       = model_type
        cfg.experiment_group = method

        # Shared params
        cfg.hidden_features = sc.get("hidden_features", cfg.hidden_features)
        cfg.hidden_layers   = sc.get("hidden_layers",   cfg.hidden_layers)
        cfg.lr              = sc.get("lr",              cfg.lr)
        cfg.steps           = sc.get("steps",           cfg.steps)
        cfg.tv_weight       = sc.get("tv_weight",       cfg.tv_weight)
        cfg.reg_weight      = sc.get("reg_weight",      cfg.reg_weight)

        # Model-specific (read regardless, each model uses only its own)
        cfg.mapping_size = sc.get("mapping_size", cfg.mapping_size)
        cfg.scale        = sc.get("scale",        cfg.scale)   # FourierMLP
        cfg.omega        = sc.get("omega",        cfg.omega)   # SirenMLP

        # Method-specific (only Ray_Batching uses these)
        if method == "Ray_Batching":
            cfg.epochs     = sc.get("epochs",     cfg.epochs)
            cfg.batch_size = sc.get("batch_size", cfg.batch_size)

        # ── Train on every target sample ──────────────────────────────────
        all_mae, all_ssim, all_rmse, all_cnr = [], [], [], []

        for sample_num, idx in enumerate(target_indices):
            sample = dataset[idx]
            cfg.sample_idx = idx

            model = _build_model(model_cls, model_type, cfg)

            # use_wandb=False: prevents step regression warnings.
            # Engine logs step=0..N internally; with multiple samples that
            # resets to 0 each time → W&B drops everything after sample 1.
            result_dict = engine_fn(
                sample=sample,
                L_matrix=dataset.L_matrix,
                model=model,
                label=model_type,
                config=cfg,
                use_wandb=False,
            )

            metrics = calculate_metrics(
                s_phys_pred=result_dict["s_phys"],
                s_gt_raw=sample["s_gt_raw"],
                grid_shape=(64, 64),
            )
            all_mae.append(metrics["MAE"])
            all_ssim.append(metrics["SSIM"])
            all_rmse.append(metrics["RMSE"])
            all_cnr.append(metrics["CNR"])

            # Per-sample metrics — step is sample_num (monotonic, safe)
            wandb.log({
                "sample/MAE":  metrics["MAE"],
                "sample/SSIM": metrics["SSIM"],
                "sample/RMSE": metrics["RMSE"],
                "sample/CNR":  metrics["CNR"],
                "sample/idx":  idx,
            }, step=sample_num)

        # ── Aggregate — what the Bayesian optimiser reads ─────────────────
        wandb.log({
            "MAE_mean":  float(np.mean(all_mae)),
            "MAE_std":   float(np.std(all_mae)),
            "RMSE_mean": float(np.mean(all_rmse)),
            "RMSE_std":  float(np.std(all_rmse)),
            "SSIM_mean": float(np.mean(all_ssim)),
            "SSIM_std":  float(np.std(all_ssim)),
            "CNR_mean":  float(np.mean(all_cnr)),
            "CNR_std":   float(np.std(all_cnr)),
        })
        wandb.finish()

    wandb.agent(
        sweep_id,
        function=_sweep_train_fn,
        count=n_runs,
        entity=entity,
        project=project,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(model_cls, model_name: str, cfg: ExperimentConfig):
    kwargs = dict(
        in_features=cfg.in_features,
        hidden_features=cfg.hidden_features,
        hidden_layers=cfg.hidden_layers,
        mapping_size=cfg.mapping_size,
    )
    if model_name == "FourierMLP":
        kwargs["scale"] = cfg.scale
    elif model_name == "SirenMLP":
        kwargs["omega"] = cfg.omega
    return model_cls(**kwargs)