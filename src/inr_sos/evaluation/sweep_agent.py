"""
sweep_config.py
---------------
W&B Bayesian sweep configurations for each model type.

Usage
-----
    import wandb
    from inr_sos.evaluation.sweep_config import get_sweep_config, run_sweep_agent

    # 1. Create the sweep (returns a sweep_id)
    sweep_id = wandb.sweep(
        get_sweep_config("FourierMLP", method="Full_Matrix"),
        project="INR-SoS-Recon"
    )

    # 2. Launch agents (can run on multiple GPUs / machines)
    run_sweep_agent(sweep_id, dataset, target_indices, n_runs=20)
"""

import wandb
import copy
import logging
import numpy as np

from inr_sos.utils.config import ExperimentConfig
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.utils.tracker import save_artifacts


# ──────────────────────────────────────────────────────────────────────────────
# Sweep search spaces
# ──────────────────────────────────────────────────────────────────────────────

_COMMON_PARAMS = {
    # Architecture
    "hidden_features": {"values": [128, 256, 512]},
    "hidden_layers":   {"values": [2, 3, 4, 5]},
    # Optimisation
    "lr":              {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
    "steps":           {"values": [1000, 2000, 3000]},
    # Regularisation
    "tv_weight":       {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
    "reg_weight":      {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
}

_MODEL_SPECIFIC = {
    "FourierMLP": {
        "mapping_size": {"values": [32, 64, 128, 256]},
        "scale":        {"distribution": "uniform", "min": 0.1, "max": 20.0},
    },
    "SirenMLP": {
        "mapping_size": {"values": [32, 64, 128]},   # unused by SIREN but kept for API compat
        "omega":        {"distribution": "uniform", "min": 10.0, "max": 60.0},
    },
    "ReluMLP": {
        "mapping_size": {"values": [64]},   # ReluMLP ignores mapping_size; placeholder
    },
}

# For ray-batching engine the epochs/batch_size matter instead of steps
_RAY_BATCHING_EXTRA = {
    "epochs":     {"values": [100, 150, 200]},
    "batch_size": {"values": [2048, 4096, 8192]},
}


def get_sweep_config(
    model_type: str,
    method: str = "Full_Matrix",
    metric_goal: str = "MAE_mean",
    metric_direction: str = "minimize",
) -> dict:
    """
    Returns a W&B sweep config dict for the given (model_type, method) pair.

    Parameters
    ----------
    model_type       : "FourierMLP" | "SirenMLP" | "ReluMLP"
    method           : "Full_Matrix" | "Sequential_SGD" | "Ray_Batching"
    metric_goal      : W&B metric name to optimise (must be logged in the sweep run)
    metric_direction : "minimize" | "maximize"
    """
    if model_type not in _MODEL_SPECIFIC:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from {list(_MODEL_SPECIFIC.keys())}")

    params = {**_COMMON_PARAMS, **_MODEL_SPECIFIC[model_type]}

    if method == "Ray_Batching":
        params.update(_RAY_BATCHING_EXTRA)

    return {
        "method": "bayes",
        "name":   f"sweep_{model_type}_{method}",
        "metric": {
            "name": metric_goal,
            "goal": metric_direction,
        },
        "parameters": params,
        # Early termination: stop runs that are clearly worse than the best so far
        "early_terminate": {
            "type":   "hyperband",
            "min_iter": 3,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Agent entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep_agent(
    sweep_id: str,
    dataset,
    target_indices: list,
    base_config: ExperimentConfig,
    model_type: str,
    method: str,
    n_runs: int = 30,
):
    """
    Launches a W&B sweep agent that calls _sweep_train_fn for each trial.

    Parameters
    ----------
    sweep_id       : returned by wandb.sweep(...)
    dataset        : USDataset instance
    target_indices : list of sample indices to average metrics over
    base_config    : ExperimentConfig used as the base (sweep overrides it)
    model_type     : "FourierMLP" | "SirenMLP" | "ReluMLP"
    method         : "Full_Matrix" | "Sequential_SGD" | "Ray_Batching"
    n_runs         : number of sweep trials
    """
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

    engine_fn = _engine_map[method]
    model_cls = _model_map[model_type]

    def _sweep_train_fn():
        """Called once per sweep trial by the W&B agent."""
        run = wandb.init()   # W&B agent fills wandb.config automatically
        sweep_cfg = wandb.config

        # ── Merge sweep params into a cloned ExperimentConfig ────────────
        cfg = copy.deepcopy(base_config)
        cfg.model_type       = model_type
        cfg.experiment_group = method
        cfg.hidden_features  = sweep_cfg.get("hidden_features", cfg.hidden_features)
        cfg.hidden_layers    = sweep_cfg.get("hidden_layers",   cfg.hidden_layers)
        cfg.lr               = sweep_cfg.get("lr",              cfg.lr)
        cfg.steps            = sweep_cfg.get("steps",           cfg.steps)
        cfg.tv_weight        = sweep_cfg.get("tv_weight",       cfg.tv_weight)
        cfg.reg_weight       = sweep_cfg.get("reg_weight",      cfg.reg_weight)
        cfg.mapping_size     = sweep_cfg.get("mapping_size",    cfg.mapping_size)

        if model_type == "FourierMLP":
            cfg.scale = sweep_cfg.get("scale", cfg.scale)
        elif model_type == "SirenMLP":
            cfg.omega = sweep_cfg.get("omega", cfg.omega)

        if method == "Ray_Batching":
            cfg.epochs     = sweep_cfg.get("epochs",     cfg.epochs)
            cfg.batch_size = sweep_cfg.get("batch_size", cfg.batch_size)

        # ── Train on every target sample, average the metrics ─────────────
        all_mae, all_ssim, all_rmse, all_cnr = [], [], [], []

        for idx in target_indices:
            sample = dataset[idx]
            cfg.sample_idx = idx

            model = _build_sweep_model(model_cls, model_type, cfg)
            result_dict = engine_fn(
                sample=sample,
                L_matrix=dataset.L_matrix,
                model=model,
                label=model_type,
                config=cfg,
                use_wandb=True,   # step-level loss curves logged here
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

        # ── Log the aggregate metric that the sweep optimises ─────────────
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

    wandb.agent(sweep_id, function=_sweep_train_fn, count=n_runs)


def _build_sweep_model(model_cls, model_name: str, cfg: ExperimentConfig):
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
    else:  # ReluMLP
        return model_cls(
            in_features=cfg.in_features,
            hidden_features=cfg.hidden_features,
            hidden_layers=cfg.hidden_layers,
            mapping_size=cfg.mapping_size,
        )