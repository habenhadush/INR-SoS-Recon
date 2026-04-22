"""
joint_sweep_agent.py
--------------------
Bayesian sweep agent for the joint denoiser + reconstructor pipeline.

Searches over denoiser config, reconstructor config, training schedule,
and lambda strategy simultaneously. Uses the same W&B Bayesian sweep
pattern as sweep_agent.py.

Search space is read from datasets.yaml under the 'joint_sweep' key.

Usage (two-step):
    # 1. Create sweep
    python scripts/create_joint_sweep.py --dataset kwave_geom --n_runs 150

    # 2. Run agent
    python scripts/run_joint_sweep.py --sweep_id <ID> --n_runs 150 --dataset kwave_geom
"""

import copy
import logging

import numpy as np
import wandb

from inr_sos.evaluation.metrics import calculate_metrics, compute_sweep_objective
from inr_sos.evaluation.sweep_agent import (
    _load_dataset_sweep_config,
    _yaml_param_to_wandb,
)
from inr_sos.models.mlp import ReluMLP, FourierMLP, GeluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.denoise_engine import DEFAULT_DENOISE_CFG
from inr_sos.training.joint_engine import optimize_joint
from inr_sos.utils.config import ExperimentConfig

_RECON_MODEL_MAP = {
    "ReluMLP": ReluMLP,
    "FourierMLP": FourierMLP,
    "GeluMLP": GeluMLP,
    "SirenMLP": SirenMLP,
}

_log = logging.getLogger(__name__)


def _sc_get(sc, key, default, cast=float):
    """Safely read a scalar from wandb.config.

    W&B sometimes passes param spec dicts (e.g. {"values": [64, 128]})
    instead of sampled scalars. This helper detects that and falls back.
    """
    val = sc.get(key, default)
    if isinstance(val, dict):
        val = default
    return cast(val)


# ─── Config builder ──────────────────────────────────────────────────────────

def _load_joint_sweep_section(dataset_key: str = None) -> dict:
    """Load the joint_sweep section from datasets.yaml."""
    import yaml
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[3] / "scripts" / "datasets.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    key = dataset_key or cfg["active"]
    ds = cfg["datasets"][key]
    section = ds.get("joint_sweep", {})
    if not section:
        raise ValueError(
            f"Dataset '{key}' has no 'joint_sweep' section in datasets.yaml."
        )
    return section


def get_joint_sweep_config(
    dataset_key: str = None,
    metric_goal: str = "MAE_mean",
    metric_direction: str = "minimize",
) -> dict:
    """Build a W&B Bayesian sweep config from the joint_sweep section."""
    section = _load_joint_sweep_section(dataset_key)
    params_yaml = section.get("parameters", {})

    wandb_params = {}
    for name, value in params_yaml.items():
        wandb_params[name] = _yaml_param_to_wandb(value)

    key_label = dataset_key or "active"
    return {
        "method": "bayes",
        "name": f"joint_sweep_{key_label}",
        "metric": {
            "name": metric_goal,
            "goal": metric_direction,
        },
        "parameters": wandb_params,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 2,
        },
    }


# ─── Agent ───────────────────────────────────────────────────────────────────

def run_joint_sweep_agent(
    sweep_id: str,
    dataset,
    grid,
    target_indices: list,
    base_time_scale: float = 1e6,
    n_runs: int = 100,
    entity: str = None,
    project: str = None,
    roi_weight: float = 0.7,
    contrast_weight: float = 0.0,
    selection_metric: str = "loss",
):
    """Launch the joint sweep agent.

    Parameters
    ----------
    sweep_id       : from wandb.sweep()
    dataset        : USDataset
    grid           : USGrid from dataset.grid
    target_indices : sample indices to average metrics over
    base_time_scale: time_scale for the dataset
    n_runs         : total Bayesian trials
    entity         : W&B entity
    project        : W&B project name
    roi_weight     : blend weight for MAE_composite. 1.0 = pure MAE_roi,
                     0.0 = pure MAE_mean, 0.7 = 70% MAE_roi + 30% MAE_mean.
    contrast_weight: multiplicative penalty for low contrast recovery.
                     0.0 = no penalty, 1.0 = doubles objective when CR=0.
    selection_metric: "loss" or "mae_roi" — model checkpoint criterion in
                     Stage 3 of optimize_joint.
    """
    project = project or "INR-SoS-Recon"

    def _sweep_train_fn():
        wandb.init()
        sc = wandb.config

        # ── Denoiser config ──────────────────────────────────────────────
        denoise_cfg = dict(DEFAULT_DENOISE_CFG)
        denoise_cfg["model_type"] = _sc_get(sc, "dn_model_type",
                                             denoise_cfg.get("model_type", "FourierMLP"), str)
        denoise_cfg["scale"] = _sc_get(sc, "dn_scale", denoise_cfg["scale"], float)
        denoise_cfg["omega"] = _sc_get(sc, "dn_omega", denoise_cfg.get("omega", 15.0), float)
        denoise_cfg["hidden_features"] = _sc_get(sc, "dn_hidden_features",
                                                  denoise_cfg["hidden_features"], int)
        denoise_cfg["hidden_layers"] = _sc_get(sc, "dn_hidden_layers",
                                                denoise_cfg["hidden_layers"], int)

        # ── Reconstructor config ─────────────────────────────────────────
        rc_model_type = _sc_get(sc, "rc_model_type", "ReluMLP", str)
        recon_cfg = ExperimentConfig(
            project_name="INR-SoS-Recon",
            experiment_group="Joint-Sweep",
            model_type=rc_model_type,
            hidden_features=_sc_get(sc, "rc_hidden_features", 256, int),
            hidden_layers=_sc_get(sc, "rc_hidden_layers", 3, int),
            mapping_size=_sc_get(sc, "rc_mapping_size", 64, int),
            lr=_sc_get(sc, "rc_lr", 1e-4, float),
            steps=_sc_get(sc, "joint_steps", 2000, int),
            early_stopping=True,
            patience=100,
            clamp_slowness=True,
            loss_type="mse",
            time_scale=base_time_scale,
            tv_weight=_sc_get(sc, "rc_tv_weight", 0.0, float),
            reg_weight=_sc_get(sc, "rc_reg_weight", 0.0, float),
        )

        # ── Training schedule ────────────────────────────────────────────
        pretrain_dn_steps = _sc_get(sc, "pretrain_dn_steps", 300, int)
        pretrain_rc_steps = _sc_get(sc, "pretrain_rc_steps", 500, int)
        joint_steps = _sc_get(sc, "joint_steps", 2000, int)
        joint_lr_factor = _sc_get(sc, "joint_lr_factor", 0.1, float)

        # ── Lambda config ────────────────────────────────────────────────
        lambda_strategy = _sc_get(sc, "lambda_strategy", "fixed", str)
        lambda_fit = _sc_get(sc, "lambda_fit", 0.1, float)

        lambda_cfg = {}
        if lambda_strategy == "cosine":
            lambda_cfg["lambda_max"] = _sc_get(sc, "lambda_max", 1.0, float)
            lambda_cfg["lambda_min"] = _sc_get(sc, "lambda_min", 0.01, float)
        elif lambda_strategy == "balanced":
            lambda_cfg["target_ratio"] = _sc_get(sc, "target_ratio", 1.0, float)
            lambda_cfg["lambda_min"] = _sc_get(sc, "lambda_min", 0.001, float)
            lambda_cfg["lambda_max"] = _sc_get(sc, "lambda_max", 10.0, float)
        elif lambda_strategy == "residual":
            lambda_cfg["alpha"] = _sc_get(sc, "alpha", 0.5, float)

        # ── Train on every target sample ─────────────────────────────────
        all_mae, all_ssim, all_rmse, all_cnr = [], [], [], []
        all_mae_roi, all_mae_bkg, all_contrast, all_composite = [], [], [], []

        for sample_num, idx in enumerate(target_indices):
            sample = dataset[idx]

            model_cls = _RECON_MODEL_MAP[rc_model_type]
            model_kwargs = dict(
                in_features=recon_cfg.in_features,
                hidden_features=recon_cfg.hidden_features,
                hidden_layers=recon_cfg.hidden_layers,
                mapping_size=recon_cfg.mapping_size,
            )
            if rc_model_type == "FourierMLP":
                model_kwargs["scale"] = _sc_get(sc, "rc_scale", 10.0, float)
            elif rc_model_type == "SirenMLP":
                model_kwargs["omega"] = _sc_get(sc, "rc_omega", 30.0, float)
            model = model_cls(**model_kwargs)

            result = optimize_joint(
                sample=sample,
                L_matrix=dataset.L_matrix,
                grid=grid,
                recon_model=model,
                recon_config=copy.deepcopy(recon_cfg),
                denoise_cfg=denoise_cfg,
                lambda_fit=lambda_fit,
                mode="staged",
                lambda_strategy=lambda_strategy,
                lambda_cfg=lambda_cfg,
                pretrain_denoiser_steps=pretrain_dn_steps,
                pretrain_recon_steps=pretrain_rc_steps,
                joint_steps=joint_steps,
                joint_lr_factor=joint_lr_factor,
                use_wandb=False,
                label=f"sweep_s{idx}",
                selection_metric=selection_metric,
                gt_for_selection=sample["s_gt_raw"] if selection_metric == "mae_roi" else None,
            )

            metrics = calculate_metrics(
                s_phys_pred=result["s_phys"],
                s_gt_raw=sample["s_gt_raw"],
                grid_shape=(64, 64),
            )
            objective = compute_sweep_objective(metrics, roi_weight, contrast_weight)

            all_mae.append(metrics["MAE"])
            all_ssim.append(metrics["SSIM"])
            all_rmse.append(metrics["RMSE"])
            all_cnr.append(metrics["CNR"])
            all_mae_roi.append(metrics["MAE_roi"])
            all_mae_bkg.append(metrics["MAE_bkg"])
            all_contrast.append(metrics["contrast_recovery"])
            all_composite.append(objective)

            wandb.log({
                "sample/MAE": metrics["MAE"],
                "sample/SSIM": metrics["SSIM"],
                "sample/RMSE": metrics["RMSE"],
                "sample/CNR": metrics["CNR"],
                "sample/MAE_roi": metrics["MAE_roi"],
                "sample/MAE_bkg": metrics["MAE_bkg"],
                "sample/contrast_recovery": metrics["contrast_recovery"],
                "sample/sweep_objective": objective,
                "sample/idx": idx,
            }, step=sample_num)

        # ── Aggregate ────────────────────────────────────────────────────
        wandb.log({
            "MAE_mean": float(np.mean(all_mae)),
            "MAE_std": float(np.std(all_mae)),
            "RMSE_mean": float(np.mean(all_rmse)),
            "RMSE_std": float(np.std(all_rmse)),
            "SSIM_mean": float(np.mean(all_ssim)),
            "SSIM_std": float(np.std(all_ssim)),
            "CNR_mean": float(np.mean(all_cnr)),
            "CNR_std": float(np.std(all_cnr)),
            "MAE_roi_mean": float(np.mean(all_mae_roi)),
            "MAE_roi_std":  float(np.std(all_mae_roi)),
            "MAE_bkg_mean": float(np.mean(all_mae_bkg)),
            "contrast_recovery_mean": float(np.mean(all_contrast)),
            "MAE_composite_mean": float(np.mean(all_composite)),
            "MAE_composite_std":  float(np.std(all_composite)),
            "roi_weight": float(roi_weight),
            "contrast_weight": float(contrast_weight),
        })
        wandb.finish()

    wandb.agent(
        sweep_id,
        function=_sweep_train_fn,
        count=n_runs,
        entity=entity,
        project=project,
    )
