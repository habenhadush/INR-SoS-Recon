"""Displacement field denoiser engine (Architecture 1).

Fits an INR to d_meas using self-supervised MSE loss.
The INR's spectral bias captures smooth (signal) components first.
Early stopping halts training before it fits high-frequency mismatch/noise.

Two-stage pipeline:
    Stage 1: denoise_displacement()  →  d_clean
    Stage 2: existing engine(sample with d_clean)  →  SoS reconstruction
"""

import copy
import logging
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from inr_sos.denoising.config import DenoiserConfig
from inr_sos.models.siren import SirenMLP
from inr_sos.models.mlp import FourierMLP

_log = logging.getLogger(__name__)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_denoiser_model(config: DenoiserConfig):
    """Instantiate the denoiser INR based on config."""
    if config.model_type == "SirenMLP":
        return SirenMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            out_features=1,
            omega=config.omega,
        )
    elif config.model_type == "FourierMLP":
        return FourierMLP(
            in_features=config.in_features,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
            mapping_size=config.mapping_size,
            scale=config.scale,
        )
    else:
        raise ValueError(f"Unknown denoiser model_type: {config.model_type}")


class _DenoiserEarlyStopper:
    """Early stopping for the denoiser.

    Splits valid rays into train/val. Tracks validation MSE.
    The validation loss minimum = optimal denoising point.
    """

    def __init__(self, mask, config: DenoiserConfig, model):
        self.enabled = config.early_stopping
        if not self.enabled:
            self.train_idx = None
            self.val_idx = None
            return

        valid_idx = torch.where(mask.flatten() > 0.5)[0]
        n_val = max(1, int(len(valid_idx) * config.val_fraction))
        perm = valid_idx[torch.randperm(len(valid_idx), device=valid_idx.device)]
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val:]

        self.train_mask = torch.zeros_like(mask.flatten())
        self.train_mask[self.train_idx] = 1.0
        self.val_mask = torch.zeros_like(mask.flatten())
        self.val_mask[self.val_idx] = 1.0

        self.patience = config.patience
        self.best_val_loss = float("inf")
        self.wait = 0
        self.best_step = 0
        self.best_state = copy.deepcopy(model.state_dict())

    def get_train_mask(self, original_mask):
        if not self.enabled:
            return original_mask.flatten()
        return self.train_mask

    def evaluate(self, d_pred, d_meas, config, model, step):
        """Compute val MSE. Returns (val_loss, should_stop)."""
        if not self.enabled:
            return None, False

        residual = (d_pred.flatten() - d_meas.flatten()) * self.val_mask
        residual_scaled = residual * config.time_scale
        n_val = self.val_mask.sum() + 1e-8
        val_loss = (residual_scaled ** 2).sum() / n_val
        val_loss_val = val_loss.item()

        if val_loss_val < self.best_val_loss:
            self.best_val_loss = val_loss_val
            self.wait = 0
            self.best_step = step
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1

        return val_loss_val, self.wait >= self.patience

    def restore_best(self, model):
        if self.enabled:
            model.load_state_dict(self.best_state)


def denoise_displacement(
    d_meas: torch.Tensor,
    mask: torch.Tensor,
    ray_features: torch.Tensor,
    config: DenoiserConfig,
    use_wandb: bool = False,
) -> dict:
    """Stage 1: Fit INR to d_meas using spectral bias as implicit denoiser.

    Parameters
    ----------
    d_meas : torch.Tensor, shape (N, 1) or (N,)
        Raw displacement field measurements.
    mask : torch.Tensor, same shape as d_meas
        Binary mask (1=valid ray, 0=invalid).
    ray_features : torch.Tensor, shape (N, 3)
        Structured ray coordinates from compute_ray_features().
    config : DenoiserConfig
        Denoiser hyperparameters.
    use_wandb : bool
        Log denoiser metrics to wandb.

    Returns
    -------
    dict with:
        'd_clean'       : (N, 1) denoised displacement
        'd_residual'    : (N, 1) removed component (d_meas - d_clean)
        'loss_history'  : list of training losses
        'val_loss_history' : list of (step, val_loss) tuples
        'best_step'     : step with best validation loss
        'model_state'   : state dict of best denoiser model
    """
    d_meas_flat = d_meas.flatten().to(_DEVICE)
    mask_flat = mask.flatten().to(_DEVICE)
    features = ray_features.to(_DEVICE)

    model = _build_denoiser_model(config).to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    n_params = sum(p.numel() for p in model.parameters())
    n_valid = int(mask_flat.sum().item())
    _log.info(f"Denoiser: {config.model_type} | {n_params:,} params | "
              f"{n_valid:,}/{len(mask_flat):,} valid rays | "
              f"omega={config.omega}")

    stopper = _DenoiserEarlyStopper(mask_flat, config, model)
    train_mask = stopper.get_train_mask(mask_flat)

    loss_history = []
    val_loss_history = []
    pbar = tqdm(range(config.steps), desc="Denoiser")

    for step in pbar:
        model.train()
        optimizer.zero_grad()

        d_pred = model(features).flatten()  # (N,)

        # MSE on train rays (no Huber — we want uniform weighting)
        residual = (d_pred - d_meas_flat) * train_mask
        residual_scaled = residual * config.time_scale
        n_train = train_mask.sum() + 1e-8
        loss = (residual_scaled ** 2).sum() / n_train

        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 50 == 0:
            pbar.set_description(f"Denoiser loss: {loss_val:.4f}")

        # Early stopping evaluation
        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                d_pred_eval = model(features).flatten()
                vl, should_stop = stopper.evaluate(
                    d_pred_eval, d_meas_flat, config, model, step
                )
            if vl is not None:
                val_loss_history.append((step, vl))
            if use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "denoiser/train_loss": loss_val,
                        "denoiser/val_loss": vl,
                    }, step=step)
                except Exception:
                    pass
            if should_stop:
                _log.info(f"Denoiser early stopped at step {step} "
                          f"(best={stopper.best_step}, patience={config.patience})")
                break

    stopper.restore_best(model)

    # Generate d_clean from best model
    model.eval()
    with torch.no_grad():
        d_clean = model(features).flatten()

    # Keep original d_meas for masked-out rays
    d_out = d_meas_flat.clone()
    d_out[mask_flat > 0.5] = d_clean[mask_flat > 0.5]

    d_clean_out = d_out.unsqueeze(-1).cpu()
    d_residual = (d_meas.flatten().cpu() - d_out.cpu()).unsqueeze(-1)

    best_step = stopper.best_step if stopper.enabled else config.steps - 1
    _log.info(f"Denoiser done | best_step={best_step} | "
              f"final_train_loss={loss_history[-1]:.4f}")

    return {
        'd_clean': d_clean_out,
        'd_residual': d_residual,
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'best_step': best_step,
        'model_state': model.state_dict(),
    }


def denoise_and_reconstruct(
    sample: dict,
    L_matrix,
    model,
    label: str,
    config,
    engine_fn,
    denoiser_config: DenoiserConfig = None,
    ray_features: torch.Tensor = None,
    use_wandb: bool = False,
) -> dict:
    """Two-stage pipeline: denoise d_meas, then reconstruct SoS.

    Parameters
    ----------
    sample : dict
        Standard USDataset sample with 'd_meas', 'mask', 'coords', etc.
    L_matrix : tensor
        Forward model matrix.
    model : nn.Module
        SoS reconstruction INR model (for Stage 2).
    label : str
        Model label for logging.
    config : ExperimentConfig
        Stage 2 reconstruction config.
    engine_fn : callable
        One of the existing engines (optimize_full_forward_operator, etc).
    denoiser_config : DenoiserConfig, optional
        If None, skips denoising and runs Stage 2 directly.
    ray_features : torch.Tensor, optional
        Precomputed ray features. If None, computed from defaults.
    use_wandb : bool
        Log metrics to wandb.

    Returns
    -------
    dict with Stage 2 results + 'denoiser_result' if denoiser was used.
    """
    denoiser_result = None

    if denoiser_config is not None:
        if ray_features is None:
            from inr_sos.denoising.ray_features import compute_ray_features
            ray_features = compute_ray_features()

        _log.info("=== Stage 1: Denoising displacement field ===")
        denoiser_result = denoise_displacement(
            d_meas=sample['d_meas'],
            mask=sample['mask'],
            ray_features=ray_features,
            config=denoiser_config,
            use_wandb=use_wandb,
        )

        # Substitute d_clean into sample (shallow copy to avoid mutating original)
        sample = {**sample, 'd_meas': denoiser_result['d_clean']}
        _log.info("=== Stage 2: Reconstruction with d_clean ===")

    # Stage 2: standard reconstruction
    result = engine_fn(
        sample=sample,
        L_matrix=L_matrix,
        model=model,
        label=label,
        config=config,
        use_wandb=use_wandb,
    )

    if denoiser_result is not None:
        result['denoiser_result'] = denoiser_result

    return result
