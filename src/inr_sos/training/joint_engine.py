"""
Joint Denoiser + Reconstructor optimization engine (Experiment 11).

Trains a per-pair denoiser INR and a reconstruction INR end-to-end.
The denoiser learns to produce measurements that help reconstruction,
constrained by a data-fidelity term to stay close to raw measurements.

Loss = ||mask * (L @ s_pred - d_denoised)||² + λ_fit * ||mask * (d_denoised - d_raw)||²

Three training modes:
  - end_to_end:  single optimizer, both INRs trained simultaneously
  - alternating: alternate fixing one INR while training the other
  - staged:      pretrain denoiser → pretrain recon → joint fine-tune
"""

import copy
import logging
import sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from inr_sos.models.mlp import FourierMLP, ReluMLP, GeluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.training.denoise_engine import (
    _build_dt_coords,
    _build_model as _build_denoiser_model,
    DEFAULT_DENOISE_CFG,
    _N_PAIRS,
    _PAIR_SIZE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("joint_engine")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SLOWNESS_MIN = 1.0 / 1800.0
_SLOWNESS_MAX = 1.0 / 1200.0


# ─── Denoiser wrapper ───────────────────────────────────────────────────────

class PairDenoiser(torch.nn.Module):
    """Wraps 8 per-pair INRs + per-pair normalization into a single module.

    Forward pass: coords_dt (16384, 2) → d_denoised (131072, 1) in physical units.
    Normalization stats are registered as buffers (non-learnable, moved with .to()).
    """

    def __init__(self, denoise_cfg, d_meas, mask):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [_build_denoiser_model(denoise_cfg) for _ in range(_N_PAIRS)]
        )

        # Precompute per-pair normalization from raw measurements
        d_flat = d_meas.flatten()
        m_flat = mask.flatten()
        means, stds = [], []
        for k in range(_N_PAIRS):
            start = k * _PAIR_SIZE
            end = (k + 1) * _PAIR_SIZE
            valid = m_flat[start:end] > 0.5
            d_valid = d_flat[start:end][valid]
            means.append(d_valid.mean())
            stds.append(d_valid.std() + 1e-10)

        self.register_buffer("d_means", torch.stack(means))       # (8,)
        self.register_buffer("d_stds", torch.stack(stds))          # (8,)

    def forward(self, coords_dt):
        """Predict denoised displacements for all 8 pairs.

        Args:
            coords_dt: (16384, 2) normalized DT coordinates.

        Returns:
            d_denoised: (131072, 1) in physical displacement units.
        """
        parts = []
        for k in range(_N_PAIRS):
            d_norm_k = self.models[k](coords_dt)        # (16384, 1)
            d_phys_k = d_norm_k * self.d_stds[k] + self.d_means[k]
            parts.append(d_phys_k)
        return torch.cat(parts, dim=0)  # (131072, 1)


# ─── Joint optimization ─────────────────────────────────────────────────────

def optimize_joint(
    sample,
    L_matrix,
    grid,
    recon_model,
    recon_config,
    denoise_cfg=None,
    lambda_fit=1.0,
    mode="end_to_end",
    pretrain_denoiser_steps=300,
    pretrain_recon_steps=500,
    joint_steps=None,
    use_wandb=False,
    label="joint",
):
    """Joint denoiser + reconstructor optimization.

    Args:
        sample:       dataset sample dict.
        L_matrix:     forward model matrix (dense or sparse).
        grid:         USGrid instance.
        recon_model:  reconstruction INR (ReluMLP etc.), freshly initialized.
        recon_config: ExperimentConfig for reconstruction.
        denoise_cfg:  dict for denoiser INR config (defaults to DEFAULT_DENOISE_CFG).
        lambda_fit:   weight for denoiser data-fidelity loss.
        mode:         'end_to_end', 'alternating', or 'staged'.
        pretrain_denoiser_steps: steps for stage 1 in 'staged' mode.
        pretrain_recon_steps:    steps for stage 2 in 'staged' mode.
        joint_steps:  total joint training steps (defaults to recon_config.steps).
        use_wandb:    whether to log to W&B.
        label:        run label for logging.

    Returns:
        dict with 's_phys', 'loss_history', 'd_denoised', 'info'.
    """
    if denoise_cfg is None:
        denoise_cfg = dict(DEFAULT_DENOISE_CFG)
    if joint_steps is None:
        joint_steps = recon_config.steps

    log.info(f"\n--- Joint Denoiser+Recon: {label} ({mode}) on {_DEVICE} ---")
    log.info(f"  lambda_fit={lambda_fit}, joint_steps={joint_steps}")

    # ── Data setup ────────────────────────────────────────────────────────
    coords_sos = sample["coords"].to(_DEVICE)
    d_raw = sample["d_meas"].to(_DEVICE)
    mask = sample["mask"].to(_DEVICE)
    s_mean = sample["s_stats"][0].item()
    s_std = sample["s_stats"][1].item()
    L = L_matrix.to(_DEVICE)

    time_scale = recon_config.time_scale

    # ── Build models ──────────────────────────────────────────────────────
    denoiser = PairDenoiser(denoise_cfg, d_raw.cpu(), mask.cpu()).to(_DEVICE)
    recon_model = recon_model.to(_DEVICE)
    coords_dt = _build_dt_coords(grid)  # (16384, 2) on device

    # ── Dispatch to training mode ─────────────────────────────────────────
    if mode == "end_to_end":
        return _train_end_to_end(
            denoiser, recon_model, coords_dt, coords_sos,
            d_raw, mask, L, s_mean, s_std, time_scale,
            recon_config, lambda_fit, joint_steps, use_wandb, label,
        )
    elif mode == "alternating":
        return _train_alternating(
            denoiser, recon_model, coords_dt, coords_sos,
            d_raw, mask, L, s_mean, s_std, time_scale,
            recon_config, lambda_fit, joint_steps, use_wandb, label,
        )
    elif mode == "staged":
        return _train_staged(
            denoiser, recon_model, coords_dt, coords_sos,
            d_raw, mask, L, s_mean, s_std, time_scale,
            recon_config, denoise_cfg, lambda_fit,
            pretrain_denoiser_steps, pretrain_recon_steps, joint_steps,
            use_wandb, label,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ─── End-to-end training ─────────────────────────────────────────────────────

def _train_end_to_end(
    denoiser, recon_model, coords_dt, coords_sos,
    d_raw, mask, L, s_mean, s_std, time_scale,
    config, lambda_fit, steps, use_wandb, label,
):
    all_params = list(denoiser.parameters()) + list(recon_model.parameters())
    optimizer = optim.Adam(all_params, lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    loss_history = []
    recon_losses = []
    fit_losses = []
    best_loss = float("inf")
    best_state_dn = copy.deepcopy(denoiser.state_dict())
    best_state_rc = copy.deepcopy(recon_model.state_dict())

    pbar = tqdm(range(steps), desc=f"Joint E2E ({label})")
    for step in pbar:
        denoiser.train()
        recon_model.train()
        optimizer.zero_grad()

        # Denoiser forward
        d_denoised = denoiser(coords_dt)  # (131072, 1)

        # Reconstruction forward
        s_norm = recon_model(coords_sos)
        s_phys = (s_norm * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
        d_pred = L @ s_phys  # (131072, 1)

        # Reconstruction loss: ||mask * (d_pred - d_denoised)||²
        recon_res = (d_pred - d_denoised) * mask * time_scale
        n_valid = mask.sum() + 1e-8
        loss_recon = (recon_res ** 2).sum() / n_valid

        # Data fidelity loss: ||mask * (d_denoised - d_raw)||²
        fit_res = (d_denoised - d_raw) * mask * time_scale
        loss_fit = (fit_res ** 2).sum() / n_valid

        # Total
        total_loss = loss_recon + lambda_fit * loss_fit
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        recon_losses.append(loss_recon.item())
        fit_losses.append(loss_fit.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dn = copy.deepcopy(denoiser.state_dict())
            best_state_rc = copy.deepcopy(recon_model.state_dict())

        if step % 50 == 0:
            pbar.set_description(
                f"L_recon={loss_recon.item():.4f} L_fit={loss_fit.item():.4f}"
            )

        if use_wandb:
            import wandb
            wandb.log({
                "total_loss": loss_val,
                "recon_loss": loss_recon.item(),
                "fit_loss": loss_fit.item(),
            }, step=step)

    # Restore best and extract results
    denoiser.load_state_dict(best_state_dn)
    recon_model.load_state_dict(best_state_rc)

    return _extract_results(
        denoiser, recon_model, coords_dt, coords_sos,
        s_mean, s_std, loss_history, recon_losses, fit_losses,
    )


# ─── Alternating training ────────────────────────────────────────────────────

def _train_alternating(
    denoiser, recon_model, coords_dt, coords_sos,
    d_raw, mask, L, s_mean, s_std, time_scale,
    config, lambda_fit, steps, use_wandb, label,
    alt_interval=100,
):
    opt_denoiser = optim.Adam(denoiser.parameters(), lr=config.lr)
    opt_recon = optim.Adam(recon_model.parameters(), lr=config.lr)

    loss_history = []
    recon_losses = []
    fit_losses = []
    best_loss = float("inf")
    best_state_dn = copy.deepcopy(denoiser.state_dict())
    best_state_rc = copy.deepcopy(recon_model.state_dict())

    pbar = tqdm(range(steps), desc=f"Joint Alt ({label})")
    for step in pbar:
        # Alternate: even blocks train recon, odd blocks train denoiser
        block = (step // alt_interval) % 2
        train_recon = (block == 0)

        denoiser.train()
        recon_model.train()

        if train_recon:
            opt_recon.zero_grad()
        else:
            opt_denoiser.zero_grad()

        # Forward (both models, but only one gets gradients)
        d_denoised = denoiser(coords_dt)
        s_norm = recon_model(coords_sos)
        s_phys = (s_norm * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
        d_pred = L @ s_phys

        recon_res = (d_pred - d_denoised) * mask * time_scale
        n_valid = mask.sum() + 1e-8
        loss_recon = (recon_res ** 2).sum() / n_valid

        fit_res = (d_denoised - d_raw) * mask * time_scale
        loss_fit = (fit_res ** 2).sum() / n_valid

        if train_recon:
            # Only reconstruction loss for recon model
            loss_recon.backward()
            opt_recon.step()
        else:
            # Both losses for denoiser
            total_loss = loss_recon + lambda_fit * loss_fit
            total_loss.backward()
            opt_denoiser.step()

        loss_val = (loss_recon + lambda_fit * loss_fit).item()
        loss_history.append(loss_val)
        recon_losses.append(loss_recon.item())
        fit_losses.append(loss_fit.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dn = copy.deepcopy(denoiser.state_dict())
            best_state_rc = copy.deepcopy(recon_model.state_dict())

        if step % 50 == 0:
            phase = "recon" if train_recon else "denoise"
            pbar.set_description(
                f"[{phase}] L_rec={loss_recon.item():.4f} L_fit={loss_fit.item():.4f}"
            )

    denoiser.load_state_dict(best_state_dn)
    recon_model.load_state_dict(best_state_rc)

    return _extract_results(
        denoiser, recon_model, coords_dt, coords_sos,
        s_mean, s_std, loss_history, recon_losses, fit_losses,
    )


# ─── Staged training ─────────────────────────────────────────────────────────

def _train_staged(
    denoiser, recon_model, coords_dt, coords_sos,
    d_raw, mask, L, s_mean, s_std, time_scale,
    config, denoise_cfg, lambda_fit,
    pretrain_dn_steps, pretrain_rc_steps, joint_steps,
    use_wandb, label,
):
    loss_history = []
    recon_losses = []
    fit_losses = []
    n_valid = mask.sum() + 1e-8

    # ── Stage 1: Pretrain denoiser (blind, MSE on raw valid pixels) ──────
    log.info(f"  Stage 1: Pretrain denoiser ({pretrain_dn_steps} steps)")
    opt_dn = optim.Adam(denoiser.parameters(), lr=denoise_cfg.get("lr", 1e-3))

    # Normalize raw data per pair for denoiser targets
    d_raw_flat = d_raw.flatten()
    mask_flat = mask.flatten()
    d_targets_norm = torch.zeros_like(d_raw_flat)
    for k in range(_N_PAIRS):
        start = k * _PAIR_SIZE
        end = (k + 1) * _PAIR_SIZE
        d_targets_norm[start:end] = (
            (d_raw_flat[start:end] - denoiser.d_means[k]) / denoiser.d_stds[k]
        )

    for step in tqdm(range(pretrain_dn_steps), desc="Stage1:Denoise"):
        denoiser.train()
        opt_dn.zero_grad()
        d_denoised = denoiser(coords_dt)  # (131072, 1)
        res = (d_denoised.flatten() - d_raw_flat) * mask_flat * time_scale
        loss = (res ** 2).sum() / n_valid
        loss.backward()
        opt_dn.step()

    # ── Stage 2: Pretrain reconstructor on denoised data ─────────────────
    log.info(f"  Stage 2: Pretrain reconstructor ({pretrain_rc_steps} steps)")
    opt_rc = optim.Adam(recon_model.parameters(), lr=config.lr)
    sched_rc = optim.lr_scheduler.CosineAnnealingLR(opt_rc, T_max=pretrain_rc_steps)

    for step in tqdm(range(pretrain_rc_steps), desc="Stage2:Recon"):
        recon_model.train()
        opt_rc.zero_grad()

        with torch.no_grad():
            d_denoised = denoiser(coords_dt)

        s_norm = recon_model(coords_sos)
        s_phys = (s_norm * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
        d_pred = L @ s_phys

        res = (d_pred - d_denoised) * mask * time_scale
        loss = (res ** 2).sum() / n_valid
        loss.backward()
        opt_rc.step()
        sched_rc.step()

    # ── Stage 3: Joint fine-tuning ───────────────────────────────────────
    log.info(f"  Stage 3: Joint fine-tuning ({joint_steps} steps)")
    all_params = list(denoiser.parameters()) + list(recon_model.parameters())
    opt_joint = optim.Adam(all_params, lr=config.lr * 0.1)  # lower LR for fine-tuning
    sched_joint = optim.lr_scheduler.CosineAnnealingLR(opt_joint, T_max=joint_steps)

    best_loss = float("inf")
    best_state_dn = copy.deepcopy(denoiser.state_dict())
    best_state_rc = copy.deepcopy(recon_model.state_dict())

    pbar = tqdm(range(joint_steps), desc="Stage3:Joint")
    for step in pbar:
        denoiser.train()
        recon_model.train()
        opt_joint.zero_grad()

        d_denoised = denoiser(coords_dt)
        s_norm = recon_model(coords_sos)
        s_phys = (s_norm * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
        d_pred = L @ s_phys

        recon_res = (d_pred - d_denoised) * mask * time_scale
        loss_recon = (recon_res ** 2).sum() / n_valid

        fit_res = (d_denoised - d_raw) * mask * time_scale
        loss_fit = (fit_res ** 2).sum() / n_valid

        total_loss = loss_recon + lambda_fit * loss_fit
        total_loss.backward()
        opt_joint.step()
        sched_joint.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        recon_losses.append(loss_recon.item())
        fit_losses.append(loss_fit.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dn = copy.deepcopy(denoiser.state_dict())
            best_state_rc = copy.deepcopy(recon_model.state_dict())

        if step % 50 == 0:
            pbar.set_description(
                f"L_rec={loss_recon.item():.4f} L_fit={loss_fit.item():.4f}"
            )

    denoiser.load_state_dict(best_state_dn)
    recon_model.load_state_dict(best_state_rc)

    return _extract_results(
        denoiser, recon_model, coords_dt, coords_sos,
        s_mean, s_std, loss_history, recon_losses, fit_losses,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_results(denoiser, recon_model, coords_dt, coords_sos,
                     s_mean, s_std, loss_history, recon_losses, fit_losses):
    """Extract final predictions from trained models."""
    denoiser.eval()
    recon_model.eval()
    with torch.no_grad():
        d_denoised = denoiser(coords_dt)
        s_norm = recon_model(coords_sos)
        s_phys = (s_norm * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)

    return {
        "s_phys": s_phys.detach().cpu(),
        "s_norm": s_norm.detach().cpu(),
        "d_denoised": d_denoised.detach().cpu(),
        "loss_history": loss_history,
        "recon_losses": recon_losses,
        "fit_losses": fit_losses,
    }
