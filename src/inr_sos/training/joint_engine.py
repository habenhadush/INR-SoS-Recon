"""
Joint Denoiser + Reconstructor optimization engine (Experiment 11).

Trains a per-pair denoiser INR and a reconstruction INR jointly via staged training:
  Stage 1: pretrain denoiser (blind, MSE on raw valid pixels)
  Stage 2: pretrain reconstructor (on denoised data, frozen denoiser)
  Stage 3: joint fine-tune (both networks, lower LR, combined loss)

Loss = ||mask * (L @ s_pred - d_denoised)||² + λ(t) * ||mask * (d_denoised - d_raw)||²

λ strategies for Stage 3:
  - fixed:    constant λ throughout training
  - cosine:   cosine decay from λ_max to λ_min
  - balanced: adaptive λ that keeps L_recon / L_fit near a target ratio
  - residual: per-sample λ set from post-Stage-2 residual ratio
"""

import copy
import logging
import math
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


# ─── Lambda scheduling ────────────────────────────────────────────────────────

def _compute_lambda(strategy, step, total_steps, loss_recon, loss_fit, state):
    """Compute λ_fit at the current step based on the chosen strategy.

    Args:
        strategy:    "fixed" | "cosine" | "balanced" | "residual"
        step:        current training step
        total_steps: total Stage 3 steps
        loss_recon:  L_recon value (scalar, detached)
        loss_fit:    L_fit value (scalar, detached)
        state:       mutable dict with strategy-specific state

    Returns:
        (lam, state) — current λ value and updated state
    """
    if strategy == "fixed":
        return state["lambda_fit"], state

    elif strategy == "cosine":
        lam_max = state["lambda_max"]
        lam_min = state["lambda_min"]
        progress = step / max(total_steps - 1, 1)
        lam = lam_min + 0.5 * (lam_max - lam_min) * (1 + math.cos(math.pi * progress))
        return lam, state

    elif strategy == "balanced":
        target = state["target_ratio"]
        eta = state.get("eta", 0.01)
        lam = state["lambda_current"]
        lam_min = state.get("lambda_min", 0.001)
        lam_max = state.get("lambda_max", 10.0)

        ratio = loss_recon / (loss_fit + 1e-10)
        if ratio > target:
            lam *= (1 - eta)  # denoiser too conservative → loosen
        else:
            lam *= (1 + eta)  # denoiser drifting → tighten
        lam = max(lam_min, min(lam_max, lam))
        state["lambda_current"] = lam
        return lam, state

    elif strategy == "residual":
        return state["lambda_sample"], state

    else:
        raise ValueError(f"Unknown lambda strategy: {strategy}")


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
    mode="staged",
    lambda_strategy="fixed",
    lambda_cfg=None,
    pretrain_denoiser_steps=300,
    pretrain_recon_steps=500,
    joint_steps=None,
    joint_lr_factor=0.1,
    use_wandb=False,
    label="joint",
):
    """Joint denoiser + reconstructor optimization (staged training).

    Args:
        sample:       dataset sample dict.
        L_matrix:     forward model matrix (dense or sparse).
        grid:         USGrid instance.
        recon_model:  reconstruction INR (ReluMLP etc.), freshly initialized.
        recon_config: ExperimentConfig for reconstruction.
        denoise_cfg:  dict for denoiser INR config (defaults to DEFAULT_DENOISE_CFG).
        lambda_fit:   base λ for data-fidelity loss (used by fixed/cosine/balanced).
        mode:         training mode (only 'staged' supported).
        lambda_strategy: "fixed" | "cosine" | "balanced" | "residual"
        lambda_cfg:   strategy-specific parameters dict. Keys depend on strategy:
                      cosine:   lambda_max, lambda_min
                      balanced: target_ratio, eta, lambda_min, lambda_max
                      residual: alpha
        pretrain_denoiser_steps: steps for stage 1.
        pretrain_recon_steps:    steps for stage 2.
        joint_steps:  total joint training steps (defaults to recon_config.steps).
        joint_lr_factor: LR multiplier for Stage 3 (default 0.1).
        use_wandb:    whether to log to W&B.
        label:        run label for logging.

    Returns:
        dict with 's_phys', 'loss_history', 'd_denoised', 'lambda_trajectory', etc.
    """
    if denoise_cfg is None:
        denoise_cfg = dict(DEFAULT_DENOISE_CFG)
    if lambda_cfg is None:
        lambda_cfg = {}
    if joint_steps is None:
        joint_steps = recon_config.steps

    if mode != "staged":
        raise ValueError(f"Only 'staged' mode is supported, got: {mode}")

    log.info(f"\n--- Joint Denoiser+Recon: {label} (staged) on {_DEVICE} ---")
    log.info(f"  lambda_strategy={lambda_strategy}, lambda_fit={lambda_fit}")
    if lambda_cfg:
        log.info(f"  lambda_cfg={lambda_cfg}")

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

    return _train_staged(
        denoiser, recon_model, coords_dt, coords_sos,
        d_raw, mask, L, s_mean, s_std, time_scale,
        recon_config, denoise_cfg, lambda_fit,
        lambda_strategy, lambda_cfg,
        pretrain_denoiser_steps, pretrain_recon_steps, joint_steps,
        joint_lr_factor, use_wandb, label,
    )


# ─── Staged training ─────────────────────────────────────────────────────────

def _train_staged(
    denoiser, recon_model, coords_dt, coords_sos,
    d_raw, mask, L, s_mean, s_std, time_scale,
    config, denoise_cfg, lambda_fit,
    lambda_strategy, lambda_cfg,
    pretrain_dn_steps, pretrain_rc_steps, joint_steps,
    joint_lr_factor, use_wandb, label,
):
    loss_history = []
    recon_losses = []
    fit_losses = []
    lambda_trajectory = []
    n_valid = mask.sum() + 1e-8

    # ── Stage 1: Pretrain denoiser (blind, MSE on raw valid pixels) ──────
    log.info(f"  Stage 1: Pretrain denoiser ({pretrain_dn_steps} steps)")
    opt_dn = optim.Adam(denoiser.parameters(), lr=denoise_cfg.get("lr", 1e-3))

    d_raw_flat = d_raw.flatten()
    mask_flat = mask.flatten()

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

    # ── Compute residuals after Stage 2 (for adaptive strategies) ────────
    with torch.no_grad():
        d_denoised_s2 = denoiser(coords_dt)
        s_norm_s2 = recon_model(coords_sos)
        s_phys_s2 = (s_norm_s2 * s_std + s_mean).clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
        d_pred_s2 = L @ s_phys_s2

        r0 = ((d_pred_s2 - d_denoised_s2) * mask * time_scale).pow(2).sum() / n_valid
        f0 = ((d_denoised_s2 - d_raw) * mask * time_scale).pow(2).sum() / n_valid
        r0_val = r0.item()
        f0_val = f0.item()

    log.info(f"  Post-Stage2 residuals: L_recon={r0_val:.6f}, L_fit={f0_val:.6f}, "
             f"ratio={r0_val / (f0_val + 1e-10):.2f}")

    # ── Build lambda state for Stage 3 ───────────────────────────────────
    lam_state = {"lambda_fit": lambda_fit}

    if lambda_strategy == "cosine":
        lam_state["lambda_max"] = lambda_cfg.get("lambda_max", 1.0)
        lam_state["lambda_min"] = lambda_cfg.get("lambda_min", 0.01)
        log.info(f"  λ strategy: cosine decay {lam_state['lambda_max']} → {lam_state['lambda_min']}")

    elif lambda_strategy == "balanced":
        lam_state["target_ratio"] = lambda_cfg.get("target_ratio", 1.0)
        lam_state["eta"] = lambda_cfg.get("eta", 0.01)
        lam_state["lambda_min"] = lambda_cfg.get("lambda_min", 0.001)
        lam_state["lambda_max"] = lambda_cfg.get("lambda_max", 10.0)
        lam_state["lambda_current"] = lambda_fit  # start from base λ
        log.info(f"  λ strategy: balanced, target_ratio={lam_state['target_ratio']}, "
                 f"η={lam_state['eta']}, init λ={lambda_fit}")

    elif lambda_strategy == "residual":
        alpha = lambda_cfg.get("alpha", 0.5)
        ratio = r0_val / (f0_val + 1e-10)
        lam_sample = alpha * ratio
        lam_state["lambda_sample"] = lam_sample
        log.info(f"  λ strategy: residual, α={alpha}, r₀/f₀={ratio:.2f}, λ_sample={lam_sample:.4f}")

    elif lambda_strategy == "fixed":
        log.info(f"  λ strategy: fixed, λ={lambda_fit}")

    # ── Stage 3: Joint fine-tuning ───────────────────────────────────────
    log.info(f"  Stage 3: Joint fine-tuning ({joint_steps} steps)")
    all_params = list(denoiser.parameters()) + list(recon_model.parameters())
    opt_joint = optim.Adam(all_params, lr=config.lr * joint_lr_factor)
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

        # Adaptive λ
        lam, lam_state = _compute_lambda(
            lambda_strategy, step, joint_steps,
            loss_recon.item(), loss_fit.item(), lam_state,
        )

        total_loss = loss_recon + lam * loss_fit
        total_loss.backward()
        opt_joint.step()
        sched_joint.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)
        recon_losses.append(loss_recon.item())
        fit_losses.append(loss_fit.item())
        lambda_trajectory.append(lam)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state_dn = copy.deepcopy(denoiser.state_dict())
            best_state_rc = copy.deepcopy(recon_model.state_dict())

        if step % 50 == 0:
            pbar.set_description(
                f"L_rec={loss_recon.item():.4f} L_fit={loss_fit.item():.4f} λ={lam:.4f}"
            )

    denoiser.load_state_dict(best_state_dn)
    recon_model.load_state_dict(best_state_rc)

    return _extract_results(
        denoiser, recon_model, coords_dt, coords_sos,
        s_mean, s_std, loss_history, recon_losses, fit_losses,
        lambda_trajectory,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_results(denoiser, recon_model, coords_dt, coords_sos,
                     s_mean, s_std, loss_history, recon_losses, fit_losses,
                     lambda_trajectory=None):
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
        "lambda_trajectory": lambda_trajectory or [],
    }
