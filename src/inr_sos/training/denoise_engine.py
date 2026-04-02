"""
Measurement-domain INR denoiser.

Trains one small INR per firing pair on the 128×128 DT displacement grid.
The INR's spectral bias smooths/denoises valid measurements and fills NaN gaps.

Usage:
    from inr_sos.training.denoise_engine import denoise_measurements
    d_denoised, info = denoise_measurements(sample, grid, config)
"""

import copy
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from inr_sos.models.mlp import FourierMLP, ReluMLP, GeluMLP
from inr_sos.models.siren import SirenMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("denoise_engine")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_N_PAIRS = 8
_PAIR_SIZE = 16384  # 128 * 128
_DT_RES = 128

_MODEL_MAP = {
    "FourierMLP": FourierMLP,
    "ReluMLP": ReluMLP,
    "GeluMLP": GeluMLP,
    "SirenMLP": SirenMLP,
}

# ─── Default denoiser config ─────────────────────────────────────────────────

DEFAULT_DENOISE_CFG = {
    "model_type": "FourierMLP",
    "hidden_features": 64,
    "hidden_layers": 2,
    "mapping_size": 32,
    "scale": 2.0,
    "omega": 15.0,
    "lr": 1e-3,
    "steps": 500,
    "patience": 50,
    "val_fraction": 0.1,
}


# ─── Coordinate helpers ──────────────────────────────────────────────────────

def _build_dt_coords(grid):
    """Build normalized (x, z) coordinates for the 128×128 DT grid.

    Uses the DT grid's own physical extent for normalization (not SoS grid).
    Flattening order: ix (outer) × iz (inner) → row_idx = ix * 128 + iz.

    Returns:
        Tensor of shape (16384, 2) on _DEVICE.
    """
    x_dt = grid.x_dt  # (128,)
    z_dt = grid.z_dt  # (128,)

    # Meshgrid: ix is outer (rows), iz is inner (cols)
    X, Z = np.meshgrid(x_dt, z_dt, indexing="ij")  # both (128, 128)
    x_flat = X.flatten()  # (16384,) — ix*128+iz ordering
    z_flat = Z.flatten()

    x_norm, z_norm = grid.normalize_dt(x_flat, z_flat)
    coords = torch.tensor(
        np.stack([x_norm, z_norm], axis=1), dtype=torch.float32
    ).to(_DEVICE)
    return coords  # (16384, 2)


def _build_model(cfg):
    """Instantiate a fresh INR model from a denoiser config dict."""
    mtype = cfg.get("model_type", "FourierMLP")
    model_cls = _MODEL_MAP[mtype]

    kwargs = dict(
        in_features=2,
        hidden_features=cfg.get("hidden_features", 64),
        hidden_layers=cfg.get("hidden_layers", 2),
        mapping_size=cfg.get("mapping_size", 32),
    )
    if mtype == "FourierMLP":
        kwargs["scale"] = cfg.get("scale", 2.0)
    elif mtype == "SirenMLP":
        kwargs["omega"] = cfg.get("omega", 15.0)

    return model_cls(**kwargs)


# ─── Single-pair denoiser ────────────────────────────────────────────────────

def _denoise_single_pair(coords, d_pair, mask_pair, cfg, pair_idx=0):
    """Train one INR on one firing pair's valid pixels, predict everywhere.

    Args:
        coords:    (16384, 2) normalized DT coordinates, on device.
        d_pair:    (16384, 1) displacement values (NaN replaced with 0).
        mask_pair: (16384, 1) validity mask (1=valid, 0=NaN).
        cfg:       denoiser config dict.
        pair_idx:  int, for logging.

    Returns:
        d_denoised: (16384, 1) denoised displacement, all pixels.
        info:       dict with loss histories, stats.
    """
    d_pair = d_pair.to(_DEVICE)
    mask_pair = mask_pair.to(_DEVICE)

    # ── Per-pair normalization ────────────────────────────────────────────
    valid_mask = mask_pair.flatten() > 0.5
    valid_d = d_pair.flatten()[valid_mask]
    d_mean = valid_d.mean()
    d_std = valid_d.std() + 1e-10
    d_norm = (d_pair - d_mean) / d_std  # (16384, 1)

    # ── Train/val split on valid pixels ───────────────────────────────────
    valid_idx = torch.where(valid_mask)[0]
    n_valid = len(valid_idx)
    n_val = max(1, int(n_valid * cfg.get("val_fraction", 0.1)))

    perm = valid_idx[torch.randperm(n_valid, device=_DEVICE)]
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_mask = torch.zeros_like(mask_pair.flatten())
    train_mask[train_idx] = 1.0
    val_mask = torch.zeros_like(mask_pair.flatten())
    val_mask[val_idx] = 1.0

    # ── Model + optimizer ─────────────────────────────────────────────────
    model = _build_model(cfg).to(_DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    steps = cfg.get("steps", 500)
    patience = cfg.get("patience", 50)

    best_val_loss = float("inf")
    wait = 0
    best_state = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []

    # ── Training loop ─────────────────────────────────────────────────────
    targets = d_norm.flatten()  # (16384,)

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        pred = model(coords).flatten()  # (16384,)
        residual = (pred - targets) * train_mask
        n_train = train_mask.sum() + 1e-8
        loss = (residual ** 2).sum() / n_train

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Val loss
        if step % 10 == 0:
            with torch.no_grad():
                val_residual = (pred - targets) * val_mask
                n_val_pixels = val_mask.sum() + 1e-8
                val_loss = (val_residual ** 2).sum() / n_val_pixels
                val_loss_val = val_loss.item()
                val_losses.append(val_loss_val)

                if val_loss_val < best_val_loss:
                    best_val_loss = val_loss_val
                    wait = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    wait += 1

                if wait >= patience:
                    log.info(
                        f"  Pair {pair_idx}: early stop at step {step} "
                        f"(val_loss={best_val_loss:.6f})"
                    )
                    break

    # ── Predict on all pixels ─────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        d_denoised_norm = model(coords).flatten()  # (16384,)

    # Denormalize
    d_denoised = d_denoised_norm * d_std + d_mean

    # Clip to ±3σ of valid pixel range for safety
    d_clip_lo = d_mean - 3 * d_std
    d_clip_hi = d_mean + 3 * d_std
    d_denoised = d_denoised.clamp(min=d_clip_lo.item(), max=d_clip_hi.item())

    n_filled = int((~valid_mask).sum().item())
    log.info(
        f"  Pair {pair_idx}: {n_valid} valid, {n_filled} filled, "
        f"steps={len(train_losses)}, best_val={best_val_loss:.6f}"
    )

    info = {
        "pair_idx": pair_idx,
        "n_valid": n_valid,
        "n_filled": n_filled,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "d_mean": d_mean.item(),
        "d_std": d_std.item(),
    }

    return d_denoised.unsqueeze(1).cpu(), info  # (16384, 1)


# ─── Public API ───────────────────────────────────────────────────────────────

def denoise_measurements(sample, grid, cfg=None):
    """Denoise displacement measurements using per-pair INR fitting.

    Args:
        sample: dataset sample dict with 'd_meas' (131072, 1) and 'mask' (131072, 1).
        grid:   USGrid instance (must have x_dt, z_dt).
        cfg:    denoiser config dict (uses DEFAULT_DENOISE_CFG if None).

    Returns:
        d_denoised: Tensor (131072, 1), denoised measurements (all valid).
        info:       dict with per-pair diagnostics.
    """
    if cfg is None:
        cfg = DEFAULT_DENOISE_CFG

    d_meas = sample["d_meas"]  # (131072, 1)
    mask = sample["mask"]      # (131072, 1)

    # Build DT coordinate grid (shared across all pairs)
    coords = _build_dt_coords(grid)  # (16384, 2) on device

    # Denoise each pair independently
    denoised_pairs = []
    pair_infos = []

    for k in range(_N_PAIRS):
        start = k * _PAIR_SIZE
        end = (k + 1) * _PAIR_SIZE
        d_pair = d_meas[start:end]    # (16384, 1)
        m_pair = mask[start:end]      # (16384, 1)

        d_denoised_k, info_k = _denoise_single_pair(
            coords, d_pair, m_pair, cfg, pair_idx=k
        )
        denoised_pairs.append(d_denoised_k)
        pair_infos.append(info_k)

    # Reassemble
    d_denoised = torch.cat(denoised_pairs, dim=0)  # (131072, 1)

    info = {
        "pairs": pair_infos,
        "n_pairs": _N_PAIRS,
        "cfg": cfg,
    }

    return d_denoised, info
