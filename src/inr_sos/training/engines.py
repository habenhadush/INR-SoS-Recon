import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import inspect
import wandb # <-- Ensure wandb is imported!

from torch.utils.data import DataLoader
from inr_sos.utils.data import RayDataset
from tqdm import tqdm
from inr_sos.utils.config import ExperimentConfig as ec

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Physical slowness bounds (s/m)
# ---------------------------------------------------------------------------
_SLOWNESS_MIN = 1.0 / 1800.0   # fastest tissue  ~5.56e-4
_SLOWNESS_MAX = 1.0 / 1200.0   # slowest tissue  ~8.33e-4


def _compute_data_loss(residual_seconds, mask, config):
    """Compute data-fidelity loss with switchable MSE / Huber.

    Args:
        residual_seconds: (d_pred - d_meas) in seconds, NOT yet masked.
        mask:             binary mask (1 = valid ray).
        config:           ExperimentConfig instance.

    Returns:
        Scalar loss tensor (mean over valid rays).
    """
    # Apply mask FIRST, then scale
    residual_masked = residual_seconds * mask
    residual_scaled = residual_masked * config.time_scale

    n_valid = mask.sum() + 1e-8

    if config.loss_type == "huber":
        # F.huber_loss with reduction='sum' then normalise manually
        target = torch.zeros_like(residual_scaled)
        loss = F.huber_loss(
            residual_scaled, target,
            delta=config.huber_delta,
            reduction="sum",
        ) / n_valid
    else:
        # Default: MSE (matches original behaviour)
        loss = (residual_scaled ** 2).sum() / n_valid

    return loss


def _maybe_clamp_slowness(s_phys, config):
    """Optionally clamp slowness to physical bounds."""
    if config.clamp_slowness:
        return s_phys.clamp(min=_SLOWNESS_MIN, max=_SLOWNESS_MAX)
    return s_phys


class _EarlyStopper:
    """Lightweight early-stopping tracker.

    Splits valid ray indices into train/val sets and tracks validation loss.
    When no improvement is seen for ``patience`` evaluation cycles, signals
    that training should stop.
    """

    def __init__(self, mask, config, model):
        """
        Args:
            mask:   1-D binary tensor on device (1 = valid ray).
            config: ExperimentConfig.
            model:  the INR model (used to snapshot best weights).
        """
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

        # Build train/val masks (same shape as original mask)
        self.train_mask = torch.zeros_like(mask)
        self.train_mask[self.train_idx] = 1.0
        self.val_mask = torch.zeros_like(mask)
        self.val_mask[self.val_idx] = 1.0

        self.patience = config.patience
        self.best_val_loss = float("inf")
        self.wait = 0
        self.best_state = copy.deepcopy(model.state_dict())

    def get_train_mask(self, original_mask):
        """Return the training-only mask (or original if ES disabled)."""
        if not self.enabled:
            return original_mask
        return self.train_mask

    def evaluate(self, residual_seconds, config, model):
        """Compute val loss and update patience counter.

        Returns:
            (val_loss_value, should_stop)
        """
        if not self.enabled:
            return None, False

        residual_val = residual_seconds * self.val_mask
        residual_val_scaled = residual_val * config.time_scale
        n_val = self.val_mask.sum() + 1e-8

        if config.loss_type == "huber":
            val_loss = F.huber_loss(
                residual_val_scaled,
                torch.zeros_like(residual_val_scaled),
                delta=config.huber_delta,
                reduction="sum",
            ) / n_val
        else:
            val_loss = (residual_val_scaled ** 2).sum() / n_val

        val_loss_val = val_loss.item()

        if val_loss_val < self.best_val_loss:
            self.best_val_loss = val_loss_val
            self.wait = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1

        return val_loss_val, self.wait >= self.patience

    def restore_best(self, model):
        """Load the best model weights (no-op if ES disabled)."""
        if self.enabled:
            model.load_state_dict(self.best_state)


def plot(result_dict, sample, grid_shape=(64, 64), title="Reconstruction"):
    """
    Visualizes the reconstruction results, comparing GT vs Prediction.
    
    Args:
        result_dict (dict): Output from the reconstruct_sos function.
                            Must contain keys: 's_phys', 'loss_history'.
        sample (dict): The data sample (containing 's_gt_raw').
        grid_shape (tuple): The shape to reshape the 1D arrays (usually 64x64).
        title (str): Title for the figure.
    """
    
    # 1. Extract Data & Convert to Numpy
    # Ground Truth (Raw Slowness)
    s_gt = sample['s_gt_raw'].view(grid_shape).detach().cpu().numpy()
    
    # Reconstruction (Physical Slowness)
    s_rec = result_dict['s_phys'].view(grid_shape).detach().cpu().numpy()
    
    # Loss History
    loss_hist = result_dict['loss_history']
    
    # 2. Convert Slowness (s/m) -> Speed of Sound (m/s)
    # v = 1 / s
    v_gt = 1.0 / (s_gt + 1e-8)
    v_rec = 1.0 / (s_rec + 1e-8)
    
    # 3. Compute Absolute Error (in m/s)
    error_map = np.abs(v_gt - v_rec)
    mae = np.mean(error_map)
    
    # 4. Create Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{title} | MAE: {mae:.2f} m/s", fontsize=16)
    
    # Plot A: Ground Truth
    im0 = axes[0].imshow(v_gt, cmap='jet', vmin=1400, vmax=1600)
    axes[0].set_title("Ground Truth (m/s)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot B: Reconstruction
    im1 = axes[1].imshow(v_rec, cmap='jet', vmin=1400, vmax=1600)
    axes[1].set_title("Reconstruction (m/s)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot C: Error Map
    # Use 'hot' or 'inferno' to highlight errors. 
    im2 = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=50) 
    axes[2].set_title("Abs. Error (m/s)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Plot D: Loss Curve
    axes[3].plot(loss_hist, label='Total Loss', color='blue')
    axes[3].set_title("Optimization Loss")
    axes[3].set_xlabel("Iterations")
    axes[3].set_ylabel("Loss (Scaled)")
    axes[3].set_yscale('log') # Log scale is usually better for convergence plots
    axes[3].grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.show()


"""
# this is too slow, thus need to figure out how to make the dataloader faster
def reconstruct_no_optimal(sample, L_matrix, model, label, config):
    warnings.warn(
            "Dataloader is very slow moving L matrix from harddrive to gpu. "
            "Use slicing instead.",
            DeprecationWarning,
            stacklevel=2
    )
    
    logging.info(f"\n--- Training {label} (Ray-Batching) on {_DEVICE} ---")
    
    # --- 1. Device & Data Setup ---
    model = model.to(_DEVICE)
    coords   = sample['coords'].to(_DEVICE)
    
    s_mean = sample['s_stats'][0].to(_DEVICE).item()
    s_std  = sample['s_stats'][1].to(_DEVICE).item()

    # --- 2. Dataloader Setup ---
    ray_dataset = RayDataset(
        L_matrix = L_matrix,
        displacement_filed=sample['d_meas'],
        mask=sample['mask']
    )

    batch_size = config.get('batch_size', 4096)
    ray_loader = DataLoader(
        ray_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=True if _DEVICE.type == 'cuda' else False,
        drop_last=False
    )

    # --- 3. Optimizer Setup ---
    epochs = config.get('epochs', 150)
    total_steps = epochs * len(ray_loader)    
    time_scale = config.get('time_scale', 1e6)

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6
    )

    loss_history = []

    # --- 4. Optimization Loop ---
    pbar = tqdm(range(epochs), desc="Epochs")
    time_scale = config.get('time_scale', 1e6)
    for epoch in pbar:
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch in ray_loader:
            # only current batches
            L_batch = batch['L_row'].to(_DEVICE)
            d_batch = batch['displacement'].to(_DEVICE)
            m_batch = batch['mask'].to(_DEVICE) 
            
            s_norm = model(coords)
            s_phys = s_norm * s_std + s_mean
            
            # (Batch_size, 4096) @ (4096, 1) -> (Batch_size, 1)
            d_pred_batch = L_batch @ s_phys

            residual = (d_pred_batch - d_batch) * m_batch
            loss_data = ((residual * time_scale) ** 2).sum() / (m_batch.sum() + 1e-8)
            reg_loss = 0.0
            if config.get('reg_weight', 0.0) > 0:
                reg_loss = config['reg_weight'] * (s_norm ** 2).mean()

            if config.get('tv_weight', 0.0) > 0:
                s_img = s_phys.reshape(64, 64)
                tv_x = torch.abs(s_img[:, 1:] - s_img[:, :-1]).mean()
                tv_z = torch.abs(s_img[1:, :] - s_img[:-1, :]).mean()
                reg_loss += config['tv_weight'] * (tv_x + tv_z)
        
        total_loss = loss_data + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss_data.item()

    # --- 5. Epoch-Level Logging ---
    avg_epoch_loss = epoch_loss / len(ray_loader)
    loss_history.append(avg_epoch_loss)
    
        
    if epoch % config.get('log_interval', 10) == 0:
            pbar.set_description(f"Avg Loss (us^2): {avg_epoch_loss:.4f}")

    # --- 6. Final Processing ---
    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = s_norm * s_std + s_mean

    return {
        's_phys': s_phys.detach().cpu(),
        's_norm': s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state': model.state_dict()
    } 
"""


def optimize_with_svd_constraint(sample, L_matrix, model, label, config: ec, use_wandb=False):
    """Phase 2: SVD-Constrained Reconstruction.
    
    Restricts the anatomy field to the stable subspace of the L-matrix
    to prevent error amplification.
    """
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} + SVD-Constraint on {_DEVICE} ---")

    model  = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask   = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std  = sample['s_stats'][1].item()
    L      = L_matrix.to(_DEVICE)

    # Load SVD components (precomputed or computed on first call)
    # Note: USDataset.load_svd is called outside or here
    # For efficiency, we assume Vt_k is provided or we get it from dataset
    # Here we'll compute it if not available, but ideally it's passed in.
    # Since we don't want to change the signature too much, we'll try to find it.
    
    # Precompute or load SVD
    # In a real run, we'd use config.svd_path
    if not hasattr(optimize_with_svd_constraint, "_Vt_k"):
        # This is a bit hacky but keeps the signature consistent with other engines
        # We assume the caller might have attached it to the model or we compute it once
        logging.info("SVD components not cached in engine. Computing...")
        L_np = L_matrix.numpy()
        U, S, Vt = np.linalg.svd(L_np, full_matrices=False)
        Vt_k = torch.tensor(Vt[:config.svd_k, :], dtype=torch.float32, device=_DEVICE)
        optimize_with_svd_constraint._Vt_k = Vt_k
    
    Vt_k = optimize_with_svd_constraint._Vt_k

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    loss_history = []
    pbar = tqdm(range(config.steps))

    for step in pbar:
        model.train()
        optimizer.zero_grad()

        # 1. Forward Anatomy
        s_norm = model(coords)
        s_phys = s_norm * s_std + s_mean
        
        # 2. Project onto Stable Subspace
        # s_stable = V_k @ (V_k^T @ s_phys)
        # Note: s_phys is (4096, 1), Vt_k is (k, 4096)
        s_stable = Vt_k.t() @ (Vt_k @ s_phys)
        
        s_stable_clamped = _maybe_clamp_slowness(s_stable, config)

        # 3. Prediction and Loss
        d_pred = L @ s_stable_clamped
        residual_seconds = d_pred - d_meas
        loss = _compute_data_loss(residual_seconds, mask, config)

        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")

        if use_wandb:
            wandb.log({
                "Total Loss": total_loss.item(),
                "Data Loss":  loss.item(),
                "Learning Rate": scheduler.get_last_lr()[0]
            }, step=step)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = s_norm * s_std + s_mean
        s_stable = Vt_k.t() @ (Vt_k @ s_phys)
        s_final = _maybe_clamp_slowness(s_stable, config)

    return {
        's_phys':       s_final.detach().cpu(),
        's_norm':       s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }


def optimize_direct_supervision(sample, L_matrix, model, label, config: ec, use_wandb=False):
    """Phase 1: Direct GT memorization (no forward model)."""
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} (Direct GT) on {_DEVICE} ---")

    model = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    s_gt_normalized   = sample['s_gt_normalized'].to(_DEVICE)
    s_mean = sample['s_stats'][0].to(_DEVICE).item()
    s_std  = sample['s_stats'][1].to(_DEVICE).item()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    loss_history = []
    pbar = tqdm(range(config.steps))

    for step in pbar:
        pbar.set_postfix({"method": inspect.currentframe().f_code.co_name, "model": label})
        optimizer.zero_grad()
        s_norm = model(coords)
        loss = loss_fn(s_norm, s_gt_normalized)

        reg_loss = 0.0
        if config.reg_weight > 0:
            reg_loss = config.reg_weight * (s_norm ** 2).mean()

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            pbar.set_description(f"Loss: {loss.item():.4e}")

        if use_wandb:
            wandb.log({
                "Total Loss":    total_loss.item(),
                "MSE Loss":      loss.item(),
                "Learning Rate": scheduler.get_last_lr()[0]
            }, step=step)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = s_norm * s_std + s_mean

    return {
        's_phys':       s_phys.detach().cpu(),
        's_norm':       s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }


def optimize_full_forward_operator(sample, L_matrix, model, label, config: ec, use_wandb=False):
    """Phase 1: Full-matrix reconstruction.

    Supports: Huber/MSE loss, slowness clamping, early stopping.
    """
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} (full-matrix) on {_DEVICE} ---")

    model  = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask   = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std  = sample['s_stats'][1].item()
    L      = L_matrix.to(_DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    # Early stopping setup
    stopper = _EarlyStopper(mask, config, model)
    train_mask = stopper.get_train_mask(mask)

    loss_history = []
    pbar = tqdm(range(config.steps))

    for step in pbar:
        pbar.set_postfix({"method": inspect.currentframe().f_code.co_name, "model": label})
        model.train()
        optimizer.zero_grad()

        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

        d_pred_seconds   = L @ s_phys
        residual_seconds = d_pred_seconds - d_meas
        loss = _compute_data_loss(residual_seconds, train_mask, config)

        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()
        if config.tv_weight > 0:
            s_img = s_phys.reshape(64, 64)
            tv_x  = ((s_img[:, 1:] - s_img[:, :-1]) ** 2).mean()
            tv_z  = ((s_img[1:, :] - s_img[:-1, :]) ** 2).mean()
            reg_loss += config.tv_weight * (tv_x + tv_z)

        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            pbar.set_description(f"Loss (us^2): {loss.item():.4f}")

        # Early stopping evaluation
        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                val_loss, should_stop = stopper.evaluate(residual_seconds.detach(), config, model)
            if use_wandb and val_loss is not None:
                wandb.log({"Val Loss": val_loss}, step=step)
            if should_stop:
                logging.info(f"Early stopping at step {step} (patience={config.patience})")
                break

        if use_wandb:
            wandb.log({
                "Total Loss":    total_loss.item(),
                "Data Loss":     loss.item(),
                "Learning Rate": scheduler.get_last_lr()[0]
            }, step=step)

    # Restore best model if early stopping was used
    stopper.restore_best(model)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

    return {
        's_phys':       s_phys.detach().cpu(),
        's_norm':       s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }


def optimize_sequential_views(sample, L_matrix, model, label, config: ec, use_wandb=False):
    """Phase 2b: Pair-by-pair reconstruction.

    Supports: Huber/MSE loss, slowness clamping, early stopping.
    Early stopping uses the *full* residual (all pairs) evaluated every log_interval.
    """
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} (Pair-by-Pair) on {_DEVICE} ---")

    model  = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask   = sample['mask'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std  = sample['s_stats'][1].item()
    L      = L_matrix.to(_DEVICE)

    n_pairs   = 8
    pair_size = L.shape[0] // n_pairs
    L_pairs   = [L[k * pair_size:(k + 1) * pair_size, :] for k in range(n_pairs)]
    d_pairs   = [d_meas[k * pair_size:(k + 1) * pair_size] for k in range(n_pairs)]
    m_pairs   = [mask[k * pair_size:(k + 1) * pair_size]   for k in range(n_pairs)]

    # Early stopping operates on the full residual
    stopper = _EarlyStopper(mask, config, model)
    # Per-pair train masks (sliced from the full train mask)
    if stopper.enabled:
        train_m_pairs = [stopper.train_mask[k * pair_size:(k + 1) * pair_size] for k in range(n_pairs)]
    else:
        train_m_pairs = m_pairs

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.steps * n_pairs, eta_min=1e-6
    )

    loss_history = []
    pbar = tqdm(range(config.steps))

    for step in pbar:
        step_pair_losses = []
        pair_order = torch.randperm(n_pairs).tolist()
        pbar.set_postfix({"method": inspect.currentframe().f_code.co_name, "model": label})

        for k in pair_order:
            model.train()
            optimizer.zero_grad()
            s_norm = model(coords)
            s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

            d_pred_k   = L_pairs[k] @ s_phys
            residual_k = d_pred_k - d_pairs[k]
            loss_k     = _compute_data_loss(residual_k, train_m_pairs[k], config)

            reg_loss = 0
            if config.reg_weight > 0:
                reg_loss = config.reg_weight * (s_norm ** 2).mean()
            if config.tv_weight > 0:
                s_img  = s_phys.reshape(64, 64)
                tv_x   = torch.abs(s_img[:, 1:] - s_img[:, :-1]).mean()
                tv_z   = torch.abs(s_img[1:, :] - s_img[:-1, :]).mean()
                reg_loss += config.tv_weight * (tv_x + tv_z)

            total_loss = loss_k + reg_loss
            total_loss.backward()
            optimizer.step()
            step_pair_losses.append(loss_k.item())

        scheduler.step()
        avg_loss = np.mean(step_pair_losses)
        loss_history.append(avg_loss)

        if step % 50 == 0:
            pbar.set_description(f"Loss (us^2): {avg_loss:.4f}")

        # Early stopping: evaluate on full residual
        if stopper.enabled and step % config.log_interval == 0:
            with torch.no_grad():
                s_norm_es = model(coords)
                s_phys_es = _maybe_clamp_slowness(s_norm_es * s_std + s_mean, config)
                d_pred_full = L @ s_phys_es
                residual_full = d_pred_full - d_meas
                val_loss, should_stop = stopper.evaluate(residual_full, config, model)
            if use_wandb and val_loss is not None:
                wandb.log({"Val Loss": val_loss}, step=step)
            if should_stop:
                logging.info(f"Early stopping at step {step} (patience={config.patience})")
                break

        if use_wandb:
            wandb.log({
                "Avg Pair Loss": avg_loss,
                "Learning Rate": scheduler.get_last_lr()[0]
            }, step=step)

    stopper.restore_best(model)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

    return {
        's_phys':       s_phys.detach().cpu(),
        's_norm':       s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }


def optimize_with_bias_absorption(sample, L_matrix, model, bias_model, label, config: ec, use_wandb=False):
    """Phase 1: Simultaneous Anatomy + Bias optimization.
    
    Decouples the smooth systematic model mismatch (bias) from the sharp anatomy.
    """
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} + BiasAbsorber on {_DEVICE} ---")

    model  = model.to(_DEVICE)
    bias_model = bias_model.to(_DEVICE)
    
    coords = sample['coords'].to(_DEVICE) # Anatomy coords (4096, 2)
    d_meas = sample['d_meas'].to(_DEVICE) # Measurements (131072, 1)
    mask   = sample['mask'].to(_DEVICE)   # Mask (131072, 1)
    s_mean = sample['s_stats'][0].item()
    s_std  = sample['s_stats'][1].item()
    L      = L_matrix.to(_DEVICE)

    # Generate Measurement Domain Coordinates (u, v) for the Bias INR
    num_rays = d_meas.shape[0]
    n_pairs = 8
    rays_per_pair = num_rays // n_pairs
    
    pair_indices = torch.arange(n_pairs, device=_DEVICE).repeat_interleave(rays_per_pair)
    ray_indices = torch.arange(rays_per_pair, device=_DEVICE).repeat(n_pairs)
    
    # Normalize to [-1, 1]
    u = 2.0 * (ray_indices.float() / (rays_per_pair - 1)) - 1.0
    v = 2.0 * (pair_indices.float() / (n_pairs - 1)) - 1.0
    bias_coords = torch.stack([u, v], dim=1) # (131072, 2)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config.lr},
        {'params': bias_model.parameters(), 'lr': config.bias_lr}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    loss_history = []
    pbar = tqdm(range(config.steps))

    for step in pbar:
        model.train()
        bias_model.train()
        optimizer.zero_grad()

        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)
        d_pred_geometric = L @ s_phys
        epsilon_bias = bias_model(bias_coords)
        d_total_pred = d_pred_geometric + epsilon_bias
        
        residual_seconds = d_total_pred - d_meas
        loss_data = _compute_data_loss(residual_seconds, mask, config)

        reg_loss = 0
        if config.reg_weight > 0:
            reg_loss += config.reg_weight * (s_norm ** 2).mean()
        if config.bias_reg_weight > 0:
            reg_loss += config.bias_reg_weight * (epsilon_bias ** 2).mean()

        total_loss = loss_data + reg_loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if step % 50 == 0:
            pbar.set_description(f"Loss: {loss_data.item():.4f}")

    return {
        's_phys':       s_phys.detach().cpu(),
        'epsilon_bias': epsilon_bias.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }


def optimize_stochastic_ray_batching(sample, L_matrix, model, label, config: ec, use_wandb=False):
    """Phase 3: Fast GPU Slicing.

    Supports: Huber/MSE loss, slowness clamping, early stopping.
    Early stopping evaluates at the end of each epoch on a held-out ray set.
    """
    logging.info(f"\n--- {inspect.currentframe().f_code.co_name}: Training {label} (Fast GPU Slicing) on {_DEVICE} ---")

    model  = model.to(_DEVICE)
    coords = sample['coords'].to(_DEVICE)
    s_mean = sample['s_stats'][0].item()
    s_std  = sample['s_stats'][1].item()
    L      = L_matrix.to(_DEVICE)
    d_meas = sample['d_meas'].to(_DEVICE)
    mask   = sample['mask'].to(_DEVICE)

    total_rays     = L.shape[0]
    steps_per_epoch = int(np.ceil(total_rays / config.batch_size))
    total_steps    = config.epochs * steps_per_epoch

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Early stopping setup
    stopper = _EarlyStopper(mask, config, model)
    train_mask = stopper.get_train_mask(mask)

    loss_history = []
    pbar = tqdm(range(config.epochs), desc="Epochs")

    for epoch in pbar:
        pbar.set_postfix({"method": inspect.currentframe().f_code.co_name, "model": label})
        epoch_loss       = 0.0
        permuted_indices = torch.randperm(total_rays, device=_DEVICE)

        for step in range(steps_per_epoch):
            model.train()
            optimizer.zero_grad()

            start_idx = step * config.batch_size
            end_idx   = min(start_idx + config.batch_size, total_rays)
            batch_idx = permuted_indices[start_idx:end_idx]

            L_batch   = L[batch_idx]
            d_batch   = d_meas[batch_idx]
            m_batch   = train_mask[batch_idx]

            s_norm  = model(coords)
            s_phys  = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)
            d_pred  = L_batch @ s_phys
            residual = d_pred - d_batch
            loss_data = _compute_data_loss(residual, m_batch, config)

            reg_loss = 0
            if config.reg_weight > 0:
                reg_loss = config.reg_weight * (s_norm ** 2).mean()
            if config.tv_weight > 0:
                s_img  = s_phys.reshape(64, 64)
                tv_x   = torch.abs(s_img[:, 1:] - s_img[:, :-1]).mean()
                tv_z   = torch.abs(s_img[1:, :] - s_img[:-1, :]).mean()
                reg_loss += config.tv_weight * (tv_x + tv_z)

            total_loss = loss_data + reg_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_data.item()

        avg_epoch_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_epoch_loss)

        if epoch % config.log_interval == 0:
            pbar.set_description(f"Avg Loss (us^2): {avg_epoch_loss:.4f}")

        # Early stopping: evaluate full residual at end of epoch
        if stopper.enabled and epoch % config.log_interval == 0:
            with torch.no_grad():
                s_norm_es = model(coords)
                s_phys_es = _maybe_clamp_slowness(s_norm_es * s_std + s_mean, config)
                d_pred_full = L @ s_phys_es
                residual_full = d_pred_full - d_meas
                val_loss, should_stop = stopper.evaluate(residual_full, config, model)
            if use_wandb and val_loss is not None:
                wandb.log({"Val Loss": val_loss}, step=epoch)
            if should_stop:
                logging.info(f"Early stopping at epoch {epoch} (patience={config.patience})")
                break

        if use_wandb:
            wandb.log({
                "Avg Epoch Loss": avg_epoch_loss,
                "Learning Rate":  scheduler.get_last_lr()[0]
            }, step=epoch)

    stopper.restore_best(model)

    model.eval()
    with torch.no_grad():
        s_norm = model(coords)
        s_phys = _maybe_clamp_slowness(s_norm * s_std + s_mean, config)

    return {
        's_phys':       s_phys.detach().cpu(),
        's_norm':       s_norm.detach().cpu(),
        'loss_history': loss_history,
        'model_state':  model.state_dict()
    }