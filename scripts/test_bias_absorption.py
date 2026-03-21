#!/usr/bin/env python3
"""
test_bias_absorption.py
-----------------------
Quick validation of the Phase 1: Bias-Absorber hypothesis.
Runs the Bias_Absorption engine on a single k-Wave sample.
"""

import torch
import logging
from pathlib import Path
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.models.siren import SirenMLP
from inr_sos.models.mlp import BiasMLP
from inr_sos.training.engines import optimize_with_bias_absorption
from inr_sos import DATA_DIR

logging.basicConfig(level=logging.INFO)

def main():
    # 1. Setup Data
    data_path = DATA_DIR + "/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat"
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    dataset = USDataset(data_path, grid_path)
    sample = dataset[0] # Test on first sample

    # 2. Config
    config = ExperimentConfig(
        steps=500, # Short run for testing
        lr=1e-4,
        bias_lr=5e-3, # Higher LR for bias to see absorption quickly
        bias_scale=0.05, # Very smooth bias
        bias_reg_weight=1e-4,
        reg_weight=1e-5
    )

    # 3. Models
    model = SirenMLP(in_features=2, hidden_features=256, hidden_layers=3, omega=30)
    bias_model = BiasMLP(in_features=2, hidden_features=64, hidden_layers=2, scale=config.bias_scale)

    # 4. Run Optimization
    print("\nStarting Bias Absorption Test...")
    result = optimize_with_bias_absorption(
        sample=sample,
        L_matrix=dataset.L_matrix,
        model=model,
        bias_model=bias_model,
        label="Siren_Test",
        config=config,
        use_wandb=False
    )

    # 5. Verify Results
    print("\nTest Complete.")
    print(f"Final slowness range: {result['s_phys'].min():.2e} to {result['s_phys'].max():.2e}")
    print(f"Bias field range: {result['epsilon_bias'].min():.2e} to {result['epsilon_bias'].max():.2e}")
    
    # Save a quick plot if on a server with display or just save to disk
    # (Visual validation is key for this hypothesis)
    plot_dir = Path("plots/test_bias")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anatomy
    v_rec = 1.0 / (result['s_phys'].reshape(64, 64) + 1e-8)
    im0 = axes[0].imshow(v_rec, cmap='jet', vmin=1400, vmax=1600)
    axes[0].set_title("Reconstructed SoS (m/s)")
    plt.colorbar(im0, ax=axes[0])
    
    # Bias (First firing pair)
    eps = result['epsilon_bias'].flatten()
    pair_size = len(eps) // 8
    bias_img = eps[:pair_size].reshape(128, 128, order='F')
    im1 = axes[1].imshow(bias_img, cmap='RdBu_r')
    axes[1].set_title("Absorbed Bias (Pair 0)")
    plt.colorbar(im1, ax=axes[1])
    
    # Loss
    axes[2].plot(result['loss_history'])
    axes[2].set_yscale('log')
    axes[2].set_title("Optimization Loss")
    
    plt.tight_layout()
    plt.savefig(plot_dir / "test_result.png")
    print(f"Results saved to {plot_dir}/test_result.png")

if __name__ == "__main__":
    main()
