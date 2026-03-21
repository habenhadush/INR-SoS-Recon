#!/usr/bin/env python3
"""
test_svd_inr.py
-----------------------
Quick validation of the Phase 2: Subspace-Constraint hypothesis.
Runs the SVD_Constraint engine on a single k-Wave sample.
"""

import torch
import logging
import yaml
import numpy as np
from pathlib import Path
from inr_sos.utils.data import USDataset
from inr_sos.utils.config import ExperimentConfig
from inr_sos.models.siren import SirenMLP
from inr_sos.training.engines import optimize_with_svd_constraint
from inr_sos import DATA_DIR

logging.basicConfig(level=logging.INFO)

def load_dataset_config(key="kwave_geom"):
    cfg_path = Path(__file__).parent / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds

def main():
    # 1. Setup Data using datasets.yaml
    ds_cfg = load_dataset_config("kwave_geom")
    grid_path = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    
    ds_kwargs = {}
    if not ds_cfg.get("has_A_matrix", True):
        matrix_file = ds_cfg.get("matrix_file")
        if matrix_file:
            ds_kwargs["matrix_path"] = DATA_DIR + matrix_file
            ds_kwargs["use_external_L_matrix"] = True
            
    dataset = USDataset(ds_cfg["data_path"], grid_path, **ds_kwargs)
    sample = dataset[0] # Test on first sample

    # 2. Config
    config = ExperimentConfig(
        steps=500, # Short run for testing
        lr=1e-4,
        svd_k=220, # Keep top 220 singular values (captures 99% energy)
        use_svd_constraint=True,
        reg_weight=1e-5
    )

    # 3. Models
    model = SirenMLP(in_features=2, hidden_features=256, hidden_layers=3, omega=30)

    # 4. Run Optimization
    print("\nStarting SVD-Constraint Test...")
    result = optimize_with_svd_constraint(
        sample=sample,
        L_matrix=dataset.L_matrix,
        model=model,
        label="Siren_SVD_Test",
        config=config,
        use_wandb=False
    )

    # 5. Verify Results
    print("\nTest Complete.")
    
    # Visual validation
    plot_dir = Path("plots/test_svd")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Anatomy GT
    s_gt_np = sample['s_gt_raw'].detach().cpu().numpy()
    v_gt = 1.0 / (s_gt_np.reshape(64, 64, order='F') + 1e-8)
    im0 = axes[0].imshow(v_gt, cmap='jet', vmin=1400, vmax=1600)
    axes[0].set_title("Ground Truth SoS (m/s)")
    plt.colorbar(im0, ax=axes[0])

    # SVD Reconstruction
    s_phys_np = result['s_phys'].detach().cpu().numpy()
    v_rec = 1.0 / (s_phys_np.reshape(64, 64) + 1e-8)
    im1 = axes[1].imshow(v_rec, cmap='jet', vmin=1400, vmax=1600)
    axes[1].set_title(f"SVD-Constrained Recon (k={config.svd_k})")
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
