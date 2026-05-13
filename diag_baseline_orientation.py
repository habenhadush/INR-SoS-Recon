import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys

# Mock joblib to avoid permission issues on server
import joblib
class MockMemory:
    def __init__(self, *args, **kwargs): pass
    def cache(self, func): return func
joblib.Memory = MockMemory

from inr_sos.utils.data import USDataset

def main():
    data_path = "scripts/data/DL-based-SoS/train_IC_10k_l2rec_l1rec_imcon.mat"
    grid_path = "scripts/data/DL-based-SoS/grid_parameters.mat"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found at {data_path}")
        return

    dataset = USDataset(data_path, grid_path)
    sample_idx = 7270 # One of the report indices
    sample = dataset[sample_idx]
    
    gt = sample["s_gt_raw"].cpu().numpy().flatten()
    l1 = sample["s_l1_recon"].cpu().numpy().flatten() if "s_l1_recon" in sample else None
    l2 = sample["s_l2_recon"].cpu().numpy().flatten() if "s_l2_recon" in sample else None

    # Test different orientations
    methods = [
        ("Reshape F", lambda x: x.reshape(64, 64, order="F")),
        ("Reshape C", lambda x: x.reshape(64, 64, order="C")),
        ("Reshape F + T", lambda x: x.reshape(64, 64, order="F").T),
        ("Reshape C + T", lambda x: x.reshape(64, 64, order="C").T),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (name, val) in enumerate([("GT", gt), ("L1", l1), ("L2", l2)]):
        if val is None: continue
        for j, (m_name, m_func) in enumerate(methods):
            ax = axes[i, j]
            try:
                img = 1.0 / (m_func(val) + 1e-8)
                im = ax.imshow(img, cmap="jet", vmin=1400, vmax=1600)
                ax.set_title(f"{name} | {m_name}")
                plt.colorbar(im, ax=ax)
            except Exception as e:
                ax.text(0.5, 0.5, str(e), ha='center', va='center')
            ax.axis("off")

    plt.tight_layout()
    save_path = "diag_orientation.png"
    plt.savefig(save_path)
    print(f"Diagnostic plot saved to {save_path}")
    print("Compare the L1/L2 rows with the GT row to find which column matches.")

if __name__ == "__main__":
    main()
