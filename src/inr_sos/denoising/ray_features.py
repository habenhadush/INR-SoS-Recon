"""Ray feature computation for the displacement field denoiser.

Each of the 131072 rays is identified by (pair_idx, tx_channel, rx_channel).
These are normalized to [-1, 1] as input features for the denoiser INR.
"""

import torch
import numpy as np


def compute_ray_features(
    n_rays: int = 131072,
    n_pairs: int = 8,
    grid=None,
) -> torch.Tensor:
    """Compute structured 3D ray features from ray indices.

    The 131072 rays are organized as:
        ray_i  →  pair = i // (128*128)
                  tx   = (i % (128*128)) // 128
                  rx   = i % 128

    Parameters
    ----------
    n_rays : int
        Total number of rays (default 131072 = 8 x 128 x 128).
    n_pairs : int
        Number of firing pairs (default 8).
    grid : USGrid, optional
        If provided, uses physical coordinates from grid.x_dt and grid.z_dt
        instead of integer channel indices.

    Returns
    -------
    torch.Tensor of shape (n_rays, 3)
        Columns: [pair_norm, tx_norm, rx_norm], each in [-1, 1].
    """
    channels_per_pair = n_rays // n_pairs  # 16384
    ch_per_dim = int(np.sqrt(channels_per_pair))  # 128

    indices = np.arange(n_rays)
    pair_idx = indices // channels_per_pair        # 0..7
    tx_idx = (indices % channels_per_pair) // ch_per_dim  # 0..127
    rx_idx = indices % ch_per_dim                  # 0..127

    if grid is not None and hasattr(grid, 'x_dt') and grid.x_dt is not None:
        # Use physical coordinates from the DT grid
        x_dt = np.asarray(grid.x_dt).flatten()
        z_dt = np.asarray(grid.z_dt).flatten()

        tx_vals = x_dt[tx_idx]
        rx_vals = z_dt[rx_idx]

        # Normalize to [-1, 1]
        tx_norm = 2.0 * (tx_vals - tx_vals.min()) / (tx_vals.max() - tx_vals.min() + 1e-10) - 1.0
        rx_norm = 2.0 * (rx_vals - rx_vals.min()) / (rx_vals.max() - rx_vals.min() + 1e-10) - 1.0
    else:
        # Integer indices normalized to [-1, 1]
        tx_norm = 2.0 * tx_idx / (ch_per_dim - 1) - 1.0
        rx_norm = 2.0 * rx_idx / (ch_per_dim - 1) - 1.0

    pair_norm = 2.0 * pair_idx / (n_pairs - 1) - 1.0

    features = np.stack([pair_norm, tx_norm, rx_norm], axis=-1)  # (131072, 3)
    return torch.tensor(features, dtype=torch.float32)
