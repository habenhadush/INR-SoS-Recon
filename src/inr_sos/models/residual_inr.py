"""Residual INR for forward-model mismatch correction (Layer 3).

Based on Gilton et al. (ICLR 2025) — untrained residual block that captures
the structured mismatch ε = d_meas - L·s_true in measurement space.

The residual INR maps ray geometry coordinates (pair_idx, row, col) ∈ [-1,1]³
to a per-ray correction value. It uses SIREN activations with a lower omega
than the reconstruction INR to capture smooth, geometry-dependent corrections.

The ray coordinate system for our L-matrix (131072 = 8 × 128 × 128):
  - pair: firing pair index (0-7), normalized to [-1, 1]
  - row:  transmit channel (0-127), normalized to [-1, 1]
  - col:  receive channel (0-127), normalized to [-1, 1]
"""

import torch
import torch.nn as nn
import numpy as np


class ResidualSiren(nn.Module):
    """SIREN-based residual INR mapping ray coordinates to mismatch correction.

    Architecture follows Gilton et al.: small, untrained network with implicit
    regularization through architecture + explicit τ-penalty on outputs.
    Uses SIREN activations for smooth, continuous corrections.

    Args:
        in_features: input dimension (3 for pair/row/col coordinates)
        hidden_features: hidden layer width
        hidden_layers: number of hidden layers
        omega_0: SIREN frequency (lower = smoother corrections)
    """

    def __init__(self, in_features=3, hidden_features=128, hidden_layers=3,
                 omega_0=10.0):
        super().__init__()
        self.omega_0 = omega_0

        layers = []
        # First layer
        layers.append(nn.Linear(in_features, hidden_features))
        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
        self.hidden = nn.ModuleList(layers)

        # Output layer — initialized small so initial correction ≈ 0
        self.output = nn.Linear(hidden_features, 1)

        self._init_weights()

    def _init_weights(self):
        """SIREN-style initialization. Output layer initialized near zero."""
        with torch.no_grad():
            # First layer
            fan_in = self.hidden[0].in_features
            self.hidden[0].weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)

            # Hidden layers
            for layer in self.hidden[1:]:
                fan_in = layer.in_features
                bound = np.sqrt(6.0 / fan_in) / self.omega_0
                layer.weight.uniform_(-bound, bound)

            # Output: small initial values → correction starts near zero
            fan_in = self.output.in_features
            bound = np.sqrt(6.0 / fan_in) / self.omega_0
            self.output.weight.uniform_(-bound * 0.1, bound * 0.1)
            self.output.bias.zero_()

    def forward(self, ray_coords):
        """
        Args:
            ray_coords: (M, 3) tensor of normalized ray coordinates [pair, row, col]

        Returns:
            (M, 1) per-ray correction values
        """
        x = torch.sin(self.omega_0 * self.hidden[0](ray_coords))
        for layer in self.hidden[1:]:
            x = torch.sin(self.omega_0 * layer(x))
        return self.output(x)


def build_ray_coordinates(n_rays, n_pairs=8, device=None):
    """Build normalized ray coordinate tensor for the L-matrix geometry.

    Our L-matrix has shape (M, N) where M = n_pairs × rows × cols.
    Each ray is indexed as: ray_i = pair * (rows*cols) + row * cols + col

    Args:
        n_rays: total number of rays (131072)
        n_pairs: number of firing pairs (8)
        device: torch device

    Returns:
        (M, 3) tensor with columns [pair_norm, row_norm, col_norm] in [-1, 1]
    """
    rays_per_pair = n_rays // n_pairs
    side = int(np.sqrt(rays_per_pair))  # 128 for our data

    indices = torch.arange(n_rays, dtype=torch.float32)
    pair = torch.floor(indices / rays_per_pair)
    within = indices % rays_per_pair
    row = torch.floor(within / side)
    col = within % side

    # Normalize to [-1, 1]
    pair_norm = 2.0 * pair / (n_pairs - 1) - 1.0
    row_norm = 2.0 * row / (side - 1) - 1.0
    col_norm = 2.0 * col / (side - 1) - 1.0

    coords = torch.stack([pair_norm, row_norm, col_norm], dim=1)
    if device is not None:
        coords = coords.to(device)
    return coords
