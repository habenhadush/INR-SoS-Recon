import torch
import torch.nn as nn
import numpy as np


class ReluMLP(nn.Module):
    """
    Standard MLP: Baseline comparison.
    """
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=1, mapping_size=64):
        super().__init__()
        
        layers = []
        # 1. Input Layer
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
        
        # 2. Hidden Layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
            
        # 3. Output Layer (No activation for regression)
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, coords):
        return self.net(coords)
    

class GaussianFourierFeatureMapping(nn.Module):
    def __init__(self, in_features=2, mapping_size=64, scale=10):
        super().__init__()
        self.B = torch.randn(in_features, mapping_size) * scale
        self.register_buffer('B_matrix', self.B)
    
    def forward(self, coords):
        x_proj = (2. * np.pi * coords) @ self.B_matrix
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class FourierMLP(nn.Module):

    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, mapping_size=64, scale=10):
        super().__init__()
        
        # 1. The Fourier Feature Encoder
        self.encoder = GaussianFourierFeatureMapping(
            in_features, 
            mapping_size, 
            scale
        )
        mlp_input_dim = mapping_size * 2

        # 2. Build the MLP Backbone
        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_features))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_features, 1))
        
        self.net = nn.Sequential(*layers)

        
    def forward(self, coords):
        features = self.encoder(coords)
        out = self.net(features)
        return out