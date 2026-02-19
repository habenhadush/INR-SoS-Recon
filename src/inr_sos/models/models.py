import torch
import torch.nn as nn
import numpy as np
import warnings

class SineLayer(nn.Module):
    """
    Fundamental building block for SIREN.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        warnings.warn(
            "OldClass is deprecated and will be removed in a future version. "
            "Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIRENMLP(nn.Module):
    """
    SIREN Network: Good for high-frequency details and derivatives.
    """
    def __init__(self, 
                 in_features=2, 
                 hidden_features=256, 
                 hidden_layers=3, 
                 out_features=1,
                 omega_0=30.0):
        warnings.warn(
            "OldClass is deprecated and will be removed in a future version. "
            "Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.net = []
        
        # 1. First Layer
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=omega_0))
        
        # 2. Hidden Layers
        for _ in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        
        # 3. Output Layer (Linear - No activation)
        self.net = nn.Sequential(*self.net)
        self.final = nn.Linear(hidden_features, out_features)
        
        # Initialize final layer for small random outputs
        with torch.no_grad():
            self.final.weight.uniform_(-np.sqrt(6 / hidden_features) / 30, 
                                        np.sqrt(6 / hidden_features) / 30)
            self.final.bias.uniform_(-1e-4, 1e-4)

    def forward(self, coords):
        x = self.net(coords)
        return self.final(x)


class ReluMLP(nn.Module):
    """
    Standard MLP: Baseline comparison.
    """
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=1):
        warnings.warn(
            "OldClass is deprecated and will be removed in a future version. "
            "Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
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
        warnings.warn(
            "OldClass is deprecated and will be removed in a future version. "
            "Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
        self.B = torch.randn(in_features, mapping_size) * scale
        self.register_buffer('B_matrix', self.B)
    
    def forward(self, coords):
        x_proj = (2. * np.pi * coords) @ self.B_matrix
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class FourierMLP(nn.Module):

    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, mapping_size=64, scale=10):
        warnings.warn(
            "OldClass is deprecated and will be removed in a future version. "
            "Use NewClass instead.",
            DeprecationWarning,
            stacklevel=2
        )
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