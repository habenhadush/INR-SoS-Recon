import torch
import torch.nn as nn
import numpy as np


class _SineLayer(nn.Module):
    """
    Fundamental building block for SIREN.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=30.0):
        super().__init__()
        self.omega = omega
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
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega, 
                                             np.sqrt(6 / self.in_features) / self.omega)
    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))


class SirenMLP(nn.Module):
    """
    SIREN Network: Good for high-frequency details and derivatives.
    """
    def __init__(self, 
                 in_features=2, 
                 hidden_features=256, 
                 hidden_layers=3, 
                 out_features=1,
                 omega=30.0,
                 mapping_size=64):
        super().__init__()
        self.net = []
        
        # 1. First Layer
        self.net.append(_SineLayer(in_features, hidden_features, is_first=True, omega=omega))
        
        # 2. Hidden Layers
        for _ in range(hidden_layers):
            self.net.append(_SineLayer(hidden_features, hidden_features, is_first=False, omega=omega))
        
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