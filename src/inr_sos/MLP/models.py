import torch
import torch.nn as nn
import numpy as np

class SIREN(nn.Module):
    """
    A single layer of the SIREN network.
    Uses sin(omega_0 * (Wx + b)) as the activation function.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # SIREN Initialization Scheme (Crucial for convergence)
        with torch.no_grad():
            if self.is_first:
                # First layer: Uniform(-1/in, 1/in)
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                # Hidden layers: Uniform(-sqrt(6/in)/omega, sqrt(6/in)/omega)
                k = np.sqrt(6 / in_features) / self.omega_0
                self.linear.weight.uniform_(-k, k)
            
            if bias:
                self.linear.bias.uniform_(-1e-4, 1e-4) # Start with near-zero bias

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class INR_Network(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=1):
        """
        The Full SIREN Network.
        Args:
            in_features: 2 (x, z coordinates)
            hidden_features: Width of the layers (e.g., 256 neurons)
            hidden_layers: Depth of the network
            out_features: 1 (Slowness value)
        """
        super().__init__()
        
        layers = []
        
        # 1. Input Layer
        layers.append(SIREN(in_features, hidden_features, is_first=True, omega_0=30))
        
        # 2. Hidden Layers
        for _ in range(hidden_layers):
            layers.append(SIREN(hidden_features, hidden_features, is_first=False, omega_0=30))
            
        # 3. Output Layer (Linear activation, not Sine)
        # We want the output to be the raw Slowness value, which isn't necessarily periodic
        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)
        
        # Init final layer
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6/hidden_features) / 30, 
                                               np.sqrt(6/hidden_features) / 30)
            self.final_linear.bias.uniform_(-1e-4, 1e-4)

    def forward(self, coords):
        # coords shape: (Batch, 4096, 2)
        # Run through Sine layers
        x = self.net(coords)
        # Final linear projection
        output = self.final_linear(x)
        return output