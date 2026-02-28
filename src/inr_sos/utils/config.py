from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    """Strongly typed configuration for INR SoS Reconstruction experiments."""
    
    # --- Experiment Metadata ---
    project_name: str = "INR-SoS-Recon"
    experiment_group: str = "Debug-Group"
    model_type: str = "FourierMLP"
    sample_idx: int = 0
    
    # --- Model Architecture ---
    in_features: int = 2
    hidden_features: int = 256
    hidden_layers: int = 3
    mapping_size: int = 64
    scale: float = 10.0      # For FourierMLP
    omega: float = 30.0      # For SIREN
    
    # --- Optimization Parameters ---
    lr: float = 1e-4
    steps: int = 2000        # Used for Phase 0, 1, 2
    epochs: int = 150        # Used for (Ray Batching)
    batch_size: int = 4096
    time_scale: float = 1e6
    log_interval: int = 50
    
    # --- Regularization ---
    reg_weight: float = 0.0
    tv_weight: float = 0.0

    # --- Loss Function ---
    loss_type: str = "mse"          # "mse" or "huber"
    huber_delta: float = 1.0        # delta for Huber loss (in scaled units)

    # --- Physics Constraints ---
    clamp_slowness: bool = False    # clamp s_phys to [1/1800, 1/1200] s/m

    # --- Early Stopping ---
    early_stopping: bool = False
    patience: int = 100             # stop if no val improvement for this many steps
    val_fraction: float = 0.1       # fraction of valid rays held out for validation

    def to_dict(self) -> dict:
        """Required for passing the config to wandb.init()"""
        return asdict(self)
