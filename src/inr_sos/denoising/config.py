from dataclasses import dataclass, asdict


@dataclass
class DenoiserConfig:
    """Configuration for the displacement-field INR denoiser (Architecture 1).

    The denoiser fits an INR to d_meas using self-supervised MSE loss.
    Spectral bias + early stopping = implicit denoising.
    """

    # --- Architecture ---
    model_type: str = "SirenMLP"    # "SirenMLP" or "FourierMLP"
    in_features: int = 3            # (pair, tx, rx) structured encoding
    hidden_features: int = 128
    hidden_layers: int = 3
    omega: float = 10.0             # SIREN frequency (low = stronger denoising)
    scale: float = 0.5              # FourierMLP scale (low = stronger denoising)
    mapping_size: int = 64          # FourierMLP random feature dimension

    # --- Training ---
    lr: float = 1e-3
    steps: int = 5000               # max steps (early stopping should trigger before)
    time_scale: float = 1e6         # residual scaling (match Stage 2)
    log_interval: int = 10          # validation evaluation frequency

    # --- Early Stopping (the denoising mechanism) ---
    early_stopping: bool = True
    patience: int = 300
    val_fraction: float = 0.15

    # --- Ray structure ---
    n_pairs: int = 8
    channels_per_dim: int = 128     # 128 tx x 128 rx per pair

    def to_dict(self) -> dict:
        return asdict(self)
