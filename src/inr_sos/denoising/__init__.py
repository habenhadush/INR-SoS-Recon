"""Displacement field denoising via INR spectral bias (Architecture 1)."""

from inr_sos.denoising.config import DenoiserConfig
from inr_sos.denoising.engine import denoise_displacement
from inr_sos.denoising.ray_features import compute_ray_features

__all__ = ["DenoiserConfig", "denoise_displacement", "compute_ray_features"]
