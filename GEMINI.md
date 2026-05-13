# INR-SoS-Recon

A research project focused on **Speed of Sound (SoS) Reconstruction** using **Implicit Neural Representations (INR)**. The project aims to improve reconstruction quality, particularly in the presence of forward model mismatches.

## Project Overview

- **Purpose:** Reconstructing Speed of Sound (SoS) maps from ultrasound displacement data using INRs (SIREN, FourierMLP).
- **Core Problem:** Mitigating forward model mismatch and improving Contrast-to-Noise Ratio (CNR) in limited-angle tomography.
- **Key Metrics:** CNR (Primary), SSIM, RMSE, MAE.
- **Tech Stack:** Python 3.12+, PyTorch, NumPy, SciPy, Weights & Biases (wandb), `uv` for dependency management.

## Architecture

- `src/inr_sos/`: Core library containing models, training engines, and utilities.
  - `models/`: Implementations of INR architectures (e.g., `mlp.py`, `siren.py`).
  - `training/`: Specialized training engines (e.g., `engines.py`, `denoise_engine.py`, `joint_engine.py`).
  - `evaluation/`: Metrics computation (`metrics.py`) and sweep agents for hyperparameter optimization.
  - `utils/`: Config management, data loading, and tracking.
- `scripts/`: Entry points for experiments, sweeps, and baseline comparisons.
- `notebooks/`: Data exploration and experimental analysis.
- `EXPERIMENT_PLAN.md`: A living document tracking experiment hypotheses, methods, and results.

## Getting Started

### Installation

This project uses `uv` for dependency management.

```bash
uv sync
# or using standard pip
pip install -e .
```

### Running Experiments

Experiments are typically run using scripts in the `scripts/` directory.

- **Standard Reconstruction:**
  ```bash
  python scripts/run_reconstruction.py --config <path_to_config>
  ```
- **Running Sweeps:**
  ```bash
  python scripts/run_sweep.py --config <path_to_config>
  ```
- **Comparing Baselines:**
  ```bash
  python scripts/compare_baselines.py
  ```

## Development Conventions

- **Experiment Tracking:** Use `EXPERIMENT_PLAN.md` to document new experiments. Every experiment should have a clear hypothesis and success criteria.
- **Metrics:** Always report CNR alongside MAE. Low MAE with low CNR often indicates a failed reconstruction that collapsed to the background.
- **Logging:** Use `wandb` for all training runs and sweeps to ensure reproducibility and easy comparison.
- **Branching:** Follow the `experiment/<experiment-name>` branching convention for new research directions.
- **Data:** Datasets are typically stored in `scripts/data` or referenced via `src/inr_sos/io/paths.py`.

## Key Files

- `EXPERIMENT_PLAN.md`: Detailed log of all research efforts and their outcomes.
- `pyproject.toml`: Project metadata and dependencies.
- `src/inr_sos/evaluation/metrics.py`: Source of truth for evaluation metrics.
- `scripts/datasets.yaml`: Configuration for different datasets used in experiments.
- `research-synthesis.md`: High-level summary of research findings and directions.
