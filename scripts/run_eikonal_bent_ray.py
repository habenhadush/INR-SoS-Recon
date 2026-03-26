"""
Experiment 5: Eikonal Bent-Ray L-Matrix Update

Iteratively improves the forward model by replacing straight-ray paths
with bent-ray paths computed from the eikonal equation.

Approach:
  Iteration 0: Train INR with original L-matrix → SoS estimate
  Iteration 1+: Solve eikonal |∇T|² = s² per source element → backtrace
                 bent rays → build L_bent → retrain INR

Usage:
    source .venv/bin/activate
    uv run python scripts/run_eikonal_bent_ray.py --dataset kwave_geom
    uv run python scripts/run_eikonal_bent_ray.py --dataset kwave_geom --n_iterations 3
"""

import argparse
import copy
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inr_sos.utils.data import USDataset
from inr_sos.evaluation.metrics import calculate_metrics, calculate_cnr
from inr_sos.io.paths import DATA_DIR
from inr_sos.utils.config import ExperimentConfig
from inr_sos.models.mlp import FourierMLP
from inr_sos.training.engines import optimize_full_forward_operator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Dataset registry ─────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "kwave_geom": {
        "data_file": f"{DATA_DIR}/DL-based-SoS/test_kWaveGeom_l2rec_l1rec_unifiedvar.mat",
        "grid_file": f"{DATA_DIR}/DL-based-SoS/forward_model_lr/grid_parameters.mat",
        "matrix_file": None,
        "use_external_L": False,
        "n_samples": 32,
    },
    "kwave_blob": {
        "data_file": f"{DATA_DIR}/DL-based-SoS/test_kWaveBlob_final.mat",
        "grid_file": f"{DATA_DIR}/DL-based-SoS/forward_model_lr/grid_parameters.mat",
        "matrix_file": f"{DATA_DIR}/DL-based-SoS/A.mat",
        "use_external_L": True,
        "n_samples": 70,
    },
}

# ── Output directory ─────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "data" / "experiment5_eikonal"

# ── Transducer geometry (from Rau et al. 2021 + notebook exploration) ────
N_ELEM = 128
PITCH = 0.3e-3  # 300 μm
DELTA_CH = 17

# Element lateral positions (centered array, z=0)
ELEM_X = (np.arange(N_ELEM) - (N_ELEM - 1) / 2) * PITCH

# Pair-element mapping (identified from L-matrix correlation in notebook)
# Format: (left_elem_idx, right_elem_idx)
# Sign convention: L_row = ray(right_elem → pixel) - ray(left_elem → pixel)
PAIR_ELEMENTS = [
    (27, 44),   # Block 0
    (37, 54),   # Block 1
    (45, 62),   # Block 2
    (54, 71),   # Block 3
    (55, 72),   # Block 4
    (63, 80),   # Block 5
    (71, 88),   # Block 6
    (80, 97),   # Block 7
]

N_PAIRS = 8
ROWS_PER_PAIR = 128 * 128  # 16384


# =========================================================================
#  Siddon's Ray Tracing
# =========================================================================

def siddon_ray(x1, z1, x2, z2, x_edges, z_edges, nx, nz):
    """Compute intersection lengths of a ray with a regular grid.

    Returns (nx*nz,) array in Fortran order (column-major: idx = ix*nz + iz).
    """
    weights = np.zeros(nx * nz, dtype=np.float64)

    dx = x2 - x1
    dz = z2 - z1
    ray_len = np.sqrt(dx**2 + dz**2)
    if ray_len < 1e-15:
        return weights

    if abs(dx) > 1e-15:
        tx = (x_edges - x1) / dx
    else:
        tx = np.array([])

    if abs(dz) > 1e-15:
        tz = (z_edges - z1) / dz
    else:
        tz = np.array([])

    t_all = np.concatenate([tx, tz])
    t_all = t_all[(t_all >= 0) & (t_all <= 1)]
    t_all = np.unique(np.sort(t_all))

    if len(t_all) == 0 or t_all[0] > 1e-12:
        t_all = np.concatenate([[0.0], t_all])
    if t_all[-1] < 1.0 - 1e-12:
        t_all = np.concatenate([t_all, [1.0]])

    for k in range(len(t_all) - 1):
        t_mid = 0.5 * (t_all[k] + t_all[k + 1])
        seg_len = (t_all[k + 1] - t_all[k]) * ray_len

        xm = x1 + t_mid * dx
        zm = z1 + t_mid * dz

        ix = np.searchsorted(x_edges, xm) - 1
        iz = np.searchsorted(z_edges, zm) - 1

        if 0 <= ix < nx and 0 <= iz < nz:
            idx = ix * nz + iz
            weights[idx] += seg_len

    return weights


# =========================================================================
#  Eikonal Solver + Bent-Ray L Construction
# =========================================================================

def solve_eikonal(slowness_2d, x_sos, z_sos, source_x, source_z=0.0):
    """Solve eikonal equation |∇T|² = s² using Fast Marching.

    Args:
        slowness_2d: (nz, nx) slowness field in s/m
        x_sos, z_sos: 1D coordinate arrays (meters)
        source_x: lateral position of source element (meters)
        source_z: depth position of source (default 0, at surface)

    Returns:
        T: (nz, nx) travel time field
    """
    import skfmm

    nx = len(x_sos)
    nz = len(z_sos)
    dx = np.abs(np.diff(x_sos).mean())

    # Initialize phi: signed distance from source (negative at source)
    X, Z = np.meshgrid(x_sos, z_sos)
    phi = np.sqrt((X - source_x)**2 + (Z - source_z)**2)

    # Mark source region: closest pixel to source
    # Source at z=0 is ABOVE the grid (grid starts at z_sos[0] > 0)
    # Use boundary condition: set phi negative at closest grid point
    ix_src = np.argmin(np.abs(x_sos - source_x))
    iz_src = 0  # top row is closest to z=0

    # Create masked source (small region around closest point)
    phi_masked = phi.copy()
    phi_masked[iz_src, ix_src] = -1.0

    # Solve eikonal: travel_time expects speed, not slowness
    speed = 1.0 / (slowness_2d + 1e-15)
    T = skfmm.travel_time(phi_masked, speed, dx=dx)

    # Correct for the gap between source (z=0) and grid top (z_sos[0])
    # Add the straight-line travel time from source to grid boundary
    gap_time = np.sqrt((x_sos - source_x)**2 + z_sos[0]**2) / (1.0 / slowness_2d[0, :].mean())
    # This is approximate — the gap is small (0.3mm) so straight-ray is fine there

    return np.array(T)


def backtrace_ray(T, x_sos, z_sos, target_x, target_z, x_edges, z_edges, nx=64, nz=64):
    """Backtrace from target pixel to source along -∇T, rasterize with Siddon.

    Uses gradient descent on the travel-time field to find the bent-ray path,
    then rasterizes the path segments onto the SoS grid.

    Args:
        T: (nz, nx) travel time field from eikonal solver
        x_sos, z_sos: 1D grid coordinates
        target_x, target_z: DT pixel position to trace from
        x_edges, z_edges: grid cell boundaries for Siddon

    Returns:
        weights: (nx*nz,) path length per pixel (Fortran order)
    """
    weights = np.zeros(nx * nz, dtype=np.float64)
    dx = np.abs(np.diff(x_sos).mean())
    step_size = dx * 0.5  # half-pixel steps for accuracy
    max_steps = 500

    # Current position
    x_cur = target_x
    z_cur = target_z

    # Grid bounds
    x_min, x_max = x_sos[0], x_sos[-1]
    z_min, z_max = z_sos[0], z_sos[-1]

    for _ in range(max_steps):
        # Check if we've left the grid (reached the source boundary)
        if z_cur <= z_min or x_cur < x_min or x_cur > x_max:
            break

        # Compute gradient of T at current position (bilinear interpolation)
        # Map to fractional grid indices
        fx = (x_cur - x_sos[0]) / dx
        fz = (z_cur - z_sos[0]) / dx  # assumes dx == dz

        ix = int(np.floor(fx))
        iz = int(np.floor(fz))

        # Clamp to valid range for gradient computation
        ix = max(0, min(ix, nx - 2))
        iz = max(0, min(iz, nz - 2))

        # Bilinear weights
        wx = fx - ix
        wz = fz - iz
        wx = max(0.0, min(1.0, wx))
        wz = max(0.0, min(1.0, wz))

        # Gradient via central differences at integer grid points, then interpolate
        # dT/dx
        if 0 < ix < nx - 1:
            dTdx_lo = (T[iz, ix + 1] - T[iz, ix - 1]) / (2 * dx)
            dTdx_hi = (T[min(iz + 1, nz - 1), ix + 1] - T[min(iz + 1, nz - 1), ix - 1]) / (2 * dx)
        elif ix == 0:
            dTdx_lo = (T[iz, 1] - T[iz, 0]) / dx
            dTdx_hi = (T[min(iz + 1, nz - 1), 1] - T[min(iz + 1, nz - 1), 0]) / dx
        else:
            dTdx_lo = (T[iz, nx - 1] - T[iz, nx - 2]) / dx
            dTdx_hi = (T[min(iz + 1, nz - 1), nx - 1] - T[min(iz + 1, nz - 1), nx - 2]) / dx

        dTdx = (1 - wz) * dTdx_lo + wz * dTdx_hi

        # dT/dz
        if 0 < iz < nz - 1:
            dTdz_lo = (T[iz + 1, ix] - T[iz - 1, ix]) / (2 * dx)
            dTdz_hi = (T[iz + 1, min(ix + 1, nx - 1)] - T[iz - 1, min(ix + 1, nx - 1)]) / (2 * dx)
        elif iz == 0:
            dTdz_lo = (T[1, ix] - T[0, ix]) / dx
            dTdz_hi = (T[1, min(ix + 1, nx - 1)] - T[0, min(ix + 1, nx - 1)]) / dx
        else:
            dTdz_lo = (T[nz - 1, ix] - T[nz - 2, ix]) / dx
            dTdz_hi = (T[nz - 1, min(ix + 1, nx - 1)] - T[nz - 2, min(ix + 1, nx - 1)]) / dx

        dTdz = (1 - wx) * dTdz_lo + wx * dTdz_hi

        grad_norm = np.sqrt(dTdx**2 + dTdz**2)
        if grad_norm < 1e-15:
            break

        # Step along -∇T (toward source)
        dir_x = -dTdx / grad_norm
        dir_z = -dTdz / grad_norm

        x_next = x_cur + step_size * dir_x
        z_next = z_cur + step_size * dir_z

        # Rasterize this segment using Siddon
        seg_weights = siddon_ray(x_cur, z_cur, x_next, z_next, x_edges, z_edges, nx, nz)
        weights += seg_weights

        x_cur = x_next
        z_cur = z_next

    return weights


def build_bent_ray_L(slowness_2d, x_sos, z_sos, x_dt, z_dt, x_edges, z_edges,
                     pair_elements, elem_x, mask_flat=None):
    """Build the full bent-ray L-matrix for all pairs.

    Args:
        slowness_2d: (64, 64) current SoS estimate in s/m
        x_sos, z_sos: SoS grid coordinates
        x_dt, z_dt: DT grid coordinates
        x_edges, z_edges: SoS pixel boundaries
        pair_elements: list of (left_elem, right_elem) tuples
        elem_x: element lateral positions
        mask_flat: (131072,) validity mask — skip invalid rows

    Returns:
        L_bent: (131072, 4096) numpy array
    """
    nx, nz = 64, 64
    n_dt = 128
    n_pairs = len(pair_elements)
    L_bent = np.zeros((n_pairs * n_dt * n_dt, nx * nz), dtype=np.float64)

    # Solve eikonal for each unique source element
    unique_elems = sorted(set(e for pair in pair_elements for e in pair))
    logger.info(f"Solving eikonal for {len(unique_elems)} unique elements...")
    travel_times = {}
    for ei in unique_elems:
        T = solve_eikonal(slowness_2d, x_sos, z_sos, elem_x[ei])
        travel_times[ei] = T

    # Build L for each pair
    for p, (e_left, e_right) in enumerate(pair_elements):
        logger.info(f"  Building L_bent for pair {p} (elems {e_left}, {e_right})...")
        T_left = travel_times[e_left]
        T_right = travel_times[e_right]

        for ix in range(n_dt):
            for iz in range(n_dt):
                row_idx = p * ROWS_PER_PAIR + ix * n_dt + iz

                # Skip invalid measurements
                if mask_flat is not None and mask_flat[row_idx] < 0.5:
                    continue

                px = x_dt[ix]
                pz = z_dt[iz]

                # Backtrace bent rays from DT pixel to each element
                w_right = backtrace_ray(T_right, x_sos, z_sos, px, pz,
                                        x_edges, z_edges, nx, nz)
                w_left = backtrace_ray(T_left, x_sos, z_sos, px, pz,
                                       x_edges, z_edges, nx, nz)

                # Differential: L = ray(right) - ray(left)
                L_bent[row_idx, :] = w_right - w_left

    return L_bent


def build_straight_ray_L(x_sos, z_sos, x_dt, z_dt, x_edges, z_edges,
                         pair_elements, elem_x, mask_flat=None):
    """Build straight-ray L from our geometry (Siddon, no eikonal)."""
    nx, nz = 64, 64
    n_dt = 128
    n_pairs = len(pair_elements)
    L_straight = np.zeros((n_pairs * n_dt * n_dt, nx * nz), dtype=np.float64)

    for p, (e_left, e_right) in enumerate(pair_elements):
        logger.info(f"  Building L_straight for pair {p} (elems {e_left}, {e_right})...")
        for ix in range(n_dt):
            for iz in range(n_dt):
                row_idx = p * ROWS_PER_PAIR + ix * n_dt + iz

                if mask_flat is not None and mask_flat[row_idx] < 0.5:
                    continue

                px = x_dt[ix]
                pz = z_dt[iz]

                w_right = siddon_ray(elem_x[e_right], 0.0, px, pz,
                                     x_edges, z_edges, nx, nz)
                w_left = siddon_ray(elem_x[e_left], 0.0, px, pz,
                                    x_edges, z_edges, nx, nz)

                L_straight[row_idx, :] = w_right - w_left

    return L_straight


# =========================================================================
#  INR Training Helper
# =========================================================================

def train_inr(sample, L_matrix_torch, config):
    """Train a FourierMLP INR and return SoS reconstruction."""
    model = FourierMLP(
        in_features=config.in_features,
        hidden_features=config.hidden_features,
        hidden_layers=config.hidden_layers,
        mapping_size=config.mapping_size,
        scale=config.scale,
    )
    result = optimize_full_forward_operator(
        sample=sample,
        L_matrix=L_matrix_torch,
        model=model,
        label="FourierMLP",
        config=config,
        use_wandb=False,
    )
    return result


def evaluate_reconstruction(s_phys, s_gt, grid_shape=(64, 64)):
    """Compute metrics for a reconstruction."""
    metrics = calculate_metrics(s_phys, s_gt, grid_shape=grid_shape)
    cnr = calculate_cnr(s_phys, s_gt, grid_shape=grid_shape)
    metrics['CNR'] = cnr
    return metrics


# =========================================================================
#  Grid Utilities
# =========================================================================

def setup_grid(grid_file):
    """Load grid and compute pixel edges for Siddon."""
    from inr_sos.io.utils import load_mat
    grid = load_mat(grid_file)

    x_sos = grid['xax_sos'].flatten()
    z_sos = grid['zax_sos'].flatten()
    x_dt = grid['xDT'].flatten()
    z_dt = grid['zDT'].flatten()

    dx = np.diff(x_sos).mean()
    dz = np.diff(z_sos).mean()
    x_edges = np.concatenate([[x_sos[0] - dx/2],
                               x_sos[:-1] + np.diff(x_sos)/2,
                               [x_sos[-1] + dx/2]])
    z_edges = np.concatenate([[z_sos[0] - dz/2],
                               z_sos[:-1] + np.diff(z_sos)/2,
                               [z_sos[-1] + dz/2]])

    return x_sos, z_sos, x_dt, z_dt, x_edges, z_edges


# =========================================================================
#  Visualization
# =========================================================================

def plot_iteration(s_phys, s_gt, iteration, metrics, save_path=None):
    """Plot reconstruction vs ground truth for one iteration."""
    s_pred_np = s_phys.detach().cpu().numpy().flatten()
    s_gt_np = s_gt.detach().cpu().numpy().flatten()

    pred_sos = 1.0 / np.clip(s_pred_np, 1/1800, 1/1200)
    gt_sos = 1.0 / np.clip(s_gt_np, 1/1800, 1/1200)

    pred_img = pred_sos.reshape(64, 64, order='F')
    gt_img = gt_sos.reshape(64, 64, order='F')
    diff_img = pred_img - gt_img

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = gt_sos.min(), gt_sos.max()

    ax = axes[0]
    im = ax.imshow(gt_img, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth SoS")
    plt.colorbar(im, ax=ax, label="m/s")

    ax = axes[1]
    im = ax.imshow(pred_img, cmap='jet', vmin=vmin, vmax=vmax)
    ax.set_title(f"Iteration {iteration}\nMAE={metrics['MAE']:.2f} m/s")
    plt.colorbar(im, ax=ax, label="m/s")

    ax = axes[2]
    vd = max(abs(diff_img.min()), abs(diff_img.max()))
    im = ax.imshow(diff_img, cmap='RdBu_r', vmin=-vd, vmax=vd)
    ax.set_title(f"Difference\nRMSE={metrics['RMSE']:.2f}")
    plt.colorbar(im, ax=ax, label="m/s")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =========================================================================
#  Main Experiment
# =========================================================================

def run_experiment(dataset_name, n_iterations=3, n_eval_samples=12):
    """Run the iterative eikonal bent-ray experiment."""
    ds_cfg = DATASET_CONFIGS[dataset_name]

    # ── Setup ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT 5: Eikonal Bent-Ray L-Matrix Update")
    logger.info(f"Dataset: {dataset_name}, Iterations: {n_iterations}")
    logger.info("=" * 60)

    # Load dataset
    dataset = USDataset(
        data_path=ds_cfg["data_file"],
        grid_path=ds_cfg["grid_file"],
        matrix_path=ds_cfg.get("matrix_file"),
        use_external_L_matrix=ds_cfg["use_external_L"],
    )
    L_original = dataset.L_matrix  # real L-matrix (torch tensor)

    # Grid setup
    x_sos, z_sos, x_dt, z_dt, x_edges, z_edges = setup_grid(ds_cfg["grid_file"])

    # INR training config (matches best known from sweeps)
    config = ExperimentConfig(
        project_name="Exp5-Eikonal",
        experiment_group="eikonal_bent_ray",
        model_type="FourierMLP",
        hidden_features=256,
        hidden_layers=4,
        mapping_size=64,
        scale=5.0,
        lr=5e-4,
        steps=500,
        time_scale=1e6,
        tv_weight=1e-2,
        reg_weight=1e-3,
        loss_type="mse",
        clamp_slowness=True,
        early_stopping=True,
        patience=100,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / dataset_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_indices = list(range(min(n_eval_samples, len(dataset))))
    all_results = {}

    for test_idx in eval_indices:
        logger.info(f"\n{'='*60}")
        logger.info(f"Sample {test_idx}/{len(eval_indices)-1}")
        logger.info(f"{'='*60}")

        sample = dataset[test_idx]
        s_gt = sample['s_gt_raw']
        mask_np = sample['mask'].numpy().flatten()

        sample_results = []

        # Current L-matrix starts as the original
        L_current = L_original

        for iteration in range(n_iterations + 1):
            logger.info(f"\n--- Iteration {iteration} ---")

            if iteration == 0:
                logger.info("Using original L-matrix")
            else:
                logger.info("Building bent-ray L from current SoS estimate...")
                # Get current SoS estimate as 2D slowness field
                s_current = prev_result['s_phys'].numpy().flatten()
                slowness_2d = s_current.reshape(64, 64, order='F')

                L_bent_np = build_bent_ray_L(
                    slowness_2d, x_sos, z_sos, x_dt, z_dt,
                    x_edges, z_edges, PAIR_ELEMENTS, ELEM_X,
                    mask_flat=mask_np,
                )
                L_current = torch.tensor(L_bent_np, dtype=torch.float32)
                logger.info(f"L_bent nnz fraction: {(L_bent_np != 0).sum() / L_bent_np.size:.4f}")

            # Train INR
            cfg = copy.deepcopy(config)
            cfg.experiment_group = f"eikonal_iter{iteration}"
            cfg.sample_idx = test_idx

            prev_result = train_inr(sample, L_current, cfg)

            # Evaluate
            metrics = evaluate_reconstruction(prev_result['s_phys'], s_gt)
            logger.info(f"  Iter {iteration}: MAE={metrics['MAE']:.2f}, "
                        f"RMSE={metrics['RMSE']:.2f}, SSIM={metrics['SSIM']:.4f}, "
                        f"CNR={metrics['CNR']:.4f}")

            sample_results.append({
                'iteration': iteration,
                **metrics,
            })

            # Save plot
            plot_path = out_dir / f"sample{test_idx}_iter{iteration}.png"
            plot_iteration(prev_result['s_phys'], s_gt, iteration, metrics,
                           save_path=str(plot_path))

        all_results[f"sample_{test_idx}"] = sample_results

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    for iteration in range(n_iterations + 1):
        maes = [all_results[f"sample_{i}"][iteration]['MAE'] for i in eval_indices]
        rmses = [all_results[f"sample_{i}"][iteration]['RMSE'] for i in eval_indices]
        logger.info(f"  Iter {iteration}: MAE = {np.mean(maes):.2f} ± {np.std(maes):.2f}  |  "
                    f"RMSE = {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")

    # Save results
    results_file = out_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
    logger.info(f"\nResults saved to {results_file}")

    return all_results


# =========================================================================
#  CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Eikonal Bent-Ray")
    parser.add_argument("--dataset", type=str, default="kwave_geom",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--n_iterations", type=int, default=3,
                        help="Number of L-update iterations (0 = baseline only)")
    parser.add_argument("--n_eval_samples", type=int, default=12,
                        help="Number of samples to evaluate")
    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        n_iterations=args.n_iterations,
        n_eval_samples=args.n_eval_samples,
    )


if __name__ == "__main__":
    main()
