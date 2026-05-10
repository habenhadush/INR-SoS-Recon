#!/usr/bin/env python3
"""
run_residual_study.py — isolated residual-INR experiment (Plan D redux).

Standalone script. Does NOT use the sweep/registry system. Trains an INR to
predict a *correction* Δs on top of a chosen baseline s_base ∈
{const(bg), L1, L2}, with two loss modes:

    --mode self   self-supervised: || L · (s_base + Δs) − d_meas ||²       (deployable)
    --mode sup    supervised:      || Δs − (s_GT − s_base) ||²              (diagnostic ceiling, GT-dep.)

Reuses USDataset + report-figure helpers from the main pipeline.

Usage examples:
    # Self-supervised, constant 1510 baseline, ReluMLP, 4000 steps
    python run_residual_study.py --dataset kwave_blob --baseline const --bg_const 1510 \
        --model relu --mode self --steps 4000 --lr 5e-4 --reg_l2 1e-4 --indices 1 21 38 \
        --report_plots

    # Diagnostic supervised ceiling on L1 baseline
    python run_residual_study.py --dataset kwave_blob --baseline l1 --model fourier \
        --mode sup --steps 4000 --lr 5e-4 --indices 1 21 38 --report_plots

Outputs land in   scripts/data/residual_study/<dataset>/<timestamp>_<tag>/
"""

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inr_sos import DATA_DIR
from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.models.mlp import FourierMLP, GeluMLP, ReluMLP
from inr_sos.models.siren import SirenMLP
from inr_sos.utils.config import ExperimentConfig
from inr_sos.utils.data import USDataset
from inr_sos.visualization.report_figures import (
    plot_method_grid,
    plot_metrics_comparison,
)

SCRIPTS_DIR = Path(__file__).parent
OUT_ROOT = SCRIPTS_DIR / "data" / "residual_study"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_SLOWNESS_MIN = 1.0 / 1800.0
_SLOWNESS_MAX = 1.0 / 1200.0
_GRID = (64, 64)

MODEL_MAP = {
    "relu":    ReluMLP,
    "gelu":    GeluMLP,
    "siren":   SirenMLP,
    "fourier": FourierMLP,
}


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_dataset_config(key=None):
    cfg_path = SCRIPTS_DIR / "datasets.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    key = key or cfg["active"]
    ds = cfg["datasets"][key]
    ds["key"] = key
    ds["data_path"] = DATA_DIR + ds["data_file"]
    return ds


def setup_logger(out_dir: Path) -> logging.Logger:
    log_path = out_dir / "run.log"
    logger = logging.getLogger("residual_study")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def zero_init_last_linear(model: torch.nn.Module) -> None:
    """Set last linear layer's weight and bias to 0 → Δs(coords) = 0 at step 0."""
    with torch.no_grad():
        if hasattr(model, "final") and isinstance(model.final, torch.nn.Linear):
            model.final.weight.zero_()
            model.final.bias.zero_()
        elif hasattr(model, "net"):
            last = model.net[-1]
            if isinstance(last, torch.nn.Linear):
                last.weight.zero_()
                last.bias.zero_()


def build_model(name: str, hidden_features: int, hidden_layers: int,
                mapping_size: int, scale: float, omega: float) -> torch.nn.Module:
    cls = MODEL_MAP[name]
    if name == "fourier":
        return cls(in_features=2, hidden_features=hidden_features,
                   hidden_layers=hidden_layers, mapping_size=mapping_size,
                   scale=scale)
    if name == "siren":
        return cls(in_features=2, hidden_features=hidden_features,
                   hidden_layers=hidden_layers, omega=omega)
    return cls(in_features=2, hidden_features=hidden_features,
               hidden_layers=hidden_layers)


def get_s_base(sample: dict, baseline: str, bg_const: float) -> torch.Tensor:
    """Return s_base as a (4096,) torch float tensor on _DEVICE."""
    if baseline == "const":
        return torch.full((np.prod(_GRID),), 1.0 / bg_const,
                          dtype=torch.float32, device=_DEVICE)
    if baseline == "l1":
        if "s_l1_recon" not in sample:
            raise ValueError("Dataset has no s_l1_recon entry.")
        return sample["s_l1_recon"].to(_DEVICE).float().flatten()
    if baseline == "l2":
        if "s_l2_recon" not in sample:
            raise ValueError("Dataset has no s_l2_recon entry.")
        return sample["s_l2_recon"].to(_DEVICE).float().flatten()
    raise ValueError(f"Unknown baseline {baseline!r}")


def total_variation(s_img: torch.Tensor) -> torch.Tensor:
    """Anisotropic TV on a 64×64 image."""
    dx = (s_img[:, 1:] - s_img[:, :-1]).abs().mean()
    dz = (s_img[1:, :] - s_img[:-1, :]).abs().mean()
    return dx + dz


# ─── Core trainer ─────────────────────────────────────────────────────────────

def train_residual(sample: dict, L_matrix: torch.Tensor, model: torch.nn.Module,
                   s_base: torch.Tensor, args, log) -> dict:
    """Train one (sample, model) pair. Returns final s_phys + history + metrics."""
    coords = sample["coords"].to(_DEVICE)
    d_meas = sample["d_meas"].to(_DEVICE).float().flatten()
    mask   = sample["mask"].to(_DEVICE).float().flatten()
    s_gt   = sample["s_gt_raw"].to(_DEVICE).float().flatten()
    n_valid = mask.sum().clamp_min(1.0)
    time_scale = args.time_scale

    model = model.to(_DEVICE)
    if args.zero_init:
        zero_init_last_linear(model)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"loss": [], "loss_data": [], "delta_max": []}
    best_loss = float("inf")
    best_state = None
    best_step = 0

    for step in range(args.steps):
        optim.zero_grad()

        delta_norm = model(coords).flatten()                 # (4096,)
        delta = delta_norm * args.delta_scale                # scale to slowness units
        s_phys = s_base + delta
        if args.clamp:
            s_phys = s_phys.clamp(_SLOWNESS_MIN, _SLOWNESS_MAX)

        if args.mode == "self":
            d_pred = L_matrix @ s_phys
            residual = (d_pred - d_meas) * mask
            loss_data = ((residual * time_scale) ** 2).sum() / n_valid
        else:  # sup
            target = s_gt - s_base
            loss_data = (((delta - target) * args.loss_scale) ** 2).mean()

        # Regularization on Δs only
        reg = torch.zeros((), device=_DEVICE)
        if args.reg_l2 > 0:
            reg = reg + args.reg_l2 * (delta ** 2).mean()
        if args.reg_l1 > 0:
            reg = reg + args.reg_l1 * delta.abs().mean()
        if args.reg_tv > 0:
            reg = reg + args.reg_tv * total_variation(delta.reshape(*_GRID))

        loss = loss_data + reg
        loss.backward()
        optim.step()

        if step % max(1, args.log_every) == 0:
            history["loss"].append(float(loss.item()))
            history["loss_data"].append(float(loss_data.item()))
            history["delta_max"].append(float(delta.detach().abs().max().item()))

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            best_step = step
            best_state = copy.deepcopy(model.state_dict())

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        delta_final = model(coords).flatten() * args.delta_scale
        s_phys_final = s_base + delta_final
        if args.clamp:
            s_phys_final = s_phys_final.clamp(_SLOWNESS_MIN, _SLOWNESS_MAX)

    s_phys_np = s_phys_final.detach().cpu().numpy()
    s_base_np = s_base.detach().cpu().numpy()
    delta_np  = delta_final.detach().cpu().numpy()

    metrics = calculate_metrics(s_phys_np, sample["s_gt_raw"].cpu().numpy())

    log.info(f"    final loss={best_loss:.4f} @ step {best_step}/{args.steps}  "
             f"|Δs|max={float(np.abs(delta_np).max()):.3e}  "
             f"MAE={metrics.get('MAE', float('nan')):.2f}  "
             f"CNR={metrics.get('CNR', float('nan')):.2f}  "
             f"SSIM={metrics.get('SSIM', float('nan')):.3f}")

    return {
        "s_phys": s_phys_np,
        "s_base": s_base_np,
        "delta":  delta_np,
        "metrics": metrics,
        "history": history,
        "best_step": best_step,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Dataset key from datasets.yaml (default: 'active' field)")
    parser.add_argument("--baseline", choices=["const", "l1", "l2"], required=True)
    parser.add_argument("--bg_const", type=float, default=1510.0,
                        help="Background SoS for --baseline const (m/s, default 1510)")
    parser.add_argument("--mode", choices=["self", "sup"], required=True,
                        help="self = self-supervised data fidelity (deployable);  "
                             "sup = supervised on (s_GT − s_base) (diagnostic, GT-dep.)")
    parser.add_argument("--model", choices=list(MODEL_MAP.keys()), default="relu")
    parser.add_argument("--hidden_features", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--mapping_size", type=int, default=64)
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Fourier feature scale (FourierMLP only)")
    parser.add_argument("--omega", type=float, default=30.0,
                        help="Siren omega (SirenMLP only)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--delta_scale", type=float, default=5e-5,
                        help="Scale factor on Δs(coords). Slowness units. "
                             "5e-5 ≈ natural slowness span for SoS 1450–1570 m/s. "
                             "Larger values let Δs wander into noise.")
    parser.add_argument("--loss_scale", type=float, default=1e6,
                        help="Multiplier on (Δs − target) before MSE in --mode sup. "
                             "Default 1e6 (~ microsecond-equivalent units) makes "
                             "gradients usable since raw target magnitudes are ~1e-5.")
    parser.add_argument("--zero_init", action="store_true", default=True,
                        help="Zero-init last layer so Δs=0 at step 0 (default ON)")
    parser.add_argument("--no_zero_init", dest="zero_init", action="store_false")
    parser.add_argument("--reg_l2", type=float, default=0.0)
    parser.add_argument("--reg_l1", type=float, default=0.0)
    parser.add_argument("--reg_tv", type=float, default=0.0)
    parser.add_argument("--clamp", action="store_true", default=True,
                        help="Clamp s_phys to [1/1800, 1/1200] (default ON)")
    parser.add_argument("--no_clamp", dest="clamp", action="store_false")
    parser.add_argument("--time_scale", type=float, default=None,
                        help="Loss scale (default = 1/pix2time from .mat)")
    parser.add_argument("--indices", nargs="+", type=int, default=None,
                        help="Sample indices (default: 8 fixed-seed indices)")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="If --indices not given, draw this many at seed 42")
    parser.add_argument("--tag", default="",
                        help="Free-form tag appended to output folder name")
    parser.add_argument("--report_plots", action="store_true",
                        help="Generate thesis-style report figures (needs GT)")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    # ── Output dir ────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_bits = [args.mode, args.baseline]
    if args.baseline == "const":
        tag_bits.append(f"bg{int(args.bg_const)}")
    tag_bits.append(args.model)
    if args.tag:
        tag_bits.append(args.tag)
    run_tag = f"{ts}_{'_'.join(tag_bits)}"
    out_dir = OUT_ROOT / "kwave_blob" / run_tag  # placeholder, set after dataset load
    # we will move/recreate after we know dataset key

    # ── Load dataset ──────────────────────────────────────────────────────
    ds_cfg = load_dataset_config(args.dataset)
    out_dir = OUT_ROOT / ds_cfg["key"] / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out_dir)

    log.info("=" * 64)
    log.info(f"  Residual study  (mode={args.mode}, baseline={args.baseline}, "
             f"model={args.model})")
    log.info(f"  Output dir : {out_dir}")
    log.info("=" * 64)

    grid_file = DATA_DIR + "/DL-based-SoS/forward_model_lr/grid_parameters.mat"
    h5_keys = ds_cfg.get("h5_keys")
    if not ds_cfg.get("has_A_matrix", True) and "matrix_file" in ds_cfg:
        matrix_path = DATA_DIR + ds_cfg["matrix_file"]
        dataset = USDataset(ds_cfg["data_path"], grid_file,
                            matrix_path=matrix_path,
                            use_external_L_matrix=True, h5_keys=h5_keys)
    else:
        dataset = USDataset(ds_cfg["data_path"], grid_file, h5_keys=h5_keys)
    log.info(f"Dataset loaded — {len(dataset)} samples ({ds_cfg['name']})")

    if args.time_scale is None:
        if hasattr(dataset, "pix2time") and dataset.pix2time is not None:
            args.time_scale = 1.0 / float(dataset.pix2time)
        else:
            args.time_scale = 1e6
    log.info(f"time_scale  : {args.time_scale:.2e}")

    if args.indices is None:
        np.random.seed(42)
        n = min(args.n_samples, len(dataset))
        args.indices = np.random.choice(len(dataset), size=n, replace=False).tolist()
    log.info(f"Indices     : {args.indices}")

    # ── L-matrix on device ────────────────────────────────────────────────
    L_t = dataset.L_matrix
    if hasattr(L_t, "to_dense"):
        L_t = L_t.to_dense()
    L_matrix = L_t.float().to(_DEVICE)
    log.info(f"L_matrix    : {tuple(L_matrix.shape)}  on {L_matrix.device}")

    # Validate baseline availability
    sample0 = dataset[args.indices[0]]
    if args.baseline == "l1" and "s_l1_recon" not in sample0:
        log.error("Baseline 'l1' requested but dataset has no s_l1_recon.")
        sys.exit(2)
    if args.baseline == "l2" and "s_l2_recon" not in sample0:
        log.error("Baseline 'l2' requested but dataset has no s_l2_recon.")
        sys.exit(2)

    # ── Run per sample ────────────────────────────────────────────────────
    label_residual = (f"Residual-{args.mode} / {args.model}"
                      f" / {args.baseline}"
                      + (f"({int(args.bg_const)})" if args.baseline == "const" else ""))
    label_baseline = f"Baseline / {args.baseline}" + (
        f"({int(args.bg_const)})" if args.baseline == "const" else ""
    )

    per_sample_residual = []
    per_sample_baseline = []
    samples_loaded = []

    t0 = time.time()
    for idx in args.indices:
        log.info(f"\n  ── sample idx={idx} ──")
        sample = dataset[idx]
        samples_loaded.append(sample)

        s_base = get_s_base(sample, args.baseline, args.bg_const)
        s_base_np = s_base.detach().cpu().numpy()
        base_metrics = calculate_metrics(s_base_np, sample["s_gt_raw"].cpu().numpy())
        log.info(f"    baseline {args.baseline}: "
                 f"MAE={base_metrics.get('MAE'):.2f}  "
                 f"CNR={base_metrics.get('CNR'):.2f}  "
                 f"SSIM={base_metrics.get('SSIM'):.3f}")
        per_sample_baseline.append({"metrics": base_metrics, "s_phys": s_base_np})

        # Fresh model per sample
        model = build_model(args.model, args.hidden_features, args.hidden_layers,
                            args.mapping_size, args.scale, args.omega)
        result = train_residual(sample, L_matrix, model, s_base, args, log)
        per_sample_residual.append({
            "metrics": result["metrics"],
            "s_phys":  result["s_phys"],
        })

        # Per-sample side-by-side PNG
        try:
            import matplotlib.pyplot as plt
            s_gt = sample["s_gt_raw"].cpu().numpy().reshape(_GRID, order="F")
            s_b  = s_base_np.reshape(_GRID, order="F")
            s_r  = result["s_phys"].reshape(_GRID, order="F")
            d_r  = result["delta"].reshape(_GRID, order="F")
            sos_gt = 1.0 / s_gt; sos_b = 1.0 / s_b; sos_r = 1.0 / s_r
            vmin = float(np.percentile(np.concatenate([sos_gt, sos_r], 0), 2))
            vmax = float(np.percentile(np.concatenate([sos_gt, sos_r], 0), 98))
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            for a, im, ttl, kw in [
                (ax[0], sos_gt, "GT (m/s)",      dict(vmin=vmin, vmax=vmax)),
                (ax[1], sos_b,  "Baseline",      dict(vmin=vmin, vmax=vmax)),
                (ax[2], sos_r,  "Residual recon",dict(vmin=vmin, vmax=vmax)),
                (ax[3], d_r,    "Δs (slowness)", dict(cmap="seismic")),
            ]:
                im_ = a.imshow(im, **kw); a.set_title(ttl); a.axis("off")
                plt.colorbar(im_, ax=a, fraction=0.045)
            fig.suptitle(f"idx={idx}  {label_residual}", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"idx{idx}_panels.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            log.warning(f"    panel plot failed: {exc}")

    elapsed = time.time() - t0
    log.info(f"\nAll samples done in {elapsed/60:.1f} min")

    # ── Report figures (reuse pipeline helpers) ──────────────────────────
    if args.report_plots and dataset.has_ground_truth:
        log.info("\n  Generating report figures ...")
        report_results = {
            label_baseline: per_sample_baseline,
            label_residual: per_sample_residual,
        }
        # also include L1/L2 for context if available
        for key, lbl in [("s_l1_recon", "L1"), ("s_l2_recon", "L2")]:
            if key in samples_loaded[0] and lbl != args.baseline.upper():
                report_results[lbl] = [
                    {"metrics": calculate_metrics(s[key], s["s_gt_raw"]), "s_phys": s[key]}
                    for s in samples_loaded
                ]
        try:
            grid_path    = out_dir / f"{run_tag}_report_comparison.svg"
            metrics_path = out_dir / f"{run_tag}_report_metrics.svg"
            plot_method_grid(
                results=report_results, samples=samples_loaded,
                save_path=grid_path,
                dataset_title={
                    "kwave_geom": "GeomSet", "kwave_blob": "BlobSet",
                    "inverse_crime": "InverseCrime",
                }.get(ds_cfg["key"], ds_cfg["key"]),
                show=False, png_fallback=True,
            )
            plot_metrics_comparison(
                results=report_results, save_path=metrics_path,
                show=False, png_fallback=True,
            )
            log.info(f"  Report grid    -> {grid_path}")
            log.info(f"  Report metrics -> {metrics_path}")
        except Exception as exc:
            log.warning(f"  Report figure generation failed: {exc}")

    # ── Save results.json ────────────────────────────────────────────────
    def _agg(per_sample):
        keys = ("MAE", "RMSE", "SSIM", "CNR")
        out = {}
        for k in keys:
            vals = [s["metrics"].get(k) for s in per_sample
                    if s["metrics"].get(k) is not None and np.isfinite(s["metrics"][k])]
            if vals:
                out[f"{k}_mean"] = float(np.mean(vals))
                out[f"{k}_std"]  = float(np.std(vals))
        return out

    results_json = {
        "timestamp": ts,
        "dataset":   ds_cfg["key"],
        "args":      {k: v for k, v in vars(args).items() if k != "indices"},
        "indices":   args.indices,
        "labels":    {"baseline": label_baseline, "residual": label_residual},
        "aggregate": {
            label_baseline: _agg(per_sample_baseline),
            label_residual: _agg(per_sample_residual),
        },
        "per_sample": {
            label_baseline: [{"idx": i, **s["metrics"]}
                             for i, s in zip(args.indices, per_sample_baseline)],
            label_residual: [{"idx": i, **s["metrics"]}
                             for i, s in zip(args.indices, per_sample_residual)],
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    log.info(f"\nresults.json  -> {out_dir / 'results.json'}")

    # ── Pretty summary ────────────────────────────────────────────────────
    log.info("\n  ── summary ──")
    for lbl, agg in results_json["aggregate"].items():
        log.info(f"    {lbl:50s} "
                 f"MAE={agg.get('MAE_mean', float('nan')):.2f}±{agg.get('MAE_std', 0):.2f}  "
                 f"SSIM={agg.get('SSIM_mean', float('nan')):.3f}  "
                 f"CNR={agg.get('CNR_mean', float('nan')):.2f}")


if __name__ == "__main__":
    main()
