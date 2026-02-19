import numpy as np
import wandb
import copy
import logging
import sys

from inr_sos.evaluation.metrics import calculate_metrics
from inr_sos.utils.config import ExperimentConfig
from inr_sos.utils.tracker import save_artifacts, log_to_local_database

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_evaluation(
    dataset, 
    model_class, 
    train_engine, 
    config: "ExperimentConfig", 
    target_indices=None, 
    use_wandb=True
):
    if target_indices is None:
        target_indices = [1] #list(range(len(dataset)))
        
    num_samples = len(target_indices)
    
    # --- 1. Initialize the SUMMARY Run immediately ---
    # We use a distinct "job_type" so we can filter these benchmark runs easily later
    if use_wandb:
        summary_run = wandb.init(
            project=config.project_name,
            group=f"{config.experiment_group}_SUMMARY",
            name=f"Benchmark_{config.model_type}",
            job_type="benchmark",
            config=config.to_dict(),
            reinit=True
        )
        
        # Create a rich table to store per-sample metrics
        # This answers the reviewer's request for "per image errors"
        benchmark_table = wandb.Table(columns=["Sample_Idx", "MAE", "RMSE", "SSIM", "CNR", "Reconstruction"])

    logging.info(f"Starting Benchmark: {config.experiment_group} | Model: {config.model_type}")

    aggregated_metrics = {"MAE": [], "RMSE": [], "SSIM": [], "CNR": []}
    
    for i, idx in enumerate(target_indices):
        logging.info(f"--- Processing Sample {i+1}/{num_samples} (Index: {idx}) ---")
        
        sample = dataset[idx]
        current_config = copy.deepcopy(config)
        current_config.sample_idx = idx

        if use_wandb:
            run = wandb.init(
                project=current_config.project_name,
                group=current_config.experiment_group, # e.g., "Full_Matrix"
                job_type=current_config.model_type,    # e.g., "FourierMLP"
                name=f"{current_config.model_type}_Sample_{idx}", 
                config=current_config.to_dict(),
                reinit=True
            )
        
        if current_config.model_type == "FourierMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size,
                scale=current_config.scale
            )
        elif current_config.model_type == "SirenMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size,
                omega=current_config.omega
            )
        elif current_config.model_type == "ReluMLP":
            model = model_class(
                in_features=current_config.in_features,
                hidden_features=current_config.hidden_features,
                hidden_layers=current_config.hidden_layers,
                mapping_size=current_config.mapping_size
            )
        
        else:
            raise ValueError(f"Unsupported model type: {current_config.model_type}")

        # Train (Engine handles logic, but NO wandb init/finish inside!)
        # Note: We are NOT creating separate runs per sample anymore to keep dashboard clean.
        # If you really need per-sample loss curves, we can add that back, 
        # but for benchmarking, we usually care about the final metrics.
        result_dict = train_engine(
            sample=sample,
            L_matrix=dataset.L_matrix,
            model=model,
            label=current_config.model_type,
            config=current_config,
            use_wandb=use_wandb # Disable engine logging to speed up benchmarking
        )
        
        # Calculate Metrics
        metrics = calculate_metrics(
            s_phys_pred=result_dict['s_phys'],
            s_gt_raw=sample['s_gt_raw'],
            grid_shape=(64, 64)
        )
        
        # Save Artifacts Locally
        plot_filepath = save_artifacts(
            result_dict=result_dict,
            sample=sample,
            config=current_config,
            metrics=metrics
        )
        
        if use_wandb:
            wandb.log({
                "Final MAE": metrics["MAE"],
                "Final SSIM": metrics["SSIM"],
                "Final CNR": metrics["CNR"],
                "Reconstruction Plot": wandb.Image(plot_filepath)
            })
            wandb.finish() # LOCK THE DOOR FOR THIS SAMPLE
        
        # Store metrics for aggregation
        for key in aggregated_metrics.keys():
            aggregated_metrics[key].append(metrics[key])
            
        # --- Add this sample's results to the W&B Table ---
        if use_wandb:
            benchmark_table.add_data(
                idx, 
                metrics["MAE"], 
                metrics["RMSE"], 
                metrics["SSIM"], 
                metrics["CNR"],
                wandb.Image(plot_filepath) # We put the image DIRECTLY in the table row!
            )
        
        logging.info(f"Sample {idx} -> MAE: {metrics['MAE']:.2f}")

    # ==========================================
    # FINAL AGGREGATION
    # ==========================================
    final_stats = {}
    
    for key, values_list in aggregated_metrics.items():
        mean_val = np.mean(values_list)
        std_val = np.std(values_list)
        final_stats[f"{key}_mean"] = mean_val
        final_stats[f"{key}_std"] = std_val

    # Log to Local CSV
    log_to_local_database(config, final_stats, target_indices)

    # Log to W&B
    if use_wandb:
        # Log the scalar averages
        wandb.log(final_stats)
        # Log the rich table (This creates the "dashboard" you want)
        wandb.log({"Benchmark_Results": benchmark_table})
        wandb.finish()

    return final_stats