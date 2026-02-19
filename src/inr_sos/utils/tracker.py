import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def generate_experiment_id(config) -> str:
    """
    Generates a unique, systematic filename based on the current date and configuration.
    Example: 20260217_1405_FourierMLP_Scale0.7_TV0.0001_Sample8
    """
    # Grab the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Extract key identifiers
    model = config.model_type
    scale = config.scale if model == "FourierMLP" else "NA"
    tv = config.tv_weight
    sample_idx = config.sample_idx
    
    # Construct the ID
    exp_id = f"{timestamp}_{model}_Scale{scale}_TV{tv}_Sample{sample_idx}"
    return exp_id


def save_reconstruction_plot(result_dict, sample, filepath, title, mae):
    """
    Generates the standard 4-panel comparison plot and saves it to disk.
    """
    grid_shape = (64, 64)
    s_gt = sample['s_gt_raw'].view(grid_shape).detach().cpu().numpy()
    s_rec = result_dict['s_phys'].view(grid_shape).detach().cpu().numpy()
    loss_hist = result_dict['loss_history']
    
    v_gt = 1.0 / (s_gt + 1e-8)
    v_rec = 1.0 / (s_rec + 1e-8)
    error_map = np.abs(v_gt - v_rec)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{title} | MAE: {mae:.2f} m/s", fontsize=16)
    
    im0 = axes[0].imshow(v_gt, cmap='jet', vmin=1400, vmax=1600)
    axes[0].set_title("Ground Truth (m/s)")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(v_rec, cmap='jet', vmin=1400, vmax=1600)
    axes[1].set_title("Reconstruction (m/s)")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(error_map, cmap='hot', vmin=0, vmax=50) 
    axes[2].set_title("Abs. Error (m/s)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    axes[3].plot(loss_hist, label='Total Loss', color='blue')
    axes[3].set_title("Optimization Loss")
    axes[3].set_xlabel("Iterations")
    axes[3].set_ylabel("Loss (Scaled)")
    axes[3].set_yscale('log')
    axes[3].grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    # Save the figure instead of showing it
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    
    # CRITICAL: Close the figure to free up RAM during batch processing
    plt.close(fig)


def save_artifacts(result_dict, sample, config, metrics, base_dir="experiments"):
    """
    Master function to save all artifacts (arrays and plots) systematically.
    """
    # 1. Create directory structure: experiments/GroupName/
    group_dir = os.path.join(base_dir, config.experiment_group)
    os.makedirs(group_dir, exist_ok=True)
    
    # 2. Generate the systematic name
    exp_id = generate_experiment_id(config)
    
    # 3. Save the raw numpy arrays
    # We save both GT and Pred so you can recalculate metrics later without the dataset
    s_phys = result_dict['s_phys'].detach().cpu().numpy()
    s_gt = sample['s_gt_raw'].detach().cpu().numpy()
    
    np.save(os.path.join(group_dir, f"{exp_id}_pred.npy"), s_phys)
    np.save(os.path.join(group_dir, f"{exp_id}_gt.npy"), s_gt)
    
    # 4. Save the visual plot
    plot_filepath = os.path.join(group_dir, f"{exp_id}_plot.png")
    save_reconstruction_plot(
        result_dict=result_dict, 
        sample=sample, 
        filepath=plot_filepath, 
        title=exp_id,
        mae=metrics['MAE']
    )
    
    # Return the filepath in case the pipeline wants to log the image to wandb
    return plot_filepath


def log_to_local_database(base_config, final_stats, target_indices, db_path="benchmark_results.csv"):
    """Appends the final aggregate statistics of a benchmark run to a master CSV ledger."""
    
    file_exists = os.path.isfile(db_path)
    
    with open(db_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # 1. Create the Header if the file is brand new
        if not file_exists:
            writer.writerow([
                "Date", "Experiment Group", "Model", "Scale", "TV Weight", 
                "Tested Indices", "MAE Mean", "MAE Std", "RMSE Mean", "RMSE Std", 
                "SSIM Mean", "SSIM Std", "CNR Mean", "CNR Std"
            ])
            
        # 2. Append the current benchmark results
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            base_config.experiment_group,
            base_config.model_type,
            base_config.scale if base_config.model_type == "FourierMLP" else "N/A",
            base_config.tv_weight,
            str(target_indices), # Log exactly which samples were tested
            f"{final_stats['MAE_mean']:.3f}", f"{final_stats['MAE_std']:.3f}",
            f"{final_stats['RMSE_mean']:.3f}", f"{final_stats['RMSE_std']:.3f}",
            f"{final_stats['SSIM_mean']:.3f}", f"{final_stats['SSIM_std']:.3f}",
            f"{final_stats['CNR_mean']:.3f}", f"{final_stats['CNR_std']:.3f}"
        ])
    
    print(f"Results successfully logged to local database: {db_path}")