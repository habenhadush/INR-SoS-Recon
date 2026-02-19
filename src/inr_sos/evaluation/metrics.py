import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import warnings


def calculate_cnr(s_phys_pred, s_gt_raw):
    """
    Calculates the Contrast-to-Noise Ratio (CNR).
    Automatically segments the Ground Truth to find the ROI mask.
    
    Formula: CNR = |mu_roi - mu_bkg| / sqrt(sigma_roi^2 + sigma_bkg^2)
    """
    # 1. Use Otsu's method on the Ground Truth to automatically find the inclusion
    # We suppress warnings in case an image is perfectly uniform (no inclusion)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            thresh = threshold_otsu(s_gt_raw)
            roi_mask = s_gt_raw > thresh
            bkg_mask = ~roi_mask
        except ValueError:
            # Fallback if image is completely uniform
            return 0.0 

    # 2. Extract the predicted pixels using the GT masks
    pred_roi = s_phys_pred[roi_mask]
    pred_bkg = s_phys_pred[bkg_mask]
    
    # 3. Handle edge cases where mask might be empty
    if len(pred_roi) == 0 or len(pred_bkg) == 0:
        return 0.0
        
    # 4. Calculate statistics
    mu_roi, sigma_roi = np.mean(pred_roi), np.std(pred_roi)
    mu_bkg, sigma_bkg = np.mean(pred_bkg), np.std(pred_bkg)
    
    # 5. Calculate CNR
    denominator = np.sqrt(sigma_roi**2 + sigma_bkg**2)
    if denominator == 0:
        return 0.0
        
    cnr = np.abs(mu_roi - mu_bkg) / denominator
    return float(cnr)


def calculate_metrics(s_phys_pred, s_gt_raw, grid_shape=(64, 64)):
    """
    Computes all standard evaluation metrics for Speed of Sound reconstruction.
    
    Args:
        s_phys_pred: Predicted Speed of Sound (or Slowness).
        s_gt_raw: Ground Truth Speed of Sound (or Slowness).
        grid_shape: Tuple representing the 2D image dimensions.
        
    Returns:
        dict: Containing MAE, RMSE, SSIM, and CNR.
    """
    # 1. Ensure inputs are standard numpy arrays of the correct shape
    if hasattr(s_phys_pred, 'detach'):
        s_phys_pred = s_phys_pred.detach().cpu().numpy()
    if hasattr(s_gt_raw, 'detach'):
        s_gt_raw = s_gt_raw.detach().cpu().numpy()
        
    pred_img = s_phys_pred.reshape(grid_shape)
    gt_img = s_gt_raw.reshape(grid_shape)
    
    # 2. Convert Slowness to Speed of Sound (m/s) if necessary
    # Assuming standard biological SoS is ~1500 m/s. 
    # If the mean is tiny (e.g., 0.0006), it's slowness (s/m) and needs inverting.
    if np.mean(gt_img) < 1.0:
        pred_img = 1.0 / (pred_img + 1e-12)
        gt_img = 1.0 / (gt_img + 1e-12)
        
    # 3. Calculate MAE & RMSE
    abs_error = np.abs(pred_img - gt_img)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error ** 2))
    
    # 4. Calculate SSIM
    # We set the dynamic range to the exact data range of the Ground Truth
    data_range = gt_img.max() - gt_img.min()
    
    # Handle homogeneous background case (data_range == 0)
    if data_range < 1e-6:
        ssim_val = 1.0 if rmse < 1e-6 else 0.0
    else:
        ssim_val = ssim(gt_img, pred_img, data_range=data_range)
        
    # 5. Calculate CNR
    cnr = calculate_cnr(pred_img, gt_img)
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "SSIM": float(ssim_val),
        "CNR": float(cnr)
    }