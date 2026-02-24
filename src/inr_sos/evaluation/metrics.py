import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import warnings

_SOS_MIN = 1200.0
_SOS_MAX = 1800.0


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


def calculate_metrics1(s_phys_pred, s_gt_raw, grid_shape=(64, 64)):
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


def calculate_metrics(s_phys_pred, s_gt_raw, grid_shape=(64, 64)):
    """
    Computes MAE, RMSE, SSIM and CNR for a Speed-of-Sound reconstruction.

    Bug fixes applied
    -----------------
    BUG A (CRITICAL — caused SSIM values of ±100s on dashboard):
      Original code computed  data_range = gt_img.max() - gt_img.min()
      using only the GT range.  When pred_img has extreme values (from a
      poorly converged model outputting negative slowness), skimage's SSIM
      is mathematically undefined in that range and returns values far
      outside [-1, 1], e.g. -700 or +220.

      Fix (two-part):
        1. Clamp s_phys_pred to physically valid slowness before inversion
           so 1/s_phys is always in [SOS_MIN, SOS_MAX] = [1200, 1800] m/s.
        2. Compute data_range from the combined GT+pred range so SSIM is
           always computed in a valid window.
        3. Hard-clip the final SSIM value to [-1, 1] as a safety net.

    BUG B (silent — no positivity guarantee on slowness):
      s_phys = s_norm * s_std + s_mean can be negative if the model
      diverges (s_norm << -s_mean/s_std ≈ -43).  1/negative = large
      negative velocity → explodes every downstream metric.
      Fix: clamp s_phys to [1/SOS_MAX, 1/SOS_MIN] before inversion.
    """
    if hasattr(s_phys_pred, 'detach'):
        s_phys_pred = s_phys_pred.detach().cpu().numpy()
    if hasattr(s_gt_raw, 'detach'):
        s_gt_raw = s_gt_raw.detach().cpu().numpy()

    pred_flat = s_phys_pred.flatten()
    gt_flat   = s_gt_raw.flatten()

    # Convert slowness (s/m) → speed of sound (m/s) if values are tiny
    if np.mean(gt_flat) < 1.0:
        # BUG A+B FIX: clamp to physically valid slowness before inverting
        slowness_min = 1.0 / _SOS_MAX   # ~5.56e-4 s/m
        slowness_max = 1.0 / _SOS_MIN   # ~8.33e-4 s/m
        pred_flat = np.clip(pred_flat, slowness_min, slowness_max)
        # gt should already be physical, but clamp for safety
        gt_flat   = np.clip(gt_flat,   slowness_min, slowness_max)

        pred_flat = 1.0 / pred_flat   # now in [SOS_MIN, SOS_MAX]
        gt_flat   = 1.0 / gt_flat

    pred_img = pred_flat.reshape(grid_shape)
    gt_img   = gt_flat.reshape(grid_shape)

    # MAE & RMSE
    abs_error = np.abs(pred_img - gt_img)
    mae  = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(abs_error ** 2)))

    # SSIM — BUG A FIX: use combined range, hard-clip result
    gt_min, gt_max     = gt_img.min(),   gt_img.max()
    pred_min, pred_max = pred_img.min(), pred_img.max()
    data_range = max(gt_max, pred_max) - min(gt_min, pred_min)

    if data_range < 1e-6:
        ssim_val = 1.0 if rmse < 1e-6 else 0.0
    else:
        raw_ssim = ssim(gt_img, pred_img, data_range=data_range)
        ssim_val = float(np.clip(raw_ssim, -1.0, 1.0))   # safety net

    # CNR
    cnr = calculate_cnr(pred_img, gt_img)

    return {
        "MAE":  mae,
        "RMSE": rmse,
        "SSIM": ssim_val,
        "CNR":  cnr,
    }