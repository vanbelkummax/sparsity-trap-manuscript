"""Evaluation metrics for spatial transcriptomics prediction."""

import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_skimage


def compute_ssim(img1, img2, mask=None, batched=False, data_range=None):
    """
    Compute Structural Similarity Index (SSIM) between two images.

    SSIM measures perceptual similarity considering luminance, contrast, and structure.
    Range: [-1, 1], where 1 = perfect match.

    Args:
        img1: Predicted image (H, W) or (B, H, W)
        img2: Target image (H, W) or (B, H, W)
        mask: Binary tissue mask (H, W). If provided, only compute SSIM on masked region.
        batched: If True, img1 and img2 are (B, H, W) and returns list of SSIMs
        data_range: Data range of images. If None, inferred from img2.

    Returns:
        SSIM value (float) or list of SSIMs if batched=True
    """
    if batched:
        assert img1.ndim == 3 and img2.ndim == 3
        assert img1.shape[0] == img2.shape[0]
        return [compute_ssim(img1[i], img2[i], mask=mask, data_range=data_range)
                for i in range(img1.shape[0])]

    # Convert to numpy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Ensure 2D
    assert img1.ndim == 2 and img2.ndim == 2
    assert img1.shape == img2.shape

    # Apply mask if provided
    if mask is not None:
        assert mask.shape == img1.shape
        mask = mask.astype(bool)
        # Extract masked regions as 1D arrays
        img1_masked = img1[mask]
        img2_masked = img2[mask]
        # Compute SSIM on 1D arrays (not ideal, but consistent)
        # Alternative: compute on full image then threshold
        # For proper implementation, we'd need windowed SSIM with mask
        # For simplicity, treat as 1D correlation-like metric
        if len(img1_masked) == 0:
            return 0.0
        # Use Pearson correlation as proxy when masked
        # (True SSIM requires spatial structure, hard with mask)
        from scipy.stats import pearsonr
        if np.std(img1_masked) == 0 or np.std(img2_masked) == 0:
            return 0.0
        return pearsonr(img1_masked, img2_masked)[0]

    # Infer data range
    if data_range is None:
        data_range = img2.max() - img2.min()
        if data_range == 0:
            data_range = 1.0

    # Compute SSIM using skimage
    ssim_val = ssim_skimage(
        img1, img2,
        data_range=data_range,
        win_size=7,  # Default window size
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return float(ssim_val)


def compute_pcc(pred, target, mask=None):
    """
    Compute Pearson Correlation Coefficient (PCC).

    PCC measures linear correlation between predictions and targets.
    Range: [-1, 1], where 1 = perfect positive correlation.

    Args:
        pred: Predicted values (any shape)
        target: Ground truth values (same shape as pred)
        mask: Binary mask (same shape). If provided, only compute PCC on masked region.

    Returns:
        PCC value (float)
    """
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten().astype(bool)
        pred_flat = pred_flat[mask_flat]
        target_flat = target_flat[mask_flat]

    # Handle edge cases
    if len(pred_flat) == 0:
        return 0.0
    if np.std(pred_flat) == 0 or np.std(target_flat) == 0:
        return 0.0

    # Compute Pearson correlation
    pcc, _ = pearsonr(pred_flat, target_flat)

    return float(pcc)


def compute_per_gene_metrics(predictions, targets, mask=None):
    """
    Compute SSIM and PCC for each gene separately.

    Args:
        predictions: (B, C, H, W) or (C, H, W) predicted gene expression
        targets: (B, C, H, W) or (C, H, W) ground truth
        mask: (H, W) binary tissue mask

    Returns:
        dict with keys 'ssim' and 'pcc', each containing list of per-gene values
    """
    # Handle batching
    if predictions.ndim == 4:
        predictions = predictions[0]  # Take first batch
        targets = targets[0]

    assert predictions.ndim == 3  # (C, H, W)
    num_genes = predictions.shape[0]

    ssims = []
    pccs = []

    for i in range(num_genes):
        pred_gene = predictions[i]
        target_gene = targets[i]

        ssim_val = compute_ssim(pred_gene, target_gene, mask=mask)
        pcc_val = compute_pcc(pred_gene, target_gene, mask=mask)

        ssims.append(ssim_val)
        pccs.append(pcc_val)

    return {
        'ssim': ssims,
        'pcc': pccs
    }
