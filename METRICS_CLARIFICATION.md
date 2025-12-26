# SSIM Computation Clarification

**Date**: December 25, 2025
**Status**: Bug fixed, manuscript results remain valid

---

## Summary

A code review identified a bug in `src/evaluation/metrics.py` where the `compute_ssim()` function returned **Pearson correlation** instead of **true SSIM** when masks were applied. However, investigation revealed that **this buggy function was never used for the manuscript results**.

The training scripts (`scripts/train_2um_cv_3fold.py`) use a **different SSIM implementation** that correctly computes structural similarity on masked images.

**Outcome**: The manuscript correctly reports SSIM values. The bug has been fixed for code accuracy, but no experimental re-runs are needed.

---

## The Bug (Lines 45-62 in Original metrics.py)

```python
if mask is not None:
    # Extract masked regions as 1D arrays
    img1_masked = img1[mask]
    img2_masked = img2[mask]
    # Use Pearson correlation as proxy when masked
    # (True SSIM requires spatial structure, hard with mask)
    from scipy.stats import pearsonr
    return pearsonr(img1_masked, img2_masked)[0]  # ❌ Returns PCC, not SSIM!
```

**Problem**: This collapses 2D masked images to 1D arrays and computes Pearson correlation, losing all spatial structure information that SSIM measures.

---

## What Was Actually Used (train_2um_cv_3fold.py, lines 436-453)

```python
from skimage.metrics import structural_similarity as ssim_metric

for b in range(pred_2um.shape[0]):
    if mask_2um[b, 0].mean() > 0.05:
        for g in range(n_genes):
            # Multiply images by mask (preserves spatial structure)
            p_img = pred_2um[b, g] * mask_2um[b, 0]
            l_img = label_2um[b, g] * mask_2um[b, 0]

            # Normalize to [0, 1]
            combined = np.concatenate([p_img.flatten(), l_img.flatten()])
            vmin, vmax = combined.min(), combined.max()

            if vmax - vmin > 1e-6:
                # Compute TRUE SSIM on masked images
                s = ssim_metric((p_img - vmin) / (vmax - vmin),
                               (l_img - vmin) / (vmax - vmin),
                               data_range=1.0)  # ✅ Real SSIM from skimage
                if not np.isnan(s):
                    ssim_2um_list.append(s)
```

**Key difference**: The training script:
1. Multiplies images by mask (preserves 2D spatial structure)
2. Calls `ssim_metric()` from skimage directly
3. Computes true structural similarity, not correlation

This is the **correct approach** for masked SSIM computation.

---

## Evidence: Saved Results Match Manuscript

Checkpoint files at `/mnt/x/mse-vs-poisson-2um-benchmark/results_cv/` contain both SSIM and PCC as separate metrics:

### Hist2ST + Poisson (`cv_summary_hist2st_poisson.json`)
- **mean_ssim_2um**: 0.5417 (manuscript: 0.542 ✓)
- **mean_pcc_2um**: 0.182 (different value → proves SSIM ≠ PCC)

### Hist2ST + MSE (`cv_summary_hist2st_mse.json`)
- **mean_ssim_2um**: 0.2001 (manuscript: 0.200 ✓)
- **mean_pcc_2um**: 0.111

**SSIM improvement**: 0.542 / 0.200 = **2.7× better** (as reported)
**PCC improvement**: 0.182 / 0.111 = 1.6× better

These are clearly different metrics being tracked independently. If the bug had been used, SSIM would equal PCC.

---

## The Fix (December 25, 2025)

Updated `src/evaluation/metrics.py` to match the training script implementation:

```python
if mask is not None:
    # Multiply images by mask (preserves spatial structure)
    img1 = img1 * mask
    img2 = img2 * mask

    # Normalize to [0, 1] range
    combined = np.concatenate([img1.flatten(), img2.flatten()])
    vmin, vmax = combined.min(), combined.max()

    if vmax - vmin < 1e-6:
        return 0.0

    img1_norm = (img1 - vmin) / (vmax - vmin)
    img2_norm = (img2 - vmin) / (vmax - vmin)

    # Compute true SSIM on masked images
    ssim_val = ssim_skimage(
        img1_norm, img2_norm,
        data_range=1.0,
        win_size=7,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )

    return float(ssim_val)  # ✅ Returns true SSIM
```

---

## Validation

### Test Case: Identical Images
```python
import numpy as np
from src.evaluation.metrics import compute_ssim

img = np.random.rand(256, 256)
mask = np.ones((256, 256), dtype=bool)

# Should return 1.0 for identical images
ssim = compute_ssim(img, img, mask=mask)
assert abs(ssim - 1.0) < 0.01, f"Expected ~1.0, got {ssim}"
```

### Test Case: Uncorrelated Random Images
```python
img1 = np.random.rand(256, 256)
img2 = np.random.rand(256, 256)
mask = np.ones((256, 256), dtype=bool)

ssim = compute_ssim(img1, img2, mask=mask)
# SSIM of random images should be near 0
# PCC of random images would also be near 0
# Use spatial structure test to differentiate:
img2_shifted = np.roll(img1, shift=10, axis=0)  # Shift identical image
ssim_shifted = compute_ssim(img1, img2_shifted, mask=mask)
# True SSIM should detect shift (lower value)
# PCC would not (similar correlation)
```

---

## Recommendations

1. **For this manuscript**: No changes needed. Results are valid as reported.

2. **For future work**: Use the fixed `compute_ssim()` function for consistency between training and evaluation code.

3. **For peer reviewers**: If questioned about SSIM computation, refer to:
   - Training script implementation (`scripts/train_2um_cv_3fold.py:436-453`)
   - Saved checkpoint files showing SSIM ≠ PCC
   - This clarification document

---

## References

- **Bug location**: `src/evaluation/metrics.py` (lines 45-62, pre-fix)
- **Correct implementation**: `scripts/train_2um_cv_3fold.py` (lines 436-453)
- **Checkpoints**: `/mnt/x/mse-vs-poisson-2um-benchmark/results_cv/`
- **Fix commit**: [To be added after push]

---

**Contact**: max.r.van.belkum@vanderbilt.edu
