#!/usr/bin/env python3
"""
Create WSI and tile-level figures for MSE vs Poisson comparison.

Features:
- Full WSI figures for top rescued genes
- Tile-level comparison showing finer detail
- Grid of patches for detailed architecture visualization
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_metric
from matplotlib.colors import PowerNorm

# Configuration
DATA_DIR = Path('/mnt/x/img2st_rotation_demo/processed_crc_raw_counts/P5')
E_PRIME_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/results_cv/model_E_prime_hist2st_poisson/fold1_testP5_20251224_105345')
D_PRIME_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/results_cv/model_D_prime_hist2st_mse/fold1_testP5_20251224_174024')
TABLES_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/tables')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures/wsi_improved')
TILE_OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures/tiles')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TILE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Color palette
COLORS = {
    'poisson': '#00b894',
    'mse': '#d63031',
}

CATEGORY_COLORS = {
    'Immune': '#9b59b6',
    'Epithelial': '#3498db',
    'Stromal': '#e74c3c',
    'Secretory': '#2ecc71',
    'Mitochondrial': '#f39c12',
    'Housekeeping': '#1abc9c',
    'Other': '#7f8c8d',
}


def load_patches_with_coords(data_dir):
    """Load patches and extract coordinates from patch IDs."""
    patches = np.load(data_dir / 'patches_raw_counts.npy', allow_pickle=True).tolist()

    coords = []
    for item in patches:
        if 'patch_row' in item and 'patch_col' in item:
            coords.append((item['patch_row'], item['patch_col']))
        elif 'patch_id' in item:
            parts = item['patch_id'].replace('patch_', '').split('_')
            coords.append((int(parts[0]), int(parts[1])))
        else:
            img_path = item.get('img_path', item.get('image_path', ''))
            name = Path(img_path).stem.replace('patch_', '')
            parts = name.split('_')
            coords.append((int(parts[0]), int(parts[1])))

    return patches, coords


def build_coord_map(coords):
    """Build coordinate map for WSI stitching."""
    rows = [c[0] for c in coords]
    cols = [c[1] for c in coords]

    return {
        'row_offset': min(rows),
        'col_offset': min(cols),
        'n_rows': max(rows) - min(rows) + 1,
        'n_cols': max(cols) - min(cols) + 1,
        'patch_coords': coords,
    }


def load_he_images(data_dir, patches):
    """Load H&E images for all patches."""
    images = []
    for item in tqdm(patches, desc='Loading H&E images'):
        img_path_key = 'img_path' if 'img_path' in item else 'image_path'
        img_name = Path(item[img_path_key]).name
        img_path = data_dir / 'images' / img_name

        if img_path.exists():
            img = np.array(Image.open(img_path).convert('RGB')) / 255.0
            images.append(img)
        else:
            images.append(np.ones((224, 224, 3)) * 0.9)

    return np.array(images)


def stitch_wsi(data, gene_idx, masks, coord_map, patch_size=128):
    """Stitch patches into WSI using coordinates, with proper masking."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.full((n_rows * patch_size, n_cols * patch_size), np.nan)

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(data):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size

            patch_data = data[patch_idx, gene_idx]
            patch_mask = masks[patch_idx, 0] if masks.ndim == 4 else masks[patch_idx]

            mask_bool = patch_mask > 0.5
            wsi_patch = np.full((patch_size, patch_size), np.nan)
            wsi_patch[mask_bool] = patch_data[mask_bool]

            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = wsi_patch

    return wsi


def stitch_he_wsi(images, coord_map, input_size=224, patch_size=128):
    """Stitch H&E patches into WSI with downsampling."""
    row_offset = coord_map['row_offset']
    col_offset = coord_map['col_offset']
    n_rows = coord_map['n_rows']
    n_cols = coord_map['n_cols']

    wsi = np.ones((n_rows * patch_size, n_cols * patch_size, 3)) * 0.95

    for patch_idx, (row, col) in enumerate(coord_map['patch_coords']):
        if patch_idx >= len(images):
            break
        r = row - row_offset
        c = col - col_offset
        if 0 <= r < n_rows and 0 <= c < n_cols:
            r_start = r * patch_size
            c_start = c * patch_size

            img = images[patch_idx]
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_resized = np.array(img_pil.resize((patch_size, patch_size), Image.LANCZOS)) / 255.0
            wsi[r_start:r_start+patch_size, c_start:c_start+patch_size] = img_resized

    return wsi


def compute_gene_metrics(pred_e, pred_d, labels, masks, gene_idx):
    """Compute PCC and SSIM for a specific gene."""
    mask_flat = masks[:, 0].flatten() > 0.5
    pred_e_flat = pred_e[:, gene_idx].flatten()[mask_flat]
    pred_d_flat = pred_d[:, gene_idx].flatten()[mask_flat]
    label_flat = labels[:, gene_idx].flatten()[mask_flat]

    pcc_e = pearsonr(pred_e_flat, label_flat)[0] if len(pred_e_flat) > 10 else 0
    pcc_d = pearsonr(pred_d_flat, label_flat)[0] if len(pred_d_flat) > 10 else 0

    # SSIM per patch
    ssim_e_list, ssim_d_list = [], []
    for b in range(pred_e.shape[0]):
        if masks[b, 0].mean() > 0.05:
            p_e = pred_e[b, gene_idx] * masks[b, 0]
            p_d = pred_d[b, gene_idx] * masks[b, 0]
            l = labels[b, gene_idx] * masks[b, 0]

            combined_e = np.concatenate([p_e.flatten(), l.flatten()])
            combined_d = np.concatenate([p_d.flatten(), l.flatten()])
            vmin_e, vmax_e = combined_e.min(), combined_e.max()
            vmin_d, vmax_d = combined_d.min(), combined_d.max()

            if vmax_e - vmin_e > 1e-6:
                try:
                    s_e = ssim_metric((p_e - vmin_e)/(vmax_e - vmin_e),
                                      (l - vmin_e)/(vmax_e - vmin_e), data_range=1.0)
                    if not np.isnan(s_e):
                        ssim_e_list.append(s_e)
                except:
                    pass

            if vmax_d - vmin_d > 1e-6:
                try:
                    s_d = ssim_metric((p_d - vmin_d)/(vmax_d - vmin_d),
                                      (l - vmin_d)/(vmax_d - vmin_d), data_range=1.0)
                    if not np.isnan(s_d):
                        ssim_d_list.append(s_d)
                except:
                    pass

    ssim_e = np.mean(ssim_e_list) if ssim_e_list else 0
    ssim_d = np.mean(ssim_d_list) if ssim_d_list else 0

    return {
        'pcc_poisson': pcc_e,
        'pcc_mse': pcc_d,
        'ssim_poisson': ssim_e,
        'ssim_mse': ssim_d,
        'delta_pcc': pcc_e - pcc_d,
        'delta_ssim': ssim_e - ssim_d,
    }


def create_wsi_figure(gene_name, gene_idx, he_wsi, gt_wsi, mse_wsi, poisson_wsi,
                      metrics, category, output_path):
    """Create a 4-panel WSI comparison figure."""

    gt_valid = gt_wsi[~np.isnan(gt_wsi)]
    mse_valid = mse_wsi[~np.isnan(mse_wsi)]
    poisson_valid = poisson_wsi[~np.isnan(poisson_wsi)]

    gt_vmax = np.percentile(gt_valid, 98) if len(gt_valid) > 0 else 1
    mse_vmin = np.percentile(mse_valid, 1) if len(mse_valid) > 0 else 0
    mse_vmax = np.percentile(mse_valid, 99) if len(mse_valid) > 0 else 1
    poisson_vmin = np.percentile(poisson_valid, 1) if len(poisson_valid) > 0 else 0
    poisson_vmax = np.percentile(poisson_valid, 99) if len(poisson_valid) > 0 else 1

    fig = plt.figure(figsize=(26, 7), facecolor='white')
    gs = GridSpec(1, 4, figure=fig, wspace=0.22)

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color='#e8e8e8')

    # Panel 1: H&E
    ax_he = fig.add_subplot(gs[0, 0])
    ax_he.imshow(he_wsi)
    ax_he.set_title('H&E', fontsize=15, fontweight='bold', pad=12)
    ax_he.axis('off')

    # Panel 2: Ground Truth
    ax_gt = fig.add_subplot(gs[0, 1])
    im_gt = ax_gt.imshow(gt_wsi, cmap=cmap, vmin=0, vmax=gt_vmax, interpolation='nearest')
    ax_gt.set_title('Ground Truth', fontsize=15, fontweight='bold', pad=12)
    ax_gt.axis('off')
    cbar_gt = plt.colorbar(im_gt, ax=ax_gt, fraction=0.04, pad=0.02, shrink=0.85)
    cbar_gt.ax.tick_params(labelsize=10)

    # Panel 3: MSE (D')
    ax_mse = fig.add_subplot(gs[0, 2])
    im_mse = ax_mse.imshow(mse_wsi, cmap=cmap, vmin=mse_vmin, vmax=mse_vmax, interpolation='nearest')
    ax_mse.set_title(f"Model D' (MSE)\nSSIM = {metrics['ssim_mse']:.3f}",
                     fontsize=14, fontweight='bold', color=COLORS['mse'], pad=12)
    ax_mse.axis('off')
    cbar_mse = plt.colorbar(im_mse, ax=ax_mse, fraction=0.04, pad=0.02, shrink=0.85)
    cbar_mse.ax.tick_params(labelsize=10)

    # Panel 4: Poisson (E')
    ax_poisson = fig.add_subplot(gs[0, 3])
    im_poisson = ax_poisson.imshow(poisson_wsi, cmap=cmap, vmin=poisson_vmin, vmax=poisson_vmax,
                                    interpolation='nearest')
    ax_poisson.set_title(f"Model E' (Poisson)\nSSIM = {metrics['ssim_poisson']:.3f}",
                         fontsize=14, fontweight='bold', color=COLORS['poisson'], pad=12)
    ax_poisson.axis('off')
    cbar_poisson = plt.colorbar(im_poisson, ax=ax_poisson, fraction=0.04, pad=0.02, shrink=0.85)
    cbar_poisson.ax.tick_params(labelsize=10)

    # Suptitle with improvement
    cat_color = CATEGORY_COLORS.get(category, '#7f8c8d')
    fig.suptitle(f"{gene_name} ({category}) — ΔSSIM = +{metrics['delta_ssim']:.3f}",
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.1)
    plt.close()

    print(f"  Created: {output_path.name}")


def compute_per_tile_ssim(pred_e, pred_d, labels, masks, gene_idx):
    """Compute per-tile SSIM for both models and return ΔSSIM per tile."""
    n_patches = pred_e.shape[0]
    tile_metrics = []

    for b in range(n_patches):
        mask = masks[b, 0]
        tissue_frac = mask.mean()

        if tissue_frac < 0.1:
            tile_metrics.append({'idx': b, 'delta_ssim': -999, 'tissue_frac': tissue_frac,
                                'ssim_poisson': 0, 'ssim_mse': 0, 'expression': 0})
            continue

        gt = labels[b, gene_idx] * mask
        mse_pred = pred_d[b, gene_idx] * mask
        poisson_pred = pred_e[b, gene_idx] * mask

        expression = gt.sum()

        # Normalize for SSIM
        combined = np.concatenate([mse_pred.flatten(), poisson_pred.flatten(), gt.flatten()])
        vmin, vmax = combined.min(), combined.max()

        if vmax - vmin < 1e-6:
            tile_metrics.append({'idx': b, 'delta_ssim': -999, 'tissue_frac': tissue_frac,
                                'ssim_poisson': 0, 'ssim_mse': 0, 'expression': expression})
            continue

        try:
            gt_norm = (gt - vmin) / (vmax - vmin)
            mse_norm = (mse_pred - vmin) / (vmax - vmin)
            poisson_norm = (poisson_pred - vmin) / (vmax - vmin)

            ssim_mse = ssim_metric(mse_norm, gt_norm, data_range=1.0)
            ssim_poisson = ssim_metric(poisson_norm, gt_norm, data_range=1.0)
            delta_ssim = ssim_poisson - ssim_mse

            tile_metrics.append({
                'idx': b,
                'delta_ssim': delta_ssim if not np.isnan(delta_ssim) else -999,
                'tissue_frac': tissue_frac,
                'ssim_poisson': ssim_poisson,
                'ssim_mse': ssim_mse,
                'expression': expression
            })
        except:
            tile_metrics.append({'idx': b, 'delta_ssim': -999, 'tissue_frac': tissue_frac,
                                'ssim_poisson': 0, 'ssim_mse': 0, 'expression': expression})

    return tile_metrics


def create_tile_comparison(gene_name, gene_idx, he_images, pred_e, pred_d, labels, masks,
                           category, metrics, output_path, n_tiles=5):
    """Create tile-level comparison showing finer detail.

    Shows 4 rows: H&E, Ground Truth, D' (MSE), E' (Poisson)
    Each panel has its own scale for proper visualization.

    TILE SELECTION: Picks tiles with HIGHEST per-tile ΔSSIM to showcase
    the clearest MSE vs Poisson differences.
    """
    # Compute per-tile SSIM to find tiles with biggest rescue
    tile_metrics = compute_per_tile_ssim(pred_e, pred_d, labels, masks, gene_idx)

    # Filter for tiles with good tissue coverage AND substantial expression
    # High expression = visible structure in ground truth
    expression_threshold = np.percentile([t['expression'] for t in tile_metrics if t['expression'] > 0], 70)

    valid_tiles = [t for t in tile_metrics
                   if t['delta_ssim'] > 0.1  # Meaningful improvement
                   and t['tissue_frac'] > 0.3
                   and t['expression'] > expression_threshold]  # High expression tiles

    if len(valid_tiles) < n_tiles:
        # Relax expression threshold
        expression_threshold = np.percentile([t['expression'] for t in tile_metrics if t['expression'] > 0], 50)
        valid_tiles = [t for t in tile_metrics
                       if t['delta_ssim'] > 0 and t['tissue_frac'] > 0.2 and t['expression'] > expression_threshold]

    if len(valid_tiles) < n_tiles:
        # Fall back to any tiles with positive ΔSSIM
        valid_tiles = [t for t in tile_metrics if t['delta_ssim'] > -900 and t['tissue_frac'] > 0.1]

    # Sort by combined score: high ΔSSIM AND high expression (both matter for visualization)
    for t in valid_tiles:
        # Normalize expression and ΔSSIM for combined scoring
        t['combined_score'] = t['delta_ssim'] * 0.5 + (t['expression'] / (expression_threshold + 1)) * 0.5

    valid_tiles.sort(key=lambda x: x['combined_score'], reverse=True)

    # Take top tiles
    good_indices = [t['idx'] for t in valid_tiles[:n_tiles]]

    if len(good_indices) < n_tiles:
        # Fall back to any tiles with tissue
        tissue_content = masks[:, 0].mean(axis=(1, 2))
        fallback = np.argsort(tissue_content)[-n_tiles:]
        good_indices = list(fallback)

    # Create figure: 4 rows (H&E, GT, MSE, Poisson) × n_tiles columns
    fig, axes = plt.subplots(4, n_tiles, figsize=(4.5*n_tiles, 16))

    cmap = plt.cm.magma.copy()
    cmap.set_bad(color='#e8e8e8')

    for col, patch_idx in enumerate(good_indices[:n_tiles]):
        # Get data
        mask = masks[patch_idx, 0]
        gt = labels[patch_idx, gene_idx]
        mse_pred = pred_d[patch_idx, gene_idx]
        poisson_pred = pred_e[patch_idx, gene_idx]

        # Apply mask
        gt_masked = np.where(mask > 0.5, gt, np.nan)
        mse_masked = np.where(mask > 0.5, mse_pred, np.nan)
        poisson_masked = np.where(mask > 0.5, poisson_pred, np.nan)

        # Individual scaling for each panel
        gt_valid = gt_masked[~np.isnan(gt_masked)]
        mse_valid = mse_masked[~np.isnan(mse_masked)]
        poisson_valid = poisson_masked[~np.isnan(poisson_masked)]

        # For GROUND TRUTH: Use power normalization to emphasize sparse expression
        # gamma < 1 compresses high values, expands low values (makes sparse data visible)
        gt_vmin = 0
        gt_vmax = np.percentile(gt_valid, 99) if len(gt_valid) > 0 else 1
        gt_vmax = max(gt_vmax, 0.001)  # Ensure positive
        gt_norm = PowerNorm(gamma=0.4, vmin=gt_vmin, vmax=gt_vmax)

        # For PREDICTIONS: Use percentile scaling to show full dynamic range
        mse_vmin = np.percentile(mse_valid, 2) if len(mse_valid) > 0 else 0
        mse_vmax = np.percentile(mse_valid, 98) if len(mse_valid) > 0 else 1
        poisson_vmin = np.percentile(poisson_valid, 2) if len(poisson_valid) > 0 else 0
        poisson_vmax = np.percentile(poisson_valid, 98) if len(poisson_valid) > 0 else 1

        # Row 0: H&E
        ax_he = axes[0, col]
        he_patch = he_images[patch_idx]
        he_pil = Image.fromarray((he_patch * 255).astype(np.uint8))
        he_resized = np.array(he_pil.resize((128, 128), Image.LANCZOS)) / 255.0
        ax_he.imshow(he_resized)
        ax_he.set_xticks([])
        ax_he.set_yticks([])
        for spine in ax_he.spines.values():
            spine.set_visible(False)

        # Row 1: Ground Truth (with power normalization for sparse data)
        ax_gt = axes[1, col]
        im_gt = ax_gt.imshow(gt_masked, cmap=cmap, norm=gt_norm, interpolation='nearest')
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        for spine in ax_gt.spines.values():
            spine.set_visible(False)
        # Colorbar below the image
        cbar_gt = fig.colorbar(im_gt, ax=ax_gt, orientation='horizontal', fraction=0.08, pad=0.02, shrink=0.8)
        cbar_gt.ax.tick_params(labelsize=8)

        # Row 2: MSE (D')
        ax_mse = axes[2, col]
        im_mse = ax_mse.imshow(mse_masked, cmap=cmap, vmin=mse_vmin, vmax=mse_vmax, interpolation='nearest')
        ax_mse.set_xticks([])
        ax_mse.set_yticks([])
        for spine in ax_mse.spines.values():
            spine.set_visible(False)
        cbar_mse = fig.colorbar(im_mse, ax=ax_mse, orientation='horizontal', fraction=0.08, pad=0.02, shrink=0.8)
        cbar_mse.ax.tick_params(labelsize=8)

        # Row 3: Poisson (E')
        ax_poisson = axes[3, col]
        im_poisson = ax_poisson.imshow(poisson_masked, cmap=cmap, vmin=poisson_vmin, vmax=poisson_vmax,
                                        interpolation='nearest')
        ax_poisson.set_xticks([])
        ax_poisson.set_yticks([])
        for spine in ax_poisson.spines.values():
            spine.set_visible(False)
        cbar_poisson = fig.colorbar(im_poisson, ax=ax_poisson, orientation='horizontal', fraction=0.08, pad=0.02, shrink=0.8)
        cbar_poisson.ax.tick_params(labelsize=8)

        # Compute per-tile SSIM
        if mask.mean() > 0.1:
            mse_norm = mse_pred * mask
            poisson_norm = poisson_pred * mask
            gt_norm = gt * mask

            combined = np.concatenate([mse_norm.flatten(), poisson_norm.flatten(), gt_norm.flatten()])
            vmin_c, vmax_c = combined.min(), combined.max()

            if vmax_c - vmin_c > 1e-6:
                try:
                    ssim_mse = ssim_metric((mse_norm - vmin_c)/(vmax_c - vmin_c),
                                          (gt_norm - vmin_c)/(vmax_c - vmin_c), data_range=1.0)
                    ssim_poisson = ssim_metric((poisson_norm - vmin_c)/(vmax_c - vmin_c),
                                               (gt_norm - vmin_c)/(vmax_c - vmin_c), data_range=1.0)

                    # Add SSIM in title of each prediction panel
                    ax_mse.set_title(f'SSIM={ssim_mse:.2f}', fontsize=10, color=COLORS['mse'],
                                    fontweight='bold', pad=2)
                    ax_poisson.set_title(f'SSIM={ssim_poisson:.2f}', fontsize=10, color=COLORS['poisson'],
                                        fontweight='bold', pad=2)
                except:
                    pass

        # Column header: tile number
        axes[0, col].text(0.5, 1.08, f'Tile {col+1}', transform=axes[0, col].transAxes,
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Row labels on left side
    row_labels = [
        ('H&E\n(Input)', 'black'),
        (f'{gene_name}\n(Ground Truth)', 'black'),
        (f"Model D'\n(Hist2ST + MSE)", COLORS['mse']),
        (f"Model E'\n(Hist2ST + Poisson)", COLORS['poisson']),
    ]

    for row, (label, color) in enumerate(row_labels):
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                         ha='right', va='center', fontsize=12, fontweight='bold',
                         color=color, rotation=0, multialignment='right')

    # Overall title
    fig.suptitle(f'{gene_name} ({category}) — Tile-Level Comparison at 2µm Resolution\n'
                 f'Model E\' (Poisson) vs Model D\' (MSE) | Overall ΔSSIM = +{metrics["delta_ssim"]:.3f}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.08, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', pad_inches=0.2)
    plt.close()

    print(f"  Created: {output_path.name}")


def main():
    print("=" * 60)
    print("Creating WSI and Tile-Level Comparison Figures")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    pred_e = np.load(E_PRIME_DIR / 'pred_2um.npy')
    pred_d = np.load(D_PRIME_DIR / 'pred_2um.npy')
    labels = np.load(E_PRIME_DIR / 'label_2um.npy')
    masks = np.load(E_PRIME_DIR / 'mask_2um.npy')

    print(f"Loaded predictions: {pred_e.shape}")

    # Load gene names
    with open(DATA_DIR / 'gene_names.json') as f:
        gene_names_data = json.load(f)
        if isinstance(gene_names_data, dict):
            gene_names = gene_names_data.get('gene_names', list(gene_names_data.keys()))
        else:
            gene_names = gene_names_data

    # Load per-gene metrics
    df = pd.read_csv(TABLES_DIR / 'table_s1_pergene_metrics.csv')

    # Load patches and build coordinate map
    patches, coords = load_patches_with_coords(DATA_DIR)
    coord_map = build_coord_map(coords)
    print(f"WSI grid: {coord_map['n_rows']} x {coord_map['n_cols']} patches")

    # Load H&E images
    he_images = load_he_images(DATA_DIR, patches)
    print(f"Loaded {len(he_images)} H&E images")

    # Stitch H&E WSI
    print("\nStitching H&E WSI...")
    he_wsi = stitch_he_wsi(he_images, coord_map)

    # CURATED GENE SELECTION: Best rescue genes across sparsity spectrum
    # Goal: showcase dramatic MSE vs Poisson differences at varied sparsity levels
    curated_genes = [
        # HIGH rescue, varied sparsity (showcase the rescue effect)
        'TSPAN8',    # +0.730 ΔSSIM, 96.2% sparsity - TOP RESCUE
        'CEACAM5',   # +0.699 ΔSSIM, 94.2% sparsity - Epithelial
        'EPCAM',     # +0.684 ΔSSIM, 94.9% sparsity - Epithelial
        'KRT8',      # +0.673 ΔSSIM, 96.8% sparsity - Epithelial
        'MUC12',     # +0.632 ΔSSIM, 95.3% sparsity - Secretory
        'JCHAIN',    # +0.459 ΔSSIM, 96.6% sparsity - Immune
        # HIGH sparsity (>98%) - shows sparsity trap most clearly
        'PYGB',      # +0.677 ΔSSIM, 98.7% sparsity - HIGH
        'LGALS3',    # +0.687 ΔSSIM, 98.0% sparsity - HIGH
        'SYNGR2',    # +0.593 ΔSSIM, 98.6% sparsity - HIGH
        # Category representatives for completeness
        'MT-ND5',    # +0.517 ΔSSIM, 96.3% sparsity - Mitochondrial
        'VIM',       # +0.180 ΔSSIM, 95.2% sparsity - Stromal (lower rescue)
        'TMSB10',    # +0.409 ΔSSIM, 93.6% sparsity - Housekeeping
    ]

    # Filter to genes that exist
    top_genes = df[df['Gene'].isin(curated_genes)].copy()

    # Sort by ΔSSIM for output order
    top_genes = top_genes.sort_values('Delta_SSIM', ascending=False)

    print(f"\nGenerating figures for {len(top_genes)} curated genes...")
    print("-" * 60)

    for _, row in top_genes.iterrows():
        gene_name = row['Gene']
        category = row['Category']

        if gene_name not in gene_names:
            print(f"Warning: {gene_name} not found in gene_names")
            continue

        gene_idx = gene_names.index(gene_name)

        print(f"\n{gene_name} ({category}):")

        # Compute metrics
        metrics = compute_gene_metrics(pred_e, pred_d, labels, masks, gene_idx)

        # Create WSI figure
        gt_wsi = stitch_wsi(labels, gene_idx, masks, coord_map)
        mse_wsi = stitch_wsi(pred_d, gene_idx, masks, coord_map)
        poisson_wsi = stitch_wsi(pred_e, gene_idx, masks, coord_map)

        wsi_path = OUTPUT_DIR / f'{gene_name}_2um_WSI_improved.png'
        create_wsi_figure(gene_name, gene_idx, he_wsi, gt_wsi, mse_wsi, poisson_wsi,
                          metrics, category, wsi_path)

        # Create tile comparison
        tile_path = TILE_OUTPUT_DIR / f'{gene_name}_tile_comparison.png'
        create_tile_comparison(gene_name, gene_idx, he_images, pred_e, pred_d, labels, masks,
                               category, metrics, tile_path, n_tiles=6)

    print("\n" + "=" * 60)
    print(f"WSI figures saved to: {OUTPUT_DIR}")
    print(f"Tile figures saved to: {TILE_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
