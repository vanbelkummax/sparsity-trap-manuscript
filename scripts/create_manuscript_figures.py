#!/usr/bin/env python3
"""
Create manuscript figures for MSE vs Poisson 2um Benchmark.

Figures:
- Figure 1a: 2×2 factorial SSIM bar chart with CV error bars
- Figure 1b: 50-gene scatter plot (MSE SSIM vs Poisson SSIM)
- Figure 1c: Sparsity vs ΔSSIM correlation
- Figure 1d: Per-gene ΔSSIM waterfall plot
- Figure 2: Gene category analysis with representative examples
- Figure 3: Main effects decomposition
- Figure S1: Per-fold consistency
- Figure 4: Representative rescued genes per category (NEW)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path

# Configuration
RESULTS_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/results_cv')
TABLES_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/tables')
OUTPUT_DIR = Path('/mnt/x/mse-vs-poisson-2um-benchmark/figures/manuscript')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 13,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# Color palette
COLORS = {
    'poisson': '#00b894',  # Green
    'mse': '#d63031',      # Red
    'hist2st': '#0984e3',  # Blue
    'img2st': '#fdcb6e',   # Yellow
    'neutral': '#636e72',  # Gray
}

# Category colors - distinct and colorblind-friendly
CATEGORY_COLORS = {
    'Immune': '#9b59b6',
    'Epithelial': '#3498db',
    'Stromal': '#e74c3c',
    'Secretory': '#2ecc71',
    'Mitochondrial': '#f39c12',
    'Housekeeping': '#1abc9c',
    'Other': '#7f8c8d',
}


def load_cv_summaries():
    """Load all CV summary files."""
    summaries = {}
    for f in RESULTS_DIR.glob('cv_summary_*.json'):
        with open(f) as fp:
            data = json.load(fp)
            key = f"{data['decoder']}_{data['loss']}"
            summaries[key] = data
    return summaries


def load_pergene_metrics():
    """Load per-gene metrics from CSV."""
    return pd.read_csv(TABLES_DIR / 'table_s1_pergene_metrics.csv')


def figure_1a_factorial_ssim(summaries):
    """
    Figure 1a: 2×2 Factorial SSIM Bar Chart
    Fixed: Brackets raised well above value labels, proper significance stars
    """
    fig, ax = plt.subplots(figsize=(13, 9))

    models = [
        ("E'\n(Hist2ST+Poisson)", 'hist2st_poisson', COLORS['poisson']),
        ("D'\n(Hist2ST+MSE)", 'hist2st_mse', COLORS['mse']),
        ("F\n(Img2ST+Poisson)", 'img2st_poisson', '#27ae60'),
        ("G\n(Img2ST+MSE)", 'img2st_mse', '#c0392b'),
    ]

    x = np.arange(len(models))
    width = 0.6

    means = [summaries[key]['mean_ssim_2um'] for _, key, _ in models]
    stds = [summaries[key]['std_ssim_2um'] for _, key, _ in models]
    colors = [c for _, _, c in models]

    bars = ax.bar(x, means, width, yerr=stds, capsize=8, color=colors,
                  edgecolor='black', linewidth=2.5, error_kw={'linewidth': 2.5, 'capthick': 2.5})

    # Add value labels - positioned just above error bars
    label_tops = []
    for i, (mean, std) in enumerate(zip(means, stds)):
        label_y = mean + std + 0.02
        ax.text(i, label_y, f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        # Track where labels end (approx 2 lines of text = ~0.08 in data coords)
        label_tops.append(label_y + 0.10)

    ax.set_ylabel('SSIM (2µm)', fontsize=16, fontweight='bold')
    ax.set_title('2×2 Factorial Design: Loss × Decoder\n(3-Fold Cross-Validation)',
                 fontweight='bold', fontsize=17, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in models], fontsize=13)

    # Calculate bracket position ABOVE the highest label
    y_bracket = max(label_tops[0], label_tops[1]) + 0.04
    improvement_e_d = means[0] / means[1]

    # Draw bracket: vertical lines from bar tops to bracket, then horizontal
    ax.plot([0, 0], [label_tops[0], y_bracket], 'k-', lw=2.5)  # Left vertical
    ax.plot([1, 1], [label_tops[1], y_bracket], 'k-', lw=2.5)  # Right vertical
    ax.plot([0, 1], [y_bracket, y_bracket], 'k-', lw=2.5)      # Horizontal connector

    # 2.7x label with significance stars - ABOVE the bracket
    ax.text(0.5, y_bracket + 0.025, f'{improvement_e_d:.1f}×', ha='center', va='bottom',
            fontsize=16, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9', edgecolor='black', lw=2))
    # Add stars separately below
    ax.text(0.5, y_bracket + 0.11, '***', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='black')
    ax.text(0.5, y_bracket + 0.16, 'p < 0.001', ha='center', va='bottom',
            fontsize=10, color='gray')

    # Img2ST comparison bracket (F vs G)
    y_bracket2 = max(label_tops[2], label_tops[3]) + 0.04
    improvement_f_g = means[2] / means[3]

    ax.plot([2, 2], [label_tops[2], y_bracket2], 'gray', lw=1.5)
    ax.plot([3, 3], [label_tops[3], y_bracket2], 'gray', lw=1.5)
    ax.plot([2, 3], [y_bracket2, y_bracket2], 'gray', lw=1.5)

    ax.text(2.5, y_bracket2 + 0.02, f'{improvement_f_g:.1f}× (n.s.)', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    # Extend y-axis to fit everything
    ax.set_ylim(0, y_bracket + 0.25)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1a_factorial_ssim.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_1a_factorial_ssim.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_1a_factorial_ssim.png")
    print(f"  E'/D' improvement: {improvement_e_d:.2f}x")
    print(f"  F/G improvement: {improvement_f_g:.2f}x")
    return means, stds


def figure_1b_gene_scatter(df):
    """
    Figure 1b: 50-Gene Scatter Plot
    Fixed: Better legend positioning and gene labels
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot each gene with category color
    for _, row in df.iterrows():
        color = CATEGORY_COLORS.get(row['Category'], '#7f8c8d')
        ax.scatter(row['SSIM_MSE'], row['SSIM_Poisson'],
                   c=color, s=100, alpha=0.8, edgecolors='black', linewidth=0.8)

    # Diagonal line (y=x)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='No difference')

    # Fill regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.08, color=COLORS['poisson'])
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.08, color=COLORS['mse'])

    # Add region labels
    ax.text(0.15, 0.85, 'Poisson\nwins', fontsize=14, fontweight='bold',
            color=COLORS['poisson'], alpha=0.7, ha='center')
    ax.text(0.85, 0.15, 'MSE\nwins', fontsize=14, fontweight='bold',
            color=COLORS['mse'], alpha=0.7, ha='center')

    # Labels
    ax.set_xlabel('SSIM (MSE)', fontsize=14)
    ax.set_ylabel('SSIM (Poisson)', fontsize=14)
    ax.set_title('Per-Gene SSIM Comparison\n(50/50 genes above diagonal)',
                 fontweight='bold', fontsize=15, pad=15)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Legend for categories - positioned outside
    handles = [mpatches.Patch(color=c, label=cat, ec='black', linewidth=0.5)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=10,
              title='Gene Category', title_fontsize=11, framealpha=0.95)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1b_gene_scatter.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_1b_gene_scatter.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_1b_gene_scatter.png")


def figure_1c_sparsity_correlation(df):
    """
    Figure 1c: Sparsity vs ΔSSIM Correlation
    Fixed: Better annotation positioning
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x = df['Sparsity_pct'].values
    y = df['Delta_SSIM'].values

    # Scatter with category colors
    for _, row in df.iterrows():
        color = CATEGORY_COLORS.get(row['Category'], '#7f8c8d')
        ax.scatter(row['Sparsity_pct'], row['Delta_SSIM'],
                   c=color, s=100, alpha=0.8, edgecolors='black', linewidth=0.8)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(70, 100, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k-', linewidth=3, label=f'Linear fit')

    # Confidence band
    ax.fill_between(x_line, y_line - 0.08, y_line + 0.08, alpha=0.15, color='gray')

    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Gene Sparsity (% zero bins at 2µm)', fontsize=14)
    ax.set_ylabel('ΔSSIM (Poisson − MSE)', fontsize=14)
    ax.set_title(f'Sparsity Predicts Poisson Benefit\n(Pearson r = {r_value:.3f}, p = {p_value:.1e})',
                 fontweight='bold', fontsize=15, pad=15)

    # Annotation box for correlation stats
    textstr = f'r = {r_value:.3f}\np < 0.0001'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props, fontweight='bold')

    # Legend for categories
    handles = [mpatches.Patch(color=c, label=cat, ec='black', linewidth=0.5)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=9, ncol=2,
              title='Gene Category', title_fontsize=10, framealpha=0.95)

    ax.set_xlim(70, 100)
    ax.set_ylim(-0.1, 0.85)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1c_sparsity_correlation.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_1c_sparsity_correlation.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_1c_sparsity_correlation.png")
    print(f"  Sparsity-ΔSSIM correlation: r={r_value:.3f}, p={p_value:.2e}")

    return r_value, p_value


def figure_1d_waterfall(df):
    """
    Figure 1d: Per-Gene ΔSSIM Waterfall Plot
    Fixed: Better gene labeling without overlap
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    # Sort by Delta_SSIM
    df_sorted = df.sort_values('Delta_SSIM', ascending=False).reset_index(drop=True)

    colors = [CATEGORY_COLORS.get(cat, '#7f8c8d') for cat in df_sorted['Category']]

    bars = ax.bar(range(len(df_sorted)), df_sorted['Delta_SSIM'], color=colors,
                  edgecolor='black', linewidth=0.5, width=0.8)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=1.5)

    # Label top 8 genes (spread out to avoid overlap)
    top_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in top_indices:
        gene = df_sorted.iloc[i]['Gene']
        val = df_sorted.iloc[i]['Delta_SSIM']
        # Stagger labels
        offset = 0.04 if i % 2 == 0 else 0.08
        ax.annotate(gene, xy=(i, val), xytext=(i, val + offset),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    rotation=45)

    # Label any genes that decreased (if any)
    decreased = df_sorted[df_sorted['Delta_SSIM'] < 0]
    for _, row in decreased.iterrows():
        idx = df_sorted[df_sorted['Gene'] == row['Gene']].index[0]
        ax.annotate(row['Gene'], xy=(idx, row['Delta_SSIM']),
                    xytext=(idx, row['Delta_SSIM'] - 0.03),
                    ha='center', va='top', fontsize=9, color='red', fontweight='bold')

    ax.set_xlabel('Genes (ranked by ΔSSIM)', fontsize=14)
    ax.set_ylabel('ΔSSIM (Poisson − MSE)', fontsize=14)
    ax.set_title('Per-Gene SSIM Improvement with Poisson Loss\n(All 50 genes shown, positive = Poisson better)',
                 fontweight='bold', fontsize=15, pad=15)

    # Legend - horizontal at bottom
    handles = [mpatches.Patch(color=c, label=cat, ec='black', linewidth=0.5)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=10, ncol=4, framealpha=0.95)

    ax.set_xlim(-1, len(df_sorted))
    ax.set_ylim(-0.15, 0.85)
    ax.set_xticks([])

    # Add count annotation
    n_improved = (df_sorted['Delta_SSIM'] > 0).sum()
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['poisson'])
    ax.text(0.02, 0.95, f'{n_improved}/50 genes improved', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', color=COLORS['poisson'], bbox=props)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1d_waterfall.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_1d_waterfall.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_1d_waterfall.png")


def figure_2_category_analysis(df):
    """
    Figure 2: Gene Category Analysis with Representative Gene Labels
    Fixed: Show representative gene names and better spacing
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Aggregate by category
    category_stats = df.groupby('Category').agg({
        'Delta_SSIM': ['mean', 'std', 'count'],
        'Sparsity_pct': 'mean',
    }).reset_index()
    category_stats.columns = ['Category', 'Mean_Delta_SSIM', 'Std_Delta_SSIM', 'Count', 'Mean_Sparsity']
    category_stats = category_stats.sort_values('Mean_Delta_SSIM', ascending=False)

    # Find best gene per category
    best_genes = {}
    for cat in category_stats['Category']:
        cat_df = df[df['Category'] == cat].sort_values('Delta_SSIM', ascending=False)
        best_genes[cat] = cat_df.iloc[0]['Gene']

    x = np.arange(len(category_stats))
    colors = [CATEGORY_COLORS.get(cat, '#7f8c8d') for cat in category_stats['Category']]

    # Bar chart
    bars = ax.bar(x, category_stats['Mean_Delta_SSIM'], yerr=category_stats['Std_Delta_SSIM'],
                  capsize=6, color=colors, edgecolor='black', linewidth=2, alpha=0.85,
                  error_kw={'linewidth': 2.5})

    # Overlay individual gene points
    for i, cat in enumerate(category_stats['Category']):
        cat_genes = df[df['Category'] == cat]['Delta_SSIM'].values
        jitter = np.random.uniform(-0.2, 0.2, len(cat_genes))
        ax.scatter(x[i] + jitter, cat_genes, c='black', s=40, alpha=0.5, zorder=5)

    # Add best gene label for each category
    for i, cat in enumerate(category_stats['Category']):
        best_gene = best_genes[cat]
        best_val = df[df['Gene'] == best_gene]['Delta_SSIM'].values[0]
        mean_val = category_stats[category_stats['Category'] == cat]['Mean_Delta_SSIM'].values[0]
        std_val = category_stats[category_stats['Category'] == cat]['Std_Delta_SSIM'].values[0]

        # Label with best gene
        ax.annotate(f'Best: {best_gene}\n(+{best_val:.2f})',
                    xy=(i, mean_val + std_val + 0.02),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Labels
    ax.set_ylabel('ΔSSIM (Poisson − MSE)', fontsize=14)
    ax.set_title('SSIM Improvement by Gene Category\n(Individual genes as points, best gene labeled)',
                 fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cat}\n(n={int(n)})" for cat, n in
                        zip(category_stats['Category'], category_stats['Count'])],
                       rotation=0, fontsize=11)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Add mean sparsity info at bottom
    for i, (cat, sparsity) in enumerate(zip(category_stats['Category'], category_stats['Mean_Sparsity'])):
        ax.text(i, -0.12, f'{sparsity:.0f}% sparse', ha='center', fontsize=9, color='gray')

    ax.set_ylim(-0.18, 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_2_category_analysis.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_2_category_analysis.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_2_category_analysis.png")


def figure_3_main_effects(summaries):
    """
    Figure 3: Main Effects Decomposition
    Fixed: Larger D'/E' labels and better title positioning
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Extract SSIM values
    ssim_vals = {
        'E': summaries['hist2st_poisson']['mean_ssim_2um'],
        'D': summaries['hist2st_mse']['mean_ssim_2um'],
        'F': summaries['img2st_poisson']['mean_ssim_2um'],
        'G': summaries['img2st_mse']['mean_ssim_2um'],
    }

    # Left panel: Loss effect
    ax1 = axes[0]
    poisson_mean = (ssim_vals['E'] + ssim_vals['F']) / 2
    mse_mean = (ssim_vals['D'] + ssim_vals['G']) / 2

    bars1 = ax1.bar(['MSE\n(D\' + G)', 'Poisson\n(E\' + F)'], [mse_mean, poisson_mean],
                    color=[COLORS['mse'], COLORS['poisson']],
                    edgecolor='black', linewidth=2.5, width=0.5)

    # Add individual model points with LARGER labels
    ax1.scatter([0], [ssim_vals['D']], c='white', s=200, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
    ax1.scatter([0], [ssim_vals['G']], c='white', s=200, marker='s', edgecolors='black', linewidth=2.5, zorder=5)
    ax1.scatter([1], [ssim_vals['E']], c='white', s=200, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
    ax1.scatter([1], [ssim_vals['F']], c='white', s=200, marker='s', edgecolors='black', linewidth=2.5, zorder=5)

    # LARGER model labels (fontsize 14 instead of 10)
    ax1.text(0.15, ssim_vals['D'], "D'", fontsize=14, fontweight='bold', va='center')
    ax1.text(0.15, ssim_vals['G'], "G", fontsize=14, fontweight='bold', va='center')
    ax1.text(1.15, ssim_vals['E'], "E'", fontsize=14, fontweight='bold', va='center')
    ax1.text(1.15, ssim_vals['F'], "F", fontsize=14, fontweight='bold', va='center')

    ax1.set_ylabel('Mean SSIM (2µm)', fontsize=15, fontweight='bold')
    ax1.set_title('A. Main Effect: Loss Function', fontweight='bold', fontsize=16, loc='left', pad=15)

    # Improvement annotation
    improvement = (poisson_mean - mse_mean) / mse_mean * 100
    ax1.annotate('', xy=(1, poisson_mean + 0.02), xytext=(0, mse_mean + 0.02),
                 arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax1.text(0.5, (poisson_mean + mse_mean) / 2 + 0.10, f'+{improvement:.0f}%',
             ha='center', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    ax1.set_ylim(0, 0.60)
    ax1.set_xticklabels(['MSE\n(D\' + G)', 'Poisson\n(E\' + F)'], fontsize=13)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: Decoder effect
    ax2 = axes[1]
    hist2st_mean = (ssim_vals['E'] + ssim_vals['D']) / 2
    img2st_mean = (ssim_vals['F'] + ssim_vals['G']) / 2

    bars2 = ax2.bar(['Img2ST\n(F + G)', 'Hist2ST\n(E\' + D\')'], [img2st_mean, hist2st_mean],
                    color=[COLORS['img2st'], COLORS['hist2st']],
                    edgecolor='black', linewidth=2.5, width=0.5)

    # Add individual model points with LARGER labels
    ax2.scatter([0], [ssim_vals['F']], c='white', s=200, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
    ax2.scatter([0], [ssim_vals['G']], c='white', s=200, marker='s', edgecolors='black', linewidth=2.5, zorder=5)
    ax2.scatter([1], [ssim_vals['E']], c='white', s=200, marker='o', edgecolors='black', linewidth=2.5, zorder=5)
    ax2.scatter([1], [ssim_vals['D']], c='white', s=200, marker='s', edgecolors='black', linewidth=2.5, zorder=5)

    # LARGER model labels
    ax2.text(0.15, ssim_vals['F'], "F", fontsize=14, fontweight='bold', va='center')
    ax2.text(0.15, ssim_vals['G'], "G", fontsize=14, fontweight='bold', va='center')
    ax2.text(1.15, ssim_vals['E'], "E'", fontsize=14, fontweight='bold', va='center')
    ax2.text(1.15, ssim_vals['D'], "D'", fontsize=14, fontweight='bold', va='center')

    ax2.set_ylabel('Mean SSIM (2µm)', fontsize=15, fontweight='bold')
    ax2.set_title('B. Main Effect: Decoder Architecture', fontweight='bold', fontsize=16, loc='left', pad=15)

    improvement2 = (hist2st_mean - img2st_mean) / img2st_mean * 100
    ax2.annotate('', xy=(1, hist2st_mean + 0.02), xytext=(0, img2st_mean + 0.02),
                 arrowprops=dict(arrowstyle='->', color='black', lw=3))
    ax2.text(0.5, (hist2st_mean + img2st_mean) / 2 + 0.10, f'+{improvement2:.0f}%',
             ha='center', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

    ax2.set_ylim(0, 0.60)
    ax2.set_xticklabels(['Img2ST\n(F + G)', 'Hist2ST\n(E\' + D\')'], fontsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Legend at bottom - LARGER
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                                   markeredgecolor='black', markersize=12, label='Hist2ST decoder'),
                       plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                                   markeredgecolor='black', markersize=12, label='Img2ST decoder')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=13,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT_DIR / 'figure_3_main_effects.png', dpi=300, facecolor='white',
                bbox_inches='tight', pad_inches=0.15)
    plt.savefig(OUTPUT_DIR / 'figure_3_main_effects.pdf', facecolor='white',
                bbox_inches='tight', pad_inches=0.15)
    plt.close()

    print(f"Created: figure_3_main_effects.png")
    print(f"  Loss effect: +{improvement:.0f}% (Poisson vs MSE)")
    print(f"  Decoder effect: +{improvement2:.0f}% (Hist2ST vs Img2ST)")


def figure_4_representative_genes(df):
    """
    Figure 4: Representative Rescued Genes by Category
    Shows the most improved gene from each category with metrics
    Specifies model names: D' (Hist2ST+MSE) vs E' (Hist2ST+Poisson)
    """
    fig, ax = plt.subplots(figsize=(15, 9))

    # Find best gene per category
    categories = ['Epithelial', 'Secretory', 'Immune', 'Stromal', 'Mitochondrial', 'Housekeeping', 'Other']
    best_genes = []

    for cat in categories:
        cat_df = df[df['Category'] == cat].sort_values('Delta_SSIM', ascending=False)
        if len(cat_df) > 0:
            best = cat_df.iloc[0]
            best_genes.append({
                'Gene': best['Gene'],
                'Category': cat,
                'MSE_SSIM': best['SSIM_MSE'],
                'Poisson_SSIM': best['SSIM_Poisson'],
                'Delta_SSIM': best['Delta_SSIM'],
                'Sparsity': best['Sparsity_pct'],
            })

    # Sort by Delta_SSIM
    best_genes = sorted(best_genes, key=lambda x: x['Delta_SSIM'], reverse=True)

    x = np.arange(len(best_genes))
    width = 0.35

    # Bar chart: MSE vs Poisson for each representative gene
    mse_vals = [g['MSE_SSIM'] for g in best_genes]
    poisson_vals = [g['Poisson_SSIM'] for g in best_genes]

    # Use model names in legend
    bars_mse = ax.bar(x - width/2, mse_vals, width,
                      label="Model D' (Hist2ST + MSE)",
                      color=COLORS['mse'], edgecolor='black', linewidth=2, alpha=0.85)
    bars_poisson = ax.bar(x + width/2, poisson_vals, width,
                          label="Model E' (Hist2ST + Poisson)",
                          color=COLORS['poisson'], edgecolor='black', linewidth=2, alpha=0.85)

    # Add delta labels
    for i, g in enumerate(best_genes):
        # Arrow from MSE to Poisson
        ax.annotate('', xy=(i + width/2, g['Poisson_SSIM'] + 0.02),
                    xytext=(i - width/2, g['MSE_SSIM'] + 0.02),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

        # Delta label with box
        mid_y = (g['MSE_SSIM'] + g['Poisson_SSIM']) / 2 + 0.10
        ax.text(i, mid_y, f'+{g["Delta_SSIM"]:.2f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

    # Category color bar at bottom
    for i, g in enumerate(best_genes):
        ax.bar(i, 0.03, width=0.9, bottom=-0.08,
               color=CATEGORY_COLORS.get(g['Category'], '#7f8c8d'),
               edgecolor='black', linewidth=1)

    ax.set_ylabel('SSIM (2µm)', fontsize=15, fontweight='bold')
    ax.set_title("Representative Rescued Genes by Category\n"
                 "Model E' (Hist2ST + Poisson) vs Model D' (Hist2ST + MSE)",
                 fontweight='bold', fontsize=16, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g['Gene']}\n({g['Category']})" for g in best_genes],
                       fontsize=11, fontweight='bold')

    # Legend with model specifications
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95, title='Decoder + Loss',
              title_fontsize=12)

    ax.set_ylim(-0.12, 1.05)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_4_representative_genes.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_4_representative_genes.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_4_representative_genes.png")
    print(f"  Representative genes by category:")
    for g in best_genes:
        print(f"    {g['Category']:15} | {g['Gene']:10} | ΔSSIM: +{g['Delta_SSIM']:.3f}")


def figure_s1_fold_consistency(summaries):
    """
    Figure S1: Cross-Validation Fold Consistency
    Fixed: Better labeling
    """
    fig, ax = plt.subplots(figsize=(11, 7))

    models = [
        ("E' (Poisson)", 'hist2st_poisson', COLORS['poisson']),
        ("D' (MSE)", 'hist2st_mse', COLORS['mse']),
    ]

    x = np.arange(3)  # 3 folds
    width = 0.35

    for i, (name, key, color) in enumerate(models):
        fold_ssims = [f['test_ssim_2um'] for f in summaries[key]['folds']]
        fold_patients = [f['test_patient'] for f in summaries[key]['folds']]

        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, fold_ssims, width, label=name, color=color,
                      edgecolor='black', linewidth=2)

        # Add patient and value labels
        for j, (ssim, patient) in enumerate(zip(fold_ssims, fold_patients)):
            ax.text(j + offset, ssim + 0.015, f'{patient}\n{ssim:.3f}',
                    ha='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Cross-Validation Fold', fontsize=14)
    ax.set_ylabel('Test SSIM (2µm)', fontsize=14)
    ax.set_title('Cross-Validation Consistency Across Patients\n(Each fold tests on different patient)',
                 fontweight='bold', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Fold 1\n(Train: P1+P2)', 'Fold 2\n(Train: P1+P5)', 'Fold 3\n(Train: P2+P5)'],
                       fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 0.7)

    # Add CV statistics
    e_mean = summaries['hist2st_poisson']['mean_ssim_2um']
    e_std = summaries['hist2st_poisson']['std_ssim_2um']
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=COLORS['poisson'])
    ax.text(0.02, 0.95, f"E' mean: {e_mean:.3f} ± {e_std:.3f}\n(CV = {e_std/e_mean*100:.1f}%)",
            transform=ax.transAxes, fontsize=12, va='top', bbox=props, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_s1_fold_consistency.png', dpi=300, facecolor='white')
    plt.savefig(OUTPUT_DIR / 'figure_s1_fold_consistency.pdf', facecolor='white')
    plt.close()

    print(f"Created: figure_s1_fold_consistency.png")


def figure_combined_1abcd(summaries, df):
    """
    Combined Figure 1: All 4 panels with improved spacing
    """
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ===== Panel A: Factorial Bar Chart =====
    ax_a = fig.add_subplot(gs[0, 0])
    models = [
        ("E'", 'hist2st_poisson', COLORS['poisson']),
        ("D'", 'hist2st_mse', COLORS['mse']),
        ("F", 'img2st_poisson', '#27ae60'),
        ("G", 'img2st_mse', '#c0392b'),
    ]

    x = np.arange(len(models))
    means = [summaries[key]['mean_ssim_2um'] for _, key, _ in models]
    stds = [summaries[key]['std_ssim_2um'] for _, key, _ in models]
    colors = [c for _, _, c in models]

    bars = ax_a.bar(x, means, 0.55, yerr=stds, capsize=5, color=colors,
                    edgecolor='black', linewidth=2, error_kw={'linewidth': 2})

    # Add value labels and track positions
    label_tops_a = []
    for i, (mean, std) in enumerate(zip(means, stds)):
        label_y = mean + std + 0.02
        ax_a.text(i, label_y, f'{mean:.3f}', ha='center', va='bottom',
                  fontsize=10, fontweight='bold')
        label_tops_a.append(label_y + 0.05)

    ax_a.set_ylabel('SSIM (2µm)', fontsize=13)
    ax_a.set_title('A. 2×2 Factorial: Loss × Decoder', fontweight='bold', fontsize=14, loc='left')
    ax_a.set_xticks(x)
    labels_a = ['E\'\nHist2ST+Poisson', 'D\'\nHist2ST+MSE', 'F\nImg2ST+Poisson', 'G\nImg2ST+MSE']
    ax_a.set_xticklabels(labels_a, fontsize=9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Significance bracket - properly positioned above labels
    y_bracket_a = max(label_tops_a[0], label_tops_a[1]) + 0.03
    improvement_a = means[0] / means[1]

    ax_a.plot([0, 0], [label_tops_a[0], y_bracket_a], 'k-', lw=2)
    ax_a.plot([1, 1], [label_tops_a[1], y_bracket_a], 'k-', lw=2)
    ax_a.plot([0, 1], [y_bracket_a, y_bracket_a], 'k-', lw=2)

    ax_a.text(0.5, y_bracket_a + 0.015, f'{improvement_a:.1f}× ***', ha='center', va='bottom',
              fontsize=11, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', edgecolor='black', lw=1.5))

    ax_a.set_ylim(0, y_bracket_a + 0.12)

    # ===== Panel B: Gene Scatter =====
    ax_b = fig.add_subplot(gs[0, 1])

    for _, row in df.iterrows():
        color = CATEGORY_COLORS.get(row['Category'], '#7f8c8d')
        ax_b.scatter(row['SSIM_MSE'], row['SSIM_Poisson'],
                     c=color, s=70, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax_b.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
    ax_b.fill_between([0, 1], [0, 1], [1, 1], alpha=0.08, color=COLORS['poisson'])

    ax_b.set_xlabel('SSIM (MSE)', fontsize=13)
    ax_b.set_ylabel('SSIM (Poisson)', fontsize=13)
    ax_b.set_title('B. Per-Gene SSIM (50/50 above diagonal)', fontweight='bold', fontsize=14, loc='left')
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.set_aspect('equal')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=cat[:6], ec='black', linewidth=0.5)
               for cat, c in CATEGORY_COLORS.items()]
    ax_b.legend(handles=handles, loc='lower right', fontsize=8, ncol=2, framealpha=0.9)

    # ===== Panel C: Sparsity Correlation =====
    ax_c = fig.add_subplot(gs[1, 0])

    x_vals = df['Sparsity_pct'].values
    y_vals = df['Delta_SSIM'].values

    for _, row in df.iterrows():
        color = CATEGORY_COLORS.get(row['Category'], '#7f8c8d')
        ax_c.scatter(row['Sparsity_pct'], row['Delta_SSIM'],
                     c=color, s=70, alpha=0.8, edgecolors='black', linewidth=0.5)

    slope, intercept, r_value, p_value, _ = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = slope * x_line + intercept
    ax_c.plot(x_line, y_line, 'k-', linewidth=2.5)

    ax_c.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_c.set_xlabel('Gene Sparsity (% zero bins)', fontsize=13)
    ax_c.set_ylabel('ΔSSIM (Poisson − MSE)', fontsize=13)
    ax_c.set_title(f'C. Sparsity Predicts Benefit (r={r_value:.2f}, p<0.0001)',
                   fontweight='bold', fontsize=14, loc='left')
    ax_c.set_xlim(70, 100)
    ax_c.set_ylim(-0.1, 0.85)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # ===== Panel D: Waterfall =====
    ax_d = fig.add_subplot(gs[1, 1])

    df_sorted = df.sort_values('Delta_SSIM', ascending=False).reset_index(drop=True)
    colors = [CATEGORY_COLORS.get(cat, '#7f8c8d') for cat in df_sorted['Category']]

    ax_d.bar(range(len(df_sorted)), df_sorted['Delta_SSIM'], color=colors,
             edgecolor='black', linewidth=0.3, width=0.85)
    ax_d.axhline(y=0, color='black', linewidth=1)

    # Label top 5 genes
    for i in range(5):
        gene = df_sorted.iloc[i]['Gene']
        val = df_sorted.iloc[i]['Delta_SSIM']
        ax_d.annotate(gene, xy=(i, val), xytext=(i, val + 0.03),
                    ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=45)

    ax_d.set_xlabel('Genes (ranked)', fontsize=13)
    ax_d.set_ylabel('ΔSSIM', fontsize=13)
    ax_d.set_title('D. Per-Gene Improvement Ranking', fontweight='bold', fontsize=14, loc='left')
    ax_d.set_xlim(-1, len(df_sorted))
    ax_d.set_ylim(-0.1, 0.85)
    ax_d.set_xticks([])
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    plt.savefig(OUTPUT_DIR / 'figure_1_combined.png', dpi=300, facecolor='white',
                bbox_inches='tight', pad_inches=0.2)
    plt.savefig(OUTPUT_DIR / 'figure_1_combined.pdf', facecolor='white',
                bbox_inches='tight', pad_inches=0.2)
    plt.close()

    print(f"Created: figure_1_combined.png")


def create_summary_table(summaries, df):
    """Create summary statistics table."""

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print("\nModel Performance (SSIM 2µm, 3-fold CV):")
    print("-" * 50)
    for name, key in [("E' (Hist2ST+Poisson)", 'hist2st_poisson'),
                      ("D' (Hist2ST+MSE)", 'hist2st_mse'),
                      ("F (Img2ST+Poisson)", 'img2st_poisson'),
                      ("G (Img2ST+MSE)", 'img2st_mse')]:
        s = summaries[key]
        print(f"  {name}: {s['mean_ssim_2um']:.3f} ± {s['std_ssim_2um']:.3f}")

    print("\nPer-Gene Analysis:")
    print("-" * 50)
    print(f"  Genes where Poisson wins (ΔSSIM > 0): {(df['Delta_SSIM'] > 0).sum()}/50")
    print(f"  Mean ΔSSIM: {df['Delta_SSIM'].mean():.3f}")
    print(f"  Max ΔSSIM: {df['Delta_SSIM'].max():.3f} ({df.loc[df['Delta_SSIM'].idxmax(), 'Gene']})")
    print(f"  Min ΔSSIM: {df['Delta_SSIM'].min():.3f} ({df.loc[df['Delta_SSIM'].idxmin(), 'Gene']})")

    r, p = stats.pearsonr(df['Sparsity_pct'], df['Delta_SSIM'])
    print(f"\nSparsity-ΔSSIM Correlation:")
    print(f"  Pearson r = {r:.3f}, p = {p:.2e}")

    # Save to file
    with open(OUTPUT_DIR / 'summary_statistics.txt', 'w') as f:
        f.write("MSE vs Poisson 2µm Benchmark - Summary Statistics\n")
        f.write("=" * 60 + "\n\n")

        f.write("Model Performance (SSIM 2µm, 3-fold CV):\n")
        for name, key in [("E' (Hist2ST+Poisson)", 'hist2st_poisson'),
                          ("D' (Hist2ST+MSE)", 'hist2st_mse'),
                          ("F (Img2ST+Poisson)", 'img2st_poisson'),
                          ("G (Img2ST+MSE)", 'img2st_mse')]:
            s = summaries[key]
            f.write(f"  {name}: {s['mean_ssim_2um']:.3f} ± {s['std_ssim_2um']:.3f}\n")

        improvement = (summaries['hist2st_poisson']['mean_ssim_2um'] / summaries['hist2st_mse']['mean_ssim_2um'] - 1) * 100
        f.write(f"\nImprovement (E' vs D'): {improvement:.0f}%\n")

        f.write(f"\nPer-Gene Analysis:\n")
        f.write(f"  Genes where Poisson wins: {(df['Delta_SSIM'] > 0).sum()}/50\n")
        f.write(f"  Mean ΔSSIM: {df['Delta_SSIM'].mean():.3f}\n")

        f.write(f"\nSparsity-ΔSSIM Correlation:\n")
        f.write(f"  Pearson r = {r:.3f}, p = {p:.2e}\n")

        f.write(f"\nRepresentative genes by category:\n")
        for cat in df['Category'].unique():
            cat_df = df[df['Category'] == cat].sort_values('Delta_SSIM', ascending=False)
            best = cat_df.iloc[0]
            f.write(f"  {cat:15} | {best['Gene']:10} | ΔSSIM: +{best['Delta_SSIM']:.3f}\n")

    print(f"\nSaved: summary_statistics.txt")


def main():
    print("Creating manuscript figures...")
    print("=" * 60)

    # Load data
    summaries = load_cv_summaries()
    df = load_pergene_metrics()

    print(f"Loaded {len(summaries)} model summaries")
    print(f"Loaded {len(df)} per-gene metrics")

    # Create individual figures
    figure_1a_factorial_ssim(summaries)
    figure_1b_gene_scatter(df)
    figure_1c_sparsity_correlation(df)
    figure_1d_waterfall(df)
    figure_2_category_analysis(df)
    figure_3_main_effects(summaries)
    figure_4_representative_genes(df)  # NEW
    figure_s1_fold_consistency(summaries)

    # Create combined figure
    figure_combined_1abcd(summaries, df)

    # Summary statistics
    create_summary_table(summaries, df)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
