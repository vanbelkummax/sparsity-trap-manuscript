# Tables Directory

CSV tables with per-gene metrics and summary statistics.

## Expected Files

- **table_s1_pergene_metrics.csv** - Per-gene SSIM, PCC, and delta values
- **table_s2_category_summary.csv** - Summary statistics by gene category
- **table_s3_sparsity_quartiles.csv** - Sparsity quartile analysis

## Schema

### table_s1_pergene_metrics.csv
| Column | Description |
|--------|-------------|
| gene | Gene name |
| category | Functional category |
| sparsity | Fraction of zeros (0-1) |
| mse_ssim | SSIM with MSE loss |
| poisson_ssim | SSIM with Poisson loss |
| delta_ssim | Improvement (poisson - mse) |
| mse_pcc | PCC with MSE loss |
| poisson_pcc | PCC with Poisson loss |

### table_s2_category_summary.csv
Summary by category: Epithelial, Immune, Stromal, Secretory, etc.

### table_s3_sparsity_quartiles.csv
Performance binned by sparsity level (Q1-Q4)

## Generation

Tables are generated during evaluation:
```bash
python scripts/evaluate_model.py --fold all
```
