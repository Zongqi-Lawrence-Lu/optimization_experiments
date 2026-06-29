# threshold_vs_lr_benefit

## Purpose

When gradient clipping is used together with learning-rate warmup, two knobs
can be turned to improve final performance: the clipping threshold τ and the
terminal (post-warmup) learning rate η.  This experiment asks:

> **Which axis gives more benefit — increasing τ or increasing η?**

The answer has practical implications for hyperparameter tuning under warmup:
if the threshold axis dominates, practitioners should sweep τ more aggressively
before worrying about η, and vice-versa.

## Design

A 2-D grid is swept over (τ, η) pairs for each of three settings:

| Setting | Task | Noise / domain |
|---------|------|----------------|
| `gaussian` | Synthetic linear regression | Light-tailed Gaussian noise |
| `heavy_tail` | Synthetic linear regression | Heavy-tailed Student-t(df=2) noise |
| `sst2` | RoBERTa-base fine-tuning on SST-2 | Real NLP classification |

For each cell, `n_seeds` (default 5) independent runs are executed with
identical (τ, η) but different random seeds, and results are aggregated to
mean ± std of best validation loss.

All runs use **linear LR warmup** for the first `warmup_fraction` of total
steps (default 10%).  τ and η are the only varied quantities; all other
hyperparameters are held fixed at the base config values.

## Comparing the axes

After the sweep, `compare.py` produces:

1. **Heatmap** — mean best val loss over the (τ × η) grid.  Darker = better.
2. **Marginal sensitivity plots** — effect of each axis averaged over the
   other.  The y-range on each panel directly shows how much improvement is
   available along that axis.
3. **Comparison summary JSON** — `comparison_summary.json` records
   `tau_axis_improvement`, `lr_axis_improvement`, their ratio, and a plain-
   English interpretation string per setting.

## File layout

```
threshold_vs_lr_benefit/
├── README.md                    (this file)
├── grid_sweep.py                (runs the 2D grid, writes results JSON)
├── compare.py                   (loads results, writes plots + summary)
├── configs/
│   ├── base_gaussian.yaml
│   ├── base_heavy_tail.yaml
│   └── base_sst2.yaml
├── job_sweep_gaussian.sbatch    (SLURM: CPU, 6 h, 16 CPUs)
├── job_sweep_heavy_tail.sbatch  (SLURM: CPU, 6 h, 16 CPUs)
├── job_sweep_sst2.sbatch        (SLURM: A40 GPU, 12 h, 8 CPUs)
└── job_compare.sbatch           (SLURM: CPU, 30 min — runs after all sweeps)
```

Outputs go to:
- `outputs/threshold_vs_lr_benefit/{setting}/` — per-trial logs
- `results/threshold_vs_lr_benefit/{setting}_grid.json` — aggregated cells
- `results/threshold_vs_lr_benefit/comparison_summary.json` — axis comparison
- `plots/threshold_vs_lr_benefit/` — heatmaps and sensitivity plots

## Running

```bash
# Manual (from project root)
python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting gaussian
python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting heavy_tail
python experiments/threshold_vs_lr_benefit/grid_sweep.py --setting sst2
python experiments/threshold_vs_lr_benefit/compare.py

# SLURM (submit as a dependency chain — see submission instructions)
```

## Default grids

| Setting | τ values | η values |
|---------|----------|----------|
| gaussian | 0.1, 0.5, 1, 2, 5, 10, 20 | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 |
| heavy_tail | 0.1, 0.5, 1, 2, 5, 10, 20 | 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 |
| sst2 | 0.1, 0.5, 1, 5, 10, 20 | 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2 |

The grids are calibrated from prior sweep results (τ prior: heavy-tail ~6.9,
gaussian ~1.1; η prior: heavy-tail ~0.04, gaussian ~0.25).
