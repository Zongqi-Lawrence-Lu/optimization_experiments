# warmup_flatness

## Purpose

Does learning-rate warmup change the loss-landscape flatness at the end of
training, and does that flatness difference explain warmup's effect on
generalization?

This experiment trains with and without warmup across a grid of learning rates,
periodically measuring **random-direction sharpness** throughout training:

    sharpness(w) = mean_v [ L(w + ρv) − L(w) ]

where v is a random unit vector and ρ is the perturbation radius.  This is the
scalar analogue of the flatness measure from Keskar et al. (2017) and
Li et al. (2018).

## Design

For each of three settings, runs are paired (warmup vs. no-warmup) across a
grid of terminal learning rates, with `n_seeds` seeds per pair:

| Setting | Task | Noise / domain |
|---------|------|----------------|
| `gaussian` | Synthetic linear regression | Light-tailed Gaussian noise |
| `heavy_tail` | Synthetic linear regression | Heavy-tailed Student-t(df=2) noise |
| `sst2` | RoBERTa-base fine-tuning on SST-2 | Real NLP classification |

The clipping threshold τ is **fixed** at 1.0 across all runs (controlled
experiment — only warmup varies).  The warmup condition uses a linear schedule
over the first `warmup_fraction` (default 10%) of total steps.

Sharpness is measured every `sharpness_every` steps (default: aligned with
`eval_every`) by perturbing parameters in 30 random unit directions and
averaging the resulting loss increase.  The measurement is also forced at the
final training step.

## What compare.py produces

Per setting, `compare.py` generates five plots and a numeric summary:

1. **Sharpness-over-time panel grid** — one column per LR, warmup vs. no-warmup
   overlaid, mean ± 1σ over seeds.
2. **Val-loss-over-time panel grid** — same layout as above.
3. **Best-per-condition sharpness** — each condition (warmup / no-warmup) uses
   its independently best LR; shows the training trajectory of sharpness.
4. **Best-per-condition val loss** — same but for validation loss.
5. **Final sharpness scatter** — final sharpness vs. final val loss, coloured
   by condition.  Reveals whether sharpness and generalization co-vary.

A numeric summary (`flatness_summary.json`) records mean ± std of final
sharpness for both conditions and whether warmup leads to a sharper or flatter
optimum.

## File layout

```
warmup_flatness/
├── README.md                    (this file)
├── run_and_measure.py           (trains + logs sharpness, writes run index)
├── compare.py                   (loads run index, writes plots + summary)
├── configs/
│   ├── base_gaussian.yaml
│   ├── base_heavy_tail.yaml
│   └── base_sst2.yaml
├── job_run_gaussian.sbatch      (SLURM: CPU, 4 h, 16 CPUs)
├── job_run_heavy_tail.sbatch    (SLURM: CPU, 4 h, 16 CPUs)
├── job_run_sst2.sbatch          (SLURM: A40 GPU, 12 h, 8 CPUs)
└── job_compare.sbatch           (SLURM: CPU, 30 min — runs after all setting jobs)
```

Outputs go to:
- `outputs/warmup_flatness/{setting}/{run_name}/eval_log.jsonl` — per-step val
  loss and sharpness (JSONL, `outer_step` field present at every row)
- `results/warmup_flatness/{setting}_runs.json` — run index (one entry per run)
- `results/warmup_flatness/flatness_summary.json` — final sharpness comparison
- `plots/warmup_flatness/` — all five plot types per setting

## Running

```bash
# Manual (from project root)
python experiments/warmup_flatness/run_and_measure.py --setting gaussian
python experiments/warmup_flatness/run_and_measure.py --setting heavy_tail
python experiments/warmup_flatness/run_and_measure.py --setting sst2
python experiments/warmup_flatness/compare.py

# SLURM (submit as a dependency chain — see submission instructions)
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--clip_upper` | 1.0 | Fixed τ — do not vary; controls for clipping |
| `--warmup_fraction` | 0.1 | Fraction of total steps used for warmup |
| `--rho` | 0.05 | Perturbation radius for sharpness measurement |
| `--n_directions` | 30 | Random unit vectors averaged per estimate |
| `--n_sharpness_batches` | 4 | Mini-batches used per sharpness estimate |
| `--sharpness_every` | 0 | 0 = align with eval_every |
| `--n_seeds` | 5 | Seeds per (lr, warmup) combination |

## Interpretation guide

- If warmup runs end up in **flatter** minima (lower sharpness) and also have
  better val loss, flatness is a plausible mechanism for warmup's benefit.
- If sharpness differs but val loss does not (or vice versa), the two effects
  decouple and flatness is not the primary explanation.
- The scatter plot (final sharpness vs. final val loss) is the sharpest
  diagnostic for this relationship across all seeds and LRs.
