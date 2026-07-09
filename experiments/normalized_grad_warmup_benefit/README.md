# normalized_grad_warmup_benefit

## Purpose

`lr_warmup_comparison` asks whether linear LR warmup helps, for raw and
clipped gradients, and whether the benefit depends on noise tail heaviness.
This experiment asks the same warmup-benefit question under a third
gradient-processing regime:

> **Does linear warmup + constant rate still help once the gradient is
> normalized, so step size is decoupled entirely from gradient magnitude?**

Under normalization, `‖g‖ = 1` at every step by construction (see
implementation note below), so the update norm each step is *exactly* `lr` —
there is no clipping threshold left to tune. The only free axis is the (max)
learning rate itself (plus `warmup_steps` for the warmup condition), which is
what motivated this experiment: instead of a 2-D (τ, η) or (clip, lr, warmup)
grid, the sweep collapses onto lr (+ warmup_steps).

## Implementation note: normalization = biclip(upper = lower = 1.0)

No changes were needed in `framework/`. `GlobalBiclipOperator`
(`framework/optimizers/clipping.py`) already implements gradient
normalization as a degenerate case:

```python
if norm > upper:   scale = upper / norm   # clip down
elif norm < lower:  scale = lower / norm  # amplify up
```

When `upper == lower == 1.0`, exactly one of these branches fires on every
step (the only way to skip both is `norm == 1.0` exactly, measure zero in
practice), so the gradient is *always* rescaled to `‖g‖ = 1.0`. Every config
in this experiment sets
`inner_optimizer.clipping = {clip_type: biclip, clip_scope: global, upper: 1.0, lower: 1.0}`
to get this behavior. The logged `grad_norm_clipped` diagnostic should sit at
≈1.0 throughout training in every run here — see
`plots/normalized_grad_warmup_benefit/heavy_tail_lr_and_grad.png`, which plots
it explicitly as a sanity check.

## Design

Following a decision to keep scope simple (matching
`lr_warmup_comparison`'s *original* 4-condition design, not its
tail-index-sweep extension): two noise settings, warmup on/off, no
tail-index (Student-t df) sweep.

| Condition | Noise | Warmup |
|---|---|---|
| `no_warmup_gaussian` | Gaussian (light-tail) | none |
| `warmup_gaussian` | Gaussian (light-tail) | linear, `warmup_steps` swept |
| `no_warmup_heavy_tail` | Student-t, df=2 (heavy-tail) | none |
| `warmup_heavy_tail` | Student-t, df=2 (heavy-tail) | linear, `warmup_steps` swept |

Model/data: `two_layer_mlp` (20 features → hidden 32 → 1), same synthetic
regression generator as `lr_warmup_comparison`, plain SGD inner optimizer
with the fixed global-normalization clipping config above. 500 outer steps
per sweep trial; final comparison re-runs each condition's sweep winner for
1000 steps across 5 seeds (42–46).

Each condition is swept independently (two-stage grid: coarse then fine
around the coarse winner), single seed (42) for the sweep itself:

- No-warmup conditions: `inner_optimizer.lr` only.
  Stage-1 grid `[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]` (log-scale) —
  wider than `lr_warmup_comparison`'s raw-gradient lr grids, since
  normalization fixes `‖g‖=1` every step (comparable to the *high end* of the
  old clip-threshold range, not the unclipped-gradient lr range), so a good
  step size plausibly sits higher here.
- Warmup conditions: same lr grid × `warmup_steps ∈ [20, 50, 100, 200, 500]`.
- Stage 2: zoom (`n_values`/`zoom_factor`) around the stage-1 winner, same
  pattern as `lr_warmup_comparison`.

Multi-seed final re-run (`compare.py`) reuses the aggregation pattern from
`lr_warmup_comparison/tail_index_compare.py` — mean ± std of best/final val
loss and test loss across 5 seeds — rather than the *original*
`lr_warmup_comparison/compare.py`'s single-seed final run, since seed
averaging is cheap here (only 4 conditions, no df axis) and removes the
single-seed variance risk the project already hit once (see
`lr_warmup_comparison/README.md`'s "Revision history").

## File layout

```
normalized_grad_warmup_benefit/
├── README.md                    (this file)
├── configs/
│   ├── no_warmup_gaussian.yaml
│   ├── warmup_gaussian.yaml
│   ├── no_warmup_heavy_tail.yaml
│   └── warmup_heavy_tail.yaml
├── sweeps/
│   ├── sweep_no_warmup_gaussian.yaml
│   ├── sweep_warmup_gaussian.yaml
│   ├── sweep_no_warmup_heavy.yaml
│   └── sweep_warmup_heavy.yaml
├── compare.py                   (multi-seed final re-run, plots, warmup_comparison.json)
├── job_sweep.sbatch             (SLURM: CPU, 12h request — see sbatch comments for basis)
└── job_final_comparison.sbatch  (SLURM: CPU, 12h request — see sbatch comments for basis)
```

Outputs go to:
- `outputs/normalized_grad_warmup_benefit/sweeps/{condition}/` — sweep artifacts
  (`stage1/`, `stage2/`, `best_config.yaml`)
- `outputs/normalized_grad_warmup_benefit/final/{condition}_final_seed{N}/` — per-seed final re-run logs
- `results/normalized_grad_warmup_benefit/warmup_comparison.json` — per-condition
  mean±std (best/final val loss, test loss, per-seed detail) plus
  `warmup_benefit.heavy_tail_val_loss_benefit` /
  `warmup_benefit.gaussian_val_loss_benefit`
  (`no_warmup_best_val_loss_mean − warmup_best_val_loss_mean`, positive = warmup wins)
- `plots/normalized_grad_warmup_benefit/`:
  - `heavy_tail_warmup_vs_no_warmup.png`, `gaussian_warmup_vs_no_warmup.png` —
    test-loss curves, warmup vs. no-warmup, per noise setting (one
    representative seed per condition)
  - `heavy_tail_lr_and_grad.png` — LR schedule + `grad_norm_clipped` sanity
    check (should read ≈1.0 throughout)
  - `all_conditions_test_loss.png` — all four conditions overlaid
  - `best_val_loss_by_condition.png` — bar chart, mean ± std over seeds, one
    bar per condition — the primary summary figure

## Running

```bash
# Manual (from project root) — sanity-check one condition locally first
python -m framework.sweep.two_stage \
    experiments/normalized_grad_warmup_benefit/sweeps/sweep_no_warmup_gaussian.yaml \
    outputs/normalized_grad_warmup_benefit/sweeps/no_warmup_gaussian

# Full sweep (all 4 conditions) + final multi-seed comparison
python -m framework.sweep.two_stage experiments/normalized_grad_warmup_benefit/sweeps/sweep_no_warmup_heavy.yaml outputs/normalized_grad_warmup_benefit/sweeps/no_warmup_heavy_tail
python -m framework.sweep.two_stage experiments/normalized_grad_warmup_benefit/sweeps/sweep_warmup_heavy.yaml outputs/normalized_grad_warmup_benefit/sweeps/warmup_heavy_tail
python -m framework.sweep.two_stage experiments/normalized_grad_warmup_benefit/sweeps/sweep_no_warmup_gaussian.yaml outputs/normalized_grad_warmup_benefit/sweeps/no_warmup_gaussian
python -m framework.sweep.two_stage experiments/normalized_grad_warmup_benefit/sweeps/sweep_warmup_gaussian.yaml outputs/normalized_grad_warmup_benefit/sweeps/warmup_gaussian
python experiments/normalized_grad_warmup_benefit/compare.py

# SLURM (recommended)
sbatch experiments/normalized_grad_warmup_benefit/job_sweep.sbatch
sbatch --dependency=afterok:<sweep_job_id> experiments/normalized_grad_warmup_benefit/job_final_comparison.sbatch
```

## Interpretation guide

- If `warmup_heavy_tail` beats `no_warmup_heavy_tail` by about as much as
  `warmup_clip` beats `clip_only` in `lr_warmup_comparison`, warmup's benefit
  under heavy-tailed noise is not primarily about taming large early raw
  gradient norms (normalization already removes that mechanism entirely) —
  something else about the schedule shape matters.
- If the benefit shrinks or vanishes here relative to the clipped/raw
  conditions, that's evidence the original warmup benefit *was* largely
  about controlling gradient magnitude early in training, which
  normalization already achieves without any schedule at all.
- Compare `gaussian_val_loss_benefit` vs. `heavy_tail_val_loss_benefit` the
  same way `lr_warmup_comparison` uses Gaussian as a light-tail control.
