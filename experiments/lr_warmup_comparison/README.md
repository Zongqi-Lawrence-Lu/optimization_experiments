# lr_warmup_comparison

## Purpose

Does linear LR warmup help synthetic convex regression, and does the benefit
depend on how heavy-tailed the label noise is?

> **Does warmup help more as the gradient noise gets heavier-tailed?**

## Original design (Gaussian vs. Student-t(df=2))

Four conditions, each swept independently (two-stage grid: coarse then fine
around the coarse winner):

| Condition | Noise | Warmup |
|---|---|---|
| `no_warmup_gaussian` | Gaussian (light-tail) | none |
| `warmup_gaussian` | Gaussian (light-tail) | linear, `warmup_steps` swept |
| `no_warmup_heavy_tail` | Student-t, df=2 (heavy-tail) | none |
| `warmup_heavy_tail` | Student-t, df=2 (heavy-tail) | linear, `warmup_steps` swept |

Model: `two_layer_mlp` (20 features → hidden 32 → 1), plain SGD inner
optimizer with fixed-threshold global upper-clipping (`upper` also swept).
500 outer steps per sweep trial; final comparison re-runs each condition's
sweep winner for 1000 steps.

Result (`results/lr_warmup_comparison/warmup_comparison.json`): warmup gave a
real benefit under heavy-tail noise (best val loss 5.60 vs. 5.80) but
essentially none under Gaussian noise (1.061 vs. 1.062) — consistent with the
config comments treating Gaussian as the control condition.

### File layout (original)

```
configs/{no_warmup,warmup}_{gaussian,heavy_tail}.yaml   — 4 standalone configs
sweeps/sweep_{no_warmup,warmup}_{gaussian,heavy}.yaml   — 4 two-stage sweep specs
compare.py            — re-runs each sweep winner, plots, writes warmup_comparison.json
job_sweep.sbatch       — runs all 4 sweeps
job_final_comparison.sbatch — runs compare.py
```

## Extension: sweeping the tail index

Fixing df=2 only tells us heavy-tail-vs-not; it doesn't show whether the
warmup benefit keeps growing, saturates, or reverses as the tail gets even
heavier. This extension sweeps the Student-t degrees of freedom itself.

### Revision history

**v1** swept `noise_df ∈ {2.0, 1.75, 1.5, 1.25, 1.1}` across two conditions
(`no_warmup`, `warmup`), single fixed seed (42) throughout. Result: warmup
benefit was small and flat from df=2.0→1.5 (+0.04 to +0.18 val loss), then
swung sharply negative at df=1.25/1.1 (−7.3, −15.1) — i.e. warmup appeared to
actively *hurt* under very heavy tails. That reversal was judged unreliable
before drawing conclusions from it: single-seed val loss under Student-t
noise with df≤1.25 is itself heavy-tailed (the loss estimate's own variance
can be dominated by rare draws), and df=1.1–1.25 is more extreme than the
tail indices typically measured in real neural-net gradient noise
(commonly-cited estimates cluster around df≈1.5–3). **v2** (current) responds
to both problems:
- Dropped df=1.1 and 1.25; kept `noise_df ∈ {2.0, 1.75, 1.5}` — narrower, but
  every point stays in a regime where a single seed's variance is plausible
  to trust for HP selection.
- Added a third condition, `warmup_only`, to separate "does warmup help"
  from "does warmup help *beyond what clipping already provides*":

| Condition | Clipping | Warmup |
|---|---|---|
| `clip_only` (was `no_warmup`) | fixed-threshold global upper-clip, swept | none |
| `warmup_clip` (was `warmup`) | swept | linear, `warmup_steps` swept |
| `warmup_only` (**new**) | `clip_type: none` (disabled) | linear, `warmup_steps` swept |

- The two-stage HP sweep itself is still single-seed (42) — re-sweeping
  every trial across seeds would multiply the trial count by the seed count.
  Instead, **each winning config is re-run across 5 seeds** (42–46) in
  `tail_index_compare.py`, and the trend now reports mean±std across seeds.

For **each** df value and **each** condition, an independent two-stage sweep
is run over that condition's own axes — `clip_only`/`warmup_clip` reuse the
original heavy-tail sweep axes (`lr`, `clipping.upper`[, `warmup_steps`]);
`warmup_only` sweeps just `lr`, `warmup_steps` (no clip axis, since clipping
is disabled) — i.e. every point on the trend is tuned exactly as hard as the
others.

### Files

```
tail_index_sweep.py     — runs the two-stage sweep for each (condition, df) pair
tail_index_compare.py   — re-runs each winner across 5 seeds x 1000 steps,
                           computes the warmup-benefit trend (mean±std), writes plots
job_sweep_tail_index.sbatch    — runs all (condition, df) sweeps in one job
job_tail_index_compare.sbatch  — aggregate + plot, run after the sweep completes
```

Outputs go to:
- `outputs/lr_warmup_comparison/tail_index/{condition}/df_{tag}/` — per-df sweep artifacts
  (`{condition}` ∈ `clip_only`, `warmup_clip`, `warmup_only`; v1's `no_warmup`/`warmup`
  directories from the 5-df run are left in place, superseded but not deleted)
- `outputs/lr_warmup_comparison/tail_index/final/` — final 1000-step re-runs, one per seed
- `results/lr_warmup_comparison/tail_index_sweep_index*.json` — per-df sweep
  winners (task-sharded when run via a SLURM array; see below)
- `results/lr_warmup_comparison/tail_index_trend.json` — per-df metrics
  (mean/std across seeds + raw per-seed values) and two benefit columns:
  `warmup_clip_val_loss_benefit`, `warmup_only_val_loss_benefit`
  (`clip_only_best_val_loss_mean − variant_best_val_loss_mean`, positive = warmup variant wins)
- `plots/lr_warmup_comparison/tail_index_val_loss_trend.png` — all three
  conditions' best val loss (mean±std error bars) vs. tail index
- `plots/lr_warmup_comparison/tail_index_benefit_trend.png` — both benefit
  columns vs. tail index

### Compute budget

Each (condition, df) pair is a full two-stage sweep: `clip_only` is
6×4 + 5×5 ≈ 49 trials, `warmup_clip` is 5×4×5 + 4×4×4 ≈ 164 trials,
`warmup_only` is 5×5 + 4×4 ≈ 41 trials — ~254 trials/df, ~762 trials total
across the 3 df values and 3 conditions. `job_sweep_tail_index.sbatch` runs
all of these serially in a single job (one process, `tail_index_sweep.py
--parallel_jobs 8`) — no SLURM array, so there's no concurrent-write race on
the index file.

Timing is calibrated from `sacct` on job 954563, the v1 sweep (5 df values,
2 conditions, `parallel_jobs=8`): it sustained **~2.8 trials/min on a
contended node** (`q001`), consistently across both the cheap (~49-trial)
and expensive (~180-trial observed, vs. ~164 planned) condition/df pairs. At
that conservative rate, 762 trials ≈ **4.8h**. The same v1 workload
completed a comparable trial count in ~2 minutes on an uncontended node
(`k001`/`k002` in that run) — a ~35x spread — so 2.8 trials/min is
deliberately the pessimistic planning figure, not an average. Per project
policy, `job_sweep_tail_index.sbatch` and `job_tail_index_compare.sbatch`
both request the 12h cluster max regardless of the conservative estimate
(sweep ~4.8h, final 45-seed-run comparison conservatively ~4.3h) falling
under it.

### Running

```bash
# See the planned trial counts before committing to a full run
python experiments/lr_warmup_comparison/tail_index_sweep.py --dry_run

# Manual (single machine, one df/condition at a time)
python experiments/lr_warmup_comparison/tail_index_sweep.py --tail_indices 2.0 --conditions warmup_only
python experiments/lr_warmup_comparison/tail_index_compare.py --seeds 42 43 44 45 46

# SLURM (recommended)
sbatch experiments/lr_warmup_comparison/job_sweep_tail_index.sbatch
sbatch --dependency=afterok:<sweep_job_id> experiments/lr_warmup_comparison/job_tail_index_compare.sbatch
```

### Note

While implementing the original (v1) extension, `run_two_stage_sweep` in
`framework/sweep/two_stage.py` was found to crash right after every stage-2
completion (invalid f-string format spec on the "Stage 2 best" print), which
would have silently broken the top-level `best_config.yaml` for every new
sweep run through it. Fixed as part of that change; the function was also
refactored into `run_two_stage_sweep_for_config(base_config, cfg, sweep_dir)`
so callers (like `tail_index_sweep.py`) can pass a programmatically-built
config instead of writing one YAML file per df value.
