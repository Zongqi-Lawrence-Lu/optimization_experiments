# Optimization Experiments

Empirical comparison of optimization methods in centralized and distributed settings, with a focus on heavy-tailed noise.

---

## Directory Structure

```
optimization_experiments/
├── data/                         # Input data
│   ├── raw/                      # Downloaded or original datasets (HuggingFace cache, WMT files, etc.)
│   └── processed/                # Pre-tokenized / partitioned datasets written to disk
│
├── outputs/                      # Training outputs (model checkpoints, logs)
│   └── <run_name>/
│       ├── config.yaml           # Exact config used for the run
│       ├── step_log.jsonl        # Per-step training diagnostics (loss, grad norms, …)
│       ├── eval_log.jsonl        # Per-eval-step validation/test metrics
│       ├── gradient_stats.jsonl  # Per-layer gradient statistics
│       ├── final_metrics.json    # Summary of best / final metrics
│       └── checkpoints/
│           ├── step_<N>.pt       # Checkpoint at outer step N
│           └── best.pt           # Best checkpoint by validation loss
│
├── results/                      # Organized JSON result summaries
│   └── <experiment_name>/
│       └── summary.json          # Aggregated metrics across methods / seeds
│
├── plots/                        # Generated figures
│   ├── training_curves/          # Loss, accuracy, param-distance vs. step
│   └── sweep_heatmaps/           # Hyperparameter sweep heatmaps
│
├── configs/                      # Reusable YAML experiment configs
│   ├── centralized_sgd.yaml
│   ├── distributed_adam.yaml
│   ├── roberta_glue.yaml
│   └── t5_wmt.yaml
│
├── framework/                    # Core library
│   ├── configs/
│   │   └── config.py             # All config dataclasses (ClippingConfig, TrainingConfig, …)
│   ├── data/
│   │   ├── synthetic.py          # Synthetic linear regression datasets
│   │   ├── loaders.py            # Generic HuggingFace / torchvision loaders
│   │   ├── glue_loader.py        # GLUE benchmark loader for RoBERTa fine-tuning
│   │   └── wmt_loader.py         # WMT translation loader for T5 fine-tuning
│   ├── models/
│   │   ├── convex.py             # LinearRegressionModel
│   │   └── wrappers.py           # ModelWrapper + build_model factory
│   ├── optimizers/
│   │   ├── clipping.py           # Clipping operators (L2, coordinate, layerwise, biclip_global, dynamic)
│   │   ├── inner.py              # SGD, Adam, Adagrad, AdagradNorm, RMSProp, AdamW (inner loop)
│   │   ├── outer.py              # Average, SGD, Adagrad, RMSProp, Adam, AdamW (outer/server)
│   │   └── registry.py           # Optimizer registry for custom extensions
│   ├── training/
│   │   ├── centralized.py        # Single-node training loop with checkpointing & resume
│   │   └── distributed.py        # Federated/distributed training loop with checkpointing & resume
│   ├── sweep/
│   │   ├── grid.py               # Cartesian-product grid search
│   │   ├── random.py             # Random hyperparameter search
│   │   ├── runner.py             # Sweep dispatcher (parallel trials)
│   │   ├── two_stage.py          # Two-stage coarse-then-fine sweep
│   │   └── analyze.py            # Sweep result analysis and pair plots
│   ├── plotting/
│   │   ├── training_curves.py    # Plot loss / accuracy / param-distance curves
│   │   └── sweep_heatmap.py      # 2-D heatmap and 1-D sensitivity plots for sweeps
│   └── tracking/
│       ├── logger.py             # RunLogger (JSONL step/eval/grad logs + checkpoints)
│       └── metrics.py            # Metric registry (MSE, accuracy, F1, BLEU, Spearman, …)
│
├── tests/                        # Unit and integration tests (pytest)
├── requirements.txt
└── setup.py
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

For the **non-convex / generative modes (c) RoBERTa-GLUE and (d) T5-WMT**, also install the
optional NLP extras (HuggingFace `transformers`/`datasets`, plus `sacrebleu`/`nltk` for BLEU/METEOR):

```bash
pip install -e ".[nlp]"
```

Without these, modes (a) and (b) still run fully; the NLP modes (and the `bleu`/`meteor` metrics)
are skipped. BLEU is only reported for seq2seq runs when `sacrebleu` is installed.

### 2. Run a centralized experiment

```bash
python -m framework.run configs/centralized_sgd.yaml
```

### 3. Run a distributed experiment

```bash
python -m framework.run configs/distributed_adam.yaml
```

### 4. Run a hyperparameter sweep

```bash
python -m framework.sweep.runner configs/sweep_example.yaml outputs/sweeps/my_sweep
```

### 5. Run a two-stage sweep (coarse → fine)

```bash
python -m framework.sweep.two_stage configs/two_stage_sweep.yaml outputs/sweeps/two_stage
```

### 6. Analyze sweep results and plot heatmaps

```bash
python -m framework.sweep.analyze outputs/sweeps/my_sweep/results_summary.csv
python -m framework.plotting.sweep_heatmap \
    --csv outputs/sweeps/my_sweep/results_summary.csv \
    --x inner_optimizer.lr --y inner_optimizer.clipping.upper \
    --metric best_val_loss --output plots/sweep_heatmaps/lr_vs_clip.png
```

### 7. Plot training curves

```bash
python -m framework.plotting.training_curves \
    --log_dir outputs/my_run \
    --output plots/training_curves/my_run.png
```

### 8. Resume a run from the latest checkpoint

Set `resume: true` in the config YAML, or pass `--resume` on the command line (future CLI).
The training loop automatically detects the latest checkpoint in `output_dir/<run_name>/checkpoints/`
and continues from there.

---

## Experiment Modes

| Setting | Dataset | Model | Config key |
|---------|---------|-------|------------|
| Convex / Gaussian | Synthetic (Gaussian noise) | Linear regression | `data.noise_distribution: gaussian` |
| Convex / Heavy-tail | Synthetic (Student-t noise) | Linear regression | `data.noise_distribution: student_t` |
| Convex / Token | Synthetic (Bernoulli-mixed features) | Linear regression | `data.feature_distribution: bernoulli_mixed` |
| Non-convex NLU | GLUE benchmark | RoBERTa-base | `configs/roberta_glue.yaml` |
| Generative NLG | WMT En→De/Fr | T5-small | `configs/t5_wmt.yaml` |

For GLUE, the classifier head size (`model.num_classes`) is **auto-derived from `data.glue_task`**
(e.g. MNLI→3, STS-B→1, others→2), so switching tasks only requires changing `glue_task`.
For T5/seq2seq runs, listing `bleu` (or `meteor`) under `metrics` triggers a generation-based
evaluation that decodes `model.generate(...)` against the references — loss is always reported too.

---

## Clipping Modes

| `clip_type` | `clip_scope` | Behaviour |
|-------------|-------------|-----------|
| `none` | — | No clipping |
| `upper` | `global` | Rescale gradient so global L2 norm ≤ upper |
| `upper` | `layerwise` | Rescale each parameter tensor so its L2 norm ≤ upper |
| `upper` | `coordinate` | Clip each element to [−upper, +upper] |
| `biclip` | `global` | Scale global norm into [lower, upper] |
| `biclip` | `layerwise` | Scale each layer norm into [lower, upper] |
| `biclip` | `coordinate` | Amplify elements below lower, clip elements above upper |

Set `dynamic: true` to compute thresholds automatically from the current gradient statistics
(percentile-based for coordinate scope; EMA-norm for global/layerwise scope).

---

## Optimizer Options

**Inner optimizer** (`inner_optimizer.name`):
`sgd` | `adam` | `adagrad` | `adagrad_norm` | `rmsprop` | `adamw`

**Outer optimizer** (`outer_optimizer.name`):
`average` | `sgd` | `adagrad` | `adagrad_norm` | `rmsprop` | `adam` | `adamw` | `clipped`

Both inner and outer optimizers support independent clipping configurations.

---

## Learning Rate Warmup

Set `warmup_steps > 0` in an optimizer config to enable warmup:

```yaml
inner_optimizer:
  name: adamw
  lr: 2.0e-05
  warmup_steps: 500        # ramp lr from 0 to warmup_max_lr over 500 steps
  warmup_max_lr: 2.0e-05  # peak learning rate (defaults to lr if unset)
  warmup_schedule: linear  # "linear" (default) or "cosine"
```

Warmup state persists across distributed rounds (it is not reset with the moment buffers).

---

## Checkpointing & Resume

- Checkpoints are saved to `output_dir/<run_name>/checkpoints/step_<N>.pt` every `checkpoint_every` steps.
- Time-based checkpointing: set `checkpoint_interval_minutes` (e.g. `10`) to also checkpoint every ~10 minutes of wall time.
- To resume: set `resume: true` in the config. The trainer finds the latest checkpoint automatically.

---

## Output Directories

| Config key    | Default    | Contents |
|---------------|------------|----------|
| `output_dir`  | `outputs/` | Checkpoints, step/eval/grad logs, config snapshot |
| `results_dir` | `results/` | `summary.json` with best/final metrics per run |
| `plots_dir`   | `plots/`   | Auto-generated training curves and gradient norm plots |

Training curves and gradient norm plots are generated automatically at the end of every run and saved to `plots/training_curves/<run_name>.png`.
