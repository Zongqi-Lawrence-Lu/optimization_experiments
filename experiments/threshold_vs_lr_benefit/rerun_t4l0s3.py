"""One-shot rerun of the single missing trial thr_sst2_t4_l0_s3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import replace
from framework.configs import load_config
from framework.run import run

os_chdir = Path(__file__).resolve().parent.parent.parent
import os; os.chdir(os_chdir)

config = load_config("experiments/threshold_vs_lr_benefit/configs/base_sst2.yaml")
steps        = config.total_outer_steps
warmup_steps = max(1, round(steps * 0.1))

clipping  = replace(config.inner_optimizer.clipping, upper=10.0)
inner_opt = replace(config.inner_optimizer, lr=1e-4, clipping=clipping,
                    warmup_steps=warmup_steps)
config    = replace(config, run_name="thr_sst2_t4_l0_s3", seed=1003,
                    inner_optimizer=inner_opt, output_dir=None)

metrics  = run(config)
val_loss = metrics.get("val_loss_best", metrics.get("best_val_loss"))
print(f"Done: thr_sst2_t4_l0_s3  val_loss_best={val_loss:.6f}")
