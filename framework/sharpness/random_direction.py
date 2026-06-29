"""Random-direction perturbation sharpness estimator.

Sharpness is estimated as the average loss increase L(w + ρv) - L(w) over K
independent random unit vectors v, evaluated on a fixed mini-batch subset.
This is the scalar analogue of the flatness measure used in Keskar et al. (2017)
and Li et al. (2018), adapted for efficiency on large models.

All three configurable coefficients (rho, n_directions, n_batches) are exposed
as arguments with defaults; none are hardcoded in the caller.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _move_batch(batch, device: str):
    if isinstance(batch, (list, tuple)):
        return (batch[0].to(device), batch[1].to(device))
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def compute_sharpness(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    rho: float = 0.05,
    n_directions: int = 30,
    n_batches: int = 4,
    seed: Optional[int] = None,
) -> float:
    """Estimate sharpness via random-direction perturbation.

    For each of ``n_directions`` random unit vectors v, perturbs all model
    parameters by ρv, computes average loss over ``n_batches`` mini-batches,
    then immediately restores the original parameters.  Returns the mean Δloss.

    Parameters
    ----------
    model        : the model to measure (must have a ``compute_loss(batch)`` method)
    loader       : data loader to draw evaluation batches from
    device       : torch device string
    rho          : perturbation radius (absolute, not scaled to weight norm)
    n_directions : number of random unit vectors to average over
    n_batches    : number of mini-batches used to estimate each perturbed loss
    seed         : optional seed for the RNG used to generate directions

    Returns
    -------
    float : mean (L(w + ρv) - L(w)) over all sampled directions
    """
    model.eval()

    # Collect a fixed set of batches once, reused across all directions.
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(_move_batch(batch, device))

    if not batches:
        model.train()
        return 0.0

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        model.train()
        return 0.0

    # Compute base loss.
    with torch.no_grad():
        base_loss = sum(model.compute_loss(b).item() for b in batches) / len(batches)

    # Flatten all parameters into a single contiguous GPU vector so that each
    # random direction requires exactly ONE GPU random-number generation and no
    # CPU→GPU transfers.  Shapes and split sizes are recorded for in-place
    # scatter-back after each perturbation.
    sizes   = [p.numel() for p in params]
    flat_orig = torch.cat([p.data.view(-1) for p in params])  # stays on device

    # Use a GPU generator seeded deterministically from the caller's seed.
    # This keeps sharpness measurements reproducible without touching the
    # global RNG state used by training.
    gpu_rng = torch.Generator(device=device)
    if seed is not None:
        gpu_rng.manual_seed(seed)

    deltas: list[float] = []
    with torch.no_grad():
        for _ in range(n_directions):
            # Single GPU random draw + single norm computation — no CPU work.
            v_flat = torch.randn(flat_orig.shape, generator=gpu_rng, device=device)
            scale  = rho / (v_flat.norm() + 1e-12)

            # Write perturbed weights back in-place using split views.
            offset = 0
            for p, sz in zip(params, sizes):
                p.data.add_(v_flat[offset: offset + sz].view_as(p.data) * scale)
                offset += sz

            pert_loss = sum(model.compute_loss(b).item() for b in batches) / len(batches)
            deltas.append(pert_loss - base_loss)

            # Restore from the single flat copy — one GPU memcpy per direction.
            offset = 0
            for p, sz in zip(params, sizes):
                p.data.copy_(flat_orig[offset: offset + sz].view_as(p.data))
                offset += sz

    model.train()
    return float(sum(deltas) / len(deltas)) if deltas else 0.0
