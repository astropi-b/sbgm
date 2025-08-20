"""Predictor–Corrector sampler with Langevin corrector.

This module implements the predictor–corrector sampling scheme as
proposed in Song et al. (2021). It alternates between a predictor
step (Euler–Maruyama) and a corrector step (Langevin dynamics) to
generate higher quality samples. The number of corrector steps and the
target signal to noise ratio (SNR) can be configured.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

from ..sde.base import SDE


@torch.no_grad()
def pc_sampler(
    model: torch.nn.Module,
    sde: SDE,
    shape: Tuple[int, ...],
    num_steps: int = 1000,
    snr: float = 0.16,
    n_corrector: int = 1,
    device: Optional[torch.device] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Draw samples using predictor–corrector with Langevin corrector.

    Parameters
    ----------
    model: torch.nn.Module
        Score network.
    sde: SDE
        Forward SDE.
    shape: Tuple[int, ...]
        Output shape.
    num_steps: int
        Number of predictor steps.
    snr: float
        Target signal to noise ratio for the Langevin corrector.
    n_corrector: int
        Number of corrector steps per predictor step.
    device: Optional[torch.device]
        Device for sampling.
    eps: float
        Minimum time.

    Returns
    -------
    torch.Tensor
        Generated samples.
    """
    if device is None:
        device = next(model.parameters()).device
    batch_size = shape[0]
    x = sde.prior_sampling(shape).to(device)
    t = torch.linspace(1.0, eps, num_steps + 1, device=device)
    for i in range(num_steps):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i
        g = sde.diffusion(t_i)
        g2 = g * g
        # Predictor step
        f = sde.drift(x, t_i.expand(batch_size))
        score = model(x, t_i.expand(batch_size))
        x_mean = x + (f - g2.view(-1, *([1] * (x.dim() - 1))) * score) * dt
        noise = torch.randn_like(x)
        x = x_mean + g.view(-1, *([1] * (x.dim() - 1))) * torch.sqrt(-dt) * noise
        # Corrector step: Langevin dynamics
        for _ in range(n_corrector):
            score = model(x, t_i.expand(batch_size))
            # Compute step size based on SNR
            # Compute norms
            grad_norm = torch.linalg.vector_norm(score.view(batch_size, -1), dim=1).mean()
            noise = torch.randn_like(x)
            noise_norm = torch.linalg.vector_norm(noise.view(batch_size, -1), dim=1).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2
            # Update
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    return x