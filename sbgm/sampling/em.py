"""Euler–Maruyama sampler for reverse time SDE.

This module implements the Euler–Maruyama discretisation of the
reverse time SDE used for sampling from score based models. It can
generate samples for both VE and VP style SDEs and works with
multi‑dimensional data by broadcasting appropriately.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

from ..sde.base import SDE


@torch.no_grad()
def euler_maruyama_sampler(
    model: torch.nn.Module,
    sde: SDE,
    shape: Tuple[int, ...],
    num_steps: int = 1000,
    device: Optional[torch.device] = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Draw samples via the Euler–Maruyama solver.

    Parameters
    ----------
    model: torch.nn.Module
        Trained score network returning the score given ``(x, t)``.
    sde: SDE
        Stochastic differential equation corresponding to the forward
        process.
    shape: Tuple[int, ...]
        Desired shape of the sample batch. Must include batch and
        channels and any spatial/temporal dimensions.
    num_steps: int
        Number of discretisation steps between ``t=1`` and ``t=eps``.
    device: Optional[torch.device]
        Device on which to perform sampling. If ``None`` uses model's
        device.
    eps: float
        Small value to avoid singularity at ``t=0``.

    Returns
    -------
    torch.Tensor
        Samples from the learned distribution.
    """
    if device is None:
        device = next(model.parameters()).device
    batch_size = shape[0]
    # Initialise from the prior distribution at t=1
    x = sde.prior_sampling(shape).to(device)
    # Create time discretisation from 1 to eps
    t = torch.linspace(1.0, eps, num_steps + 1, device=device)
    for i in range(num_steps):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i  # negative value
        g = sde.diffusion(t_i)
        g2 = g * g
        f = sde.drift(x, t_i.expand(batch_size))
        score = model(x, t_i.expand(batch_size))
        # Euler–Maruyama update: reverse SDE
        x_mean = x + (f - g2.view(-1, *([1] * (x.dim() - 1))) * score) * dt
        noise = torch.randn_like(x)
        x = x_mean + g.view(-1, *([1] * (x.dim() - 1))) * torch.sqrt(-dt) * noise
    return x