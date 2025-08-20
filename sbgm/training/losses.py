"""Loss functions for training score models.

Implements denoising score matching loss for continuous time SDEs as
described in Song et al. (2021). The loss draws random times from
``(eps, 1]``, constructs the noisy samples using the closed form
marginal distribution and trains the model to predict the score of the
marginal distribution at those times. A weighting based on the
diffusion coefficient is applied depending on the SDE type.
"""

from __future__ import annotations

import torch
from typing import Callable, Optional

from ..sde.base import SDE


def dsm_loss(
    model: torch.nn.Module,
    sde: SDE,
    x0: torch.Tensor,
    eps: float = 1e-5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute the continuous time denoising score matching loss.

    Parameters
    ----------
    model: torch.nn.Module
        Score network taking ``(x_t, t)`` and returning a tensor of the
        same shape as ``x0``.
    sde: SDE
        Stochastic differential equation defining the forward process.
    x0: torch.Tensor
        Clean data batch. Shape ``(B, C, ...)``.
    eps: float
        Small positive constant to avoid sampling exactly at ``t=0``.
    device: Optional[torch.device]
        Device on which to perform the computation.

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the loss averaged over the batch.
    """
    B = x0.size(0)
    if device is None:
        device = x0.device
    # Sample times uniformly from (eps, 1]
    t = torch.rand(B, device=device) * (1.0 - eps) + eps
    # Sample standard normal noise
    noise = torch.randn_like(x0)
    # Obtain mean and std of x(t)
    mean, std = sde.marginal_prob(x0, t)
    x_t = mean + std * noise
    # Ground truth score of the marginal distribution: grad log p(x_t | x0)
    # For Gaussian p(x_t | x0): -(x_t - mean) / (std^2)
    target = -(x_t - mean) / (std ** 2 + 1e-12)
    # Model prediction
    pred = model(x_t, t)
    # Weighting: use diffusion coefficient squared
    g = sde.diffusion(t).view(-1, *([1] * (x0.dim() - 1)))
    weight = g ** 2
    loss = (weight * (pred - target) ** 2).view(B, -1).mean(dim=1)
    return loss.mean()