"""Time discretisation and noise schedules.

This module contains helper functions for creating discretised time
schedules for sampling and training. It includes Karras style sigma
schedules for variance exploding SDEs as well as beta schedules for
variance preserving (VP) diffusion models with both linear and
cosine forms.
"""

from __future__ import annotations

import torch
import math
from typing import Tuple


def karras_sigmas(sigma_min: float, sigma_max: float, num_steps: int) -> torch.Tensor:
    """Generate a Karras style noise schedule for VE SDE.

    The schedule is geometric in the sigma domain: equally spaced in
    log space between ``sigma_min`` and ``sigma_max``. This schedule
    often yields improved sample quality for VE models (Karras et al.,
    2022).

    Parameters
    ----------
    sigma_min: float
        Minimum sigma (smallest noise level).
    sigma_max: float
        Maximum sigma (largest noise level).
    num_steps: int
        Number of timesteps in the schedule.

    Returns
    -------
    torch.Tensor
        1D tensor of length ``num_steps`` containing the noise levels.
    """
    sigmas = torch.logspace(math.log10(sigma_min), math.log10(sigma_max), steps=num_steps)
    return sigmas.flip(0)  # descending order (from high to low noise)


def linear_beta(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Linear beta schedule for VP SDE.

    Beta grows linearly from ``beta_min`` at ``t=0`` to ``beta_max`` at
    ``t=1``. The integral of beta over t can be used to compute
    ``alpha_bar`` analytically.
    """
    return beta_min + (beta_max - beta_min) * t


def linear_alpha_bar(t: torch.Tensor, beta_min: float, beta_max: float) -> torch.Tensor:
    """Compute ``alpha_bar(t)`` for the linear beta schedule.

    Given ``beta(t) = beta_min + (beta_max - beta_min) * t`` the
    integral ``∫_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2``. The
    resulting ``alpha_bar(t)`` is ``exp(-∫ beta ds)``.
    """
    integral = beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2
    return torch.exp(-integral)


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Nichol–Dhariwal cosine schedule for ``alpha_bar``.

    Implements the noise schedule proposed in "Improved Denoising
    Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021). The
    ``alpha_bar`` function is defined as ``cos^2((t + s)/(1 + s) * pi/2)``.

    Parameters
    ----------
    t: torch.Tensor
        Normalised time in [0,1].
    s: float
        Small offset to avoid singularities near t=0.

    Returns
    -------
    torch.Tensor
        Values of ``alpha_bar(t)``.
    """
    f = (t + s) / (1 + s)
    return torch.cos((math.pi / 2) * f) ** 2


def cosine_beta(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Compute ``beta(t)`` from the cosine ``alpha_bar`` schedule.

    Derived from the formula ``beta(t) = d/dt [ -log(alpha_bar(t)) ]``. See
    the derivation in the module documentation. For the cosine schedule
    this reduces to ``beta(t) = (pi / (1 + s)) * tan(u)`` where
    ``u = ((t + s) / (1 + s)) * pi / 2``.

    Parameters
    ----------
    t: torch.Tensor
        Normalised time in [0,1].
    s: float
        Offset as in :func:`cosine_alpha_bar`.

    Returns
    -------
    torch.Tensor
        Values of ``beta(t)``.
    """
    f = (t + s) / (1 + s)
    u = (math.pi / 2) * f
    beta = (math.pi / (1 + s)) * torch.tan(u)
    return beta