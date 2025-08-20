"""Variance preserving SDE (VP‑SDE).

Implements the variance preserving SDE used in diffusion models
(Song et al., 2021). The forward SDE is

.. math::

    dx = -\tfrac{1}{2} \beta(t) x \, dt + \sqrt{\beta(t)} \, dW_t,

where ``beta(t)`` is a positive function controlling the noise rate.
Two schedules are supported: a linear schedule ``beta(t) = beta_min +
(beta_max - beta_min) t`` and the cosine schedule from Nichol &
Dhariwal (2021). The closed‑form marginal distribution is
``x(t) = \alpha(t) x(0) + \sigma(t) z`` where ``\alpha(t) =
\sqrt{\alpha_bar(t)}`` and ``\sigma(t) = \sqrt{1 - \alpha_bar(t)}``.
"""

from __future__ import annotations

import torch
import math
from typing import Tuple

from .base import SDE
from ..utils import schedulers


class VPSDE(SDE):
    """Variance preserving SDE."""

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        schedule: str = "linear",
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> None:
        super().__init__(t0, t1)
        self.beta_min = beta_min
        self.beta_max = beta_max
        if schedule not in {"linear", "cosine"}:
            raise ValueError(f"Unknown beta schedule: {schedule}")
        self.schedule = schedule

    def sde_type(self) -> str:
        return "vp"

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            return schedulers.linear_beta(t, self.beta_min, self.beta_max)
        else:
            return schedulers.cosine_beta(t)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule == "linear":
            return schedulers.linear_alpha_bar(t, self.beta_min, self.beta_max)
        else:
            return schedulers.cosine_alpha_bar(t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.beta(t))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta(t).view(-1, *([1] * (x.dim() - 1)))
        return -0.5 * beta_t * x

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_bar_t = self.alpha_bar(t).view(-1, *([1] * (x0.dim() - 1)))
        alpha = torch.sqrt(alpha_bar_t)
        sigma = torch.sqrt(1.0 - alpha_bar_t)
        mean = alpha * x0
        std = sigma
        return mean, std

    def prior_sampling(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape)