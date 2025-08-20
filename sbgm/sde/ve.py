"""Variance exploding SDE (VE‑SDE).

Implements the SDE

.. math::

    dx = g(t) \, dW_t,

where ``g(t) = sigma(t) \sqrt{2\log(sigma_{\max}/sigma_{\min})}`` and
``sigma(t) = sigma_{\min} (sigma_{\max}/sigma_{\min})^t``. The forward
process corresponds to adding Gaussian noise with increasing variance
over time. The closed‑form marginal distribution is simply
``x(t) = x(0) + sigma(t) z`` for ``z ~ N(0, I)``.
"""

from __future__ import annotations

import math
import torch

from .base import SDE


class VESDE(SDE):
    """Variance exploding SDE as in Song et al. (2021)."""

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, t0: float = 0.0, t1: float = 1.0) -> None:
        super().__init__(t0, t1)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # Precompute constant for diffusion coefficient
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sde_type(self) -> str:
        return "ve"

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise scale ``sigma(t)`` as a function of time."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient ``g(t)``.

        Derived from ``Var[x(t)-x(0)] = sigma(t)^2`` leading to
        ``g(t) = sigma(t) * sqrt(2 * log(sigma_max / sigma_min))``.
        """
        sigma_t = self.sigma(t)
        return sigma_t * math.sqrt(2.0 * self.log_ratio)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # VE SDE has zero drift
        return torch.zeros_like(x)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x0
        std = self.sigma(t).view(-1, *([1] * (x0.dim() - 1)))
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape)