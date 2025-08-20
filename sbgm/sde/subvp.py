"""subVP SDE used in Song et al. (2021).

The sub-variance preserving SDE modifies the diffusion term of the VP
SDE to better match the discrete time forward process used in
discrete diffusion models. The forward SDE is

.. math::

    dx = -\tfrac{1}{2} \beta(t) x \, dt + \sqrt{\beta(t)\frac{1 - \alpha_{\bar{t}}(t)}{\alpha_{\bar{t}}(t)}} \, dW_t,

where ``\alpha_{\bar{t}}(t)`` is the cumulative product of
``1 - beta(s) ds``. The marginal distribution is identical to that of
the VP SDE.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .base import SDE
from .vp import VPSDE


class subVPSDE(SDE):
    """Sub-variance preserving SDE."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, schedule: str = "linear") -> None:
        super().__init__(0.0, 1.0)
        self.vp_sde = VPSDE(beta_min=beta_min, beta_max=beta_max, schedule=schedule)

    def sde_type(self) -> str:
        return "subvp"

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.vp_sde.beta(t)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.vp_sde.alpha_bar(t)

    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        beta_t = self.beta(t)
        alpha_bar_t = self.alpha_bar(t)
        # g(t) = sqrt(beta(t) * (1 - alpha_bar(t)) / alpha_bar(t))
        return torch.sqrt(beta_t * (1.0 - alpha_bar_t) / (alpha_bar_t + 1e-7))

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.vp_sde.drift(x, t)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vp_sde.marginal_prob(x0, t)

    def prior_sampling(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape)