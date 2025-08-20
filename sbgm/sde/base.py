"""Base classes and common utilities for SDEs.

All SDE implementations derive from :class:`SDE`, which defines the
interface for drift and diffusion functions as well as helper
functions for sampling from the marginal distribution. Concrete SDEs
must override the drift and diffusion functions and provide
closed‑form expressions for the marginal mean and standard deviation of
``x(t)`` given ``x(0)``.
"""

from __future__ import annotations

import abc
from typing import Tuple

import torch


class SDE(abc.ABC):
    """Abstract base class for stochastic differential equations."""

    def __init__(self, t0: float = 0.0, t1: float = 1.0) -> None:
        self.t0 = t0
        self.t1 = t1

    @abc.abstractmethod
    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift term ``f(x,t)`` of the forward SDE."""

    @abc.abstractmethod
    def diffusion(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient ``g(t)`` of the forward SDE."""

    @abc.abstractmethod
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and standard deviation of ``x(t)`` given ``x(0)``.

        Should return a tuple ``(mean, std)`` of tensors with the same
        leading dimensions as ``t`` and spatial dimensions matching
        ``x0``.
        """

    @abc.abstractmethod
    def prior_sampling(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Sample from the prior distribution at ``t=1``."""

    @abc.abstractmethod
    def sde_type(self) -> str:
        """Return a string identifier for the SDE type."""

    def sample_marginal(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample ``x(t)`` in closed form.

        Parameters
        ----------
        x0: torch.Tensor
            Initial data at time ``t=0``.
        t: torch.Tensor
            Times at which to sample, shape ``(batch,)``.
        noise: torch.Tensor
            Standard normal noise of the same shape as ``x0``.

        Returns
        -------
        torch.Tensor
            Sampled ``x(t)``.
        """
        mean, std = self.marginal_prob(x0, t)
        return mean + std * noise