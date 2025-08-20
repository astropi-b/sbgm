"""Exponential moving average for model parameters.

Maintaining a running average of model parameters during training
improves sampling quality for diffusion models. This helper class
implements a simple decay based exponential moving average which can be
swapped in before running the sampler and swapped back afterwards.
"""

from __future__ import annotations

import copy
from typing import Dict, Any, Iterator

import torch


class EMA:
    """Maintain an exponential moving average (EMA) of model parameters.

    The EMA stores a shadow copy of parameters which are updated after
    every optimisation step. The decay factor controls how much weight
    is given to older parameter values: a value close to one means that
    the EMA changes slowly, while a value close to zero means that the
    shadow weights follow the instantaneous weights closely. Use
    :meth:`copy_to` to load the averaged parameters into a model and
    :meth:`store`/``restore`` to temporarily swap weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Create a copy of the parameters for the shadow buffer
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        # Storage for saving and restoring parameters
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update the shadow weights from the current model parameters.

        Parameters
        ----------
        model: torch.nn.Module
            Model whose parameters to track.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_val = param.detach()
                old_val = self.shadow[name]
                # Exponential moving average update
                self.shadow[name] = self.decay * old_val + (1.0 - self.decay) * new_val

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy shadow (EMA) weights into the given model.

        Typically called before sampling so that the model uses the
        stabilised EMA parameters. After sampling call
        :meth:`restore` to put the original training weights back.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Restore the original parameters that were overwritten by
        :meth:`copy_to`.
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}