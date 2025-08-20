"""Checkpointing utilities.

This module defines simple functions to save and load model and
optimizer state dictionaries during training. A separate function is
provided to track the best model by monitoring a validation metric
across epochs.
"""

from __future__ import annotations

import os
import torch
from typing import Any, Dict, Optional


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    """Serialize a training state to disk.

    Parameters
    ----------
    path: str
        Output file path (will be created or overwritten).
    state: Dict[str, Any]
        A dictionary containing at least ``'model'`` and possibly
        ``'optimizer'``, ``'epoch'`` and other fields. All tensors will
        be moved to CPU before saving.
    """
    # Move tensors to CPU to avoid device specific issues
    cpu_state: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        elif isinstance(v, dict):
            cpu_state[k] = {kk: vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
        else:
            cpu_state[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cpu_state, path)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """Load a checkpoint previously saved with :func:`save_checkpoint`.

    Parameters
    ----------
    path: str
        Path to the checkpoint file.
    map_location: Optional[str]
        Device mapping for the loaded tensors. Defaults to ``None`` which
        lets PyTorch decide.

    Returns
    -------
    Dict[str, Any]
        The training state dictionary.
    """
    return torch.load(path, map_location=map_location)