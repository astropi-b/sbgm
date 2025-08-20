"""Device management utilities.

This module centralises logic for determining the appropriate PyTorch
device (CPU, CUDA or Apple MPS) and exposes helper functions for
mixed precision training. When the user does not specify a device
explicitly the code will prefer CUDA if available, otherwise Apple
Silicon's Metal Performance Shaders (MPS) backend if available and
finally fall back to the CPU.
"""

from __future__ import annotations

import contextlib
import torch
from typing import Optional


def get_device(user_device: Optional[str] = None) -> torch.device:
    """Determine the torch.device to be used.

    Parameters
    ----------
    user_device: Optional[str]
        Optional string provided by the user, for example ``"cpu"``,
        ``"cuda"`` or ``"mps"``. If ``None`` the device will be
        auto‑detected.

    Returns
    -------
    torch.device
        The device that should be used for tensors and models.
    """
    user_device = None if user_device is None else user_device.lower()
    if user_device is not None:
        return torch.device(user_device)
    # Auto detect device
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is only available on Apple Silicon with recent PyTorch versions
    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def autocast_context(device: torch.device, enabled: bool = True):
    """Context manager for automatic mixed precision.

    On CUDA, mixed precision is enabled via ``torch.cuda.amp.autocast``.
    On MPS the current PyTorch release does not yet support AMP so
    autocasting is disabled. On CPU the context is also disabled. Use
    this context manager when computing forward passes in training to
    avoid cluttering the code with backend specific conditions.

    Parameters
    ----------
    device: torch.device
        Device on which the computation is executed.
    enabled: bool
        If ``True`` AMP will be used when supported, otherwise the
        context acts as a no‑op.
    """
    if device.type == "cuda" and enabled:
        with torch.cuda.amp.autocast():
            yield
    else:
        # MPS and CPU currently do not support autocast in PyTorch
        yield