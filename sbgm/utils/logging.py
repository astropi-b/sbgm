"""Logging utilities for training and sampling.

This module provides helper classes to manage progress reporting via
``tqdm`` and optional TensorBoard logging if ``tensorboard`` is
available. The functions and classes here are intentionally lightweight
so as not to impose a specific logging framework on users. Use
``TrainingLogger`` to record scalar metrics during training and to
accumulate loss curves for later visualisation.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, List

from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter as _TBWriter  # type: ignore
except Exception:
    _TBWriter = None  # type: ignore


class TrainingLogger:
    """Simple logger for training loops.

    Handles printing progress to the console using ``tqdm`` and
    optionally writing scalars to TensorBoard. All logged metrics are
    stored in memory so they can be plotted after training.
    """

    def __init__(self, total_steps: int, log_dir: Optional[str] = None) -> None:
        self.pbar = tqdm(total=total_steps, desc="Training", leave=True)
        self.history: Dict[str, List[float]] = {}
        self.step = 0
        if log_dir is not None and _TBWriter is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.tb = _TBWriter(log_dir)
        else:
            self.tb = None

    def log(self, metrics: Dict[str, float]) -> None:
        """Log a dictionary of scalar metrics for the current step.

        Parameters
        ----------
        metrics: Dict[str, float]
            Mapping from metric names to scalar values. For each key
            ``'loss'``, ``'grad_norm'`` or other arbitrary names the value is
            recorded and optionally sent to TensorBoard.
        """
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))
            if self.tb is not None:
                self.tb.add_scalar(k, v, self.step)
        self.pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()}, refresh=False)
        self.pbar.update(1)
        self.step += 1

    def close(self) -> None:
        """Close the logger.

        Stops the tqdm progress bar and closes the TensorBoard writer if
        it was created.
        """
        self.pbar.close()
        if self.tb is not None:
            self.tb.close()