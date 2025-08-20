"""Plotting helpers for sbgm.

This module centralises all matplotlib usage so that the rest of the
code does not need to worry about figure creation or saving.
Functions are provided to visualise image samples, time series
traces, SDE schedules and training curves. All plots are saved to
the supplied filenames and the paths are returned for convenience.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.utils as vutils


def _ensure_dir(fname: str) -> None:
    os.makedirs(os.path.dirname(fname), exist_ok=True)


def plot_image_grid(samples: torch.Tensor, fname: str, nrow: int = 8) -> str:
    """Save a grid of image samples.

    Parameters
    ----------
    samples: torch.Tensor
        Batch of images with shape ``(B, C, H, W)`` scaled to [-1, 1].
    fname: str
        Output filename (PNG is recommended).
    nrow: int
        Number of images per row in the grid.

    Returns
    -------
    str
        The path to the saved figure.
    """
    _ensure_dir(fname)
    # Denormalize from [-1,1] to [0,1]
    grid = vutils.make_grid((samples + 1) / 2, nrow=nrow, padding=2)
    ndarr = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(ndarr.squeeze(), cmap='gray' if samples.shape[1] == 1 else None)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_timeseries(samples: torch.Tensor, fname: str, max_plots: int = 8) -> str:
    """Plot a batch of one‑dimensional sequences.

    Parameters
    ----------
    samples: torch.Tensor
        Tensor of shape ``(B, C, T)`` where ``C`` is typically 1.
    fname: str
        Output filename.
    max_plots: int
        Maximum number of traces to plot. Excess samples are ignored.

    Returns
    -------
    str
        The path to the saved figure.
    """
    _ensure_dir(fname)
    b, c, t = samples.shape
    num = min(b, max_plots)
    plt.figure(figsize=(6, 2 * num))
    time = np.arange(t)
    for i in range(num):
        ax = plt.subplot(num, 1, i + 1)
        seq = samples[i].detach().cpu().numpy().squeeze()
        ax.plot(time, seq)
        ax.set_xlim(0, t - 1)
        ax.set_ylabel(f"Sample {i}")
        if i == 0:
            ax.set_title("Generated Time Series")
    plt.xlabel("Time index")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_schedule(ts: torch.Tensor, values: torch.Tensor, fname: str, label: str) -> str:
    """Plot an SDE schedule such as sigma(t) or beta(t).

    Parameters
    ----------
    ts: torch.Tensor
        1D tensor of time points in [0,1].
    values: torch.Tensor
        Corresponding values of the schedule.
    fname: str
        Output filename.
    label: str
        Label for the y‑axis.

    Returns
    -------
    str
        The path to the saved figure.
    """
    _ensure_dir(fname)
    t_np = ts.detach().cpu().numpy()
    v_np = values.detach().cpu().numpy()
    plt.figure()
    plt.plot(t_np, v_np)
    plt.xlabel('t')
    plt.ylabel(label)
    plt.title(f"{label} schedule")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_loss_curve(losses: List[float], fname: str, label: str = "Loss") -> str:
    """Plot a loss curve over training steps.

    Parameters
    ----------
    losses: List[float]
        List of loss values.
    fname: str
        Output filename.
    label: str
        Label for the y‑axis.

    Returns
    -------
    str
        The path to the saved figure.
    """
    _ensure_dir(fname)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel(label)
    plt.title(f"{label} over time")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname