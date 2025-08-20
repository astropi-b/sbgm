"""Synthetic 1D time series generator.

Generates random univariate sequences composed of a mixture of sine
waves, occasional bursts and additive Gaussian noise. The resulting
dataset can be used to train score models on synthetic data without
needing to load an external file. Each sequence has a single channel
dimension compatible with Conv1d based architectures.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    """Dataset of synthetic time series.

    Parameters
    ----------
    num_samples: int
        Number of sequences to generate.
    length: int
        Length of each sequence.
    freq_range: Tuple[float, float]
        Range of frequencies for the sinusoidal components.
    noise_std: float
        Standard deviation of the additive Gaussian noise.
    burst_prob: float
        Probability of adding a burst to a sequence.
    """

    def __init__(self,
                 num_samples: int = 1000,
                 length: int = 100,
                 freq_range: Tuple[float, float] = (1.0, 3.0),
                 noise_std: float = 0.05,
                 burst_prob: float = 0.1) -> None:
        self.num_samples = num_samples
        self.length = length
        self.freq_range = freq_range
        self.noise_std = noise_std
        self.burst_prob = burst_prob
        self.samples = [self._generate_sample() for _ in range(num_samples)]

    def _generate_sample(self) -> torch.Tensor:
        t = np.linspace(0, 1, self.length)
        # Random number of sine waves between 1 and 3
        num_waves = np.random.randint(1, 4)
        signal = np.zeros_like(t)
        for _ in range(num_waves):
            freq = np.random.uniform(*self.freq_range)
            phase = np.random.uniform(0, 2 * np.pi)
            amp = np.random.uniform(0.5, 1.0)
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        # Optional burst: a Gaussian pulse centred at a random position
        if np.random.rand() < self.burst_prob:
            centre = np.random.uniform(0.2, 0.8)
            width = np.random.uniform(0.02, 0.05)
            burst = np.exp(-0.5 * ((t - centre) / width) ** 2)
            amp_burst = np.random.uniform(1.0, 2.0)
            signal += amp_burst * burst
        # Additive Gaussian noise
        signal += np.random.randn(*t.shape) * self.noise_std
        return torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)  # shape (1, length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def get_synthetic_loader(
    num_samples: int,
    length: int,
    batch_size: int,
    num_workers: int = 0
) -> DataLoader:
    """Return a DataLoader for the synthetic dataset.

    Parameters
    ----------
    num_samples: int
        Number of sequences to generate.
    length: int
        Length of each sequence.
    batch_size: int
        Batch size.
    num_workers: int
        Number of worker processes.

    Returns
    -------
    DataLoader
        DataLoader yielding batches of synthetic sequences.
    """
    dataset = SyntheticDataset(num_samples=num_samples, length=length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)