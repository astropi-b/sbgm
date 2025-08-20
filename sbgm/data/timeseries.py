"""CSV based time series loading.

This module defines a dataset for loading univariate time series from
CSV files. The values are normalised and split into sliding windows
that form the training examples. Each window is returned with a
channel dimension of 1 so that it can be fed into a 1D convolutional
network directly.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CSVTimeSeriesDataset(Dataset):
    """Dataset of sliding windows extracted from a CSV time series.

    Parameters
    ----------
    path: str
        Path to a CSV file containing a single column of numeric
        observations.
    window_size: int
        Length of each window in samples.
    stride: int
        Step size between successive windows. A stride of 1 produces
        maximal overlap while larger values reduce overlap.
    normalize: bool
        If ``True`` the sequence will be normalised to zero mean and
        unit variance before windowing.
    """

    def __init__(self, path: str, window_size: int, stride: int = 1, normalize: bool = True) -> None:
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        if data.ndim > 1:
            data = data[:, 0]
        if normalize:
            self.mean = float(data.mean())
            self.std = float(data.std() + 1e-8)
            data = (data - self.mean) / self.std
        else:
            self.mean, self.std = 0.0, 1.0
        # Create sliding windows
        self.windows = []
        for start in range(0, len(data) - window_size + 1, stride):
            end = start + window_size
            self.windows.append(torch.from_numpy(data[start:end]).unsqueeze(0))  # shape (1, window_size)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def get_timeseries_loader(
    path: str,
    batch_size: int,
    window_size: int,
    stride: int = 1,
    num_workers: int = 0,
    normalize: bool = True,
) -> DataLoader:
    """Convenience function returning a DataLoader for CSV time series.

    Parameters
    ----------
    path: str
        Path to the CSV file.
    batch_size: int
        Batch size for iteration.
    window_size: int
        Size of each sliding window.
    stride: int
        Step between windows.
    num_workers: int
        Number of workers used by the DataLoader.
    normalize: bool
        Whether to normalise the series.

    Returns
    -------
    DataLoader
        PyTorch DataLoader instance.
    """
    dataset = CSVTimeSeriesDataset(path, window_size=window_size, stride=stride, normalize=normalize)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader