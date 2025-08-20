"""Datasets for sbgm.

This package contains data loading utilities for both 2D image
datasets (e.g. MNIST) and one‑dimensional time series. Synthetic
datasets are also provided. Each submodule exposes functions or
classes returning PyTorch ``Dataset`` objects and corresponding
``DataLoader`` factory functions when appropriate.
"""

from .mnist import get_mnist_dataloaders
from .timeseries import CSVTimeSeriesDataset
from .synthetic import SyntheticDataset

__all__ = [
    "get_mnist_dataloaders",
    "CSVTimeSeriesDataset",
    "SyntheticDataset",
]