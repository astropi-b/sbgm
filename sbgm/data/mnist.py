"""MNIST dataset loading.

Provides helper functions to load the MNIST dataset from torchvision
and return PyTorch dataloaders. The images are normalised to the
range [-1, 1] to match the default output range expected by the
score network.
"""

from __future__ import annotations

from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloaders(
    batch_size: int,
    root: str = "./data",
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return training and test dataloaders for MNIST.

    Parameters
    ----------
    batch_size: int
        Mini‑batch size.
    root: str
        Directory where MNIST is stored or will be downloaded.
    num_workers: int
        Number of subprocesses for data loading.
    download: bool
        If ``True`` download the dataset if not present.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        A pair ``(train_loader, test_loader)``.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])
    train_dataset = datasets.MNIST(root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root, train=False, transform=transform, download=download)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, test_loader