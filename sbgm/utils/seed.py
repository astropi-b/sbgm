"""Random seed utilities.

Setting a deterministic random seed is essential for reproducible
experiments. This helper exposes a single function that sets the seed
for Python's ``random`` module, NumPy and PyTorch (including CUDA).
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all random number generators with the given seed.

    Parameters
    ----------
    seed: int
        The global seed. Should be a positive integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False