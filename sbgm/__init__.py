"""Top level package for the sbgm library.

This package collects all modules required to implement score based
diffusion models in the stochastic differential equation (SDE) formalism.
The high level API is exposed via the command line interface located
under :mod:`sbgm.cli.main`. Importing this package makes it possible to
access individual components such as SDE implementations, model
definitions and training utilities directly.

Examples
--------
>>> from sbgm.sde.ve import VESDE
>>> sde = VESDE()
>>> x = sde.prior_sampling((16, 1, 28, 28))
>>> print(x.shape)
(16, 1, 28, 28)

The version number of the package is available as ``sbgm.__version__``.
"""

from importlib.metadata import version as _package_version

# Public API: import key subpackages so they appear under sbgm.*
from . import config, data, models, sde, training, sampling, utils, cli

try:
    __version__ = _package_version(__name__)
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "config", "data", "models", "sde", "training", "sampling", "utils", "cli",
    "__version__"
]