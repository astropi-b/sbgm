"""Configuration handling for sbgm.

This module defines a simple configuration system built on top of
Python's standard :mod:`dataclasses` and YAML files. A configuration
contains nested dictionaries for various subsystems such as the model,
dataset, SDE, training hyper‑parameters and the sampler. The
configuration can be loaded from a YAML file and overridden via
command line arguments passed through the CLI. In addition a global
random seed can be configured to ensure deterministic behaviour.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import yaml

from .utils.seed import set_seed


@dataclass
class Config:
    """Top level configuration object.

    Each attribute corresponds to a subsystem of the project. Nested
    dictionaries are used deliberately instead of deeply nested
    dataclasses to keep merging of command line overrides simple. All
    keys are optional and reasonable defaults should be provided in the
    code using the configuration. See the default YAML files under
    ``configs/`` for examples.
    """

    dataset: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    sde: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    sampler: Dict[str, Any] = field(default_factory=dict)
    device: Optional[str] = None
    seed: Optional[int] = None

    def update_from_args(self, args: argparse.Namespace) -> None:
        """Override configuration values from CLI arguments.

        Parameters
        ----------
        args: argparse.Namespace
            Parsed command line arguments.
        """
        # Flatten the args namespace into a dictionary for easier
        # inspection. Only override keys that are not None.
        args_dict = vars(args)
        for key, value in args_dict.items():
            if value is None:
                continue
            if key in {"train", "sample", "config"}:
                continue  # flags handled by CLI
            if key == "device":
                self.device = value
            elif key == "seed":
                self.seed = int(value)
            # Override nested configuration entries if the key
            # contains a dot, e.g. "training.epochs".
            elif "." in key:
                section, subkey = key.split(".", 1)
                if section not in asdict(self):
                    continue
                getattr(self, section)[subkey] = value
            else:
                # Top level keys that map to dicts
                if key in asdict(self):
                    setattr(self, key, value)

        # Set seed if provided
        if self.seed is not None:
            set_seed(self.seed)


def load_config(path: str) -> Config:
    """Load a configuration from a YAML file.

    Parameters
    ----------
    path: str
        Path to a YAML configuration file.

    Returns
    -------
    Config
        A ``Config`` instance populated with the values from the YAML
        file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg