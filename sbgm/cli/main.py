"""Command line interface for sbgm.

The entry point exposes a unified interface for training and
sampling. Configuration is primarily specified via YAML files but
can be overridden on the command line using dotted keys. For
example::

    python -m sbgm.cli.main --config configs/mnist_ve.yaml --train \
        --training.epochs 2 --model.base_channels 32

If ``--sample`` is passed without ``--train`` the script will load
the last checkpoint from the output directory specified in the
configuration and generate samples according to the ``sampler`` section.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import torch

from ..config import load_config, Config
from ..training import train_image, train_timeseries
from ..utils.device import get_device
from ..models.unet2d import UNet2D
from ..models.unet1d import UNet1D
from ..sde import VESDE, VPSDE, subVPSDE
from ..sampling import euler_maruyama_sampler, pc_sampler, ode_sampler
from ..utils.checkpoints import load_checkpoint
from ..utils.plot import plot_image_grid, plot_timeseries


def parse_overrides(overrides: list[str]) -> Dict[str, Any]:
    """Parse dotted key overrides from the CLI.

    Accepts a list like ``["training.lr", "1e-4", "model.base_channels", "64"]``
    and returns a dictionary mapping keys to values. Values are kept as
    strings since the configuration loader will interpret them.
    """
    result: Dict[str, Any] = {}
    it = iter(overrides)
    for key in it:
        if not key.startswith("--"):
            # Skip positional arguments
            continue
        k = key.lstrip("-")
        try:
            v = next(it)
        except StopIteration:
            raise ValueError(f"No value provided for override {key}")
        result[k] = v
    return result


def build_model_and_sde(config: Config) -> tuple[torch.nn.Module, Any]:
    # Instantiate SDE
    sde_cfg = config.sde
    sde_type = sde_cfg.get("type", "ve").lower()
    if sde_type == "ve":
        sde = VESDE(sigma_min=sde_cfg.get("sigma_min", 0.01), sigma_max=sde_cfg.get("sigma_max", 50.0))
    elif sde_type == "vp":
        sde = VPSDE(beta_min=sde_cfg.get("beta_min", 0.1), beta_max=sde_cfg.get("beta_max", 20.0), schedule=sde_cfg.get("schedule", "linear"))
    else:
        sde = subVPSDE(beta_min=sde_cfg.get("beta_min", 0.1), beta_max=sde_cfg.get("beta_max", 20.0), schedule=sde_cfg.get("schedule", "linear"))
    # Instantiate model based on dataset dimension
    dataset_type = config.dataset.get("type", "mnist").lower()
    if dataset_type in {"mnist", "image", "images"}:
        model = UNet2D(
            in_channels=int(config.model.get("in_channels", 1)),
            base_channels=int(config.model.get("base_channels", 64)),
            channel_mults=tuple(config.model.get("channel_mults", [1, 2, 4])),
            num_res_blocks=int(config.model.get("num_res_blocks", 2)),
            attn_resolutions=tuple(config.model.get("attn_resolutions", [7])),
            time_embed_dim=int(config.model.get("time_embed_dim", 128)),
        )
    else:
        model = UNet1D(
            in_channels=int(config.model.get("in_channels", 1)),
            base_channels=int(config.model.get("base_channels", 64)),
            channel_mults=tuple(config.model.get("channel_mults", [1, 2, 4])),
            num_res_blocks=int(config.model.get("num_res_blocks", 2)),
            attn_lengths=tuple(config.model.get("attn_lengths", [])),
            time_embed_dim=int(config.model.get("time_embed_dim", 128)),
        )
    return model, sde


def sample_only(config: Config) -> None:
    """Sample using a saved checkpoint without training."""
    device = get_device(config.device)
    model, sde = build_model_and_sde(config)
    # Determine checkpoint path
    output_dir = config.training.get("output_dir") or config.sampler.get("output_dir")
    if output_dir is None:
        raise ValueError("output_dir must be specified in the config for sampling")
    ckpt_path = os.path.join(output_dir, "last.ckpt")
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    # Use EMA weights if present
    if "ema" in state:
        for name, param in model.named_parameters():
            if name in state["ema"]:
                param.data.copy_(state["ema"][name])
    model.to(device)
    model.eval()
    # Determine sample shape
    num_samples = int(config.sampler.get("num_samples", 64))
    dataset_type = config.dataset.get("type", "mnist").lower()
    if dataset_type in {"mnist", "image", "images"}:
        shape = (num_samples, model.in_channels, 28, 28)
    else:
        seq_length = int(config.dataset.get("length", config.dataset.get("window_size", 100)))
        shape = (num_samples, model.in_channels, seq_length)
    sampler_type = config.sampler.get("method", "em").lower()
    steps = int(config.sampler.get("steps", 100))
    if sampler_type == "em":
        samples = euler_maruyama_sampler(model, sde, shape, num_steps=steps, device=device)
    elif sampler_type == "pc":
        samples = pc_sampler(model, sde, shape, num_steps=steps, device=device)
    else:
        samples = ode_sampler(model, sde, shape, num_steps=steps, device=device)
    # Plot samples
    out_path = os.path.join(output_dir, f"samples_{sampler_type}_only.png")
    if dataset_type in {"mnist", "image", "images"}:
        plot_image_grid(samples.cpu()[:64], out_path, nrow=8)
    else:
        plot_timeseries(samples.cpu()[:min(8, num_samples)], out_path)
    print(f"Samples saved to {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score based generative modelling via SDEs")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--sample", action="store_true", help="Generate samples without training")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    # Accept arbitrary overrides after the known args
    args, unknown = parser.parse_known_args(argv)
    config = load_config(args.config)
    # Merge CLI overrides
    overrides = parse_overrides(unknown + [] if unknown else [])
    # Add device and seed to overrides
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = str(args.seed)
    # Set flags for train/sample into the namespace; they will be ignored by update_from_args
    overrides["train"] = args.train
    overrides["sample"] = args.sample
    # Update config
    # Convert overrides to argparse.Namespace to reuse update_from_args
    namespace = argparse.Namespace(**{k.replace('.', '_'): v for k, v in overrides.items()})
    config.update_from_args(namespace)
    if args.train:
        # Determine whether to train images or time series
        dataset_type = config.dataset.get("type", "mnist").lower()
        if dataset_type in {"mnist", "image", "images"}:
            train_image(config)
        else:
            train_timeseries(config)
    elif args.sample:
        sample_only(config)
    else:
        parser.error("Specify either --train or --sample")


if __name__ == "__main__":
    main()