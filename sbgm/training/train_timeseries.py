"""Training loop for 1D time series data.

This module defines a function to train a UNet based score model on
univariate time series using continuous time denoising score
matching. The dataset can be either synthetic or loaded from a CSV
file. After training, the sampler is used to generate new synthetic
sequences which are saved as plots.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import Config
from ..utils.device import get_device, autocast_context
from ..utils.ema import EMA
from ..utils.logging import TrainingLogger
from ..utils.checkpoints import save_checkpoint
from ..utils.plot import plot_timeseries, plot_loss_curve
from ..data.timeseries import CSVTimeSeriesDataset, get_timeseries_loader
from ..data.synthetic import SyntheticDataset, get_synthetic_loader
from ..models.unet1d import UNet1D
from ..sde import VESDE, VPSDE, subVPSDE
from .losses import dsm_loss
from ..sampling.em import euler_maruyama_sampler
from ..sampling.pc import pc_sampler
from ..sampling.ode import ode_sampler


def _instantiate_sde(sde_cfg: Dict[str, Any]) -> Any:
    sde_type = sde_cfg.get("type", "ve").lower()
    if sde_type == "ve":
        return VESDE(sigma_min=sde_cfg.get("sigma_min", 0.01), sigma_max=sde_cfg.get("sigma_max", 50.0))
    elif sde_type == "vp":
        return VPSDE(beta_min=sde_cfg.get("beta_min", 0.1), beta_max=sde_cfg.get("beta_max", 20.0), schedule=sde_cfg.get("schedule", "linear"))
    elif sde_type == "subvp":
        return subVPSDE(beta_min=sde_cfg.get("beta_min", 0.1), beta_max=sde_cfg.get("beta_max", 20.0), schedule=sde_cfg.get("schedule", "linear"))
    else:
        raise ValueError(f"Unknown SDE type: {sde_type}")


def train_timeseries(config: Config) -> None:
    device = get_device(config.device)
    # Dataset selection
    dataset_cfg = config.dataset
    dataset_type = dataset_cfg.get("type", "synthetic").lower()
    batch_size = int(config.training.get("batch_size", 64))
    if dataset_type == "synthetic":
        num_samples = int(dataset_cfg.get("num_samples", 1000))
        length = int(dataset_cfg.get("length", 100))
        train_loader = get_synthetic_loader(num_samples=num_samples, length=length, batch_size=batch_size, num_workers=int(dataset_cfg.get("num_workers", 0)))
    elif dataset_type == "csv":
        path = dataset_cfg.get("path", "examples/tiny_timeseries.csv")
        window_size = int(dataset_cfg.get("window_size", 100))
        stride = int(dataset_cfg.get("stride", 1))
        train_loader = get_timeseries_loader(path=path, batch_size=batch_size, window_size=window_size, stride=stride, num_workers=int(dataset_cfg.get("num_workers", 0)), normalize=bool(dataset_cfg.get("normalize", True)))
    else:
        raise ValueError(f"Unknown time series dataset type: {dataset_type}")
    # Model
    model_cfg = config.model
    model = UNet1D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_lengths=tuple(model_cfg.get("attn_lengths", [])),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
    ).to(device)
    # SDE
    sde = _instantiate_sde(config.sde)
    # Optimiser
    lr = float(config.training.get("lr", 2e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.training.get("weight_decay", 0.0)))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config.training.get("epochs", 1)) * len(train_loader), eta_min=lr * 0.01)
    ema = EMA(model, decay=float(config.training.get("ema_decay", 0.999)))
    use_amp = bool(config.training.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_steps = int(config.training.get("epochs", 1)) * len(train_loader)
    output_dir = config.training.get("output_dir", os.path.join("outputs", time.strftime("%Y%m%d_%H%M%S")))
    log_dir = os.path.join(output_dir, "logs") if config.training.get("tensorboard", False) else None
    logger = TrainingLogger(total_steps=total_steps, log_dir=log_dir)
    os.makedirs(output_dir, exist_ok=True)
    clip_norm = float(config.training.get("grad_clip", 1.0))
    epochs = int(config.training.get("epochs", 1))
    for epoch in range(epochs):
        for batch in train_loader:
            # batch shape (B, C, T)
            x = batch.to(device)
            optimizer.zero_grad()
            with autocast_context(device, enabled=use_amp):
                loss = dsm_loss(model, sde, x, eps=float(config.sde.get("eps", 1e-5)), device=device)
            if use_amp:
                scaler.scale(loss).backward()
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
            scheduler.step()
            ema.update(model)
            logger.log({"loss": loss.item()})
    logger.close()
    ckpt_path = os.path.join(output_dir, "last.ckpt")
    save_checkpoint(ckpt_path, {"model": model.state_dict(), "ema": ema.shadow, "config": config.__dict__})
    # Sampling
    num_samples = int(config.sampler.get("num_samples", 64))
    seq_length = int(dataset_cfg.get("length", dataset_cfg.get("window_size", 100)))
    sampler_type = config.sampler.get("method", "em").lower()
    ema.copy_to(model)
    shape = (num_samples, model.in_channels, seq_length)
    if sampler_type == "em":
        samples = euler_maruyama_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    elif sampler_type == "pc":
        samples = pc_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    elif sampler_type == "ode":
        samples = ode_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    else:
        raise ValueError(f"Unknown sampler method: {sampler_type}")
    ema.restore(model)
    # Save traces
    plot_path = os.path.join(output_dir, f"samples_{sampler_type}.png")
    plot_timeseries(samples[:min(8, num_samples)].cpu(), plot_path)
    # Save loss curve
    loss_curve_path = os.path.join(output_dir, "loss.png")
    plot_loss_curve(logger.history["loss"], loss_curve_path)
    print(f"Training complete. Outputs saved to {output_dir}")