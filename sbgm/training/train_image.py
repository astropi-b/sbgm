"""Training loop for image data (e.g. MNIST).

This script defines a function to train a UNet based score model on
2D image datasets using the continuous time denoising score matching
loss. It supports mixed precision training, gradient clipping,
exponential moving averages and checkpointing. At the end of
training the script optionally samples a batch of images using the
configured sampler and saves a figure to disk.
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
from ..utils.plot import plot_image_grid, plot_loss_curve
from ..data.mnist import get_mnist_dataloaders
from ..models.unet2d import UNet2D
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


def train_image(config: Config) -> None:
    device = get_device(config.device)
    # Data
    batch_size = int(config.training.get("batch_size", 64))
    train_loader, _ = get_mnist_dataloaders(
        batch_size=batch_size,
        root=config.dataset.get("root", "./data"),
        num_workers=int(config.dataset.get("num_workers", 0)),
        download=True,
    )
    # Model
    model_cfg = config.model
    model = UNet2D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        num_res_blocks=int(model_cfg.get("num_res_blocks", 2)),
        attn_resolutions=tuple(model_cfg.get("attn_resolutions", [7])),
        time_embed_dim=int(model_cfg.get("time_embed_dim", 128)),
    ).to(device)
    # SDE
    sde = _instantiate_sde(config.sde)
    # Optimiser
    lr = float(config.training.get("lr", 2e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config.training.get("weight_decay", 0.0)))
    # LR scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config.training.get("epochs", 1)) * len(train_loader), eta_min=lr * 0.01)
    # EMA
    ema_decay = float(config.training.get("ema_decay", 0.999))
    ema = EMA(model, decay=ema_decay)
    # AMP
    use_amp = bool(config.training.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # Logging
    total_steps = int(config.training.get("epochs", 1)) * len(train_loader)
    output_dir = config.training.get("output_dir", os.path.join("outputs", time.strftime("%Y%m%d_%H%M%S")))
    log_dir = os.path.join(output_dir, "logs") if config.training.get("tensorboard", False) else None
    logger = TrainingLogger(total_steps=total_steps, log_dir=log_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Training loop
    global_step = 0
    clip_norm = float(config.training.get("grad_clip", 1.0))
    epochs = int(config.training.get("epochs", 1))
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            with autocast_context(device, enabled=use_amp):
                loss = dsm_loss(model, sde, x, eps=float(config.sde.get("eps", 1e-5)), device=device)
            if use_amp:
                scaler.scale(loss).backward()
                # Gradient clipping
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
            global_step += 1
    logger.close()
    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "last.ckpt")
    save_checkpoint(ckpt_path, {"model": model.state_dict(), "ema": ema.shadow, "config": config.__dict__})
    # Sampling after training
    sample_batch = int(config.sampler.get("num_samples", 64))
    sampler_type = config.sampler.get("method", "em").lower()
    ema.copy_to(model)
    shape = (sample_batch, model.in_channels, 28, 28)
    if sampler_type == "em":
        samples = euler_maruyama_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    elif sampler_type == "pc":
        samples = pc_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    elif sampler_type == "ode":
        samples = ode_sampler(model, sde, shape, num_steps=int(config.sampler.get("steps", 100)), device=device)
    else:
        raise ValueError(f"Unknown sampler method: {sampler_type}")
    ema.restore(model)
    # Save samples to image grid
    grid_path = os.path.join(output_dir, f"samples_{sampler_type}.png")
    plot_image_grid(samples[:64].cpu(), grid_path, nrow=8)
    # Plot loss curve
    loss_curve_path = os.path.join(output_dir, "loss.png")
    plot_loss_curve(logger.history["loss"], loss_curve_path)
    print(f"Training complete. Checkpoints and samples saved to {output_dir}")