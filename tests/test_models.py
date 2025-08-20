"""Tests for UNet models."""

import torch
from sbgm.models.unet2d import UNet2D
from sbgm.models.unet1d import UNet1D


def test_unet2d_shape():
    model = UNet2D(in_channels=1, base_channels=16, channel_mults=(1, 2), num_res_blocks=1, attn_resolutions=(7,), time_embed_dim=32)
    x = torch.randn(2, 1, 28, 28)
    t = torch.rand(2)
    y = model(x, t)
    assert y.shape == x.shape
    # Ensure parameter count > 0
    assert sum(p.numel() for p in model.parameters()) > 0


def test_unet1d_shape():
    model = UNet1D(in_channels=1, base_channels=16, channel_mults=(1, 2), num_res_blocks=1, attn_lengths=(), time_embed_dim=32)
    x = torch.randn(2, 1, 100)
    t = torch.rand(2)
    y = model(x, t)
    assert y.shape == x.shape
    assert sum(p.numel() for p in model.parameters()) > 0