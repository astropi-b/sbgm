"""UNet architecture for 1D time series.

This module defines a U‑Net adapted to 1D sequences. It follows the
same overall structure as the 2D variant: a series of residual
blocks with time embeddings for the down path, a bottleneck and a
corresponding up path with skip connections. Self‑attention can
optionally be inserted at selected sequence lengths.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

from .blocks import get_timestep_embedding, TimeMLP, ResBlock1D, Attention1D


class UNet1D(nn.Module):
    """Score network based on a UNet for 1D sequences."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_lengths: Iterable[int] = (),
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = TimeMLP(time_embed_dim)
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        # Down blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        length = 100  # default synthetic length; actual length does not change skip mapping
        self.lengths: List[int] = []
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock1D(channels, out_channels, time_embed_dim))
                channels = out_channels
                if length in attn_lengths:
                    self.down_blocks.append(Attention1D(channels))
            self.lengths.append(length)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1))
                length = length // 2
        # Middle
        self.middle_blocks = nn.ModuleList([
            ResBlock1D(channels, channels, time_embed_dim),
            Attention1D(channels),
            ResBlock1D(channels, channels, time_embed_dim),
        ])
        # Up blocks
        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.up_blocks.append(ResBlock1D(channels + out_channels, out_channels, time_embed_dim))
                channels = out_channels
                if self.lengths[len(channel_mults) - 1 - i] in attn_lengths:
                    self.up_blocks.append(Attention1D(channels))
            if i != 0:
                self.up_blocks.append(nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1))
        self.output_norm = nn.GroupNorm(8, channels)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(t, self.time_embed_dim)
        temb = self.time_mlp(temb)
        h = self.input_conv(x)
        hs: List[torch.Tensor] = []
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock1D):
                h = layer(h, temb)
            elif isinstance(layer, Attention1D):
                h = layer(h)
            else:
                hs.append(h)
                h = layer(h)
            if isinstance(layer, ResBlock1D) or isinstance(layer, Attention1D):
                hs.append(h)
        for layer in self.middle_blocks:
            if isinstance(layer, ResBlock1D):
                h = layer(h, temb)
            else:
                h = layer(h)
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock1D):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, temb)
            elif isinstance(layer, Attention1D):
                h = layer(h)
            else:
                h = layer(h)
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)
        return h