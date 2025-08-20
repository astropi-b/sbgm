"""UNet architecture for 2D images.

This module defines a moderately deep U‑Net that operates on 2D
images. It uses residual blocks with time embeddings and optional
self‑attention at selected resolutions. The architecture is flexible
enough to be used as the score network for diffusion models on
datasets like MNIST (28×28 images).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

from .blocks import get_timestep_embedding, TimeMLP, ResBlock2D, Attention2D


class UNet2D(nn.Module):
    """Score network based on a UNet for 2D images.

    Parameters
    ----------
    in_channels: int
        Number of input channels (e.g. 1 for grayscale, 3 for RGB).
    base_channels: int
        Number of channels in the first convolution layer. Channel
        multipliers determine deeper layers.
    channel_mults: Iterable[int]
        Multiplicative factors for the base_channels at each downsample
        level. The length of this iterable determines the depth of
        the U‑Net.
    num_res_blocks: int
        Number of residual blocks at each resolution.
    attn_resolutions: Iterable[int]
        Spatial resolutions at which to apply self‑attention. For
        example, passing ``[7]`` will insert attention at the 7×7
        resolution stage (MNIST downsampled twice).
    time_embed_dim: int
        Dimensionality of the sinusoidal time embedding processed by a
        two‑layer MLP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Iterable[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Iterable[int] = (7,),
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.time_embed_dim = time_embed_dim
        self.time_mlp = TimeMLP(time_embed_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # Downsampling layers
        self.down_blocks = nn.ModuleList()
        self.resolutions: List[int] = []
        channels = base_channels
        resolution = 28  # assume MNIST shape; if different dataset adjust accordingly
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock2D(channels, out_channels, time_embed_dim))
                channels = out_channels
                if resolution in attn_resolutions:
                    self.down_blocks.append(Attention2D(channels))
            self.resolutions.append(resolution)
            # add downsample except for last level
            if i != len(channel_mults) - 1:
                self.down_blocks.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
                resolution = resolution // 2
        # Middle block
        self.middle_blocks = nn.ModuleList([
            ResBlock2D(channels, channels, time_embed_dim),
            Attention2D(channels),
            ResBlock2D(channels, channels, time_embed_dim),
        ])
        # Upsampling layers
        self.up_blocks = nn.ModuleList()
        # Iterate in reverse for upsampling
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            for _ in range(num_res_blocks):
                # Skip connections: add channels from down path
                self.up_blocks.append(ResBlock2D(channels + out_channels, out_channels, time_embed_dim))
                channels = out_channels
                if self.resolutions[len(channel_mults) - 1 - i] in attn_resolutions:
                    self.up_blocks.append(Attention2D(channels))
            if i != 0:
                # Upsample before moving to next resolution
                self.up_blocks.append(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1))
        # Final convolutions
        self.output_norm = nn.GroupNorm(8, channels)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, H, W)``.
        t: torch.Tensor
            1D tensor of shape ``(B,)`` containing continuous times in [0,1].

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as ``x`` containing the
            predicted score field.
        """
        # Compute time embedding
        temb = get_timestep_embedding(t, self.time_embed_dim)
        temb = self.time_mlp(temb)
        # Down pass
        h = self.input_conv(x)
        hs: List[torch.Tensor] = []
        i = 0
        for layer in self.down_blocks:
            if isinstance(layer, ResBlock2D):
                h = layer(h, temb)
            elif isinstance(layer, Attention2D):
                h = layer(h)
            else:
                # downsample layer
                hs.append(h)
                h = layer(h)
            # Always append after residual/attention block; ensures one skip per res stage
            if isinstance(layer, ResBlock2D) or isinstance(layer, Attention2D):
                hs.append(h)
        # Middle
        for layer in self.middle_blocks:
            if isinstance(layer, ResBlock2D):
                h = layer(h, temb)
            else:
                h = layer(h)
        # Up pass
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock2D):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, temb)
            elif isinstance(layer, Attention2D):
                h = layer(h)
            else:
                # Upsample
                h = layer(h)
        h = self.output_norm(h)
        h = self.output_act(h)
        h = self.output_conv(h)
        return h