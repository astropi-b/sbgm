"""Building blocks for score models.

This module defines reusable neural network components used to build
UNet architectures for 1D time series and 2D images. The key
ingredients are residual blocks that incorporate time embeddings,
group normalisation and SiLU activations, as well as self‑attention
blocks. A sinusoidal time embedding is provided to condition the
network on the continuous noise level.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import math


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    The implementation follows the one used in the original Transformer
    paper and in diffusion models. Given a 1D tensor of timesteps in
    [0, 1], this function returns a tensor of shape ``(batch,
    embedding_dim)`` containing sinusoidal embeddings. The frequencies
    decay exponentially so that each dimension spans a different
    timescale.

    Parameters
    ----------
    timesteps: torch.Tensor
        A 1D tensor of values in [0,1].
    embedding_dim: int
        Dimension of the embedding.

    Returns
    -------
    torch.Tensor
        Embedding tensor of shape ``(len(timesteps), embedding_dim)``.
    """
    assert timesteps.dim() == 1, "timesteps should be a 1D tensor"
    half_dim = embedding_dim // 2
    # Generate exponentially decreasing frequencies
    exponent = torch.arange(half_dim, dtype=timesteps.dtype, device=timesteps.device)
    exponent = exponent / half_dim
    freqs = 10.0 ** (-4.0 * exponent)
    # (batch, half_dim)
    angles = timesteps[:, None] * freqs[None, :] * math.pi
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding_dim % 2 == 1:
        # Zero pad if embedding dim is odd
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class TimeMLP(nn.Module):
    """Two layer MLP to process sinusoidal time embeddings."""

    def __init__(self, embedding_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 4
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class ResBlock1D(nn.Module):
    """Residual block for 1D sequences with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        # Add time embedding (broadcast to sequence length)
        emb_out = self.emb_proj(emb).unsqueeze(-1)
        h = h + emb_out
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class ResBlock2D(nn.Module):
    """Residual block for 2D images with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return h + self.skip(x)


class Attention1D(nn.Module):
    """Self‑attention for 1D sequences."""

    def __init__(self, channels: int, heads: int = 1) -> None:
        super().__init__()
        self.heads = heads
        self.scale = channels ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        qkv = self.qkv(x)  # (b, 3c, t)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each (b,c,t)
        # Reshape for multihead: (b, heads, c//heads, t)
        q = q.view(b, self.heads, c // self.heads, t)
        k = k.view(b, self.heads, c // self.heads, t)
        v = v.view(b, self.heads, c // self.heads, t)
        # Compute attention scores
        attn = torch.einsum('bhct,bhcs->bhts', q, k) * self.scale  # (b, heads, t, t)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhts,bhcs->bhct', attn, v)
        out = out.reshape(b, c, t)
        out = self.proj(out)
        return x + out


class Attention2D(nn.Module):
    """Self‑attention for 2D feature maps.

    The 2D feature map of shape ``(B, C, H, W)`` is flattened into a
    sequence of length ``H*W`` and passed through a standard scaled dot
    product self‑attention. The output is reshaped back to the 2D
    spatial layout.
    """

    def __init__(self, channels: int, heads: int = 1) -> None:
        super().__init__()
        self.heads = heads
        self.scale = channels ** -0.5
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each (b, c, h, w)
        # Flatten spatial dimensions
        q = q.reshape(b, self.heads, c // self.heads, n)
        k = k.reshape(b, self.heads, c // self.heads, n)
        v = v.reshape(b, self.heads, c // self.heads, n)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale  # (b, heads, n, n)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return x + out