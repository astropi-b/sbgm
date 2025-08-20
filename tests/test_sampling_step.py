"""Tests for samplers."""

import torch
from sbgm.sampling.em import euler_maruyama_sampler
from sbgm.sampling.ode import ode_sampler
from sbgm.sde.ve import VESDE


class ZeroModel(torch.nn.Module):
    """Dummy score model returning zeros."""
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_em_shapes():
    sde = VESDE(sigma_min=0.01, sigma_max=0.02)
    model = ZeroModel()
    shape = (2, 1, 28, 28)
    samples = euler_maruyama_sampler(model, sde, shape, num_steps=10)
    assert samples.shape == torch.Size(shape)


def test_ode_deterministic_zero_model():
    sde = VESDE(sigma_min=0.01, sigma_max=0.02)
    model = ZeroModel()
    shape = (2, 1, 5)
    # ODE with zero score should leave the sample unchanged
    # Sample initial x
    x0 = sde.prior_sampling(shape)
    # Use fixed step solver; copy x0 to device
    x_final = ode_sampler(model, sde, shape, num_steps=10)
    # Since drift and score are zero, x stays constant (no change)
    assert torch.allclose(x_final, x0, atol=1e-5)