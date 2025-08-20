"""Tests for SDE marginal distributions."""

import torch
from sbgm.sde.ve import VESDE
from sbgm.sde.vp import VPSDE
from sbgm.sde.subvp import subVPSDE


def test_ve_marginal():
    sde = VESDE(sigma_min=0.01, sigma_max=1.0)
    x0 = torch.randn(4, 3)
    t = torch.rand(4)
    mean, std = sde.marginal_prob(x0, t)
    sigma_t = sde.sigma(t)
    assert torch.allclose(mean, x0)
    assert std.shape == mean.shape
    # std is sigma(t)
    assert torch.allclose(std.view(-1), sigma_t.repeat_interleave(x0[0].numel()), atol=1e-6)


def test_vp_marginal():
    sde = VPSDE(beta_min=0.1, beta_max=0.1, schedule="linear")
    x0 = torch.randn(5, 2)
    t = torch.rand(5)
    mean, std = sde.marginal_prob(x0, t)
    alpha_bar = sde.alpha_bar(t)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1 - alpha_bar)
    assert torch.allclose(mean, x0 * alpha.view(-1, 1))
    assert torch.allclose(std.view(-1), sigma.repeat_interleave(x0[0].numel()), atol=1e-6)


def test_subvp_marginal_matches_vp():
    vp = VPSDE(beta_min=0.1, beta_max=0.2, schedule="linear")
    subvp = subVPSDE(beta_min=0.1, beta_max=0.2, schedule="linear")
    x0 = torch.randn(3, 2)
    t = torch.rand(3)
    mean_vp, std_vp = vp.marginal_prob(x0, t)
    mean_subvp, std_subvp = subvp.marginal_prob(x0, t)
    assert torch.allclose(mean_vp, mean_subvp)
    assert torch.allclose(std_vp, std_subvp)