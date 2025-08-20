"""Probability flow ODE sampler.

This module implements a fixed step fourth order Runge–Kutta (RK4)
solver for the probability flow ODE associated with a stochastic
differential equation. The ODE is solved backward in time from
``t=1`` to ``t=eps`` without any stochasticity. If the optional
dependency ``torchdiffeq`` is installed, the adaptive solver
``torchdiffeq.odeint`` can be used instead by specifying
``method='torchdiffeq'``.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

from ..sde.base import SDE

try:
    from torchdiffeq import odeint as _odeint  # type: ignore
    _has_torchdiffeq = True
except Exception:
    _has_torchdiffeq = False


@torch.no_grad()
def ode_sampler(
    model: torch.nn.Module,
    sde: SDE,
    shape: Tuple[int, ...],
    num_steps: int = 1000,
    device: Optional[torch.device] = None,
    eps: float = 1e-3,
    method: str = "rk4",
) -> torch.Tensor:
    """Sample from the probability flow ODE.

    Parameters
    ----------
    model: torch.nn.Module
        Score network.
    sde: SDE
        Forward SDE.
    shape: Tuple[int, ...]
        Output shape.
    num_steps: int
        Number of steps for the fixed step solver (ignored if using
        ``torchdiffeq`` method).
    device: Optional[torch.device]
        Device for computation.
    eps: float
        Minimum time.
    method: str
        Solver to use: ``'rk4'`` for built in fixed step solver or
        ``'torchdiffeq'`` for the adaptive solver (requires
        ``torchdiffeq`` installed).

    Returns
    -------
    torch.Tensor
        Samples from the learned distribution.
    """
    if device is None:
        device = next(model.parameters()).device
    batch_size = shape[0]
    x0 = sde.prior_sampling(shape).to(device)
    if method == "torchdiffeq" and _has_torchdiffeq:
        # Define ODE function
        def ode_func(t, x_flat):
            # x_flat shape (batch*channels*dim)
            x = x_flat.view(x0.shape)
            t_tensor = torch.full((batch_size,), float(t), device=device)
            score = model(x, t_tensor)
            drift = sde.drift(x, t_tensor)
            g = sde.diffusion(t_tensor)
            g2 = g * g
            dxdt = drift - 0.5 * g2.view(-1, *([1] * (x.dim() - 1))) * score
            return dxdt.view(-1)
        ts = torch.tensor([1.0, eps], device=device)
        x_flat = x0.view(-1)
        x_t = _odeint(ode_func, x_flat, ts, rtol=1e-5, atol=1e-5)[-1]
        return x_t.view(x0.shape)
    # Fixed step RK4 solver
    x = x0
    t = torch.linspace(1.0, eps, num_steps + 1, device=device)
    for i in range(num_steps):
        t_i = t[i]
        t_next = t[i + 1]
        dt = t_next - t_i
        t_mid = t_i + 0.5 * dt
        t_vals = [t_i, t_mid, t_mid, t_next]
        ks = []
        x_temp = x
        for ti in t_vals:
            tt = ti
            t_batch = tt.expand(batch_size)
            score = model(x_temp, t_batch)
            drift = sde.drift(x_temp, t_batch)
            g = sde.diffusion(t_batch)
            g2 = g * g
            ode_drift = drift - 0.5 * g2.view(-1, *([1] * (x.dim() - 1))) * score
            ks.append(ode_drift)
            if ti != t_vals[-1]:
                # For RK4 intermediate steps
                x_temp = x + 0.5 * dt * ks[-1]
        # RK4 combination
        k1, k2, k3, k4 = ks
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x