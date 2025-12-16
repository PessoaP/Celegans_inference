# tests for simulator.py

import pytest
import math
import numpy as np
import torch
from utils.simulator import sample, step

def test_no_colonization_stays_zero():
    device = torch.device("cuda")
    torch.manual_seed(0)

    N = 2000
    alpha = 0.0
    mu = 0.25
    k = 1e5
    d = 0.1
    params = torch.tensor([alpha, mu, k, d], dtype=torch.float32)

    T = 72.0
    t, E = sample(params, N=N, T=T, device=device)

    assert torch.all(E == 0), "With alpha=0 and E0=0, population should remain zero."

@pytest.mark.slow
def test_colonization_fraction_matches_poisson():
    # If we “turn off” growth and death (μ = d = 0, k huge so nonlinear ≈ 0):
    # The process before first colonization is just a Poisson process with rate α.
    # P(no colonization by time T) = exp(−α T)
    device = torch.device('cuda')
    N = 100000  # smaller is okay, but big N gives tight CI
    alpha = 0.05
    mu = 0
    d = 0
    k = 1e9   # huge so nonlinear is negligible
    params = torch.tensor([alpha, mu, k, d])

    T = 24.0
    t, E = sample(params, N=N, T=T, device=device)

    # colonized = E > 0
    frac_colonized = (E > 0).float().mean().item()

    # theorem: P(no colonization) = exp(-alpha * T)
    expected_colonized = 1 - np.exp(-alpha * T)

    # allow ~1–2% tolerance because of finite N
    assert abs(frac_colonized - expected_colonized) < 0.02

def test_E_never_negative_under_tau_leap():
    # Check that tau-leaping step never produces negative E.
    device = torch.device('cuda')
    N = 5000
    params = torch.tensor([0.05, 0.25, 1e5, 0.1])
    T = 72.0

    # Start with some large initial E to ensure we are in tau-leap regime
    E_initial = torch.full((N,), 2000, dtype=torch.int32, device=device)
    params_b = params.to(device).unsqueeze(-1).repeat(1, N)

    t = torch.zeros(N, device=device)
    E = E_initial.clone()

    while torch.any(t < T):
        t, E = step(params_b, E, t, torch.full((N,), T, device=device), E_tol=0)
        assert torch.all(E >= 0), "Tau-leap produced negative E!"

@pytest.mark.slow
def test_linear_mean_matches_ode():
    device = torch.device("cuda")
    torch.manual_seed(3)

    N = 30000  # need a lot to reduce noise
    alpha = 0.05
    mu = 0.25
    d = 0.10
    k = 1e9  # huge => nonlinear term tiny
    params = torch.tensor([alpha, mu, k, d], dtype=torch.float32)

    T = 12.0  # not too long, keep variance manageable
    _, E = sample(params, N=N, T=T, device=device)

    empirical_mean = E.float().mean().item()

    r = mu - d
    expected_mean = (alpha / r) * (math.exp(r * T) - 1.0)

    # relative tolerance: 10-20% is okay given noise
    rel_err = abs(empirical_mean - expected_mean) / expected_mean
    assert rel_err < 0.2, f"Relative error too large: {rel_err}"




