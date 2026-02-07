# for Time-Limited feeding experiments 

import numpy as np
import pandas as pd
import torch

# Reaction stoichiometry vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# NOTE: S is created on the default compute device.
# This file assumes all stochastic simulation is run on that same device.
S = torch.tensor((1, 1, -1, -1),device=device).int()

def get_rates(E, params, t=None, switch_time=24.0):
    """
    params: (5, N) = [alpha, mu, k, d, rho]
    t: (N,) current time in hours
    """
    E = E.to(params.device)
    alpha, mu_E, k, dell, rho = params

    if t is None:
        alpha_eff = alpha
    else:
        t = t.to(device=params.device, dtype=alpha.dtype)
        alpha_eff = torch.where(t >= switch_time, rho * alpha, alpha)

    rates = torch.stack((
        alpha_eff,
        mu_E * E,
        ((mu_E - dell) / k) * E * E,
        dell * E
    ), dim=-1)

    return rates

def ez_sample_exp(rates):
    # Sample exponential waiting times for given reaction rates.
    U = torch.rand_like(rates)
    return -torch.log(U) / (rates)

def ez_sample_poisson(r):
    # Sample from Poisson distribution for given rate array.
    return torch.poisson(r)


def Gillespie_step(params, E, dt_max, t0, switch_time=24.0):
    rates = get_rates(E, params, t=t0, switch_time=switch_time)
    dt_prop = ez_sample_exp(rates)
    dt, reacts = dt_prop.min(axis=1)

    change = dt < dt_max
    dt[~change] = dt_max[~change]

    dE = torch.zeros_like(E)
    dE[change] += S[reacts][change]
    return dt, dE

def tau_leap(params, E, dt_max, t0, dt_min=0., switch_time=24.0):
    rates = get_rates(E, params, t=t0, switch_time=switch_time)

    exp_change = torch.abs((rates * S).sum(axis=1))
    exp_change = torch.clamp(exp_change, min=1e-12)  # avoid divide-by-zero
    dt = (E / 101) / exp_change

    dt = dt.clamp(min=dt_min, max=0.1)
    dt = torch.minimum(dt, dt_max)

    num_reac = ez_sample_poisson(rates * dt[:, None])
    dE = (num_reac * S).sum(axis=1).int()
    return dt, dE

def step(params, E, t0, T, E_tol=250, switch_time=24.0):
    move_forward = t0 < T
    use_Gillespie = move_forward & (E <= E_tol)
    use_tau = move_forward & (E > E_tol)

    dt = torch.zeros_like(t0)
    dE = torch.zeros_like(E)

    if torch.any(use_Gillespie):
        dt[use_Gillespie], dE[use_Gillespie] = Gillespie_step(
            params[:, use_Gillespie],
            E[use_Gillespie],
            (T - t0)[use_Gillespie],
            t0=t0[use_Gillespie],
            switch_time=switch_time,
        )

    if torch.any(use_tau):
        dt[use_tau], dE[use_tau] = tau_leap(
            params[:, use_tau],
            E[use_tau],
            (T - t0)[use_tau],
            t0=t0[use_tau],
            switch_time=switch_time,
        )

    E_new = torch.clamp(E + dE, min=0)
    return t0 + dt, E_new

def sample(params, E_initial=None, T=48, N=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
           switch_time=24.0):
    
    params = params.to(device)

    # Infer N from E_initial if not provided
    if N is None:
        if E_initial is None:
            raise ValueError("Either N or E_initial must be provided.")
        elif not isinstance(E_initial, torch.Tensor):
            raise TypeError("E_initial must be a torch.Tensor if N is not given.")
        N = E_initial.shape[0]
    else:
        if E_initial is not None and E_initial.shape[0] != N:
            raise ValueError(f"Incompatible E_initial shape {E_initial.shape} with N={N}")

    # Ensure parameters are shape (5, N)
    if isinstance(params, torch.Tensor):
        if params.dim() == 1:
            params = params.to(device).unsqueeze(-1).repeat(1, N)
        elif params.shape != (5, N):
            raise ValueError(f"Expected param shape (5, {N}), got {tuple(params.shape)}")
    else:
        raise TypeError("params must be a torch.Tensor")

    if params.shape[0] != 5:
        raise ValueError(f"Expected 5 params [alpha, mu, k, d, rho], got shape {tuple(params.shape)}")

    # Handle scalar or broadcast T
    if isinstance(T, (float, int)):
        T = torch.full((N,), T, device=device).float()
    elif isinstance(T, torch.Tensor):
        if T.numel() == 1:
            T = T.to(device).expand(N).float()

    # Initialize time and E
    t = torch.zeros(N, device=device)
    if E_initial is None:
        E = torch.zeros(N, device=device).int()
    else:
        E = E_initial.to(device).int()

    while torch.any(t < T):
        t, E = step(params, E, t, T, switch_time=switch_time)  # switch_time can be left default or passed

    return t, E

def sample_record(params, record_times, N, E_initial=None,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), switch_time=24.0):
    params = params.to(device)

    if not isinstance(record_times, torch.Tensor):
        record_times = torch.tensor(record_times, device=device, dtype=torch.float32)
    else:
        record_times = record_times.to(device=device, dtype=torch.float32)

    record_times = record_times.reshape(-1)
    record_times, _ = torch.sort(record_times)

    if params.dim() == 1:
        params = params.unsqueeze(-1).repeat(1, N)
    elif params.shape != (5, N):
        raise ValueError(f"Expected params shape (5,{N}), got {tuple(params.shape)}")

    t = torch.zeros(N, device=device, dtype=torch.float32)
    if E_initial is None:
        E = torch.zeros(N, device=device, dtype=torch.int32)
    else:
        E = E_initial.to(device=device).to(torch.int32)

    snapshots = []
    for T_next in record_times:
        Tvec = torch.full_like(t, T_next)
        while torch.any(t < Tvec):
            t, E = step(params, E, t, Tvec, switch_time=switch_time)
        snapshots.append(E.clone())

    return snapshots