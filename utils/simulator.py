# for non-time-limited feeding experiments 

import numpy as np
import pandas as pd


"""
simulator.py

Hybrid stochastic simulation of a 1D birth-death process with nonlinear rates
using Gillespie (SSA) and tau-leaping methods in PyTorch.

Functions:
- ez_sample_exp: Sample from exponential distributions.
- ez_sample_poisson: Sample Poisson-distributed reaction counts.
- get_rates: Compute reaction rates given the state and parameters.
- Gillespie_step: Perform one SSA step for eligible trajectories.
- tau_leap: Perform one tau-leap step for eligible trajectories.
- step: Apply one adaptive step (SSA or tau-leap) based on E tolerance.
- sample: Run full trajectories from t=0 to t=T across N parallel systems.

Assumes 4 reactions:
    R1: ∅ → E      (rate: α)
    R2: E → 2E     (rate: μE)
    R3: 2E → E (rate: (μ-d)/k * E^2)
    R4: E → ∅      (rate: dE)
"""
import torch

# Reaction stoichiometry vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# NOTE: S is created on the default compute device.
# This file assumes all stochastic simulation is run on that same device.
S = torch.tensor((1, 1, -1, -1),device=device).int()

@torch.jit.script
def get_rates(E, params):
    """
    Compute vector of reaction rates for each system.
    Parameters
    ----------
    E : torch.Tensor, shape (N,)
        Current value of species E for N systems.
    params : torch.Tensor, shape (4, N)
        Parameters: [α, μ, k, d] for each system.
    Returns
    -------
    torch.Tensor, shape (N, 4)
        Rates for 4 reactions.
    """
    # Ensure E and params are on same device
    E = E.to(params.device)
    #alpha, mu_E, k, dell = params
    alpha = params[0]
    mu_E  = params[1]
    k     = params[2]
    dell  = params[3]
    rates = torch.stack((alpha,
                         mu_E * E,
                        ((mu_E-dell) / k) * E * E,
                        dell * E), dim=-1)
    return rates


@torch.jit.script
def ez_sample_exp(rates):
    # Sample exponential waiting times for given reaction rates.
    U = torch.rand_like(rates)
    return -torch.log(U) / (rates)


@torch.jit.script
def ez_sample_poisson(r):
    # Sample from Poisson distribution for given rate array.
    return torch.poisson(r)


@torch.jit.script
def Gillespie_step(params, E, dt_max,S=S):
    """
    One Gillespie step for systems below E threshold.
    Parameters
    ----------
    params : torch.Tensor, shape (4, N)
    E : torch.Tensor, shape (N,)
    dt_max : torch.Tensor, shape (N,)
    Returns
    -------
    dt : torch.Tensor, shape (N,)
        Time advanced for each system.
    dE : torch.Tensor, shape (N,)
        Change in E per system.
    """
    rates = get_rates(E, params)
    dt_prop = ez_sample_exp(rates)

    dt, reacts = dt_prop.min(dim=1)

    change = dt < dt_max
    dt[~change] = dt_max[~change]
    
    dE = torch.zeros_like(E)
    dE[change] += S[reacts][change]
    return dt, dE


@torch.jit.script
def tau_leap(params, E, dt_max,S=S):
    """
    One tau-leap step for high E systems.
    Parameters
    ----------
    params : torch.Tensor, shape (4, N)
    E : torch.Tensor, shape (N,)
    dt_max : torch.Tensor, shape (N,)
    Returns
    -------
    dt : torch.Tensor, shape (N,)
    dE : torch.Tensor, shape (N,)
    """
    rates = get_rates(E, params) 

    # Compute the expected net change in total entity count per unit time
    exp_change = torch.abs((rates*S).sum(dim=1))
    exp_change = exp_change.clamp_min(1e-12)
    # Choose dt such that the expected *net* change in E during dt is approximately E / 100
    dt = (E / 101) / exp_change

    # enforce min and max
    dt = dt.clamp(max=0.1)

    # also respect dt_max
    dt = torch.minimum(dt, dt_max)

    num_reac = ez_sample_poisson(rates * dt[:, None])
    dE = (num_reac * S).sum(dim=1).int()

    return dt, dE

E_tol=250*torch.ones(1, device=device) # Threshold to switch between SSA and tau-leap

@torch.jit.script
def step(params, E, t0, T, E_tol=E_tol):
    """
    Single adaptive step (Gillespie or tau-leap depending on E).
    Parameters
    ----------
    params : torch.Tensor, shape (4, N)
    E : torch.Tensor, shape (N,)
    t0 : torch.Tensor, shape (N,)
        Current time per system.
    T : torch.Tensor, shape (N,)
        Final time per system.
    E_tol : float
        Threshold value to switch from SSA to tau-leap.
    Returns
    -------
    t1 : torch.Tensor, shape (N,)
        Updated times.
    E1 : torch.Tensor, shape (N,)
        Updated values of E.
    """
    move_forward = t0 < T
    use_Gillespie = move_forward & (E <= E_tol)
    use_tau = move_forward & (E > E_tol)

    dt = torch.zeros_like(t0)
    dE = torch.zeros_like(E)

    if torch.any(use_Gillespie):
        dt[use_Gillespie], dE[use_Gillespie] = Gillespie_step(params[:, use_Gillespie], 
                                                              E[use_Gillespie], 
                                                              (T - t0)[use_Gillespie])
    if torch.any(use_tau):
        dt[use_tau], dE[use_tau] = tau_leap(params[:, use_tau], 
                                            E[use_tau], 
                                            (T - t0)[use_tau])
    
    E_new = torch.clamp(E + dE, min=0)  # Ensure E doesn't go negative

    return t0 + dt, E_new


def sample(params, E_initial=None, T=48, N=None, 
           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Run full trajectory simulations for N replicates.

    Parameters
    ----------
    params : torch.Tensor, shape (4,) or (4, N)
        Model parameters per system or shared.
    E_initial : torch.Tensor, shape (N,), optional
        Initial states. If None, initialized to zero.
    T : float or torch.Tensor
        Final time per trajectory.
    N : int, optional
        Number of systems to simulate. If None, inferred from E_initial.
    device : torch.device
        PyTorch device.

    Returns
    -------
    t : torch.Tensor, shape (N,)
    E : torch.Tensor, shape (N,)
    """
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

    # Ensure parameters are shape (4, N)
    if isinstance(params, torch.Tensor):
        if params.dim() == 1:
            params = params.to(device).unsqueeze(-1).repeat(1, N)
        elif params.shape[1] != N:
            raise ValueError(f"Expected param shape (4, {N}), got {params.shape}")
    else:
        raise TypeError("params must be a torch.Tensor")

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
        t, E = step(params, E, t, T)

    return t, E

def sample_record(params, record_times, N, E_initial=None,
                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    params = params.to(device)

    if not isinstance(record_times, torch.Tensor):
        record_times = torch.tensor(record_times, device=device, dtype=torch.float32)
    else:
        record_times = record_times.to(device=device, dtype=torch.float32)

    record_times = record_times.reshape(-1)
    record_times, _ = torch.sort(record_times)

    if params.dim() == 1:
        params = params.unsqueeze(-1).repeat(1, N)
    elif params.shape != (4, N):
        raise ValueError(f"Expected params shape (4,{N}), got {tuple(params.shape)}")

    t = torch.zeros(N, device=device, dtype=torch.float32)
    if E_initial is None:
        E = torch.zeros(N, device=device, dtype=torch.int32)
    else:
        E = E_initial.to(device=device).to(torch.int32)

    snapshots = []
    for T_next in record_times:
        Tvec = torch.full_like(t, T_next)
        while torch.any(t < Tvec):
            t, E = step(params, E, t, Tvec)
        snapshots.append(E.clone())

    return snapshots

