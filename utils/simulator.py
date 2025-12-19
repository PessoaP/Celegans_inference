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
    R3: 2E → E (rate: (μE-d)/k * E^2)
    R4: E → ∅      (rate: dE)
"""
import torch

# Reaction stoichiometry vector
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S = torch.tensor((1, 1, -1, -1),device=device).int()

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
    alpha, mu_E, k, dell = params
    rates = torch.stack((alpha,
                         mu_E * E,
                        ((mu_E-dell) / k) * E * E,
                        dell * E), dim=-1)
    return rates


def ez_sample_exp(rates):
    # Sample exponential waiting times for given reaction rates.
    U = torch.rand_like(rates)
    return -torch.log(U) / (rates)

def ez_sample_poisson(r):
    # Sample from Poisson distribution for given rate array.
    return torch.poisson(r)


def Gillespie_step(params, E, dt_max):
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
    dt, reacts = dt_prop.min(axis=1)

    change = dt < dt_max
    dt[~change] = dt_max[~change]
    
    dE = torch.zeros_like(E)
    dE[change] += S[reacts][change]
    return dt, dE

def tau_leap(params, E, dt_max):
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
    exp_change = torch.abs((rates*S).sum(axis=1))
    # Choose dt such that the expected *net* change in E during dt is approximately E / 100
    dt = (E / 101) / exp_change
    dt = dt.clamp(max=.1)

    change = dt < dt_max
    dt[~change] = dt_max[~change]

    num_reac = ez_sample_poisson(rates * dt[:, None])
    dE = (num_reac * S).sum(axis=1).int()
    return dt, dE

def step(params, E, t0, T, E_tol=1e3):
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


def integrate_mass_action(params, Ts, dt=0.1,device='cpu'):
    """
    Simulate ODE trajectories from E=0 using Euler method.
    """
    params = params.to(device)
    S_ma = S.to(device)
    E = torch.zeros(1,device=device)
    t = 0

    dEdt = lambda Ex: (get_rates(Ex,params.reshape(-1,1))*S_ma).sum(axis=1)
    
    # Make sure Ts is a tensor on cpu
    if not isinstance(Ts, torch.Tensor):
        Ts = torch.tensor(Ts, device='cpu').float()
    else:
        Ts = Ts.to('cpu').float()

    Es = []
    for T in Ts:
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, device='cpu').float()
        else:
            T = T.to('cpu').float()
        while t + dt < T:
            E = E + dEdt(E) * dt
            t = t + dt
        E = E + dEdt(E) * (T - t)
        t = T
        Es.append(E.clone())
    return torch.stack(Es)