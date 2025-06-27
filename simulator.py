"""
hybrid_sim.py

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
    alpha, mu_E, k, d = params
    rates = torch.stack((
        alpha,
        mu_E * E,
        ((mu_E-d) / k) * E * E,
        d * E
    ), dim=-1)
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
    dt.clamp(max=.1)

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
        dt[use_Gillespie], dE[use_Gillespie] = Gillespie_step(
            params[:, use_Gillespie], E[use_Gillespie], (T - t0)[use_Gillespie]
        )
    if torch.any(use_tau):
        dt[use_tau], dE[use_tau] = tau_leap(
            params[:, use_tau], E[use_tau], (T - t0)[use_tau]
        )

    return t0 + dt, E + dE

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


class ConstrainedLogNormalPrior:
    def __init__(self, loc, scale):
        self.base = torch.distributions.LogNormal(loc, scale)

    def log_prob(self, x):
        if x[1] < x[3]:
            return torch.tensor(float('-inf'))
        return self.base.log_prob(x).sum()

    def sample(self):
        for _ in range(1000):
            x = self.base.sample()
            if x[1] >= x[3]:
                return x
        raise RuntimeError("Failed to sample satisfying x[1] >= x[3] after 100 attempts.")



class SyntheticSimulator:
    """
    Simulates counts from a stochastic growth model at different times,
    applies dilution, and exports results to CSV.
    """

    def __init__(self, params, cutoff=300, 
                 dil_schedule=20.0 * torch.pow(10, torch.arange(4)), name='default', 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """Initialize simulator with model parameters, cutoff, dilution schedule, and device."""
        self.params = params
        self.cutoff = cutoff
        self.dil_schedule = dil_schedule 
        self.name = name
        self.device = device 
        os.makedirs("synthetic_data", exist_ok=True)

    def multinomial_with_completion(self, n, probs):
        """Draw from a multinomial with possibly incomplete probability mass."""
        sum_probs = probs.sum()
        if sum_probs < 1:
            probs = np.append(probs, 1 - sum_probs)
        samples = np.random.multinomial(n, probs)
        return samples[:-1] if sum_probs < 1 else samples

    def make_data(self, n_sam):
        """Simulate diluted observed counts from true counts using dilution schedule."""
        cts = np.zeros_like(n_sam)
        dil = np.zeros_like(n_sam)*1.
        probs = 1 / self.dil_schedule.cpu().numpy()

        for i in range(n_sam.size):
            ks = self.multinomial_with_completion(n_sam[i], probs)
            idx = np.argmax(ks <= self.cutoff) if np.any(ks <= self.cutoff) else len(ks) - 1
            cts[i] = ks[idx]
            dil[i] = self.dil_schedule[idx].item()

        return cts, dil

    def sample_n(self, size=None, T=[48]):
        """Simulate true counts at time T for a given sample size."""
        _, E = sample(self.params, T=T, N=size, device=self.device)
        return E
    
    def sample_data(self, size=None, Ts=[48]):
        """Simulate and dilute counts for multiple timepoints in a single batch."""
        #Ts = torch.tensor(Ts, device=self.device).float()
        Ts = Ts.clone().to(self.device).float()
        T_batch = Ts.repeat_interleave(size)  # shape: (len(Ts) * size,)
        n_sam = self.sample_n(T=T_batch,size=T_batch.numel()).cpu().numpy()

        if isinstance(self.dil_schedule, torch.Tensor):
            cts, dils = self.make_data(n_sam)
        else:
            dils = np.ones_like(n_sam) * self.dil_schedule
            cts = np.random.binomial(n_sam, 1.0 / dils)


        df = pd.DataFrame({
            'Time': T_batch.cpu().numpy(),
            'Counts': cts,
            'Dilution': dils
        })

        return df

    def sample_save(self, size=100, Ts=[48], filename=''):
        """Run simulation and save the output as a CSV with counts and dilutions."""
        if filename == '':
            filename = f'synthetic_data/synth_{self.name}.csv'
        df = self.sample_data(size, Ts)
        df.to_csv(filename, index=False)



if __name__ == "__main__":
    import sys, os
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt


    try:
        seed = int(sys.argv[1])
    except:
        seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    value = torch.tensor((1/20, #alpha
                          1/4, #mu
                          1e5, #k
                          .1 #d
                        )).to(device) 
    if seed == 0:
        
        E = torch.zeros(20)
        Es = []
        dT = .2
        ts = torch.arange(0,4*24,dT)

        for t in ts:
            tt,E = sample(value,E,dT)
            Es.append(E)
        for es in torch.vstack(Es).T[-20:]:
            plt.plot(ts,es.cpu())
        plt.xlim(0,ts[-1])  
        plt.xlabel('Time(h)') 
        plt.ylabel('Bacterial number')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        os.makedirs("synthetic_data", exist_ok=True)
        plt.savefig('synthetic_data/example.png',dpi=500)  

        params = value
        np.savetxt('synthetic_data/gt_map.csv', np.hstack((np.array((seed)),value.cpu())))


    else:
        prior = ConstrainedLogNormalPrior(torch.log(value),torch.ones_like(value))
        params = prior.sample()
        np.savetxt('synthetic_data/gt_map.csv', 
                   np.vstack((np.loadtxt('synthetic_data/gt_map.csv'),
                              np.hstack((np.array((seed)),params.cpu())))) )

    print(params)
    sim = SyntheticSimulator(params=params,
                             name=f'seed{seed}',
                             dil_schedule=20. * torch.pow(10, torch.arange(4)),
                             device=device)

    sim.sample_save(size=100, Ts=torch.arange(5)*48 + 24)


def integrate_mass_action(params, Ts, dt=0.1,device='cpu'):
    """
    Simulate ODE trajectories from E=0 using Euler method.
    """
    S_ma = S.to(device)
    E = torch.zeros(1,device=device)
    t = 0

    dEdt = lambda Ex: (get_rates(Ex,params.reshape(-1,1))*S_ma).sum(axis=1)
    
    Es = []
    for T in Ts:
        while t + dt <T:
            E = E + dEdt(E)*dt
            t += dt

        E = E + dEdt(E)*(T-t)
        t = T

        Es.append(E*1)
    return torch.stack(Es)
