import os
import torch
import numpy as np
from simulator import ConstrainedLogNormalPrior

# === Prior Construction from ODE Initialization ===
def make_prior_from_initial_guess(dataset, frac_error=0.5, device=None):
    """
    Construct a log-normal prior centered at ODE-initialized parameter estimates,
    with fractional uncertainty in linear space.

    Parameters:
        dataset (TimeSeriesInferenceDataset): Combined dataset across all days
        frac_error (float): Relative uncertainty in linear space (e.g., 0.5 means ±50%)
        device (torch.device or None): Optionally move prior parameters to a device

    Returns:
        prior (ConstrainedLogNormalPrior): Prior centered on ODE init with log-scale stddev
        init_guess (torch.Tensor): The ODE-initialized parameter vector
    """
    default_guess = torch.tensor([1/20, 1/4, 2e5, 0.1], dtype=torch.float32, device=device)  # hardcoded for 4 parameters
    baseline_prior = ConstrainedLogNormalPrior(torch.log(default_guess), torch.ones_like(default_guess))

    init_guess = dataset.ode_initialization(baseline_prior.sample()).detach()
    if not torch.all(init_guess > 0):
        init_guess = torch.clamp(init_guess, min=1e-8) # make sure all the parameter values are >0
    if device is not None:
        init_guess = init_guess.to(device)

    logmean = torch.log(init_guess)
    logstd  = torch.log(torch.tensor(1 + frac_error, device=device)) * torch.ones_like(logmean)
    prior = ConstrainedLogNormalPrior(logmean, logstd)
    return prior, init_guess

# === Logposterior Constructor ===
def make_logposterior(data, prior):
    def logposterior(params):
        return data.loglike(params) + prior.log_prob(params).sum()
    return logposterior

# === Proposal Function ===
def proposal(th, L):
    lth = torch.log(th)
    noise = torch.randn_like(th)
    lprop = lth + noise @ L.T
    return torch.exp(lprop)

# === Covariance Adapter ===
def adapt_covariance(sample_history, epsilon=1e-3, min_samples=100, fill_std=1e-2, output_dir=None):
    # Ensure tensor type first
    if isinstance(sample_history, np.ndarray):
        sample_history = torch.from_numpy(sample_history).float()
    
    # Check shape and device
    N, d = sample_history.shape
    d = int(d)
    N = int(N)
    device = sample_history.device if hasattr(sample_history, "device") else "cpu"

    assert torch.all(sample_history > 0), "Sample history contains non-positive values!"
    log_samples = torch.log(sample_history)

    if output_dir is not None:
        np.savetxt(os.path.join(output_dir, 'prefill_cholesky.csv'), log_samples.cpu().numpy())

    if N < min_samples:
        mean = log_samples.mean(dim=0)
        extra = mean + fill_std * torch.randn((min_samples - N, d), device=device)
        log_samples = torch.vstack([log_samples, extra])

    # Defensive: after filling, make sure log_samples is float32 (for torch.cov on GPU)
    log_samples = log_samples.float()

    cov = torch.cov(log_samples.T)
    scaling = (2.4 ** 2) / d
    scaled_cov = scaling * cov + epsilon * torch.eye(d, device=device, dtype=log_samples.dtype)

    if output_dir is not None:
        np.savetxt(os.path.join(output_dir, 'postfill_cholesky.csv'), log_samples.cpu().numpy())

    return torch.linalg.cholesky(scaled_cov)

# === SamplerState Object ===
class SamplerState:
    def __init__(self, L, iter_num=0):
        self.L = L
        self.iter = iter_num

    def state_dict(self):
        return {'L': self.L, 'iter': self.iter}

    def load_state_dict(self, state):
        self.L = state['L']
        self.iter = state['iter']

# === MCMC Step ===
def next_MCMC_sample(logposterior, params, lp, state, greedy=False, adapt=False, sample_history=None, output_dir=None):
    state.iter += 1

    if adapt:
        state.L = adapt_covariance(sample_history, output_dir)

    if state.iter % 2 == 0:
        lp = logposterior(params)

    params_prop = proposal(params, state.L)
    lp_prop = logposterior(params_prop)

    if (
        not torch.isfinite(lp_prop)
        or not torch.all(torch.isfinite(params_prop))
        or not torch.all(params_prop > 0)   # make sure only positive proposal values are ever accepted 
    ):
        print(
            f"[Warning] Rejected bad proposal at iteration {state.iter}: "
            f"non-finite or non-positive values detected."
        )
        return params, lp, state, False

    accept = lp_prop > lp if greedy else torch.log(torch.rand(1)).item() < (lp_prop - lp).item()
    if accept:
        return params_prop, lp_prop, state, True
    return params, lp, state, False

