import os
import torch
import numpy as np
from simulator import ConstrainedLogNormalPrior

# === Prior Construction from ODE Initialization (CPU) ===
def make_prior_from_initial_guess(dataset, frac_error=0.5, device=None):
    """
    Build a log-normal prior around the ODE-initialized guess.
    ODE initialization is forced on CPU; the result is moved to `device` afterward.
    """
    # default guess on CPU
    default_guess = torch.tensor([1/20, 1/4, 2e5, 0.1], dtype=torch.float32, device='cpu')
    baseline_prior = ConstrainedLogNormalPrior(torch.log(default_guess), torch.ones_like(default_guess))

    # ODE init on CPU
    init_guess = dataset.ode_initialization(baseline_prior.sample().cpu()).detach()
    if not torch.all(init_guess > 0):
        init_guess = torch.clamp(init_guess, min=1e-8)

    # move to target device if provided
    if device is not None:
        init_guess = init_guess.to(device)

    logmean = torch.log(init_guess)
    # Build logstd on same device/dtype as logmean
    logstd_val = torch.log(torch.tensor(1 + frac_error, dtype=logmean.dtype, device=logmean.device))
    logstd  = torch.full_like(logmean, logstd_val)
    prior = ConstrainedLogNormalPrior(logmean, logstd)
    return prior, init_guess

# === Space transforms ===
def to_u(theta):      # θ -> u
    return torch.log(theta)

def to_theta(u):      # u -> θ
    return torch.exp(u)

def logabsdet_J_exp(u):
    # θ = exp(u): diag(exp(u_i)) => log|J| = sum u_i  (scalar for vector u)
    return u.sum()

# === Target in u-space (posterior ∘ exp + Jacobian) ===
def make_logposterior_u(data, prior):
    def logposterior_u(u):
        theta = to_theta(u)
        # guard invalid θ before model/prior
        if torch.any(theta <= 0) or torch.any(~torch.isfinite(theta)):
            return torch.tensor(-float('inf'), device=u.device)
        lp = prior.log_prob(theta).sum()
        if not torch.isfinite(lp):
            return torch.tensor(-float('inf'), device=u.device)
        ll = data.loglike(theta)
        if not torch.isfinite(ll):
            return torch.tensor(-float('inf'), device=u.device)
        return lp + ll + logabsdet_J_exp(u)
    return logposterior_u

# === Proposals in u-space (full & block) ===
def proposal_u(u, L):
    """
    u: (d,), L: (d,d) Cholesky so that cov = L L^T in u-space
    """
    z = torch.randn_like(u)
    return u + (z @ L.T)

def block_proposal_u(u, L, update_idx):
    if not update_idx:  # None or empty
        return proposal_u(u, L)
    up  = u.clone()
    idx = torch.as_tensor(update_idx, device=u.device, dtype=torch.long)
    Lb  = L.index_select(0, idx).index_select(1, idx)    # (k,k)
    zb  = torch.randn(idx.numel(), device=u.device)
    up.index_add_(0, idx, (zb @ Lb.T))
    return up

# === Adapt in u-space ===
def adapt_covariance_u(u_history, epsilon=1e-3, min_samples=100, fill_std=1e-2, output_dir=None):
    """
    Adapts a Cholesky in u-space from a history of u-samples (rows = samples).
    """
    if isinstance(u_history, np.ndarray):
        u_history = torch.from_numpy(u_history).float()
    N, d = u_history.shape
    device = u_history.device if hasattr(u_history, "device") else "cpu"

    uh = u_history
    if N < min_samples:
        mean = uh.mean(dim=0)
        extra = mean + fill_std * torch.randn((min_samples - N, d), device=device)
        uh = torch.vstack([uh, extra])

    uh = uh.float()
    cov = torch.cov(uh.T)  # needs >= 2 rows; ensured by prefill
    scaling = (2.4 ** 2) / d
    scaled_cov = scaling * cov + epsilon * torch.eye(d, device=device, dtype=uh.dtype)
    L = torch.linalg.cholesky(scaled_cov)
    return L.to(dtype=u_history.dtype, device=u_history.device)

# === Sampler state ===
class SamplerState:
    def __init__(self, L, iter_num=0):
        self.L = L
        self.iter = iter_num
    def state_dict(self):
        return {'L': self.L, 'iter': self.iter}
    def load_state_dict(self, state):
        self.L = state['L']
        self.iter = state['iter']

# === Single MH step in u-space ===
def next_MCMC_sample_u(logposterior_u, u, lp_u, state,
                       greedy=False, adapt=False, u_history=None,
                       output_dir=None, update_idx=None):
    """
    One MH step with state carried in u-space.
    Returns (u_next, lp_u_next, state, accepted_bool)
    """
    state.iter += 1

    if adapt and (u_history is not None):
        state.L = adapt_covariance_u(u_history, output_dir=output_dir)

    # Always align L to u
    state.L = state.L.to(dtype=u.dtype, device=u.device)

    up = block_proposal_u(u, state.L, update_idx)
    lp_up = logposterior_u(up)

    if not torch.isfinite(lp_up):
        return u, lp_u, state, False

    if greedy:
        accept = (lp_up > lp_u).item()
    else:
        logu = torch.log(torch.rand((), device=u.device))
        accept = (logu < (lp_up - lp_u)).item()

    if accept:
        return up, lp_up, state, True
    else:
        return u, lp_u, state, False
