import os
import torch
import numpy as np
from simulator import ConstrainedLogNormalPrior

# === Prior Construction from ODE Initialization (CPU) ===
def make_prior_from_initial_guess(dataset, frac_error=0.5, device=None):
    # fixed default on CPU
    default_guess = torch.tensor([1/20, 1/4, 2e5, 0.1], dtype=torch.float32, device='cpu')

    # ODE init on CPU from a fixed point (no randomness)
    init_guess = dataset.ode_initialization(default_guess).detach()
    init_guess = torch.clamp(init_guess, min=1e-8)

    if device is not None:
        init_guess = init_guess.to(device)

    logmean = torch.log(init_guess)
    logstd_val = torch.log(torch.tensor(1 + frac_error, dtype=logmean.dtype, device=logmean.device))
    logstd = torch.full_like(logmean, logstd_val)
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
        theta = to_theta(u)    # exponentiate to ensure positivity
        # guard invalid θ before model/prior
        if torch.any(theta <= 0) or torch.any(~torch.isfinite(theta)):
            return torch.tensor(-float('inf'), device=u.device)
            
        lp = prior.log_prob(theta).sum()  # log-prior evaluated in θ-space
        if not torch.isfinite(lp):
            return torch.tensor(-float('inf'), device=u.device)
            
        ll = data.loglike(theta)  # log-likelihood at the θ that corresponds to current u
        if not torch.isfinite(ll):
            return torch.tensor(-float('inf'), device=u.device)
        jacobian_correction = logabsdet_J_exp(u) # Jacobian correction for changing variables from u to θ
        return lp + ll + jacobian_correction
    return logposterior_u

# === Proposals in u-space (full & block) ===
def full_proposal(u, L):
    z = torch.randn_like(u)          # z ~ N(0, I_d)
    return u + L @ z                 # ~ N(u, Σ)

def block_proposal_u(u, L, update_idx):
    if not update_idx:  # None or empty
        return full_proposal(u, L) # default is to infer all 4 parameters if no subset specified. Faster.
    up  = u.clone()
    idx = torch.as_tensor(update_idx, device=u.device, dtype=torch.long)
    Lb  = L.index_select(0, idx).index_select(1, idx)    # (k,k)
    zb  = torch.randn(idx.numel(), device=u.device)
    up.index_add_(0, idx, (zb @ Lb.T))
    return up

# === Adapt in u-space ===
def adapt_covariance_u(u_history,
                       epsilon=1e-3,
                       min_samples=100,
                       fill_std=1e-2,
                       max_jitter_tries=6,
                       output_dir=None):
    """
    Return lower-triangular Cholesky L for a proposal in u-space.
    u_history: (N,d) tensor/ndarray of past u-samples.
    """
    # --- to tensor
    if isinstance(u_history, np.ndarray):
        u_history = torch.from_numpy(u_history)
    u_history = u_history.detach().to(torch.float32)

    device = u_history.device
    N, d = u_history.shape

    # --- remove non-finite rows
    mask_finite = torch.isfinite(u_history).all(dim=1)
    uh = u_history[mask_finite]

    # --- if nothing/too little is finite, synthesize around 0 with floor std
    if uh.shape[0] == 0:
        uh = torch.zeros((1, d), dtype=torch.float32, device=device)

    # --- compute per-dim std on the finite part (fallback to fill_std)
    with torch.no_grad():
        mu = uh.mean(dim=0)
        sd = uh.std(dim=0, unbiased=True)
        sd = torch.where(torch.isfinite(sd), sd, torch.zeros_like(sd))
        sd = torch.clamp(sd, min=fill_std)

    # --- ensure at least min_samples rows by prefilling around mean
    if uh.shape[0] < min_samples:
        need = min_samples - uh.shape[0]
        # per-dimension noise, not isotropic
        extra = mu + sd * torch.randn((need, d), device=device, dtype=uh.dtype)
        uh = torch.vstack([uh, extra])

    # --- covariance in float64, then scale + regularize
    uh64 = uh.to(torch.float64)
    cov = torch.cov(uh64.T)  # (d,d), unbiased N-1 normalization
    # numerical symmetrization
    cov = 0.5 * (cov + cov.T)

    # scale factor (Roberts/Gelman optimal for RW in d dims)
    scaling = (2.4 ** 2) / float(d)

    # base regularization magnitude relative to variance scale
    # use median diag to set epsilon scale if tiny
    diag_cov = torch.diag(cov)
    med = torch.median(torch.clamp(diag_cov, min=1e-16)).item()
    base_eps = max(epsilon, 1e-6 * med)

    C = scaling * cov + base_eps * torch.eye(d, dtype=torch.float64, device=device)

    # --- clamp tiny/negative diag and jitter escalations
    # floor diag to avoid exact zeros
    diag = torch.diag(C).clone()
    diag = torch.clamp(diag, min=max(1e-12, 1e-4 * med))
    C[range(d), range(d)] = diag

    # escalate jitter until PD
    eye = torch.eye(d, dtype=torch.float64, device=device)
    for i in range(max_jitter_tries):
        try:
            L = torch.linalg.cholesky(C)
            return L.to(dtype=u_history.dtype, device=device)
        except RuntimeError:
            # add multiplicative jitter each try
            jitter = (10.0 ** i) * max(1e-12, 1e-4 * med)
            C = C + jitter * eye

    # final fallback: diagonal proposal
    C = torch.diag(torch.clamp(torch.diag(C), min=max(1e-10, 1e-6 * med)))
    L = torch.linalg.cholesky(C)
    return L.to(dtype=u_history.dtype, device=device)


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
    Takes 
    Returns (u_next, lp_u_next, state, accepted_bool)
    """
    state.iter += 1

    if adapt and (u_history is not None):
        state.L = adapt_covariance_u(u_history, output_dir=output_dir)

    # Always align L to u
    state.L = state.L.to(dtype=u.dtype, device=u.device)

    # Calculate log-posterior of the proposed step
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
