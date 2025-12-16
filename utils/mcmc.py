import os
import torch
import numpy as np

# === Prior Construction from ODE Initialization (CPU) ===
class ConstrainedLogNormalPrior:
    def __init__(self, loc, scale):
        self.base = torch.distributions.LogNormal(loc, scale)

    def log_prob(self, x):
        if x[1] < x[3]:
            return torch.tensor(float('-inf'), device=x.device)
        return self.base.log_prob(x).sum()

    def sample(self):
        for _ in range(1000):
            x = self.base.sample().to(self.base.loc.device)
            if x[1] >= x[3]:
                return x
        raise RuntimeError("Failed to sample satisfying x[1] >= x[3] after 100 attempts.")

def make_prior_from_initial_guess(dataset, frac_error=0.5, device=None, min_logstd=1e-2):
    """
    Build a lognormal prior around the ODE initialization.
    - frac_error: scalar or length-4 iterable. Interpreted as multiplicative std: logstd = log(1 + frac_error).
    - min_logstd: small floor to avoid over-tight priors.
    """
    # fixed default on CPU for deterministic ODE init
    default_guess = torch.tensor([1/20, 1/4, 2e5, 0.1], dtype=torch.float32, device='cpu')

    # ODE init from a fixed point (no randomness)
    init_guess = dataset.ode_initialization(default_guess).detach()
    init_guess = torch.clamp(init_guess, min=1e-12)

    if device is not None:
        init_guess = init_guess.to(device)

    logmean = torch.log(init_guess)

    # allow scalar or per-parameter values
    if np.isscalar(frac_error):
        frac_error = [float(frac_error)] * logmean.numel()
    frac_error = torch.as_tensor(frac_error, dtype=logmean.dtype, device=logmean.device)

    # multiplicative std in log-space
    logstd = torch.log1p(frac_error).clamp_min(min_logstd)

    prior = ConstrainedLogNormalPrior(logmean, logstd)
    return prior, init_guess


# === Space transforms ===
def to_u(theta):      # θ -> u
    return torch.log(theta)

def to_theta(u):      # u -> θ
    return torch.exp(u)

def logabsdet_J_exp(u):
    # θ = exp(u): diag(exp(u_i)) => log|J| = sum u_i
    return u.sum()


# === Target in u-space (posterior ∘ exp + Jacobian) ===
def make_logposterior_u(data, prior, debug=False):
    def logposterior_u(u):
        theta = to_theta(u)                      # positivity
        if torch.any(theta <= 0) or torch.any(~torch.isfinite(theta)):
            return torch.tensor(-float('inf'), device=u.device)

        lp = prior.log_prob(theta)         # prior in θ-space
        ll = data.loglike(theta)           # likelihood at θ(u)
        jac = logabsdet_J_exp(u)           # Jacobian for θ = exp(u)
        
        if debug:
            print("theta:", theta.detach().cpu().numpy())
            print("  lp (prior):", float(lp))
            print("  ll (likelihood):", float(ll))
            print("  jacobian:", float(jac))
            print("  total:", float(lp + ll + jac))
            print("-" * 40)

        if not torch.isfinite(lp) or not torch.isfinite(ll):
            return torch.tensor(-float('inf'), device=u.device)

        return lp + ll + jac
    return logposterior_u


# === Proposals in u-space (full & block) with step scaling ===
def full_proposal(u, L, step_scale=1.0):
    """
    Random-walk proposal: u' = u + (step_scale * L) @ N(0, I).\
    L: lower-triangular Cholesky of covariance in u-space.
    step_scale: scalar multiplier for step size.
    Used for simultaneous updates of all parameters.
    """
    z = torch.randn_like(u)
    return u + (L * step_scale) @ z

def block_proposal_u(u, L, update_idx, step_scale=1.0):
    """
    Block/coordinate proposal on indices update_idx (or full if None).
    Used to only update a subset of parameters per MCMC step.
    """
    if update_idx is None or len(update_idx) == 0:  # None or empty -> full update
        return full_proposal(u, L, step_scale)

    up  = u.clone()
    idx = torch.as_tensor(update_idx, device=u.device, dtype=torch.long)
    Lb  = L.index_select(0, idx).index_select(1, idx)    # (k,k)
    zb  = torch.randn(idx.numel(), device=u.device)      # (k,)
    up[idx] = up[idx] + (Lb * step_scale) @ zb           # (k,)
    return up

# Create a mixture proposal to help escape stickiness 
def propose_mixture(u, L, update_idx=None, small=0.8, big=2.0, p_big=0.12):
    """
    Mixture-of-scales proposal:
      with prob (1 - p_big): step_scale = small
      with prob p_big:       step_scale = big
    """
    step_scale = big if (torch.rand((), device=u.device) < p_big) else small
    return block_proposal_u(u, L, update_idx, step_scale=step_scale)


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
        extra = mu + sd * torch.randn((need, d), device=device, dtype=uh.dtype)
        uh = torch.vstack([uh, extra])

    # --- covariance in float64, then scale + regularize
    uh64 = uh.to(torch.float64)
    cov = torch.cov(uh64.T)  # (d,d), unbiased N-1 normalization
    cov = 0.5 * (cov + cov.T)  # symmetrize

    scaling = (2.4 ** 2) / float(d)

    diag_cov = torch.diag(cov)
    med = torch.median(torch.clamp(diag_cov, min=1e-16)).item()
    base_eps = max(epsilon, 1e-6 * med)   # to make sure step size doesn't collapse

    C = scaling * cov + base_eps * torch.eye(d, dtype=torch.float64, device=device)

    # clamp tiny/negative diag and jitter escalations
    diag = torch.diag(C).clone()
    diag = torch.clamp(diag, min=max(1e-12, 1e-4 * med))
    C[range(d), range(d)] = diag

    eye = torch.eye(d, dtype=torch.float64, device=device)
    for i in range(max_jitter_tries):
        try:
            L = torch.linalg.cholesky(C)
            return L.to(dtype=u_history.dtype, device=device)
        except RuntimeError:
            jitter = (10.0 ** i) * max(1e-12, 1e-4 * med)
            C = C + jitter * eye # add jitter until Cholesky works

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
                       output_dir=None, update_idx=None,
                       proposal_small=0.8, proposal_big=2.0, proposal_p_big=0.12):
    """
    One MH step with state carried in u-space.
    Uses a mixture-of-scales random-walk proposal in u.
      - proposal_small: typical (small) step scale (multiplies Cholesky L)
      - proposal_big:   occasional larger step scale
      - proposal_p_big: probability of taking the big step
    Returns (u_next, lp_u_next, state, accepted_bool)
    """
    state.iter += 1

    # optional covariance adaptation
    if adapt and (u_history is not None):
        state.L = adapt_covariance_u(u_history, output_dir=output_dir)

    # align L to u
    state.L = state.L.to(dtype=u.dtype, device=u.device)

    # propose (coordinate or full) with a small/large mixture
    up = propose_mixture(u, state.L, update_idx,
                         small=proposal_small, big=proposal_big, p_big=proposal_p_big)

    # log-posterior at proposal
    lp_up = logposterior_u(up)
    if not torch.isfinite(lp_up):
        return u, lp_u, state, False

    # accept / reject
    if greedy:
        accept = (lp_up > lp_u).item()
    else:
        logu = torch.log(torch.rand((), device=u.device))
        accept = (logu < (lp_up - lp_u)).item()

    if accept:
        return up, lp_up, state, True
    else:
        return u, lp_u, state, False
