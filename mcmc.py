import torch
from simulator import ConstrainedLogNormalPrior

#Define log posterior and check up in the lp_gt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


value = torch.tensor((1/20, #alpha
                      1/4, #mu
                      2*1e5, #k
                      .1 #d
                      )).to(device) 

prior = ConstrainedLogNormalPrior(torch.log(value),torch.ones_like(value))
logposterior = lambda value,data: data.loglike(value)  + prior.log_prob(value).sum()

#L = 1e-2 *torch.diag(torch.tensor((1,1,1,5),device=device))
L = 1e-2 *torch.tensor((1,1,1,5),device=device)
total_iter = 0

# ---- Forward & inverse transforms -------------------------------------------
def _forward_transform(theta):
    """
    theta: (..., 4) = (colonization, replication, capacity, expulsion)
    Returns tuple of tensors (u0, d, u2, r) where:
      u0 = log(colonization)
       d = replication - expulsion
      u2 = log(capacity)
       r = log(replication / expulsion)
    """
    th0, th1, th2, th3 = theta[..., 0], theta[..., 1], theta[..., 2], theta[..., 3]
    u0 = torch.log(th0)        # colonization in log-space
    d  = th1 - th3             # replication - expulsion
    u2 = torch.log(th2)        # capacity in log-space
    r  = torch.log(th1) - torch.log(th3)  # log(replication / expulsion)
    return (u0, d, u2, r)

def _inverse_transform(u0, d, u2, r, eps=1e-12):
    """
    Map (u0, d, u2, r) back to (th0, th1, th2, th3) =
    (colonization, replication, capacity, expulsion).

    Uses expm1 for numerical stability when r ~ 0.
    """
    th0 = torch.exp(u0)   # colonization
    th2 = torch.exp(u2)   # capacity

    # Recover replication (th1) and expulsion (th3) from d and r:
    er_minus_1 = torch.expm1(r)
    small = er_minus_1.abs() < eps
    denom = torch.where(small, r, er_minus_1)  # e^r - 1 ~ r when r ~ 0

    th3 = d / denom             # expulsion
    th1 = th3 * torch.exp(r)    # replication

    return torch.stack([th0, th1, th2, th3], dim=-1)

# ---- Proposal in transformed space ------------------------------------------
def proposal(theta, L):
    """
    Random-walk proposal using a Gaussian in transformed coords.

    theta: (..., 4) = (colonization, replication, capacity, expulsion)
    L: (4, 4) Cholesky-like factor; covariance in transformed space is Σ = L L^T

    Returns: tuple (th0', th1', th2', th3') =
             (colonization', replication', capacity', expulsion')
    """
    u0, d, u2, r = _forward_transform(theta)

    # Add independent Gaussian noise to each transformed variable
    u0p = u0 + L[0] * torch.randn_like(u0)  # colonization (log)
    dp  = d  + L[1] * torch.randn_like(d)   # replication - expulsion
    u2p = u2 + L[2] * torch.randn_like(u2)  # capacity (log)
    rp  = r  + L[3] * torch.randn_like(r)   # log(replication/expulsion)

    return _inverse_transform(u0p, dp, u2p, rp)



# # ---- Covariance Adapter ----
# def adapt_covariance(sample_history, epsilon=1e-3, min_samples=100, fill_std=1e-2):
#     """
#     Adapt covariance using log-space empirical samples.

#     Parameters:
#         sample_history (torch.Tensor): Tensor of shape (N, d), in original (not log) space.
#         epsilon (float): Stability jitter for Cholesky.
#         min_samples (int): Minimum number of samples to build a reliable covariance.
#         fill_std (float): Std of Gaussian noise used to fill when n < min_samples.

#     Returns:
#         torch.Tensor: Cholesky factor (L) of scaled covariance matrix.
#     """
#     N, d = sample_history.shape
#     log_samples = torch.log(sample_history)

#     if N < min_samples:
#         mean = log_samples.mean(dim=0)
#         extra = mean + fill_std * torch.randn((min_samples - N, d), device=sample_history.device)
#         log_samples = torch.vstack([log_samples, extra])

#     cov = torch.cov(log_samples.T)
#     scaling = (2.4 ** 2) / d
#     scaled_cov = scaling * cov + epsilon * torch.eye(d, device=sample_history.device)

#     return torch.linalg.cholesky(scaled_cov)

# ---- Single MCMC Step with Optional Adaptation ----
def next_MCMC_sample(logposterior, params, lp, greedy=False, adapt=False, sample_history=None):
    global L 
    global total_iter
    total_iter +=1

    # if adapt:
    #     L = adapt_covariance(sample_history)

    if total_iter%2 == 0:
        lp = logposterior(params)

    params_prop = proposal(params, L)
    lp_prop = logposterior(params_prop)

    if greedy:
        accept = lp_prop > lp
    else:
        accept = torch.log(torch.rand(1)).item() < (lp_prop - lp).item()
    if accept:
        return (params_prop, lp_prop)
    return (params, lp)
