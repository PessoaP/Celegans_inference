import torch

#Define log posterior and check up in the lp_gt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


value = torch.tensor((1/20, #alpha
                      1/4, #mu
                      2*1e5, #k
                      .1 #d
                      )).to(device) 

prior = torch.distributions.LogNormal(torch.log(value),torch.ones_like(value))
logposterior = lambda value,data: data.loglike(value)  + prior.log_prob(value).sum()

L = 1e-2 *torch.diag(torch.tensor((1,1,1,5),device=device))
total_iter = 0
# ---- Proposal Function ----

def proposal(th, L):
    lth = torch.log(th)
    noise = torch.randn_like(th)
    lprop = lth + noise @ L.T  # use matrix product if L is a matrix
    return torch.exp(lprop)

# ---- Covariance Adapter ----
def adapt_covariance(sample_history, epsilon=1e-6, min_samples=100, fill_std=1e-2):
    """
    Adapt covariance using log-space empirical samples.

    Parameters:
        sample_history (torch.Tensor): Tensor of shape (N, d), in original (not log) space.
        epsilon (float): Stability jitter for Cholesky.
        min_samples (int): Minimum number of samples to build a reliable covariance.
        fill_std (float): Std of Gaussian noise used to fill when n < min_samples.

    Returns:
        torch.Tensor: Cholesky factor (L) of scaled covariance matrix.
    """
    N, d = sample_history.shape
    log_samples = torch.log(sample_history)

    if N < min_samples:
        mean = log_samples.mean(dim=0)
        extra = mean + fill_std * torch.randn((min_samples - N, d), device=sample_history.device)
        log_samples = torch.vstack([log_samples, extra], dim=0)

    cov = torch.cov(log_samples.T)
    scaling = (2.4 ** 2) / d
    scaled_cov = scaling * cov + epsilon * torch.eye(d, device=sample_history.device)

    return torch.linalg.cholesky(scaled_cov)

# ---- Single MCMC Step with Optional Adaptation ----
def next_MCMC_sample(logposterior, params, lp, greedy=False, adapt=False, sample_history=None):
    global L 
    global total_iter
    total_iter +=1

    if adapt:
        assert sample_history is not None and len(sample_history) > 1, "Need history for adaptation"
        L = adapt_covariance(sample_history)

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
