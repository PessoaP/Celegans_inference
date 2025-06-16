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

# ---- Proposal Function ----
def proposal(th, L):
    lth = torch.log(th)
    noise = torch.randn_like(th)

    lprop = lth + noise * L


    return torch.exp(lprop)

# ---- Single MCMC Step ----
def next_MCMC_sample(logposterior, params, lp, L, greedy=False):
    if (torch.rand(1)).item() <.5:
        lp = logposterior(params)
    params_prop = proposal(params, L)
    lp_prop = logposterior(params_prop)

    if greedy:
        accept = lp_prop > lp
    else:
        accept = torch.log(torch.rand(1)).item() < (lp_prop - lp).item()

    return (params_prop, lp_prop) if accept else (params, lp)
