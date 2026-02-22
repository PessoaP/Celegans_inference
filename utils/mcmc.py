import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


#setting up prior
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gamma_logpdf(x, k, rate):
    return (k * torch.log(rate) 
            - torch.lgamma(torch.tensor(k, dtype=x.dtype, device=x.device))
            + (k - 1) * torch.log(x) - rate * x )

def beta_logpdf(x, a, b):
    logB = torch.lgamma(a) + torch.lgamma(b)- torch.lgamma(a+b)
    
    return (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x) - logB
    

gamma_shape_alpha=2*torch.ones(1, device=device)  
gamma_rate_alpha=.1*torch.ones(1, device=device) 
gamma_shape_mu=2*torch.ones(1, device=device) 
gamma_rate_mu=.1*torch.ones(1, device=device) 
beta_a=1.5*torch.ones(1, device=device) 
beta_b=1.5*torch.ones(1, device=device)

def logprior(alpha, mu, omega):
    return (gamma_logpdf(alpha, gamma_shape_alpha, gamma_rate_alpha) +
            gamma_logpdf(mu, gamma_shape_mu, gamma_rate_mu) +
            beta_logpdf(omega, beta_a, beta_b))

#likelihood function for MCMC (takes in alpha, mu, omega and computes d, then calls dataset.loglike)
def loglike_alpha_mu_omega(dataset, alpha, mu, omega):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    omega = torch.as_tensor(omega, device=dataset.device, dtype=torch.float32)

    d = omega * mu
    theta_phys = torch.stack([alpha, mu, d])
    with torch.no_grad():
        return dataset.loglike(theta_phys)
    

#proposal function
def reflect_unit_interval_scalar(x):
    y = torch.remainder(x, 2.0)  # now in [0,2)
    if y <= 1.0:
        return y
    else:
        return 2.0 - y

def proposal(alpha,mu,omega, jumpsize_proposal):

    # --- log RW for positive params ---
    dlog_alpha = jumpsize_proposal[0] * torch.randn_like(alpha)
    dlog_mu    = jumpsize_proposal[1] * torch.randn_like(mu)

    alpha_prop = torch.exp(torch.log(alpha) + dlog_alpha)
    mu_prop    = torch.exp(torch.log(mu)    + dlog_mu)

    # --- symmetric RW in omega-space + reflection ---
    domega = jumpsize_proposal[2] * torch.randn_like(omega)
    omega_prop = reflect_unit_interval_scalar(omega + domega)

    # --- MH correction (only log-space params need it) ---
    log_q_ratio = dlog_alpha + dlog_mu

    return torch.stack((alpha_prop, mu_prop, omega_prop)), log_q_ratio

def next_MCMC_sample(logposterior, theta, lp, jumpsize_proposal):
    th_prop, log_q_ratio = proposal(*theta,jumpsize_proposal)
    lp_prop = logposterior(th_prop)
    logA = (lp_prop - lp) + log_q_ratio 
    if  torch.log(torch.rand(())).item() < logA.item():
        return th_prop, lp_prop
    else:
        return theta, lp
    
