import torch

#Define log posterior and check up in the lp_gt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
value = torch.tensor( (1/20, 1/4, 2*1e5, 1e-3) ).to(device)
prior = torch.distributions.LogNormal(torch.log(value),torch.ones_like(value))
logposterior = lambda value,data: data.loglike(value) ## + prior.log_prob(value).sum()

# ---- Proposal Function ----
def proposal(th, L):
    lth = torch.log(th)
    noise = torch.randn_like(th)

    if isinstance(L, float):  # Scalar isotropic
        lprop = lth + noise * L
    else:  # Matrix L (Cholesky)
        lprop = lth + L @ noise

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

# # ---- Naive Hessian with coordinate-wise dx ----
# def naive_hessian_torch(foo, x, dx=None):
#     """
#     Central difference Hessian with coordinate-wise step sizes.

#     Parameters
#     ----------
#     foo : function
#         Scalar function foo(x), where x is a 1D torch tensor.
#     x : torch.Tensor
#         Point at which to evaluate the Hessian (1D tensor).
#     dx : torch.Tensor or float
#         Step sizes (same shape as x), default = 1e-4 * |x|

#     Returns
#     -------
#     H : torch.Tensor
#         Hessian matrix.
#     """
#     x = x.clone().detach()
#     n = x.numel()

#     if dx is None:
#         dx = (1e-4 * x.abs())
#     else:
#         dx = dx.clone().detach()

#     H = torch.zeros(n, n, dtype=torch.double)

#     for i in range(n):
#         for j in range(n):
#             dx_i, dx_j = dx[i], dx[j]

#             x_ijp = x.clone(); x_ijp[i] += dx_i; x_ijp[j] += dx_j
#             x_ijm = x.clone(); x_ijm[i] += dx_i; x_ijm[j] -= dx_j
#             x_imj = x.clone(); x_imj[i] -= dx_i; x_imj[j] += dx_j
#             x_imm = x.clone(); x_imm[i] -= dx_i; x_imm[j] -= dx_j

#             H[i, j] = (
#                 foo(x_ijp) - foo(x_ijm) - foo(x_imj) + foo(x_imm)
#             ) / (4 * dx_i * dx_j)

#     return H