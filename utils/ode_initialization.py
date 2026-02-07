
import torch
import tqdm
from utils import simulator

def ode_initialization(self, init=None, iters=300, lr=0.03):
    """
    Rough CPU ODE init for prior centering. Returns (alpha, mu, d).
    """
    target = (self.counts * self.dils).detach().cpu().float().reshape(-1)
    Ts     = self.Ts.detach().cpu().float().reshape(-1)

    # --- initial guesses
    if init is None:
        alpha0 = torch.tensor(1/24, dtype=torch.float32)
        r0     = torch.tensor(1/6,  dtype=torch.float32)   # mu-d ~ 0.166/h as a starting point
        d0     = torch.tensor(0.1,  dtype=torch.float32)
    else:
        init = init.detach().cpu().float()
        alpha0 = init[0]
        mu0    = init[1]
        d0     = init[2]
        r0     = (mu0 - d0).clamp_min(1e-3)

    # optimize in log space for positivity
    lalpha = torch.log(alpha0.clamp_min(1e-12)).requires_grad_()
    lr_    = torch.log(r0.clamp_min(1e-12)).requires_grad_()
    ld     = torch.log(d0.clamp_min(1e-12)).requires_grad_()

    opt = torch.optim.Adam([lalpha, lr_, ld], lr=lr)

    # time weights: downweight earliest points (more stochastic colonization)
    # e.g. weight ~ (t / t_max)^2
    tmax = Ts.max().clamp_min(1e-6)
    w = (Ts / tmax) ** 2
    w = w / w.mean().clamp_min(1e-12)

    eps = 1.0 

    best = None
    best_loss = float("inf")

    k_fixed = torch.median(self.kappa_samples.detach().cpu()).clamp_min(1.0)

    for it in tqdm(range(iters), desc="ODE init", leave=False):
        opt.zero_grad()

        alpha = torch.exp(lalpha)
        r     = torch.exp(lr_)          # r = mu - d
        d     = torch.exp(ld)
        mu    = r + d

        theta_ode = torch.stack([alpha, mu, k_fixed, d])  # (4,)
        n_ode = simulator.integrate_mass_action(theta_ode, Ts, dt=0.1, device='cpu').reshape(-1)
        n_ode = n_ode.clamp_min(eps)

        # relative error, weighted toward later times
        rel = (target / n_ode - 1.0)
        loss = torch.sqrt((w * rel * rel).mean())

        if torch.isfinite(loss):
            loss.backward()
            opt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best = torch.stack([alpha, mu, d]).detach().clone()

    if best is None:
        best = torch.stack([alpha.detach(), (r+d).detach(), d.detach()])

    return best.to(self.device)
