"""
mcmc.py

Annealed Metropolis–Hastings in log-space (u = log(theta)) for positive parameters.

Key features:
- Start from user-chosen theta0 (no ODE init)
- Fixed dataset internals (kappa_samples, rho, t_switch) baked into dataset construction
- Annealing: beta ramps from beta0 -> 1 over warmup_frac * n_steps
- No adaptive covariance
- Simple checkpointing + resume
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import torch


# ============================================================
# Priors
# ============================================================

class LogNormalPrior:
    def __init__(self, logmean: torch.Tensor, logstd: torch.Tensor):
        self.logmean = logmean
        self.logstd = logstd
        self.base = torch.distributions.LogNormal(self.logmean, self.logstd)

    def to(self, device: Union[str, torch.device]) -> "LogNormalPrior":
        device = torch.device(device)
        self.logmean = self.logmean.to(device)
        self.logstd = self.logstd.to(device)
        self.base = torch.distributions.LogNormal(self.logmean, self.logstd)
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.log_prob(x).sum()


def make_lognormal_prior_centered(
    center_theta: torch.Tensor,
    frac_error: Union[float, Sequence[float]] = 0.7,
    min_logstd: float = 1e-2,
) -> LogNormalPrior:
    """
    LogNormal prior centered at center_theta with multiplicative uncertainty.
    """
    center_theta = torch.as_tensor(center_theta, dtype=torch.float32).clamp_min(1e-12)
    logmean = torch.log(center_theta)

    if np.isscalar(frac_error):
        frac_error = [float(frac_error)] * logmean.numel()
    frac_error = torch.as_tensor(frac_error, dtype=logmean.dtype)

    logstd = torch.log1p(frac_error).clamp_min(min_logstd)
    return LogNormalPrior(logmean=logmean, logstd=logstd)


# ============================================================
# Transforms
# ============================================================

def to_u(theta: torch.Tensor) -> torch.Tensor:
    return torch.log(theta)

def to_theta(u: torch.Tensor) -> torch.Tensor:
    return torch.exp(u)

def logabsdet_J_exp(u: torch.Tensor) -> torch.Tensor:
    return u.sum()


# ============================================================
# Annealing schedule
# ============================================================

def make_beta_schedule(n_steps: int, warmup_frac: float = 0.35, beta0: float = 0.05) -> torch.Tensor:
    """
    Linear ramp beta0 -> 1 over warmup portion; then stays at 1.
    """
    n_warm = max(1, int(warmup_frac * n_steps))
    betas = torch.ones(n_steps, dtype=torch.float32)
    betas[:n_warm] = torch.linspace(float(beta0), 1.0, n_warm)
    return betas


# ============================================================
# Proposals in u-space
# ============================================================

def full_proposal(u: torch.Tensor, L: torch.Tensor, step_scale: float = 1.0) -> torch.Tensor:
    z = torch.randn_like(u)
    return u + (L * step_scale) @ z

def propose_mixture(
    u: torch.Tensor,
    L: torch.Tensor,
    small: float = 0.8,
    big: float = 2.0,
    p_big: float = 0.12,
) -> torch.Tensor:
    step_scale = big if (torch.rand((), device=u.device) < p_big) else small
    return full_proposal(u, L, step_scale=step_scale)


# ============================================================
# Log-posterior in u-space
# ============================================================

def make_logposterior_u_annealed(dataset, prior: LogNormalPrior, betas: torch.Tensor, debug: bool = False):
    """
    dataset: your TimeSeriesInferenceDataset-like object with:
        dataset.loglike(theta_phys) -> scalar Tensor
    """
    def logposterior_u(u: torch.Tensor, i: int) -> torch.Tensor:
        theta = to_theta(u)  # theta_phys
        if torch.any(theta <= 0) or torch.any(~torch.isfinite(theta)):
            return torch.tensor(-float("inf"), device=u.device)

        lp = prior.log_prob(theta)
        ll = dataset.loglike(theta)
        jac = logabsdet_J_exp(u)
        beta = betas[i]

        if debug:
            print(f"[i={i}] beta={float(beta):.4f} theta={theta.detach().cpu().numpy()} lp={float(lp)} ll={float(ll)} jac={float(jac)}")

        if (not torch.isfinite(lp)) or (not torch.isfinite(ll)):
            return torch.tensor(-float("inf"), device=u.device)

        return lp + beta * ll + jac

    return logposterior_u


@torch.no_grad()
def next_mh_step_annealed(
    logposterior_u,
    u: torch.Tensor,
    lp_u: torch.Tensor,
    i: int,
    L: torch.Tensor,
    proposal_small: float = 0.8,
    proposal_big: float = 2.0,
    proposal_p_big: float = 0.12,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    up = propose_mixture(u, L, small=proposal_small, big=proposal_big, p_big=proposal_p_big)
    lp_up = logposterior_u(up, i)
    if not torch.isfinite(lp_up):
        return u, lp_u, False

    logu = torch.log(torch.rand((), device=u.device))
    accept = (logu < (lp_up - lp_u)).item()

    if accept:
        return up, lp_up, True
    return u, lp_u, False


# ============================================================
# Runner + checkpointing
# ============================================================

@dataclass
class MCMCConfig:
    n_steps: int = 50_000
    warmup_frac: float = 0.35
    beta0: float = 0.05

    prior_frac_error: float = 0.7
    prior_min_logstd: float = 1e-2

    init_step_u: float = 0.15
    proposal_small: float = 0.8
    proposal_big: float = 2.0
    proposal_p_big: float = 0.12

    save_every: int = 1000
    out_dir: str = "mcmc_out"
    run_name: str = "chain"

    # If True, store u + logpost to disk as we go (recommended for long runs)
    checkpoint: bool = True


def _default_device(device: Union[str, torch.device]) -> torch.device:
    d = torch.device(device)
    if d.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return d


def checkpoint_paths(out_dir: str, run_name: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{run_name}.pt")
    tmp_path = os.path.join(out_dir, f"{run_name}.tmp.pt")
    return ckpt_path, tmp_path


def save_checkpoint_atomic(path: str, tmp_path: str, payload: Dict[str, Any]) -> None:
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, map_location: Optional[Union[str, torch.device]] = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def run_mcmc_annealed(
    dataset,
    theta0_phys: Sequence[float],
    config: Optional[MCMCConfig] = None,
    device: Union[str, torch.device] = "cuda",
    resume_from: Optional[str] = None,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float, str]:
    """
    Returns:
      thetas_cpu: (n_saved, d) tensor on CPU
      logpost_cpu: (n_saved,) tensor on CPU
      accept_rate: float
      ckpt_path: where the latest checkpoint is saved
    """
    if config is None:
        config = MCMCConfig()

    device = _default_device(device)

    ckpt_path, tmp_path = checkpoint_paths(config.out_dir, config.run_name)

    # --- initialize or resume
    if resume_from is not None:
        ck = load_checkpoint(resume_from, map_location="cpu")
        start_i = int(ck["i_next"])
        us = ck["u"].clone()              # CPU
        lps = ck["logpost"].clone()       # CPU
        acc = int(ck["acc"])
        theta_dim = int(us.shape[1])

        u = us[-1].to(device=device, dtype=torch.float32)
        lp_u = lps[-1].to(device=device, dtype=torch.float32)

        theta0 = torch.exp(us[0]).to(device=device, dtype=torch.float32)  # for metadata only
    else:
        start_i = 0
        theta0 = torch.as_tensor(theta0_phys, dtype=torch.float32, device=device).clamp_min(1e-12)
        theta_dim = int(theta0.numel())

        us = torch.empty((0, theta_dim), dtype=torch.float32, device="cpu")
        lps = torch.empty((0,), dtype=torch.float32, device="cpu")
        acc = 0

        u = to_u(theta0)
        lp_u = None  # set below after logpost defined

    # --- prior + betas (must be identical if resuming; we store config in ckpt)
    prior = make_lognormal_prior_centered(
        center_theta=theta0.detach().cpu(),
        frac_error=config.prior_frac_error,
        min_logstd=config.prior_min_logstd,
    ).to(device)

    betas = make_beta_schedule(config.n_steps, warmup_frac=config.warmup_frac, beta0=config.beta0).to(device)

    logpost_u = make_logposterior_u_annealed(dataset, prior, betas, debug=debug)

    if lp_u is None:
        lp_u = logpost_u(u, 0)

    # --- proposal Cholesky (diagonal)
    L = torch.eye(theta_dim, device=device, dtype=torch.float32) * float(config.init_step_u)

    # --- storage preallocation (on CPU, append in chunks)
    # We keep simple append-to-list semantics with occasional concatenation.
    u_list = [us] if us.numel() else []
    lp_list = [lps] if lps.numel() else []

    for i in range(start_i, config.n_steps):
        u, lp_u, accepted = next_mh_step_annealed(
            logposterior_u=logpost_u,
            u=u,
            lp_u=lp_u,
            i=i,
            L=L,
            proposal_small=config.proposal_small,
            proposal_big=config.proposal_big,
            proposal_p_big=config.proposal_p_big,
        )
        acc += int(accepted)

        u_list.append(u.detach().cpu().reshape(1, -1))
        lp_list.append(lp_u.detach().cpu().reshape(1))

        # checkpoint
        if config.checkpoint and config.save_every > 0:
            if ((i + 1) % config.save_every == 0) or ((i + 1) == config.n_steps):
                u_cat = torch.cat(u_list, dim=0)
                lp_cat = torch.cat(lp_list, dim=0)

                payload = {
                    "u": u_cat,
                    "logpost": lp_cat,
                    "acc": acc,
                    "i_next": i + 1,
                    "config": asdict(config),
                }
                save_checkpoint_atomic(ckpt_path, tmp_path, payload)

    # final concat
    u_cat = torch.cat(u_list, dim=0)
    lp_cat = torch.cat(lp_list, dim=0)
    thetas = torch.exp(u_cat)

    accept_rate = acc / float(config.n_steps)
    return thetas, lp_cat, accept_rate, ckpt_path
