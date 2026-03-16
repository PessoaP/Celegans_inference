import os, sys, csv, argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data

def parse_scalar_or_range(name: str):
    """
    Accepted formats:
      - "2.5"          -> float(2.5)
      - "0:3" or "0,3" -> (0.0, 3.0)
    """
    def _parse(s: str):
        s = s.strip()
        if ":" in s:
            lo, hi = s.split(":", 1)
            return (float(lo), float(hi))
        if "," in s:
            lo, hi = s.split(",", 1)
            return (float(lo), float(hi))
        return float(s)

    _parse.__name__ = f"scalar_or_range_{name}"
    return _parse

def is_range(x):
    return isinstance(x, tuple)


def proposal(mu, rho, mu_range=None, rho_range=None, jumpsize_proposal=0.01):
    """
    Propose new mu/rho.
    If a parameter range is None, that parameter is fixed.
    """
    if mu_range is not None:
        lo, hi = mu_range
        q_mu = (mu - lo) / (hi - lo)
        q_mu_prop = q_mu + jumpsize_proposal * torch.randn_like(q_mu)
        mu_prop = q_mu_prop * (hi - lo) + lo
    else:
        mu_prop = mu

    if rho_range is not None:
        lo, hi = rho_range
        q_rho = (rho - lo) / (hi - lo)
        q_rho_prop = q_rho + jumpsize_proposal * torch.randn_like(q_rho)
        rho_prop = q_rho_prop * (hi - lo) + lo
    else:
        rho_prop = rho

    return mu_prop, rho_prop


def prior(mu, rho, mu_range=None, rho_range=None):
    """
    Uniform prior over specified ranges.
    Fixed parameters (range=None) contribute constant 0.
    """
    # if mu_range is not None:
    #     lo, hi = mu_range
    #     if mu.item() < lo or mu.item() > hi:
    #         return -float("inf")

    # if rho_range is not None:
    #     lo, hi = rho_range
    #     if rho.item() < lo or rho.item() > hi:
    #         return -float("inf")

    return 0

def maybe_scalar_string(x):
    if is_range(x):
        return f"range={x}"
    return f"fixed={x}"


def init_value_and_range(x):
    """
    If x is a range (lo, hi), initialize at midpoint and return:
        init_value, (lo, hi)
    If x is a scalar, return:
        scalar_value, None
    """
    if is_range(x):
        lo, hi = x
        return 0.5 * (lo + hi), (lo, hi)
    return float(x), None


def to_tensor(x, device):
    return torch.tensor(float(x), device=device, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("real_data_path", type=str)

    p.add_argument("--n-samples", type=int, default=10000,
                   help="Number of MCMC samples (default: 10000)")

    p.add_argument("--alpha", type=parse_scalar_or_range("alpha"), default=1.0,
                   help="Either a number (e.g. 1.0) or an interval 'lo:hi' (e.g. 0.5:2.0)")
    p.add_argument("--mu", type=parse_scalar_or_range("mu"), default=(0.0, 3.0),
                   help="Either a number or an interval 'lo:hi'. Default is 0:3")
    p.add_argument("--omega", type=parse_scalar_or_range("omega"), default=0.0,
                   help="Either a number or an interval 'lo:hi'. Default is 0")
    p.add_argument("--rho", type=parse_scalar_or_range("rho"), default=(0.0, 1.0),
                   help="Either a number or an interval 'lo:hi'. Default is 0:1")

    p.add_argument("--t-switch", type=float, default=None,
                   help="time (hours) when feeding stops / regime switches. None = no switch")

    p.add_argument("--kappa-samples-path", type=str,
                   help="Path to .npz file containing kappa samples")
    
    p.add_argument("--outfile", type=str, default="grid_zomega.csv")

    p.add_argument("--jumpsize", type=float, default=0.01,
                   help="Proposal jump size (default: 0.01)")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    
    # Initialize parameter values and optional ranges
    alpha_init, alpha_range = init_value_and_range(args.alpha)
    mu_init, mu_range       = init_value_and_range(args.mu)
    omega_init, omega_range = init_value_and_range(args.omega)
    rho_init, rho_range     = init_value_and_range(args.rho)

    # Convert current values to tensors
    alpha = to_tensor(alpha_init, device)
    mu    = to_tensor(mu_init, device)
    omega = to_tensor(omega_init, device)
    rho   = to_tensor(rho_init, device)

    print("Parameter setup:")
    print(f"  alpha: {maybe_scalar_string(args.alpha)} -> init={alpha.item():.6f}")
    print(f"  mu:    {maybe_scalar_string(args.mu)} -> init={mu.item():.6f}")
    print(f"  omega: {maybe_scalar_string(args.omega)} -> init={omega.item():.6f}")
    print(f"  rho:   {maybe_scalar_string(args.rho)} -> init={rho.item():.6f}")


    # --- load and set up dataset ---
    basename = os.path.splitext(os.path.basename(args.real_data_path))[0]
    output_dir = os.path.join("grid_likelihood_outputs", basename)
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean_real_data(args.real_data_path, cutoff=300)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load(args.kappa_samples_path)["kappa_samples"]
    print(f"Loaded {len(kappa_np)} kappa samples from {args.kappa_samples_path} file")

    dataset = TimeSeriesInferenceDataset(
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=300,
        t_switch=args.t_switch,
        device=device,
    )

    out_path = os.path.join(output_dir, args.outfile)
    print(out_path)

    # --- output file (single) ---


    mus_sample, rhos_sample, lls = [], [], []
    flush_every = 10

    for iter in tqdm(range(args.n_samples)):
        mu_prop, rho_prop = proposal(mu, rho, mu_range, rho_range, jumpsize_proposal=args.jumpsize)

        ll_prop = dataset.loglike(alpha, mu_prop, omega, rho_prop)
        ll_curr = dataset.loglike(alpha, mu, omega, rho)

        log_accept_ratio = (ll_prop - ll_curr).item() + prior(mu_prop, rho_prop) - prior(mu, rho)

        if torch.log(torch.rand(1)) < log_accept_ratio:
            mu = mu_prop
            rho = rho_prop
            ll_curr = ll_prop


        mus_sample.append(float(mu.item()))
        rhos_sample.append(float(rho.item()))
        lls.append(float(ll_curr.item()))

        if (iter+1) % flush_every == 0:
            pd.DataFrame({"mu": np.array(mus_sample).round(6).reshape(-1), 
                           "rho": np.array(rhos_sample).round(6).reshape(-1), 
                           "ll": np.array(lls).round(6).reshape(-1)}).to_csv(out_path, index=False)

if __name__ == "__main__":
    main()



