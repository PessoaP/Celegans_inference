import os, sys, csv, argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data


def parse_scalar_or_range(name: str, default: str):
    """
    Returns a function usable as argparse `type=...`.
    Accepted formats:
      - "2.5"          -> np.array([2.5])
      - "0:3" or "0,3" -> (lo, hi) tuple, later expanded to linspace
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

def expand(x, n: int, purge_zeros=False):
    """float -> [float]; (lo,hi) -> linspace(lo,hi,n)"""
    if isinstance(x, tuple):
        lo, hi = x
        if purge_zeros and lo<=0:
            ans = np.linspace(lo, hi, n+1).round(8)
            return ans[ans > 0.0]
        return np.linspace(lo, hi, n+1).round(8)
    return np.array([float(x)], dtype=float)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("real_data_path", type=str)

    p.add_argument("--n-samples", type=int, default=50,
                   help="Number of samples used when an interval is provided (default: 50)")

    p.add_argument("--alpha", type=parse_scalar_or_range("alpha", "1.0"), default=1.0,
                   help="Either a number (e.g. 1.0) or an interval 'lo:hi' (e.g. 0.5:2.0)")
    p.add_argument("--mu", type=parse_scalar_or_range("mu", "0:3"), default=(0.0, 3.0),
                   help="Either a number or an interval 'lo:hi'. Default is 0:3")
    p.add_argument("--omega", type=parse_scalar_or_range("omega", "0.0"), default=0.0,
                   help="Either a number or an interval 'lo:hi'. Default is 0")
    p.add_argument("--rho", type=parse_scalar_or_range("rho", "0:1"), default=(0.0, 1.0),
                   help="Either a number or an interval 'lo:hi'. Default is 0:1")

    p.add_argument("--t-switch", type=float, default=None,
                   help="time (hours) when feeding stops / regime switches. None = no switch")

    p.add_argument("--kappa-samples-path", type=str,
                   help="Path to .npz file containing kappa samples")
    
    p.add_argument("--outfile", type=str, default="grid_zomega.csv")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # --- expand grid axes (singleton if scalar; linspace if interval) ---
    alphas = expand(args.alpha, args.n_samples,purge_zeros=True)
    mus    = expand(args.mu, args.n_samples,purge_zeros=True)
    omegas = expand(args.omega, args.n_samples)
    rho    = expand(args.rho, args.n_samples)

    grid = [(a, m, o, r) for a in alphas for m in mus for o in omegas for r in rho]

    print(f"alphas: {alphas[:3]} ... (len={len(alphas)})")
    print(f"mus:    {mus[:3]} ... (len={len(mus)})")
    print(f"omegas: {omegas[:3]} ... (len={len(omegas)})")
    print(f"rho:    {rho[:3]} ... (len={len(rho)})")
    print(f"grid size = {len(grid)}")


    # --- load and set up dataset ---
    basename = os.path.splitext(os.path.basename(args.real_data_path))[0]
    output_dir = os.path.join("grid_likelihood_outputs", basename)
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean_real_data(args.real_data_path, cutoff=300)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load("kappa_samples_stratified_highpH.npz")["kappa_samples"]
    print(f"Loaded {len(kappa_np)} kappa samples from .npz file")

    dataset = TimeSeriesInferenceDataset(
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=300,
        t_switch=args.t_switch,
        device=device,
    )

    # --- output file (single) ---

    already_done = 0
    flush_every = 10

    #print(f"[single] grid points = {len(grid)}; resuming at row {already_done}")

    dic = []

    for i in tqdm(range(already_done, len(grid))):
        alpha, mu, omega, rho = grid[i]
        with torch.no_grad():
            ll = dataset.loglike(alpha, mu, omega, rho)
        ll_val = float(ll.detach().cpu())

        dic.append({"idx": i, "alpha": alpha, "mu": mu, "omega": omega, "rho": rho, "loglike": ll_val})

        if (i+1) % flush_every == 0:
            pd.DataFrame(dic).to_csv(out_path, index=False)

    pd.DataFrame(dic).to_csv(out_path, index=False)

    print(f"Done. Wrote {len(grid) - already_done} rows to {out_path}")


if __name__ == "__main__":
    main()



