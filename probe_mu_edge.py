#!/usr/bin/env python3
"""
probe_mu_edge.py

Purpose
-------
When a 3D grid-search MLE hits the mu upper edge (e.g. mu=1.0),
this script probes whether the likelihood continues improving for mu > 1
*while holding alpha and omega fixed at the current MLE values*.

It does two things:
  (1) Spot-check selected mu values > 1 (alpha, omega fixed at MLE).
  (2) Sweep a 1D line in mu (alpha, omega fixed at MLE), save CSV + optional plot.

Inputs
------
- real_data_path: path to your real dataset (same as grid script)
- --grid-csv: path to the existing grid_loglike.csv that contains columns:
      alpha, mu, omega, loglike
  (We use it to recover (alpha*, mu*, omega*) from the maximum loglike row.)

Outputs
-------
Creates:
  grid_likelihood_outputs/<basename_of_real_data_path>/mu_probe/
      mu_checks.csv
      mu_sweep.csv
      mu_sweep.png (if --plot)

Notes
-----
- Uses your parameterization d = omega * mu (omega fixed => d changes with mu).
- If your dataset.loglike is stochastic due to internal simulation, you may want
  to add replicates per mu and average/log-mean-exp. This script currently does
  a single call per mu (like your grid script).
"""

import os, csv, argparse
import numpy as np
import torch

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data


def loglike_alpha_mu_omega(dataset, alpha, mu, omega):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    omega = torch.as_tensor(omega, device=dataset.device, dtype=torch.float32)

    d = omega * mu
    theta_phys = torch.stack([alpha, mu, d])
    with torch.no_grad():
        return dataset.loglike(theta_phys)


def read_grid_mle(grid_csv_path: str):
    """
    Returns (alpha_hat, mu_hat, omega_hat, loglike_hat) from the max-loglike row,
    ignoring non-numeric rows and ignoring non-finite loglikes (nan/inf/-inf).
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(grid_csv_path)
    for col in ["alpha", "mu", "omega", "loglike"]:
        if col not in df.columns:
            raise ValueError(f"grid CSV missing required column {col!r}. Found: {list(df.columns)}")

    # Coerce numeric (repeated header rows become NaN)
    for c in ["alpha", "mu", "omega", "loglike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ll = df["loglike"].to_numpy(dtype=float)
    finite = np.isfinite(ll)
    if not np.any(finite):
        raise ValueError("All values in 'loglike' are non-finite (nan/inf/-inf).")

    df_f = df.loc[finite].copy()
    idx = df_f["loglike"].idxmax()
    row = df_f.loc[idx]

    return float(row["alpha"]), float(row["mu"]), float(row["omega"]), float(row["loglike"])



def build_dataset(real_data_path: str, t_switch: float | None, rho: float, cutoff: int, device):
    df = load_and_clean_real_data(real_data_path, cutoff=cutoff)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load("kappa_samples_Exp1_4gauss.npz")["kappa_samples"]

    dataset = TimeSeriesInferenceDataset(
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=cutoff,
        t_switch=t_switch,
        rho=rho,
        device=device,
    )
    return dataset


def write_rows_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("real_data_path", type=str)
    p.add_argument("--grid-csv", type=str, required=True,
                   help="Path to existing grid_loglike.csv (used to recover alpha*, mu*, omega*).")
    p.add_argument("--t-switch", type=float, default=None,
                   help="time (hours) when feeding stops / regime switches. None = no switch")
    p.add_argument("--rho", type=float, default=1.0,
                   help="colonization scaling after t_switch (often 0.0 for time-limited feeding)")
    p.add_argument("--cutoff", type=int, default=300)

    # (1) selected mu checks (> 1)
    p.add_argument("--mu-checks", type=float, nargs="*", default=[1.1, 1.25, 1.5, 2.0],
                   help="Specific mu values to evaluate at fixed alpha*, omega*.")

    # (2) dense sweep line in mu
    p.add_argument("--mu-min", type=float, default=None,
                   help="mu sweep min. Default: recovered mu* from grid.")
    p.add_argument("--mu-max", type=float, default=3.0,
                   help="mu sweep max.")
    p.add_argument("--mu-num", type=int, default=81,
                   help="Number of points in mu sweep (linspace).")

    # output / misc
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--plot", action="store_true", help="Save a simple mu_sweep.png")
    args = p.parse_args()

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seeds (match your grid behavior)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # output dir
    basename = os.path.splitext(os.path.basename(args.real_data_path))[0]
    base_out = os.path.join("grid_likelihood_outputs", basename, "mu_probe")
    os.makedirs(base_out, exist_ok=True)

    # recover MLE from grid CSV
    alpha_hat, mu_hat, omega_hat, ll_hat = read_grid_mle(args.grid_csv)
    print(f"[MLE from grid]")
    print(f"  alpha* = {alpha_hat:g}")
    print(f"  mu*    = {mu_hat:g}")
    print(f"  omega* = {omega_hat:g}")
    print(f"  loglike* = {ll_hat:g}")

    # build dataset once
    dataset = build_dataset(
        real_data_path=args.real_data_path,
        t_switch=args.t_switch,
        rho=args.rho,
        cutoff=args.cutoff,
        device=device,
    )

    # ---------------------------
    # (1) spot checks: mu > 1
    # ---------------------------
    check_rows = []
    for mu in args.mu_checks:
        ll = loglike_alpha_mu_omega(dataset, alpha_hat, mu, omega_hat)
        ll_val = float(ll.detach().cpu())
        check_rows.append([mu, alpha_hat, omega_hat, omega_hat * mu, ll_val])

    checks_csv = os.path.join(base_out, "mu_checks.csv")
    write_rows_csv(
        checks_csv,
        header=["mu", "alpha_fixed", "omega_fixed", "d=omega*mu", "loglike"],
        rows=check_rows
    )
    print(f"\nWrote checks: {checks_csv}")

    # ---------------------------
    # (2) 1D sweep: mu line scan
    # ---------------------------
    mu_min = mu_hat if args.mu_min is None else args.mu_min
    mu_grid = np.linspace(mu_min, args.mu_max, args.mu_num)

    sweep_rows = []
    best = (-np.inf, None)  # (ll, mu)
    for mu in mu_grid:
        ll = loglike_alpha_mu_omega(dataset, alpha_hat, mu, omega_hat)
        ll_val = float(ll.detach().cpu())
        sweep_rows.append([mu, alpha_hat, omega_hat, omega_hat * mu, ll_val])
        if ll_val > best[0]:
            best = (ll_val, float(mu))

    sweep_csv = os.path.join(base_out, "mu_sweep.csv")
    write_rows_csv(
        sweep_csv,
        header=["mu", "alpha_fixed", "omega_fixed", "d=omega*mu", "loglike"],
        rows=sweep_rows
    )
    print(f"Wrote sweep:  {sweep_csv}")
    print(f"Best on sweep: mu = {best[1]:g}  (loglike = {best[0]:g})")

    # optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        mus = np.array([r[0] for r in sweep_rows], dtype=float)
        lls = np.array([r[4] for r in sweep_rows], dtype=float)

        plt.figure()
        plt.plot(mus, lls, marker="o", markersize=3, linewidth=1)
        plt.axvline(mu_hat, linestyle="--")
        plt.xlabel("mu (alpha, omega fixed at MLE)")
        plt.ylabel("loglike")
        plt.title("1D mu sweep (fixed alpha*, omega*)")
        plt.tight_layout()

        fig_path = os.path.join(base_out, "mu_sweep.png")
        plt.savefig(fig_path, dpi=200)
        print(f"Saved plot:   {fig_path}")


if __name__ == "__main__":
    main()

# How to run for regular feeding:
# python probe_mu_edge.py real_data/Exp_1_live_lowpH.csv \
#   --grid-csv recovery_tests/grid_Exp1/grid_loglike_Exp1_full.csv \
#   --mu-checks 1.1 1.25 1.5 2.0 3.0 \
#   --mu-max 3.0 --mu-num 50 --plot
