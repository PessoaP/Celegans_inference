import os, sys, csv, argparse
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import gc


# Force Python garbage collection
gc.collect()

# Clear PyTorch CUDA cache
torch.cuda.empty_cache()


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("real_data_path", type=str)
    p.add_argument("--t-switch", type=float, default=None,
                   help="time (hours) when feeding stops / regime switches. None = no switch")
    p.add_argument("--rho", type=float, default=1.0,
                   help="colonization scaling after t_switch (often 0.0 for time-limited feeding)")

    args = p.parse_args()

    # --- single process => pick one GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    np.random.seed(42)

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
        rho=args.rho,
        device=device,
    )

    # --- define grid ---
    alphas = ( .06*np.linspace(0.0, 1, 51)[1:]).round(6)    
    mus    = (.1 + .5*np.linspace(0.0, 1, 51)[1:]).round(6) 
    omegas = np.zeros(1)    

    grid = [(a, m, o)  for o in omegas for m in mus for a in alphas]

    # --- output file (single) ---
    out_path = os.path.join(output_dir, "grid_zomega.csv")

    already_done = 0
    flush_every = 10

    print(f"[single] grid points = {len(grid)}; resuming at row {already_done}")

    dic = []

    for i in tqdm(range(already_done, len(grid))):
        alpha, mu, omega = grid[i]
        ll = loglike_alpha_mu_omega(dataset, alpha, mu, omega)
        ll_val = float(ll.detach().cpu())

        dic.append({"idx": i, "alpha": alpha, "mu": mu, "omega": omega, "loglike": ll_val})

        if (i+1) % flush_every == 0:
            pd.DataFrame(dic).to_csv(out_path, index=False)

    pd.DataFrame(dic).to_csv(out_path, index=False)

    print(f"Done. Wrote {len(grid) - already_done} rows to {out_path}")


if __name__ == "__main__":
    main()


