"""
run_mcmc.py
Experiment-specific driver for annealed MH MCMC.
"""

import os, argparse
import numpy as np
import torch

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.mcmc import MCMCConfig, run_mcmc_annealed




def main():
    # -----------------------
    # Experiment config
    # -----------------------
    REAL_DATA_PATH = "path/to/real_data.csv"   # <-- CHANGE

    rho = 0.1
    t_switch = 72.0  # hours, or None
    cutoff = 300

    KAPPA_NPZ = "kappa_samples_Exp1_4gauss.npz"
    N_KAPPA = None        # fixed CRN size; None = use all
    KAPPA_SEED = 0

    theta0 = [0.05, 0.5, 0.12]  # alpha, mu, d

    # -----------------------
    # Device
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Dataset
    # -----------------------
    df = load_and_clean_real_data(args.real_data_path, cutoff=300)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load("kappa_samples_Exp1_4gauss.npz")["kappa_samples"]

    dataset = TimeSeriesInferenceDataset(
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=300,
        t_switch=args.t_switch,
        rho=args.rho,
        device=device,
    )
    # -----------------------
    # MCMC config
    # -----------------------
    basename = os.path.splitext(os.path.basename(REAL_DATA_PATH))[0]
    config = MCMCConfig(
        n_steps=5_000,
        warmup_frac=0.1,
        beta0=0.05,
        prior_frac_error=0.7,
        init_step_u=0.15,
        save_every=1000,
        out_dir=os.path.join("mcmc_out", basename),
        run_name=f"chain_rho{rho:g}_switch{('none' if t_switch is None else t_switch)}_K{('all' if N_KAPPA is None else N_KAPPA)}",
        checkpoint=True,
    )

    # -----------------------
    # Run MCMC
    # -----------------------
    samples, logpost, acc_rate, ckpt_path = run_mcmc_annealed(
        dataset=dataset,
        theta0_phys=theta0,
        config=config,
        device=device,
        resume_from=None,
        debug=False,
    )

    print(f"Done. accept_rate={acc_rate:.3f}")
    print(f"Checkpoint saved at: {ckpt_path}")
    print(f"Samples shape: {tuple(samples.shape)}  logpost shape: {tuple(logpost.shape)}")


if __name__ == "__main__":
    main()
