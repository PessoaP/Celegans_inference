import math
import numpy as np
import torch
import matplotlib.pyplot as plt

import repop  
from utils.load_and_clean_real_data import load_and_clean_real_data



def main():
    # -----------------------------
    # Choose a small diagnostic setup
    # -----------------------------
    device = torch.device("cpu")

    # === Load and preprocess data ===
    df = load_and_clean_real_data("real_data/Exp_1_live_lowpH.csv", cutoff=300)

    # Load on CPU; dataset class moves to GPU after pre-processing
    ts = torch.tensor(df["Day"].values * 24)     
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)
    # Ensure counts/dils are column vectors: (ndata, 1)
    counts = counts.to(device=device, dtype=torch.float64).reshape(-1, 1)
    dils   = dils.to(device=device, dtype=torch.float64).reshape(-1, 1)

    print("Loaded real data:")
    print("  counts range:", counts.min().item(), counts.max().item())
    print("  dils unique:", torch.unique(dils).tolist())
    print("  n datapoints:", counts.shape[0])
    
    Nmax = 20000

    # Ensure n is a ROW vector so it broadcasts with (ndata,1)
    n = torch.arange(Nmax, device=device, dtype=torch.long).reshape(1, -1)

    cutoff = 300  # try also -1 to compare

    # -----------------------------
    # Compute lpkdil_n
    # -----------------------------
    with torch.no_grad():
        lpk = repop.get_lpkdil_n(counts, dils, n, cutoff, Nmax)
        lpk_raw = repop.get_lpkdil_n(counts, dils, n, -1, Nmax)

    # Ensure CPU numpy for plotting
    lpk_np = lpk.detach().cpu().numpy()
    lpk_raw_np = lpk_raw.detach().cpu().numpy()
    n_np = n.detach().cpu().numpy().reshape(-1) 

    # -----------------------------
    # Quick numeric diagnostics
    # -----------------------------
    print("\n=== Diagnostics ===")
    print("cutoff =", cutoff, "Nmax =", Nmax)
    print("lpk shape:", lpk.shape)
    print("min(lpk):", float(np.nanmin(lpk_np)), "max(lpk):", float(np.nanmax(lpk_np)))
    print("min(lpk_raw):", float(np.nanmin(lpk_raw_np)), "max(lpk_raw):", float(np.nanmax(lpk_raw_np)))

    # Where does it go positive?
    pos = lpk_np > 1e-9
    if pos.any():
        rows = np.where(pos.any(axis=1))[0]
        print(f"WARNING: positive entries found in corrected lpk for rows {rows.tolist()}")
        for r in rows[:5]:
            nn = np.where(pos[r])[0]
            print(f"  row {r}: max={lpk_np[r].max():.4g} at n={nn[np.argmax(lpk_np[r, nn])]} ; first positive n={nn[0]}")
    else:
        print("OK: no positive entries detected in corrected lpk (within tolerance).")

    # -----------------------------
    # Plot: lpk vs n (corrected) and raw counts_loglike vs n
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # Plot corrected
    ax = axes[0]
    for i in range(lpk_np.shape[0]):
        ax.plot(n_np, lpk_np[i], label=f"k={int(counts[i].item())}, dil={int(dils[i].item())}")
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("lpkdil_n (corrected)")
    ax.set_title("repop.get_lpkdil_n vs n")
    ax.legend(fontsize=8, ncol=2)

    # Plot raw (no cutoff correction)
    ax = axes[1]
    for i in range(lpk_raw_np.shape[0]):
        ax.plot(n_np, lpk_raw_np[i], label=f"k={int(counts[i].item())}, dil={int(dils[i].item())}")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("n")
    ax.set_ylabel("counts_loglike mode (cutoff=-1)")
    ax.set_title("Baseline: get_lpkdil_n with cutoff = -1")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Optional: decompose the correction term
    # (only works if these names exist in repop namespace)
    # -----------------------------
    try:
        with torch.no_grad():
            lpk_unnorm = repop.counts_loglike(counts, n, dils)
            logZ, lpdil_n = repop.dils_switch(dils, Nmax, cutoff)
            corr = (-logZ + lpdil_n)

        lpk_unnorm_np = lpk_unnorm.detach().cpu().numpy()
        corr_np = corr.detach().cpu().numpy()
        logZ_np = logZ.detach().cpu().numpy()
        lpdil_np = lpdil_n.detach().cpu().numpy()

        print("\n=== Decomposition ===")
        print("max(lpk_unnorm):", float(lpk_unnorm_np.max()))
        print("max(-logZ + lpdil_n):", float(corr_np.max()))
        print("max(logZ):", float(logZ_np.max()), "min(logZ):", float(logZ_np.min()))
        print("max(lpdil_n):", float(lpdil_np.max()), "min(lpdil_n):", float(lpdil_np.min()))

        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

        ax = axes[0]
        for i in range(lpk_unnorm_np.shape[0]):
            ax.plot(n_np, lpk_unnorm_np[i], label=f"row {i}")
        ax.axhline(0.0, linewidth=1)
        ax.set_ylabel("counts_loglike (unnorm)")
        ax.set_title("Decomposition: lpk_unnorm")

        ax = axes[1]
        for i in range(corr_np.shape[0]):
            ax.plot(n_np, corr_np[i], label=f"row {i}")
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("n")
        ax.set_ylabel("(-logZ + lpdil_n)")
        ax.set_title("Decomposition: correction term")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\n[Optional decomposition skipped]")
        print("Could not compute counts_loglike/dils_switch decomposition:", repr(e))

    i = 50  # choose a datapoint index to inspect

    ell = lpk_np[i].astype(np.float64)   # log curve over n
    n = n_np.astype(np.int64)            # x-axis, shape (Nmax,)

    # 1) relative likelihood in real space, max-scaled to 1
    m = np.nanmax(ell[np.isfinite(ell)])          # max over finite values
    rel = np.exp(ell - m)                         # exp(-inf) -> 0 automatically

    # 2) optional: normalize over n to get a proper pmf over n (conditional on this datapoint)
    Z = rel.sum()
    pmf = rel / Z if Z > 0 else rel

    # Plot
    plt.figure(figsize=(9, 3))
    plt.plot(n, rel)
    plt.ylim(-0.02, 1.05)
    plt.xlabel("n")
    plt.ylabel(r"exp(lpk - max)")
    plt.title(f"Relative real-space curve (row {i}): max-scaled likelihood")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 3))
    plt.plot(n, pmf)
    plt.xlabel("n")
    plt.ylabel("normalized pmf over n")
    plt.title(f"Normalized over n (row {i})")
    plt.tight_layout()
    plt.show()

    idxs = [0, 10, 50, 100]  # pick a few

    plt.figure(figsize=(9, 3))
    for i in idxs:
        ell = lpk_np[i].astype(np.float64)
        m = np.max(ell[np.isfinite(ell)])
        rel = np.exp(ell - m)
        plt.plot(n_np, rel, label=f"i={i}, k={int(counts[i])}, dil={int(dils[i])}")
    plt.ylim(-0.02, 1.05)
    plt.xlabel("n")
    plt.ylabel(r"exp(lpk - max)")
    plt.title("Relative likelihood curves (max-scaled)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
