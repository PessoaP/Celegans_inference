import os
import sys
import torch
import numpy as np
import pandas as pd

# Match driver imports
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils import simulator

# --- copy of simulator wrapper (for main inference framework)---
def simulate_for_likelihood(params, times, Nsamples=2**15):
    times = times.to(params.device)

    # First segment
    t, E = simulator.sample(params, N=Nsamples, T=times[0])
    simulations = [E.to(torch.long)]

    # Subsequent segments
    for T in times[1:]:
        dt = T - t
        delta_t, E = simulator.sample(params, E_initial=E, N=Nsamples, T=dt)
        t = t + delta_t
        simulations.append(E.to(torch.long))

    return simulations


def summarize_support(ns_list, Nmax, times):
    """
    ns_list: list of tensors, one per timepoint, each shape (Nsamples,) (or similar)
    """
    rows = []
    Nmax = int(Nmax)

    for i, (t, ns) in enumerate(zip(times, ns_list)):
        ns = ns.detach()
        ns_cpu = ns.to("cpu")

        invalid_low  = (ns_cpu < 0)
        invalid_high = (ns_cpu >= Nmax)
        invalid = invalid_low | invalid_high

        frac_invalid = invalid.float().mean().item()
        frac_high = invalid_high.float().mean().item()
        frac_low  = invalid_low.float().mean().item()

        # Only compute quantiles on valid ones (if any)
        valid_ns = ns_cpu[~invalid]
        if valid_ns.numel() > 0:
            q = torch.quantile(valid_ns.float(), torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
            nmin, nmed, n90, n99, nmax = [float(x) for x in q]
        else:
            nmin = nmed = n90 = n99 = nmax = float("nan")

        rows.append({
            "time": float(t.cpu()),
            "Nsamples": int(ns_cpu.numel()),
            "Nmax_support": Nmax,
            "frac_invalid": frac_invalid,
            "frac_invalid_low": frac_low,
            "frac_invalid_high": frac_high,
            "valid_min": nmin,
            "valid_median": nmed,
            "valid_p90": n90,
            "valid_p99": n99,
            "valid_max": nmax,
        })

    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_support_escape.py <real_data_path> [repeats] [Nsamples]")
        sys.exit(1)

    real_data_path = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    Nsamples = int(sys.argv[3]) if len(sys.argv) >= 4 else 2**15

    torch.manual_seed(15)
    np.random.seed(15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- load dataset exactly like driver ---
    df = load_and_clean_real_data(real_data_path, cutoff=300)
    ts = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)
    dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)

    # --- choose theta to test ---
    theta = torch.tensor([1/20, 0.25, 2e5, 0.1], dtype=torch.float32, device=device)
    print("theta:", theta.detach().cpu().numpy())
    print("dataset.Nmax:", dataset.Nmax)
    print("unique times:", dataset.times.detach().cpu().numpy().reshape(-1))

    all_frames = []
    for r in range(repeats):
        print(f"\n=== repeat {r+1}/{repeats} ===")
        ns_list = simulate_for_likelihood(theta, dataset.times, Nsamples=Nsamples)
        frame = summarize_support(ns_list, dataset.Nmax, dataset.times)
        frame["repeat"] = r
        print(frame[["time","frac_invalid","frac_invalid_high","valid_p99","valid_max"]].to_string(index=False))
        all_frames.append(frame)

    out = pd.concat(all_frames, ignore_index=True)
    out_csv = "support_escape_report.csv"
    out.to_csv(out_csv, index=False)
    print(f"\nSaved per-time support report to {out_csv}")

    # Aggregate across repeats (mean/std of invalid fractions)
    grp = out.groupby("time")[["frac_invalid","frac_invalid_high","frac_invalid_low"]].agg(["mean","std","max"])
    print("\n=== aggregated across repeats ===")
    print(grp)

if __name__ == "__main__":
    main()
