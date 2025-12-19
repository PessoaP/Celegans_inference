import os, sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils import simulator

def simulate_for_likelihood(params, times, Nsamples=2**15):
    times = times.to(params.device)
    t, E = simulator.sample(params, N=Nsamples, T=times[0])
    sims = [E.to(torch.long)]
    for T in times[1:]:
        dt = T - t
        delta_t, E = simulator.sample(params, E_initial=E, N=Nsamples, T=dt)
        t = t + delta_t
        sims.append(E.to(torch.long))
    return sims

def logmeanexp(x, dim=0):
    return torch.logsumexp(x, dim=dim) - torch.log(torch.tensor(x.shape[dim], device=x.device, dtype=x.dtype))

def main(path):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_and_clean_real_data(path, cutoff=300)
    ts = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)
    dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)

    theta = torch.tensor([1/20, 0.25, 2e5, 0.1], dtype=torch.float32, device=device)
    print("theta:", theta.detach().cpu().numpy())

    # choose one datapoint index to inspect
    idx = 0
    k = int(dataset.counts[idx].item())
    phi = float(dataset.dils[idx].item())
    T_idx = int(dataset.T_index[idx].item())
    T = float(dataset.Ts[idx].item())
    print(f"Inspect datapoint idx={idx}: k={k}, phi={phi}, T={T} (time bin {T_idx})")

    # simulate ns at all unique times; pick the one for this datapoint
    ns_list = simulate_for_likelihood(theta, dataset.times, Nsamples=2**15)
    ns = ns_list[T_idx]  # (M,)

    # grab lpkdil_n row for this datapoint (shape: (Nmax,))
    row = dataset.lpkdil_n[idx]  # (Nmax,)
    # map samples to log p(k|n,phi)
    lps = row[ns.clamp_min(0).clamp_max(row.numel()-1)]  # should all be valid if your support test passed

    # compute log-mean-exp estimator
    est = logmeanexp(lps, dim=0)
    print("log E[p(k|n)] estimate:", float(est))

    # diagnose weight degeneracy:
    # normalized weights proportional to exp(lps)
    m = torch.max(lps)
    w = torch.exp(lps - m)
    wsum = torch.sum(w)
    w_norm = w / (wsum + 1e-30)

    ess = 1.0 / torch.sum(w_norm**2)
    top1 = torch.max(w_norm)
    top10 = torch.topk(w_norm, k=10).values.sum()

    print("ESS:", float(ess))
    print("top-1 weight mass:", float(top1))
    print("top-10 weight mass:", float(top10))
    print("lps (log p(k|n)) stats:",
          "max", float(torch.max(lps)),
          "median", float(torch.median(lps)),
          "min", float(torch.min(lps)))

if __name__ == "__main__":
    main(sys.argv[1])
