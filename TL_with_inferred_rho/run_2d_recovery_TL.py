# run_2d_recovery_TL.py
import os, csv, argparse
import numpy as np
import torch

from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.dataclass_TL import TimeSeriesInferenceDataset


# =========================
# CONFIG DEFAULTS
# =========================

DATA_PATH_DEFAULT = "synthetic_data/synthetic_data_TL.csv"
KAPPA_NPZ_DEFAULT = "synthetic_data/kappa_samples_Exp1_4gauss.npz"
OUTDIR_DEFAULT    = "recovery_tests/grid"

CUTOFF_DEFAULT = 300

# Ground truth (synthetic generation)
ALPHA_TRUE = 0.05
MU_TRUE = 0.5
D_TRUE = 0.12
RHO_TRUE = 0.1

# Grids
ALPHAS = np.linspace(0.0, 0.1, 11)[1:]   # 10
MUS    = np.linspace(0.2, 1.0, 11)[1:]  # 10
DS     = np.linspace(0.0, 0.2, 11)[1:]  # 10
RHOS   = np.linspace(0.0, 0.4, 11)[1:]  # 10


# =========================
# Likelihood wrapper
# =========================

def loglike_theta(dataset, alpha, mu, d, rho):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    d     = torch.as_tensor(d,     device=dataset.device, dtype=torch.float32)
    rho   = torch.as_tensor(rho,   device=dataset.device, dtype=torch.float32)
    theta = torch.stack([alpha, mu, d, rho])  # (4,)
    with torch.no_grad():
        ll = dataset.loglike(theta)
    return float(ll.detach().cpu())


# =========================
# Resume helpers
# =========================
def max_idx_in_csv(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return -1
    mx = -1
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            try:
                mx = max(mx, int(row[0]))
            except Exception:
                pass
    return mx

def count_rows_minus_header(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", newline="") as f:
        # subtract header if present
        n = sum(1 for _ in f)
    return max(0, n - 1)

def ensure_header(csv_path: str, header):
    new_file = (not os.path.exists(csv_path)) or (count_rows_minus_header(csv_path) == 0)
    if new_file:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

def flush_safely(f):
    f.flush()
    os.fsync(f.fileno())


# =========================
# 2D sweep core
# =========================

def run_2d_sweep(
    dataset,
    name: str,
    w1: str, g1: np.ndarray,
    w2: str, g2: np.ndarray,
    truth=(ALPHA_TRUE, MU_TRUE, D_TRUE, RHO_TRUE),
    outdir=OUTDIR_DEFAULT,
    flush_every: int = 50,
    job_idx: int = 0,
    job_count: int = 1,
):
    """
    Computes a 2D grid over (w1,w2), holding the other parameters fixed at truth.
    Resumable, and supports "job slicing" by assigning rows where idx % job_count == job_idx.

    Output CSV columns: idx, alpha, mu, d, rho, loglike
    """
    out_path = os.path.join(outdir, f"{name}.csv")
    header = ["idx", "alpha", "mu", "d", "rho", "loglike"]
    ensure_header(out_path, header)

    # deterministic grid ordering
    grid = []
    idx = 0
    for x in g1:
        for y in g2:
            grid.append((idx, float(x), float(y)))
            idx += 1

    # resume point = how many rows already written
    start_idx = count_rows_minus_header(out_path)

    print(f"[{name}] out={out_path}")
    print(f"[{name}] total grid points={len(grid)}  start_idx={start_idx}  job={job_idx}/{job_count}")

    a0, m0, d0, r0 = truth

    def assign_param(val, which, a, m, d, r):
        if which == "alpha": return val, m, d, r
        if which == "mu":    return a, val, d, r
        if which == "d":     return a, m, val, r
        if which == "rho":   return a, m, d, val
        raise ValueError(f"bad param name: {which}")

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)

        wrote = 0
        for k in range(start_idx, len(grid)):
            idx, x, y = grid[k]

            # job slicing
            if (idx % job_count) != job_idx:
                continue

            a, m, d, r = a0, m0, d0, r0
            a, m, d, r = assign_param(x, w1, a, m, d, r)
            a, m, d, r = assign_param(y, w2, a, m, d, r)

            ll = loglike_theta(dataset, a, m, d, r)
            w.writerow([idx, a, m, d, r, ll])
            wrote += 1

            if wrote % flush_every == 0:
                flush_safely(f)
                print(f"[{name}] wrote {wrote} rows (idx up to ~{idx})")

        flush_safely(f)

    print(f"[{name}] done. newly wrote={wrote}\n")

# =========================
# Main
# =========================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA_PATH_DEFAULT)
    p.add_argument("--kappa", default=KAPPA_NPZ_DEFAULT)
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    p.add_argument("--cutoff", type=int, default=CUTOFF_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--flush-every", type=int, default=50)

    # Optional: if later you want to split across GPUs/nodes
    p.add_argument("--job-idx", type=int, default=0)
    p.add_argument("--job-count", type=int, default=1)

    args = p.parse_args()

    torch.manual_seed(15)
    np.random.seed(15)

    device = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    print("DEVICE:", device)

    df = load_and_clean_real_data(args.data, cutoff=args.cutoff)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load(args.kappa)["kappa_samples"]

    dataset = TimeSeriesInferenceDataset(
        ts=ts,
        counts=counts,
        dils=dils,
        kappa_samples=kappa_np,
        cutoff=args.cutoff,
        device=device,
    )

    truth = (ALPHA_TRUE, MU_TRUE, D_TRUE, RHO_TRUE)

    # ---- run all 6 2D sweeps ----
    run_2d_sweep(dataset, "alpha_mu",  "alpha", ALPHAS, "mu",  MUS,  truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)

    run_2d_sweep(dataset, "alpha_d",   "alpha", ALPHAS, "d",   DS,   truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)

    run_2d_sweep(dataset, "alpha_rho", "alpha", ALPHAS, "rho", RHOS, truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)

    run_2d_sweep(dataset, "mu_d",      "mu",    MUS,    "d",   DS,   truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)

    run_2d_sweep(dataset, "mu_rho",    "mu",    MUS,    "rho", RHOS, truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)

    run_2d_sweep(dataset, "d_rho",     "d",     DS,     "rho", RHOS, truth=truth, outdir=args.outdir,
                flush_every=args.flush_every, job_idx=args.job_idx, job_count=args.job_count)


if __name__ == "__main__":
    main()
