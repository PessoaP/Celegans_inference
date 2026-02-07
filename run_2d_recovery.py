# run_2d_recovery.py
import os, csv, argparse
import numpy as np
import torch

from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.dataclass import TimeSeriesInferenceDataset


# =========================
# CONFIG DEFAULTS
# =========================

DATA_PATH_DEFAULT = "synthetic_data/synthetic_data.csv"
KAPPA_NPZ_DEFAULT = "synthetic_data/kappa_samples_Exp1_4gauss.npz"
OUTDIR_DEFAULT    = "recovery_tests/grid"

CUTOFF_DEFAULT = 300

# Ground truth (synthetic generation)
ALPHA_TRUE = 0.05
MU_TRUE = 0.5
D_TRUE = 0.12

# Grids
ALPHAS = np.linspace(0.0, 0.1, 11)[1:]   # 10
MUS    = np.linspace(0.2, 1.0, 11)[1:]  # 10
DS     = np.linspace(0.0, 0.2, 11)[1:]  # 10


# =========================
# Likelihood wrapper
# =========================

def loglike_alpha_mu_d(dataset, alpha, mu, d):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    d     = torch.as_tensor(d,     device=dataset.device, dtype=torch.float32)
    theta = torch.stack([alpha, mu, d])  # (3,)
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
    truth=(ALPHA_TRUE, MU_TRUE, D_TRUE),
    outdir=OUTDIR_DEFAULT,
    flush_every: int = 50,
):
    """
    Computes a 2D grid over (w1,w2), holding the third parameter fixed at truth.

    Output CSV columns: idx, alpha, mu, d, loglike
    """
    out_path = os.path.join(outdir, f"{name}.csv")
    header = ["idx", "alpha", "mu", "d", "loglike"]

    ensure_header(out_path, header)

    # Full deterministic grid (idx increases with nested loops)
    grid = []
    idx = 0
    for x in g1:
        for y in g2:
            grid.append((idx, float(x), float(y)))
            idx += 1

    start_row = count_rows_minus_header(out_path)

    print(f"[{name}] out={out_path}")
    print(f"[{name}] resume_row={start_row}")

    a0, m0, d0 = truth

    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)

        wrote = 0
        idx = 0
        for x in g1:
            for y in g2:
                if idx < start_row:
                    idx += 1
                    continue

                a, m, d = a0, m0, d0

                if w1 == "alpha": a = float(x)
                elif w1 == "mu":  m = float(x)
                elif w1 == "d":   d = float(x)
                else: raise ValueError(f"bad w1={w1}")

                if w2 == "alpha": a = float(y)
                elif w2 == "mu":  m = float(y)
                elif w2 == "d":   d = float(y)
                else: raise ValueError(f"bad w2={w2}")

                ll = loglike_alpha_mu_d(dataset, a, m, d)
                w.writerow([idx, a, m, d, ll])
                wrote += 1
                idx += 1

                if wrote % flush_every == 0:
                    flush_safely(f)
                    print(f"[{name}] wrote {wrote} rows (last idx={idx-1})")

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
    p.add_argument("--t-switch", type=float, default=None,
              help="time (hours) when feeding stops / regime switches. None = no switch")
    p.add_argument("--rho", type=float, default=1.0,
                help="colonization scaling after t_switch")

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
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=args.cutoff,
        t_switch=args.t_switch,
        rho=args.rho,
        device=device,
    )

    truth = (ALPHA_TRUE, MU_TRUE, D_TRUE)
    print(f"Truth: alpha={truth[0]}, mu={truth[1]}, d={truth[2]}\n")

    # ---- run the three 2D sweeps ----
    run_2d_sweep(dataset, "alpha_mu", "alpha", ALPHAS, "mu", MUS,
                truth=truth, outdir=args.outdir, flush_every=args.flush_every)

    run_2d_sweep(dataset, "alpha_d", "alpha", ALPHAS, "d", DS,
                truth=truth, outdir=args.outdir, flush_every=args.flush_every)

    run_2d_sweep(dataset, "mu_d", "mu", MUS, "d", DS,
                truth=truth, outdir=args.outdir, flush_every=args.flush_every)

    print("All 2D sweeps complete.")

if __name__ == "__main__":
    main()
