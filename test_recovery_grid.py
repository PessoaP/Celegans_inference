import os, csv
import numpy as np
import torch

from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.dataclass import TimeSeriesInferenceDataset  


# =========================
# CONFIG 
# =========================

DATA_PATH = "synthetic_data/synthetic_data.csv"  
KAPPA_NPZ  = "synthetic_data/kappa_samples_Exp1_4gauss.npz"

OUTDIR = "recovery_tests/grid"

CUTOFF = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ground truth (what you used to generate synthetic data)
ALPHA_TRUE = 0.01
MU_TRUE    = 0.48
D_TRUE     = 0.038461538461538464 * MU_TRUE  

# Grids (keep these biologically feasible + coarse first)
ALPHAS = np.linspace(0.0, 1/4, 26)[1:]  # 0 excluded
MUS    = np.linspace(0.2, 1.5, 26)[1:]
DS     = np.linspace(0.0, 0.05,26)[1:]  


# =========================
# Likelihood wrapper
# =========================

def loglike_alpha_mu_d(dataset, alpha, mu, d):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    d     = torch.as_tensor(d,     device=dataset.device, dtype=torch.float32)

    theta_phys = torch.stack([alpha, mu, d])  # (3,)
    with torch.no_grad():
        return dataset.loglike(theta_phys)


# =========================
# Grid helpers
# =========================

def write_grid_csv(out_path, header, rows_iter):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows_iter:
            w.writerow(row)


def argmax_from_rows(rows, ll_col_idx):
    # rows: list of tuples; return (best_row, best_ll)
    best = None
    best_ll = -np.inf
    for r in rows:
        ll = r[ll_col_idx]
        if ll > best_ll:
            best_ll = ll
            best = r
    return best, best_ll


# =========================
# Main test runner
# =========================

def main():
    torch.manual_seed(15)
    np.random.seed(15)

    os.makedirs(OUTDIR, exist_ok=True)

    # ---- load data ----
    df = load_and_clean_real_data(DATA_PATH, cutoff=CUTOFF)
    ts     = torch.tensor(df["Day"].values * 24)       # hours
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load(KAPPA_NPZ)["kappa_samples"]

    dataset = TimeSeriesInferenceDataset(
        ts=ts,
        counts=counts,
        dils=dils,
        kappa_samples=kappa_np,
        cutoff=CUTOFF,
        device=DEVICE,
    )

    truth = (ALPHA_TRUE, MU_TRUE, D_TRUE)
    print(f"\nTruth: alpha={truth[0]:.6g}, mu={truth[1]:.6g}, d={truth[2]:.6g}\n")

    # -----------------------
    # 1D tests
    # -----------------------
    tests_1d = [
        ("alpha_only", "alpha", ALPHAS),
        ("mu_only",    "mu",    MUS),
        ("d_only",     "d",     DS),
    ]

    for name, which, grid in tests_1d:
        out_path = os.path.join(OUTDIR, f"{name}.csv")
        rows = []
        for i, x in enumerate(grid):
            a, m, d = truth
            if which == "alpha": a = float(x)
            if which == "mu":    m = float(x)
            if which == "d":     d = float(x)

            ll = float(loglike_alpha_mu_d(dataset, a, m, d).detach().cpu())
            rows.append((i, a, m, d, ll))

        write_grid_csv(out_path, ["idx", "alpha", "mu", "d", "loglike"], rows)
        best, best_ll = argmax_from_rows(rows, ll_col_idx=4)

        print(f"[1D] {name}: best alpha={best[1]:.6g}, mu={best[2]:.6g}, d={best[3]:.6g}  (ll={best_ll:.3f})")
        print(f"     wrote: {out_path}")

    # -----------------------
    # 2D tests
    # -----------------------
    tests_2d = [
        ("alpha_mu", "alpha", ALPHAS, "mu", MUS),
        ("alpha_d",  "alpha", ALPHAS, "d",  DS),
        ("mu_d",     "mu",    MUS,    "d",  DS),
    ]

    for name, w1, g1, w2, g2 in tests_2d:
        out_path = os.path.join(OUTDIR, f"{name}.csv")
        rows = []
        idx = 0
        for x in g1:
            for y in g2:
                a, m, d = truth
                if w1 == "alpha": a = float(x)
                if w1 == "mu":    m = float(x)
                if w1 == "d":     d = float(x)

                if w2 == "alpha": a = float(y)
                if w2 == "mu":    m = float(y)
                if w2 == "d":     d = float(y)

                ll = float(loglike_alpha_mu_d(dataset, a, m, d).detach().cpu())
                rows.append((idx, a, m, d, ll))
                idx += 1

        write_grid_csv(out_path, ["idx", "alpha", "mu", "d", "loglike"], rows)
        best, best_ll = argmax_from_rows(rows, ll_col_idx=4)

        print(f"[2D] {name}: best alpha={best[1]:.6g}, mu={best[2]:.6g}, d={best[3]:.6g}  (ll={best_ll:.3f})")
        print(f"     wrote: {out_path}")

    # -----------------------
    # 3D test
    # -----------------------
    out_path = os.path.join(OUTDIR, "alpha_mu_d.csv")
    rows = []
    idx = 0
    for a in ALPHAS:
        for m in MUS:
            for d in DS:
                ll = float(loglike_alpha_mu_d(dataset, float(a), float(m), float(d)).detach().cpu())
                rows.append((idx, float(a), float(m), float(d), ll))
                idx += 1

    write_grid_csv(out_path, ["idx", "alpha", "mu", "d", "loglike"], rows)
    best, best_ll = argmax_from_rows(rows, ll_col_idx=4)

    print(f"[3D] alpha_mu_d: best alpha={best[1]:.6g}, mu={best[2]:.6g}, d={best[3]:.6g}  (ll={best_ll:.3f})")
    print(f"     wrote: {out_path}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
