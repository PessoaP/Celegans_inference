# For use in calculating the likelihood of different combinations of parameters on a grid of biologically feasible values
'''
Docstring for utils.grid_likelihood_calculation
Parameters:
-> k: capacity; preselected vector from capacity_sampling_stratified.py
-> alpha: colonization rate, represeting rate of attachment of bacteria consumed from the environment onto the gut; 
          ranges from (0,1/4]
-> mu: replication rate of bacteria adhered to the gut; ranges from (0,2]
-> omega: ratio that scales down mu , representing the decrease from expulsion/death; ranges from (0,1)
   --> in this sense, d = omega*mu
'''
import os, sys, csv
import torch
import numpy as np

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data

def loglike_alpha_mu_omega(dataset, alpha, mu, omega):
    # all scalars (python floats or 0-d tensors)print("Device selected:", device)
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    omega = torch.as_tensor(omega, device=dataset.device, dtype=torch.float32)

    d = omega * mu
    theta_phys = torch.stack([alpha, mu, d])  # (3,)

    with torch.no_grad():
        return dataset.loglike(theta_phys)

# === Set up what to run ===
def main(real_data_path):
    torch.manual_seed(15)
    np.random.seed(15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basename = os.path.splitext(os.path.basename(real_data_path))[0]
    output_dir = os.path.join("grid_likelihood_outputs", basename)
    os.makedirs(output_dir, exist_ok=True)

    # === Load and preprocess data ===
    df = load_and_clean_real_data(real_data_path, cutoff=300)

    # Load on CPU; dataset class moves to GPU after pre-processing
    ts = torch.tensor(df["Day"].values * 24)     
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)

    # 1. Load κ samples from preselected stratified samples
    kappa_np = np.load("kappa_samples_stratified.npz")["kappa_samples"]

    # 2. Construct dataset
    dataset = TimeSeriesInferenceDataset(
        ts=ts,
        counts=counts,
        dils=dils,
        kappa_samples=kappa_np,
        cutoff=300,
        device=device,
    )

    alphas = np.linspace(0, 1/4, 26)[1:]
    mus = np.linspace(0, 2, 26)[1:]
    omegas = np.linspace(0, 1, 27)[1:-1]
    grid = [(a, m, o) for a in alphas for m in mus for o in omegas]

    out_path = os.path.join(output_dir, "grid_loglike.csv")

    # --- resume support: count existing rows (minus header) ---
    already_done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", newline="") as f:
            already_done = max(0, sum(1 for _ in f) - 1)

    write_header = not os.path.exists(out_path) or already_done == 0

    flush_every = 125

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["idx", "alpha", "mu", "omega", "loglike"])

        for i in range(already_done, len(grid)):
            alpha, mu, omega = grid[i]
            ll = loglike_alpha_mu_omega(dataset, alpha, mu, omega)
            ll_val = float(ll.detach().cpu())

            writer.writerow([i, alpha, mu, omega, ll_val])

            # periodic flush just in case job needs to be killed early
            if (i + 1) % flush_every == 0:
                f.flush()
                os.fsync(f.fileno())

    print(f"Done. Wrote {len(grid) - already_done} new rows to {out_path}")

if __name__ == "__main__":
    main(sys.argv[1])
