import os, sys, csv, argparse
import torch
import numpy as np

from dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data

def loglike_alpha_mu_omega(dataset, alpha, mu, omega):
    alpha = torch.as_tensor(alpha, device=dataset.device, dtype=torch.float32)
    mu    = torch.as_tensor(mu,    device=dataset.device, dtype=torch.float32)
    omega = torch.as_tensor(omega, device=dataset.device, dtype=torch.float32)

    d = omega * mu
    theta_phys = torch.stack([alpha, mu, d])
    with torch.no_grad():
        return dataset.loglike(theta_phys)

def chunk_indices(n, num_tasks, task_idx0):
    base = n // num_tasks
    rem = n % num_tasks
    start = task_idx0 * base + min(task_idx0, rem)
    end = start + base + (1 if task_idx0 < rem else 0)
    return start, end

def main():
    p = argparse.ArgumentParser()
    p.add_argument("real_data_path", type=str)
    p.add_argument("--task-id", type=int, default=None, help="1-based task id (SLURM_ARRAY_TASK_ID)")
    p.add_argument("--num-tasks", type=int, default=None, help="total number of tasks (e.g. 4)")
    p.add_argument("--t-switch", type=float, default=None,
               help="time (hours) when feeding stops / regime switches. None = no switch")
    p.add_argument("--rho", type=float, default=1.0,
                help="colonization scaling after t_switch (often 0.0 for time-limited feeding)")

    args = p.parse_args()

    # --- infer task split from Slurm if not provided ---
    env_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if args.task_id is None and env_task is not None:
        args.task_id = int(env_task)  # 1-based
    if args.task_id is None:
        args.task_id = 1

    if args.num_tasks is None:
        env_range = os.environ.get("SLURM_ARRAY_TASK_MAX")
        env_min   = os.environ.get("SLURM_ARRAY_TASK_MIN")
        if env_range is not None and env_min is not None:
            args.num_tasks = int(env_range) - int(env_min) + 1
        else:
            args.num_tasks = 4  # fallback

    task_idx0 = args.task_id - 1  # convert to 0-based

    # --- single process => pick one GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(15 + task_idx0)
    np.random.seed(15 + task_idx0)

    basename = os.path.splitext(os.path.basename(args.real_data_path))[0]
    output_dir = os.path.join("grid_likelihood_outputs", basename)
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean_real_data(args.real_data_path, cutoff=300)
    ts     = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils   = torch.tensor(df["Dilution"].values)

    kappa_np = np.load("kappa_samples_stratified_dropleq1000.npz")["kappa_samples"]

    dataset = TimeSeriesInferenceDataset(
        ts=ts, counts=counts, dils=dils,
        kappa_samples=kappa_np,
        cutoff=300,
        t_switch=args.t_switch,
        rho=args.rho,
        device=device,
    )

    # --- define grid ---
    alphas_full = np.linspace(0, 1/4, 26)[1:-1]   # 24
    alphas = alphas_full[:12] # job 1
    mus    = np.linspace(0, 2,   51)[1:]     # 50
    omegas = np.linspace(0, 1,   27)[1:-1]   # 25

    # --- split alphas by array task ---
    a_start, a_end = chunk_indices(len(alphas), args.num_tasks, task_idx0)
    my_alphas = alphas[a_start:a_end]

    # safety check for your intended 6 each (when 24 and 4 tasks)
    print(f"[task {args.task_id}/{args.num_tasks}] alpha idx [{a_start}:{a_end}] (n={len(my_alphas)})")

    grid = [(a, m, o) for a in my_alphas for m in mus for o in omegas]

    # --- task-specific output file ---
    out_path = os.path.join(output_dir, f"grid_loglike_task{args.task_id:02d}.csv")

    already_done = 0
    if os.path.exists(out_path):
        with open(out_path, "r", newline="") as f:
            already_done = max(0, sum(1 for _ in f) - 1)

    write_header = not os.path.exists(out_path) or already_done == 0
    flush_every = 125

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["task_id", "idx_local", "alpha", "mu", "omega", "loglike"])

        for i in range(already_done, len(grid)):
            alpha, mu, omega = grid[i]
            ll = loglike_alpha_mu_omega(dataset, alpha, mu, omega)
            ll_val = float(ll.detach().cpu())
            writer.writerow([args.task_id, i, alpha, mu, omega, ll_val])

            if (i + 1) % flush_every == 0:
                f.flush()
                os.fsync(f.fileno())

    print(f"Done. Wrote {len(grid) - already_done} rows to {out_path}")

if __name__ == "__main__":
    main()
