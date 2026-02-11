# make_synthetic_data.py
import os
import numpy as np
import pandas as pd
import torch

from utils.simulator import sample


# =========================
# CONFIG 
# =========================

# Output
OUT_CSV = "synthetic_data/synthetic_data_TL_seed10.csv"

# Simulation design
WORMS_PER_DAY = 75
DAYS = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)  # in days; code converts to hours

# Forward-model params (alpha, mu, d are scalars; kappa comes from distribution)

ALPHA = 0.05
MU = 0.5
D = 0.12

# Kappa distribution file
KAPPA_NPZ = "synthetic_data/kappa_samples_Exp1_4gauss.npz"

# RNG seed for reproducibility of kappa draws
SEED = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helpers
# =========================

def load_kappa_pool(npz_path: str) -> np.ndarray:
    """
    Load a 1D pool of kappa samples from an .npz.
    """
    z = np.load(npz_path)
    return z["kappa_samples"].reshape(-1)


def build_params_per_trajectory(
    alpha: float,
    mu: float,
    d: float,
    kappa_pool: np.ndarray,
    N: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns params of shape (4, N): [alpha, mu, k, d] per trajectory.
    Draws k ~ kappa_pool with replacement.
    """
    rng = np.random.default_rng(seed)
    k_draws = rng.choice(kappa_pool, size=N, replace=True)

    params = torch.empty((4, N), device=device, dtype=torch.float32)
    params[0, :] = float(alpha)
    params[1, :] = float(mu)
    params[2, :] = torch.tensor(k_draws, device=device, dtype=torch.float32)
    params[3, :] = float(d)

    return params


def simulate_and_dilute(params_4xN: torch.Tensor, T_hours: torch.Tensor,
                        t_switch: float = 24.0, rho: float = 0.1):
    N = T_hours.numel()
    device = params_4xN.device
    T_hours = T_hours.to(device=device, dtype=torch.float32).reshape(-1)

    # Base params (stage 1): alpha unchanged
    params_pre = params_4xN

    # Stage 2 params: alpha scaled
    params_post = params_4xN.clone()
    params_post[0, :] = params_post[0, :] * float(rho)

    t0 = float(t_switch)

    # Everyone runs stage 1 up to min(T, t_switch)
    T1 = torch.minimum(T_hours, torch.full_like(T_hours, t0))
    _, E1 = sample(params_pre, T=T1, N=N, device=device)

    # If no one goes past switch, we’re done
    mask_post = T_hours > t0
    E_final = E1

    # For those with T > t_switch, continue from E(t_switch) for duration (T - t_switch)
    if torch.any(mask_post):
        T2 = (T_hours[mask_post] - t0)
        _, E2 = sample(
            params_post[:, mask_post],
            E_initial=E1[mask_post],
            T=T2,
            N=int(mask_post.sum().item()),
            device=device,
        )
        E_final = E_final.clone()
        E_final[mask_post] = E2

    n0 = E_final.to(torch.float32)

    # dilution steps exactly as you wrote...
    tube1 = torch.distributions.Binomial(total_count=n0, probs=10 / 200).sample()
    tube2 = torch.distributions.Binomial(total_count=tube1, probs=10 / 100).sample()
    c1 = tube1 - tube2
    tube3 = torch.distributions.Binomial(total_count=tube2, probs=10 / 100).sample()
    c2 = tube2 - tube3
    tube4 = torch.distributions.Binomial(total_count=tube3, probs=10 / 100).sample()
    c3 = tube3 - tube4

    return c1.to(torch.int32).cpu(), c2.to(torch.int32).cpu(), c3.to(torch.int32).cpu()

# =========================
# Main
# =========================

def main():
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)

    # Load kappa distribution
    kappa_pool = load_kappa_pool(KAPPA_NPZ)
    print(f"Loaded kappa pool from {KAPPA_NPZ} (n={kappa_pool.size}). "
          f"min={kappa_pool.min():.3g}, max={kappa_pool.max():.3g}")

    # Build times: each day repeated WORMS_PER_DAY times
    days = DAYS.clone()
    n_timepoints = days.numel()
    N = int(WORMS_PER_DAY * n_timepoints)

    # Convert to hours and expand to per-trajectory vector
    T_hours = (days.to(torch.float32).repeat_interleave(WORMS_PER_DAY) * 24.0).to(DEVICE)

    # Build params per trajectory with kappa draws
    params_4xN = build_params_per_trajectory(
        alpha=ALPHA, mu=MU, d=D, kappa_pool=kappa_pool, N=N, seed=SEED, device=DEVICE
    )

    # Forward simulate + dilute
    c1, c2, c3 = simulate_and_dilute(params_4xN, T_hours, t_switch=24.0, rho=0.1)

    # Assemble dataframe (wide format)
    df = pd.DataFrame({
        "Worm #": np.tile(np.arange(1, WORMS_PER_DAY + 1), n_timepoints).astype(int),
        "CFU_22": c1.numpy().astype(int),
        "CFU_222": c2.numpy().astype(int),
        "CFU_2222": c3.numpy().astype(int),
        "Day": np.repeat(days.cpu().numpy().astype(int), WORMS_PER_DAY).astype(int),
    })

    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote synthetic data: {OUT_CSV}")
    print(df.head(10))


if __name__ == "__main__":
    main()