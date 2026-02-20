# Goal: sample capacity by using equal-mass strata, but many draws with randomness inside each stratum, to total Nsamples

import numpy as np

def stratified_sample_from_pmf_N(p: np.ndarray, Nsamples: int, J: int, seed: int = 0, kappa_cutoff: int = 1000) -> np.ndarray:
    """
    Draw Nsamples indices from discrete pmf p using stratification in CDF space.

    - Split [0,1) into J equal-mass strata (quantile strata).
    - Draw S = Nsamples//J uniforms per stratum.
    - Invert CDF for all uniforms.

    Returns: (Nsamples,) indices
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=np.float64)
    p *= np.arange(len(p)) > kappa_cutoff  # zero out small capacities
    p = p / p.sum()

    if Nsamples % J != 0:
        raise ValueError(f"Nsamples={Nsamples} must be divisible by J={J}")

    cdf = np.cumsum(p)
    cdf[-1] = 1.0

    S = Nsamples // J

    # J strata, each gets S random uniforms
    # u_{j,s} ~ Uniform(j/J, (j+1)/J)
    u = (np.arange(J)[:, None] + rng.random((J, S))) / J  # shape (J, S)
    u = u.reshape(-1)  # (Nsamples,)

    idx = np.searchsorted(cdf, u, side="left")
    idx = np.clip(idx, 0, len(p) - 1)
    return idx.astype(np.int64)

def load_capacity_pmf(npz_path: str):
    """
    Loads capacity distribution. Supports either:
      - saved as (kappa, p), or
      - saved as (p, Nmax)
    """
    data = np.load(npz_path)
    if "p" not in data:
        raise ValueError("NPZ must contain key 'p'.")

    p = data["p"]
    if "kappa" in data:
        kappa = data["kappa"].astype(np.int64)
    else:
        kappa = np.arange(len(p), dtype=np.int64)
    return kappa, p

def draw_fixed_kappa_samples(npz_path: str, Nsamples: int, J: int, seed: int = 0) -> np.ndarray:
    kappa, p = load_capacity_pmf(npz_path)
    idx = stratified_sample_from_pmf_N(p=p, Nsamples=Nsamples, J=J, seed=seed)
    return kappa[idx]


if __name__ == "__main__":
    import os, argparse

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity_npz", required=True)
    parser.add_argument("--J", type=int, required=True)
    parser.add_argument("--Nsamples", type=int, default=2**15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "kappa_samples_stratified.npz"),
    )
    args = parser.parse_args()

    kappa_samples = draw_fixed_kappa_samples(
        npz_path=args.capacity_npz,
        Nsamples=args.Nsamples,
        J=args.J,
        seed=args.seed,
    )

    np.savez(
        args.out,
        kappa_samples=kappa_samples,
        J=args.J,
        Nsamples=args.Nsamples,
        seed=args.seed,
    )

    print(f"Saved {args.out} (n={len(kappa_samples)})")

# How to run: 
# python utils/capacity_sampling_stratified.py \
#     --capacity_npz data/capacity_day9_repop_precalibration.npz \
#     --J 128 \
#     --Nsamples 32768 \
#     --seed 0 \
#     --out data/kappa_samples_stratified.npz