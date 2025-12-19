# due to the stochastic nature of the log-likelihood estimator,
# we want to test whether using common random numbers (CRN)
# reduces the variance across multiple calls to loglike().

import os, sys, time
import torch
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def snapshot_rng():
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state

def restore_rng(state):
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])

def main(real_data_path: str, repeats: int = 8, Nsamples: int = 2**15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_and_clean_real_data(real_data_path, cutoff=300)
    ts = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)
    dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)

    # pick a theta in a reasonable range
    theta = torch.tensor([1/20, 0.25, 2e5, 0.1], dtype=torch.float32, device=device)
    print("theta:", theta.detach().cpu().numpy())
    print("Nsamples:", Nsamples, "repeats:", repeats)

    # ----------------------------
    # 1) No CRN: natural repeats
    # ----------------------------
    print("\n=== No-CRN repeats (fresh randomness each call) ===")
    vals_no = []
    for i in range(repeats):
        t0 = time.time()
        ll = dataset.loglike(theta, Nsamples=Nsamples)
        ll = float(ll)  # sync + materialize
        vals_no.append(ll)
        print(f"run {i:02d}: ll={ll:.6f}  (dt={time.time()-t0:.2f}s)")
    vals_no = np.array(vals_no)
    print("No-CRN mean:", vals_no.mean(), "std:", vals_no.std(), "range:", vals_no.max()-vals_no.min())

    # -------------------------------------------------
    # 2) CRN-style: reset RNG state before *each* call
    # -------------------------------------------------
    print("\n=== CRN-style repeats (identical RNG state each call) ===")
    set_all_seeds(123)                # choose one seed
    rng0 = snapshot_rng()             # freeze RNG state

    vals_crn = []
    for i in range(repeats):
        restore_rng(rng0)             # <-- this is the CRN trick
        t0 = time.time()
        ll = dataset.loglike(theta, Nsamples=Nsamples)
        ll = float(ll)
        vals_crn.append(ll)
        print(f"run {i:02d}: ll={ll:.6f}  (dt={time.time()-t0:.2f}s)")
    vals_crn = np.array(vals_crn)
    print("CRN mean:", vals_crn.mean(), "std:", vals_crn.std(), "range:", vals_crn.max()-vals_crn.min())

    # -------------------------------------------------
    # 3) Partial-CRN: same seed, but DON'T restore state
    # (tests whether randomness is purely seed-based)
    # -------------------------------------------------
    print("\n=== Same-seed only (seed once, no restore) ===")
    set_all_seeds(123)
    vals_seed = []
    for i in range(repeats):
        t0 = time.time()
        ll = dataset.loglike(theta, Nsamples=Nsamples)
        ll = float(ll)
        vals_seed.append(ll)
        print(f"run {i:02d}: ll={ll:.6f}  (dt={time.time()-t0:.2f}s)")
    vals_seed = np.array(vals_seed)
    print("Seed-once mean:", vals_seed.mean(), "std:", vals_seed.std(), "range:", vals_seed.max()-vals_seed.min())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_crn_smoketest.py <real_data_path> [repeats] [Nsamples]")
        sys.exit(1)
    path = sys.argv[1]
    repeats = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
    Nsamples = int(sys.argv[3]) if len(sys.argv) >= 4 else 2**15
    main(path, repeats=repeats, Nsamples=Nsamples)
