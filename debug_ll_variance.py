import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import traceback

# fix random seeds for reproducibility
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


# === Imports ===
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils import mcmc
from utils.load_and_clean_real_data import load_and_clean_real_data

# load data exactly the same way
real_data_path = sys.argv[1]
df = load_and_clean_real_data(real_data_path, cutoff=300)

# Load on CPU; dataset class moves to GPU after pre-processing
ts = torch.tensor(df["Day"].values * 24)     
counts = torch.tensor(df["Counts"].values)
dils = torch.tensor(df["Dilution"].values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)
prior, init = mcmc.make_prior_from_initial_guess(full_dataset, device=device)

u = mcmc.to_u(init)
theta = mcmc.to_theta(u)

vals = []
for _ in range(10):
    ll = float(full_dataset.loglike(theta))
    vals.append(ll)

print("ll mean:", np.mean(vals))
print("ll std :", np.std(vals))
print("ll range:", np.ptp(vals))

theta_fixed = torch.tensor([1/20, 0.25, 2e5, 0.2], device=device, dtype=torch.float32) # ground truth theta

def summarize(theta, label):
    vals = [float(full_dataset.loglike(theta)) for _ in range(10)]
    print(label)
    print(" theta:", theta.detach().cpu().numpy())
    print(" mean:", np.mean(vals), "std:", np.std(vals), "range:", np.ptp(vals))

summarize(theta, "ODE init theta")
summarize(theta_fixed, "Fixed theta")

theta_fixed = torch.tensor([1/20, 0.25, 2e5, 0.1], device=device, dtype=torch.float32)

def summarize(theta, label):
    vals = [float(full_dataset.loglike(theta)) for _ in range(10)]
    print(label)
    print(" theta:", theta.detach().cpu().numpy())
    print(" mean:", np.mean(vals), "std:", np.std(vals), "range:", np.ptp(vals))

summarize(theta, "ODE init theta")
summarize(theta_fixed, "Fixed theta")

_ = full_dataset.loglike(theta)
for ti, n in enumerate(full_dataset.last_simulated_ns):
    n_cpu = n.detach().cpu().numpy()
    print(ti, "n mean", n_cpu.mean(), "std", n_cpu.std(), "max", n_cpu.max(), "p99", np.percentile(n_cpu, 99))

