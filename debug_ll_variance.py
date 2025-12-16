import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import traceback

# === Imports ===
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils import mcmc
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.configure_plotting import configure_plotting
from plot_helpers import plot_intermediate_histograms, plot_logposterior_trace

# load data exactly the same way
real_data_path = sys.argv[1]
df = load_and_clean_real_data(real_data_path, cutoff=300)

# Load on CPU; dataset class moves to GPU after pre-processing
ts = torch.tensor(df["Day"].values * 24)     
counts = torch.tensor(df["Counts"].values)
dils = torch.tensor(df["Dilution"].values)
full_dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300)

prior, init = mcmc.make_prior_from_initial_guess(full_dataset, device="cuda")
u = mcmc.to_u(init)
theta = mcmc.to_theta(u)

vals = []
for _ in range(10):
    ll = float(full_dataset.loglike(theta))
    vals.append(ll)

print("ll mean:", np.mean(vals))
print("ll std :", np.std(vals))
print("ll range:", np.ptp(vals))
