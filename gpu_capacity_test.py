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
from dataclass import TimeSeriesInferenceDataset
import mcmc
from simulator import ConstrainedLogNormalPrior
from load_and_clean_real_data import load_and_clean_real_data
from configure_plotting import configure_plotting

# === Configuration ===
torch.manual_seed(15)
np.random.seed(15)
#random.seed(15)

# Ensure GPU reproducibility by forcing bitwise identical outputs (use for unit tests)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

configure_plotting()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device selected:", device)

# === Load real data path from argument ===
real_data_path = sys.argv[1]
basename = os.path.splitext(os.path.basename(real_data_path))[0]
output_dir = os.path.join("inference_outputs", basename)
os.makedirs(output_dir, exist_ok=True)

logfile_path = os.path.join(output_dir, "log.txt")
checkpoint_path = os.path.join(output_dir, "checkpoint.pt")

def log(msg):
    print(msg, flush=True)
    with open(logfile_path, 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

log(f"Starting inference on: {real_data_path}")
log(f"Output dir: {output_dir}")

# === Load and preprocess data ===
df = load_and_clean_real_data(real_data_path, cutoff=300)

# Load on CPU; dataset class moves to GPU after pre-processing
ts = torch.tensor(df["Day"].values * 24)     
counts = torch.tensor(df["Counts"].values)
dils = torch.tensor(df["Dilution"].values)
full_dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300)

# === ODE Initialization for prior/initial guess ===
#prior, initial_guess = mcmc.make_prior_from_initial_guess(full_dataset, frac_error=0.5, device=device)
#infer_idx = [1,2,3]  # Example: infer everything but colonization
ground_truth = torch.tensor([1/20, 1/4, 1e5, 0.1], device=device) 

simulation_times, lls, Nsamples = [],[],[]
for i in range(10,12):#30):
    for j in range(10):
        start = time.time()
        ll = full_dataset.loglike(ground_truth,Nsamples=2**i)
        end = time.time()

        simulation_times.append(end - start)
        lls.append(ll.item())   
        Nsamples.append(2**i)
    log(f"Nsamples: {Nsamples[-1]}, Mean LL: {np.mean(lls[-10:])}, Std LL: {np.std(lls[-10:])}, Time: {np.mean(simulation_times[-10:])}")

df = pd.DataFrame({
    "Nsamples": np.array(Nsamples),
    "LL": np.array(lls),
    "Time": np.array(simulation_times)})
df.to_csv("simulation_times.csv")