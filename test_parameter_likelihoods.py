# Goal: Understand if there's degeneracy due to correlation between specific varaibles (sensitivity scan).
# Method: Only allow one parameter to vary at a time, fixing the other parameters. 
# Then graph log-likelihood vs parameter value (1D profile likelihood)

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataclass import TimeSeriesInferenceDataset
from load_and_clean_real_data import load_and_clean_real_data

# === Argument: real data path ===
real_data_path = sys.argv[1]
output_dir = os.path.join("parameter_sweep_outputs")
os.makedirs(output_dir, exist_ok=True)

# === Load synthetic dataset ===
dataset = load_and_clean_real_data(real_data_path)

# === Define base parameter set ===
base_params = torch.tensor([0.05, 0.2, 1e5, 0.1])  # ground truth used to generate synthetic dataset
base_params = base_params.to(device)

param_names = ["alpha", "mu", "capacity", "d"]
n_points = 30  # how many points to evaluate the log-likelihood at for each parameter
span = 2.0   # sweep from 1/2x to 2x the unfixed parameter value

# === Sweep function ===
def profile_likelihood_scan(dataset, base_params, param_names, n_points=20, span=2.0, logscale=True):
    results = {}
    for i, name in enumerate(param_names):
        scan_vals = []
        loglikes = []

        base_val = base_params[i].item()
        if logscale:
            scan_range = torch.logspace(
                np.log10(base_val / span), np.log10(base_val * span), n_points
            )
        else:
            scan_range = torch.linspace(
                base_val / span, base_val * span, n_points
            )

        for val in scan_range:
            test_params = base_params.clone()
            test_params[i] = val
            ll = dataset.loglike(test_params).item()
            scan_vals.append(val.item())
            loglikes.append(ll)

        results[name] = (scan_vals, loglikes)

        # Save CSV
        np.savetxt(os.path.join(output_dir, f"{name}_scan.csv"),
                   np.column_stack((scan_vals, loglikes)),
                   delimiter=",", header="param_value,log_likelihood", comments='')

    return results

# === Run sweep and plot ===
results = profile_likelihood_scan(dataset, base_params, param_names,
                                  n_points=n_points, span=span)

for name, (x, y) in results.items():
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xscale('log')
    plt.xlabel(name)
    plt.ylabel("Log-likelihood")
    plt.title(f"Profile log-likelihood: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_profile.png"))
    plt.close()

