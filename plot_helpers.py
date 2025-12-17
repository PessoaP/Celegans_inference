# plot_helpers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# === Ground truth for overlaying on plots ===
ground_truth = torch.tensor([1/20, 0.25, 1e6, 0.2])  # Whatever the data was generated with

# === Intermediate plotting functions ===
def plot_intermediate_histograms(samples, step, output_dir, n_burn1, n_burn2):
    samples_np = samples.cpu().numpy()
    burn_cutoff = n_burn1 + n_burn2

    if step <= burn_cutoff:
        samples_to_plot = samples_np[:step]
        phase = "burnin"
    else:
        samples_to_plot = samples_np[burn_cutoff:step]
        phase = "posterior"

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    xlabels = ['Colonization rate (/h)', 'Replication rate (/h)', 'Capacity', 'Expulsion rate (/h)']
    for i in range(4):
        data = samples_to_plot[:, i]
        unique_vals = np.unique(data)
        if unique_vals.size > 1:
            n_bins = min(30, unique_vals.size)
            ax[i].hist(data, bins=n_bins, density=True, alpha=0.7)
        else:
            ax[i].bar(unique_vals[0], height=1.0, width=0.1, alpha=0.7)
            ax[i].set_xlim([unique_vals[0] - 0.5, unique_vals[0] + 0.5])
            ax[i].set_ylim(bottom=0)
            ax[i].text(unique_vals[0], 0.5, 'All samples identical', ha='center', va='center', color='red')
        ax[i].set_xlabel(xlabels[i])
        
        # Overlay initial ODE guess as dotted vertical line
        ax[i].axvline(float(ground_truth[i]), color='k', linestyle=':', linewidth=2, label='Ground Truth')
    
    fig.suptitle(f"{phase.capitalize()} Histograms up to Step {step}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = os.path.join(output_dir, f"hist_step{step:04d}_{phase}.png")
    svg_path = os.path.join(output_dir, f"hist_step{step:04d}_{phase}.svg")
    fig.savefig(png_path, dpi=300)
    fig.savefig(svg_path, format='svg')
    plt.close(fig)

def plot_logposterior_trace(mcmc_lps, step, output_dir, actual_n_burn1, n_burn2):
    lps = np.array(mcmc_lps)
    burn_cutoff = actual_n_burn1 + n_burn2
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(np.arange(len(lps)), lps, label='Log-posterior')
    ax.axvline(burn_cutoff, color='red', linestyle='--', label='End of burn-in')
    ax.set_xlabel('Step')
    ax.set_ylabel('Log posterior')
    ax.set_title(f'Log-posterior trace up to step {step}')
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'logposterior_step{step:04d}.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, f'logposterior_step{step:04d}.svg'), format='svg')
    plt.close(fig)
