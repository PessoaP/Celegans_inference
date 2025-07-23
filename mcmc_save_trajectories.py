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
prior, initial_guess = mcmc.make_prior_from_initial_guess(full_dataset, frac_error=0.5, device=device)
dim = initial_guess.shape[0]
logposterior = mcmc.make_logposterior(full_dataset, prior) # calculate logposterior based on all data from all days

init_L = 1e-2 * torch.diag(torch.tensor((1, 1, 1, 5), device=device))

# === MCMC Settings ===
n_burn1 = 2000
n_burn2 = 0
n_mcmc  = 8000
sample_window = 200
total_steps = n_burn1 + n_burn2 + n_mcmc

mcmc_params = torch.zeros((total_steps, dim), device=device)
mcmc_lps = []

start_step = 0

# === Resume from checkpoint if exists ===
if os.path.exists(checkpoint_path):
    log("Resuming from previous checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcmc_params = torch.zeros((total_steps, dim), device=device)
    mcmc_lps = []
    mcmc_params[:ckpt['step']] = ckpt['param_history']
    mcmc_lps = ckpt['log_lps']
    params = mcmc_params[ckpt['step']-1]
    lp = mcmc_lps[-1]
    if 'L' in ckpt:
        state = mcmc.SamplerState(L=ckpt['L'], iter_num=ckpt['step'])
    else:
        state = mcmc.SamplerState(L=init_L, iter_num=ckpt['step'])
    state.load_state_dict(ckpt['state_dict'])
    start_step = ckpt['step']
    # Restore actual_n_burn1 or default to planned n_burn1
    actual_n_burn1 = ckpt.get('actual_n_burn1', n_burn1)
    log(f"Resuming from step {start_step}")
else:
    params = initial_guess
    lp = logposterior(params)
    state = mcmc.SamplerState(L=init_L)
    actual_n_burn1 = n_burn1
    log(f"Starting from ODE-initialized params: {initial_guess.cpu().numpy()}")


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
    titles = ['Colonization rate (/h)', 'Replication rate (/h)', 'Capacity', 'Expulsion rate (/h)']
    for i in range(4):
        ax[i].hist(samples_to_plot[:, i], bins=30, density=True, alpha=0.7)
        ax[i].set_xlabel(titles[i])
        ax[i].set_ylabel('Density')
        ax[i].ticklabel_format(style='sci', axis='both', scilimits=(2, 3))
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


# === Main MCMC Loop ===
log("Starting MCMC sampling...")

stall_thresh = 20  # How many consecutive steps without movement triggers exit from greedy
stalled_steps = 0  # Counter for consecutive unmoved steps
auto_greedy = True  # Flag for whether to override greedy based on stalling
actual_n_burn1 = n_burn1  # Track the true number of greedy steps performed

# Set up file to save summary statistics of parameter proposal Gillespie runs at Day 1, 3, 5, 7, and 9
summary_path = os.path.join(output_dir, 'all_proposals_summaries.csv')
if not os.path.exists(summary_path):
    pd.DataFrame(columns=['proposal_idx','colonization','replication','capacity','expulsion',
                          'time','mean','std','median','q25','q75']).to_csv(summary_path, index=False)

def save_summaries(proposal_idx, params, sim_times, ns, target_days=[1, 3, 5, 7, 9]):
    target_times = np.array(target_days) * 24
    pop_mat = np.stack([n.cpu().numpy() for n in ns])  # shape: (n_sim_times, n_samples)
    interp_pops = np.empty((len(target_times), pop_mat.shape[1]))  # (n_targets, n_samples)
    for sample_idx in range(pop_mat.shape[1]):
        interp_pops[:, sample_idx] = np.interp(target_times, sim_times, pop_mat[:, sample_idx])
    rows = []
    for i, t in enumerate(target_times):
        vals = interp_pops[i, :]
        row = {
            'proposal_idx': proposal_idx,
            'colonization': float(params[0].cpu()),
            'replication': float(params[1].cpu()),
            'capacity': float(params[2].cpu()),
            'expulsion': float(params[3].cpu()),
            'time': float(t),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'median': float(np.median(vals)),
            'q25': float(np.percentile(vals, 25)),
            'q75': float(np.percentile(vals, 75)),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, mode='a', header=False, index=False)


try:
    for s in tqdm(range(start_step, total_steps)):
        # Standard greedy/adapt calculation
        greedy = s < n_burn1
        adapt  = (not greedy) and (s < n_burn1 + n_burn2)

        # Override greedy if we've detected a stall
        if auto_greedy and greedy and (stalled_steps >= stall_thresh):
            greedy = False
            auto_greedy = False
            actual_n_burn1 = s + 1  # +1 since s is 0-based
            log(f"[Auto-exit] Greedy phase ended early at step {s+1} after {stall_thresh} stalled steps.")
            
        # Dynamically update the length of the burn-in    
        start_post = actual_n_burn1 + n_burn2

        # Use actual_n_burn1 in history calculation to reflect early exit if it happened
        sample_history = mcmc_params[max(actual_n_burn1, s - sample_window):s] if adapt else None

        params_new, lp_new, state, accepted = mcmc.next_MCMC_sample(
            logposterior, params, lp, state,
            greedy=greedy, adapt=adapt, sample_history=sample_history, output_dir=output_dir
        )

        # Detect movement: only reset stall count if params actually changed
        if accepted and not torch.allclose(params_new, params, atol=1e-10):
            stalled_steps = 0
        else:
            stalled_steps += 1

        params = params_new
        lp = lp_new

        if not torch.isfinite(lp) or not torch.all(torch.isfinite(params)):
            log(f"Invalid sample at step {s+1}, skipping or stopping.")
            break

        mcmc_params[s] = params
        mcmc_lps.append(float(lp))
        
        # Save simualted Gillespie trajectories, but only after burn-in (posterior samples)
        if s >= start_post:
            proposal_idx = s - start_post
            ns = full_dataset.last_simulated_ns
            sim_times = full_dataset.times.cpu().numpy().reshape(-1)
            save_summaries(proposal_idx, params, sim_times, ns)
            
        # Checkpoint every 100 steps
        if (s + 1) % 100 == 0 or (s + 1) == total_steps:
            log(f"[{s+1}] lp = {float(lp):.3f}")
            ckpt = {
                'step': s + 1,
                'state_dict': state.state_dict(),
                'param_history': mcmc_params[:s+1],
                'log_lps': mcmc_lps,
                'L': state.L,
                'actual_n_burn1': actual_n_burn1 
            }
            torch.save(ckpt, checkpoint_path)

            # Lightweight .csv snapshot
            np.savetxt(os.path.join(output_dir, 'samples_so_far.csv'), mcmc_params[:s+1].cpu().numpy())
            np.savetxt(os.path.join(output_dir, 'logposterior_so_far.csv'), np.array(mcmc_lps))

        # Plotting every 200 steps 
        if ((s + 1) % 200 == 0) or ((s + 1) == total_steps):
            plot_intermediate_histograms(mcmc_params[:s+1], s+1, output_dir, actual_n_burn1, n_burn2) # Plot histograms of parms up to this step
            plot_logposterior_trace(mcmc_lps, s+1, output_dir, actual_n_burn1, n_burn2)                   # Plot log-posterior trace

except Exception as e:
    log(f"ERROR at step {s+1}: {e}")
    traceback.print_exc()
    if 'ckpt' in locals():
        torch.save(ckpt, os.path.join(output_dir, f"crashdump_step{s+1}.pt"))
    raise

# After loop finishes, you can use actual_n_burn1 for burn-in
print(f"Actual number of greedy steps: {actual_n_burn1}")

# === Save Final Outputs (and check to see if anything was cut short) ===
# Calculate the end index for the posterior region
start_post = actual_n_burn1 + n_burn2
end_post = start_post + n_mcmc

# Check if available samples are enough, and trim if not
max_available = len(mcmc_params)
if end_post > max_available:
    print(f"WARNING: Only {max_available - start_post} posterior samples available (requested {n_mcmc}).")
    end_post = max_available

posterior_samples = mcmc_params[start_post:end_post].cpu().numpy()
np.savetxt(os.path.join(output_dir, 'samples.csv'), posterior_samples)
np.savetxt(os.path.join(output_dir, 'logposterior.csv'), mcmc_lps[start_post:end_post])

df_summary = pd.DataFrame(posterior_samples, columns=['Colonization', 'Replication', 'Capacity', 'Expulsion'])
df_summary.describe().to_csv(os.path.join(output_dir, 'posterior_summary.csv'))

# === Plot Outputs ===
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Colonization rate (/h)', 'Replication rate (/h)', 'Capacity', 'Expulsion rate (/h)']
for i in range(4):
    ax[i].hist(posterior_samples[:, i], bins=30, density=True, alpha=0.7)
    ax[i].set_xlabel(titles[i])
    ax[i].set_ylabel('Density')
    ax[i].ticklabel_format(style='sci', axis='both', scilimits=(2, 3))
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'histograms.png'), dpi=600)
plt.savefig(os.path.join(output_dir, 'histograms.svg'), format='svg')

plt.figure()
plt.plot(mcmc_lps)
plt.ylabel('Log posterior')
plt.xlabel('MCMC Step')
plt.savefig(os.path.join(output_dir, 'logposterior.png'), dpi=600)
plt.savefig(os.path.join(output_dir, 'logposterior.svg'), format='svg')
plt.close()

log("Inference complete. Final outputs saved.")