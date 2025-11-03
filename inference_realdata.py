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
ts = torch.as_tensor(df["Day"].values * 24, dtype=torch.float64)   # hours
counts = torch.as_tensor(df["Counts"].values, dtype=torch.long)    # MUST be integer for indexing
dils = torch.as_tensor(df["Dilution"].values, dtype=torch.float64) # dilution factors
full_dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300)


# === ODE Initialization for prior/initial guess ===
prior, initial_guess = mcmc.make_prior_from_initial_guess(full_dataset, frac_error=[0.8, 0.8, 1.5, 0.8], device=device) # loosen prior parameter to avoid sticking
infer_idx = None # all parameters will be auto-inferred if there's no specified subset
# infer_idx = [0]  # Example: infer only colonization 
ground_truth = torch.tensor([1/20, 1/4, 1e5, 0.1], device=device) 

# Overwrite non-inferred params with ground truth if they exist
if infer_idx is not None:
    for i in range(len(initial_guess)):
        if i not in infer_idx:
            initial_guess[i] = ground_truth[i]

# Initialize L
init_L = 1e-3 * torch.diag(torch.tensor((1, 1, 1, 5), device=device))
        
# === u-space target and initial state ===
logposterior_u = mcmc.make_logposterior_u(full_dataset, prior) # the posterior in u = log(θ) space
u = mcmc.to_u(initial_guess)           # carry state in u
lp_u = logposterior_u(u)
state = mcmc.SamplerState(L=init_L)  


# === MCMC Settings ===
n_burn1 = 1000
n_burn2 = 4000
n_mcmc  = 10000
sample_window = 500
total_steps = n_burn1 + n_burn2 + n_mcmc

dim = initial_guess.shape[0]
mcmc_params = torch.zeros((total_steps, dim), device=device)  # θ history
mcmc_u      = torch.zeros((total_steps, dim), device=device)  # u history
mcmc_lps    = []
start_step  = 0


# === Resume from checkpoint if exists ===
if os.path.exists(checkpoint_path):
    log("Resuming from previous checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcmc_params[:ckpt['step']] = ckpt['param_history']
    mcmc_lps = ckpt['log_lps']
    # Prefer stored u-history if present; else reconstruct from last θ
    if 'u_history' in ckpt:
        mcmc_u[:ckpt['step']] = ckpt['u_history']
        u = mcmc_u[ckpt['step']-1]
    else:
        u = mcmc.to_u(mcmc_params[ckpt['step']-1])
    lp_u = mcmc.make_logposterior_u(full_dataset, prior)(u)
    state = mcmc.SamplerState(L=ckpt.get('L', init_L), iter_num=ckpt['step'])
    state.load_state_dict(ckpt['state_dict'])
    start_step = ckpt['step']
    actual_n_burn1 = ckpt.get('actual_n_burn1', n_burn1)
    log(f"Resuming from step {start_step}")
else:
    state = mcmc.SamplerState(L=init_L)
    actual_n_burn1 = n_burn1
    log(f"ODE guess, with fixed params set to GT: {initial_guess.cpu().numpy()}")


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
        ax[i].axvline(float(initial_guess[i]), color='k', linestyle=':', linewidth=2, label='ODE initial guess')
    
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
        greedy = s < n_burn1
        adapt  = (not greedy) and (s < n_burn1 + n_burn2)

        if auto_greedy and greedy and (stalled_steps >= stall_thresh):
            greedy = False
            auto_greedy = False
            actual_n_burn1 = s + 1
            log(f"[Auto-exit] Greedy phase ended early at step {s+1} after {stall_thresh} stalled steps.")

        start_post = actual_n_burn1 + n_burn2
        u_history = mcmc_u[max(actual_n_burn1, s - sample_window):s] if adapt else None

        any_moved = False
        
        dim = initial_guess.shape[0]
        all_idx = list(range(dim)) if infer_idx is None else infer_idx

        # Per outer step:
        if infer_idx is None:
            # single full update
            u_new, lp_u_new, state, accepted = mcmc.next_MCMC_sample_u(
                logposterior_u, u, lp_u, state,
                greedy=greedy, adapt=adapt, u_history=u_history,
                output_dir=output_dir, update_idx=None
            )
            if accepted and not torch.allclose(u_new, u, atol=1e-10):
                any_moved = True
                u, lp_u = u_new, lp_u_new
        else:
         # coordinate/block updates
            for i in infer_idx:
                u_new, lp_u_new, state, accepted = mcmc.next_MCMC_sample_u(
                    logposterior_u, u, lp_u, state,
                    greedy=greedy, adapt=adapt, u_history=u_history,
                    output_dir=output_dir, update_idx=[i]
                )
                if accepted and not torch.allclose(u_new, u, atol=1e-10):
                    any_moved = True
                    u, lp_u = u_new, lp_u_new
                if not torch.isfinite(lp_u) or not torch.all(torch.isfinite(u)):
                    log(f"Invalid u-sample at step {s+1}, skipping or stopping.")
                    break

        # Convert current u to θ once per outer step for storage / summaries
        params = mcmc.to_theta(u)
        lp = float(lp_u)  # for plots/traces below

        if any_moved:
            stalled_steps = 0
        else:
            stalled_steps += 1

        mcmc_params[s] = params
        mcmc_u[s]      = u
        mcmc_lps.append(lp)

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
                'u_history':     mcmc_u[:s+1],
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

# After loop finishes, use actual_n_burn1 for burn-in
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
xlabels = ['Colonization', 'Replication', 'Capacity', 'Expulsion']

for i in range(4):
    data = posterior_samples[:, i]
    unique_vals = np.unique(data)
    if unique_vals.size > 1:
        n_bins = min(30, unique_vals.size)
        ax[i].hist(data, bins=n_bins, density=True, alpha=0.7)
    else:
        ax[i].bar(unique_vals[0], height=1.0, width=0.1, alpha=0.7)
        ax[i].set_xlim([unique_vals[0] - 0.5, unique_vals[0] + 0.5])
        ax[i].set_ylim(bottom=0)
        ax[i].text(unique_vals[0], 0.5, 'All samples identical',
                   ha='center', va='center', color='red')

    # Overlay ODE initial guess as dotted vertical line
    ax[i].axvline(float(initial_guess[i]), color='k', linestyle=':', linewidth=2, label='ODE initial guess')
    
    # Overlay posterior mean as solid vertical line
    mean_val = np.mean(data)
    ax[i].axvline(mean_val, color='r', linestyle='-', linewidth=2,
                  label='Posterior mean')

    ax[i].set_xlabel(xlabels[i])   # short names on x-axis
    ax[i].ticklabel_format(style='sci', axis='x', scilimits=(2, 3))

    if i == 0:
        ax[i].set_ylabel('Density')     # one shared y-axis label
        ax[i].legend(loc='best')

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'histograms.png'), dpi=600)
plt.savefig(os.path.join(output_dir, 'histograms.svg'), format='svg')
plt.close(fig)

lps = np.array(mcmc_lps)
burn_cutoff = actual_n_burn1 + n_burn2
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(np.arange(len(lps)), lps, label='Log-posterior')
ax.axvline(burn_cutoff, color='red', linestyle='--', label='End of burn-in')
ax.set_xlabel('Step')
ax.set_ylabel('Log posterior')
ax.set_title('Log-posterior trace')
ax.legend()
plt.savefig(os.path.join(output_dir, 'logposterior.png'), dpi=600)
plt.savefig(os.path.join(output_dir, 'logposterior.svg'), format='svg')
plt.close(fig)

log("Inference complete. Final outputs saved.")