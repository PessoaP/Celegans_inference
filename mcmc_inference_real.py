import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import ticker

# === Imports ===
from dataclass import dataset
import mcmc
from load_and_clean_real_data import load_and_clean_real_data
from configure_plotting import configure_plotting

# === Configuration ===
torch.manual_seed(42)
configure_plotting()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device selected:", device)

# === Load real data path from argument ===
real_data_path = sys.argv[1]
basename = os.path.splitext(os.path.basename(real_data_path))[0]
output_dir = os.path.join("inference_outputs", basename)
os.makedirs(output_dir, exist_ok=True)
print(f"Running inference on: {real_data_path}")
print(f"Output dir: {output_dir}")

# === Load and preprocess data ===
df = load_and_clean_real_data(real_data_path, cutoff=300)
days = [1, 3, 5, 7, 9]

# === Construct per-day datasets ===
ts_datasets = []
for day in days:
    df_day = df[df["Day"] == day]
    ts = np.full(len(df_day), day * 24)
    counts = df_day["Counts"].to_numpy()
    dils = df_day["Dilution"].to_numpy()
    ts_datasets.append(dataset(ts, counts, dils, cutoff=300))

# === Logposterior wrapper ===
def logposterior(params):
    return sum(ds.loglike(params) for ds in ts_datasets) + mcmc.prior.log_prob(params).sum()

# === ODE Initialization ===
initial_guess = ts_datasets[0].ode_initialization(mcmc.prior.sample()).detach()
params = initial_guess.clone()
lp = logposterior(params)

# === MCMC Settings ===
n_burn1 = 2000
n_burn2 = 2000
n_mcmc  = 5000
sample_window = 200
dim = 4

total_steps = n_burn1 + n_burn2 + n_mcmc
mcmc_params = torch.zeros((total_steps, dim), device=device)
mcmc_lps = []

# === MCMC Sampling ===
print("Starting MCMC...")
for s in tqdm(range(total_steps)):
    greedy = s < n_burn1
    adapt  = (not greedy) and (s < n_burn1 + n_burn2)

    if adapt:
        start = max(n_burn1, s - sample_window)
        sample_history = mcmc_params[start:s]
    else:
        sample_history = None

    params, lp = mcmc.next_MCMC_sample(
        logposterior, params, lp,
        greedy=greedy,
        adapt=adapt,
        sample_history=sample_history
    )

    mcmc_params[s] = params
    mcmc_lps.append(lp.item())

    if (s + 1) % 100 == 0:
        print(f"Saving intermediate logs at step {s+1}")
        print(f"[{s+1}] lp = {lp.item():.3f}")
        print(params.cpu())
        np.savetxt(os.path.join(output_dir, 'samples_so_far.csv'), mcmc_params[:s].cpu().numpy())
        np.savetxt(os.path.join(output_dir, 'logposterior_so_far.csv'), np.array(mcmc_lps))

# === Save Final Outputs ===
posterior_samples = mcmc_params[-n_mcmc:].cpu().numpy()
np.savetxt(os.path.join(output_dir, 'samples.csv'), posterior_samples[-n_mcmc:]) #Samples with burnin removed
np.savetxt(os.path.join(output_dir, 'logposterior.csv'), mcmc_lps[-n_mcmc:])

# === Optional: Posterior Summary Table ===
df_summary = pd.DataFrame(posterior_samples, columns=['Colonization', 'Replication', 'Capacity', 'Expulsion'])
df_summary.describe().to_csv(os.path.join(output_dir, 'posterior_summary.csv'))


# === Plot Parameter Histograms ===
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Colonization rate (/h)', 'Replication rate (/h)', 'Capacity', 'Expulsion rate (/h)']
for i in range(4):
    ax[i].hist(posterior_samples[:, i], bins=30, density=True, alpha=0.7)
    ax[i].set_xlabel(titles[i])
    ax[i].set_ylabel('Density')
    ax[i].ticklabel_format(style='sci', axis='both', scilimits=(2, 3))

fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'histograms.png'), dpi=600)

# === Plot Log Posterior Trace ===
plt.figure()
plt.plot(mcmc_lps)
plt.ylabel('Log posterior')
plt.xlabel('MCMC Step')
plt.savefig(os.path.join(output_dir, 'logposterior.png'), dpi=600)
plt.close()

print("Finished!")
print(f"Output saved to: {output_dir}")
