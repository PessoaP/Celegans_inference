# %%
import os,sys
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils.dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils.mcmc import logprior, loglike_alpha_mu_omega, next_MCMC_sample


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# %%
grid_estimation = torch.tensor((0.04,1.,0.615)).to(device)
jumpsize_proposal = torch.tensor((1e-2,1e-2,1e-2)).to(device)


# %%
try: #read from when you called the python script the first time, to continue from the real data path
    real_data_path = sys.argv[1]
except IndexError:
    real_data_path = "real_data/Exp_1_live_lowpH.csv"
filename = os.path.basename(real_data_path) 
mcmc_output_path = os.path.join("mcmc_results", filename)

os.makedirs("mcmc_results", exist_ok=True)


# %%
df = load_and_clean_real_data(real_data_path, cutoff=300)
ts     = torch.tensor(df["Day"].values * 24)
counts = torch.tensor(df["Counts"].values)
dils   = torch.tensor(df["Dilution"].values)

kappa_np = np.load("kappa_samples_stratified.npz")["kappa_samples"]
kappa_np.sort()
kappa_np = kappa_np[np.arange(len(kappa_np)) % 2 == 0]

dataset = TimeSeriesInferenceDataset(
    ts=ts, counts=counts, dils=dils,
    kappa_samples=kappa_np,
    cutoff=300,
    t_switch=None,
    rho=1.,
    device=device,
)


# %%
N_samples=10000

samples = []
lps = []

logposterior = lambda th: loglike_alpha_mu_omega(dataset, *th) + logprior(*th)

start = grid_estimation.clone()
theta = start
lp = logposterior(theta)



# %%
for i in tqdm(range(N_samples)):
    lp = logposterior(theta)
    theta, lp = next_MCMC_sample(logposterior, theta, lp, jumpsize_proposal)
    samples.append(theta.clone().cpu().numpy())
    lps.append(lp.item()*1.)
    

    if (i + 1) % 10 == 0:
        samples_tensor = np.vstack(samples)

        df = pd.DataFrame({
            "alpha": samples_tensor[:, 0],
            "mu": samples_tensor[:, 1],
            "omega": samples_tensor[:, 2],
            "logposterior": np.array(lps)
        })

        df.to_csv(mcmc_output_path)

        print(f"Saved at iteration {i+1}")
        print (theta,lp.item())



