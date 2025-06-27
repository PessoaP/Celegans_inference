# %%
import torch
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import ticker 

torch.manual_seed(42)

from dataclass import dataset
import mcmc

# %% [markdown]
# $$
# \begin{array}{|c|l|c|c|}
# \hline
#  & \text{Reaction} & \text{Stoichiometry} & \text{Rate} \\
# \hline
# \text{Colonization} & \emptyset \to E & +1 & \alpha \\
# \hline
# \text{Replication} & E \to 2E & +1 & \mu E \\
# \hline
# \text{Competition} & E \to \emptyset & -1 & \left(\frac{\mu}{K}\right) E^2 \\
# \hline
# \text{Expulsion} & E \to \emptyset & -1 & dE \\
# \hline
# \end{array}
# $$
# 

# %%
# Loading the synthetic dataset
dseed = 0
df = pd.read_csv('synthetic_data/synth_seed{}.csv'.format(dseed))
df

# %%
#Loading the ground truth

gt = pd.read_csv('synthetic_data/gt_map.csv', header=None, delim_whitespace=True).to_numpy()
gt = torch.tensor(gt[gt[:,0]==dseed][0,1:]).float()
gt

# %%
# The dataset need to be send to the `dataset` class
Ts     = df['Time'].to_numpy()
counts = df['Counts'].to_numpy()
dils   = df['Dilution'].to_numpy()
data = dataset(Ts,counts,dils)
print('It should run on: ',data.device)


logposterior = lambda x: mcmc.logposterior(x,data) 

# %%
lp_gt = logposterior(gt.to(data.device))
lp_gt

# %%
params = mcmc.prior.sample()

params = data.ode_initialization(params)
lp = logposterior(params)

# %%
params

# %%
### Does the initialization makes sense?

t = df['Time'].to_numpy()
naive_n = df['Counts'].to_numpy()*df['Dilution'].to_numpy()

import simulator
import numpy as np

y = simulator.integrate_mass_action(gt,Ts).reshape(-1)
plt.plot(Ts,np.log10(y),color='k',label='GT mass action')
y = simulator.integrate_mass_action(params.cpu(),Ts).reshape(-1)
plt.plot(Ts,np.log10(y),color='r',label=' initialization mass action')

plt.scatter(Ts,np.log10(naive_n))
plt.legend()
plt.savefig('Initialization_result.png',dpi=600)

# %%
initial_guess=params 

n_burn1 = 200
n_burn2 = 500
n_mcmc = 2000 
sample_window = 200

    
dim = 4
params = initial_guess.clone().detach()#.double()
lp = logposterior(params)

mcmc_params = torch.zeros((n_burn1+n_burn2+n_mcmc,dim),device=params.device)
mcmc_lps = []

# scalar proposal for burn1

total_steps = n_burn1 + n_burn2 + n_mcmc

# %%
greedy = False
for s in tqdm(range(total_steps)):
    greedy = s < n_burn1
    adapt = (not greedy) and (s < n_burn1+n_burn2)

    # ---- Step and store ---
    if adapt:
        start = max(n_burn1, s - sample_window) -1 # To initate the Gaussian noise padding during adaptive steps if the sample history isn't long enough yet
        sample_history = (mcmc_params[start:s])
    else:
        sample_history = None   # for when we want to not call an adaptive step
        
    params, lp = mcmc.next_MCMC_sample(logposterior, params, lp, 
                                       greedy=greedy, adapt=adapt,
                                       sample_history=sample_history)
    mcmc_params[s] = 1*params
    mcmc_lps.append(lp.item())

    if (s + 1) % 10 == 0:
        print(f"[{s+1}] lp = {lp.item():.3f}")
        print(params.cpu())
        print('If it is running on GPU it should say CUDA:', params.device )

        #It will save the full chain so far
        posterior_samples = mcmc_params[:s].cpu().numpy()
        np.savetxt('partial_samples.csv',posterior_samples)
        np.savetxt('partial_lps.csv',mcmc_lps)

posterior_samples = mcmc_params[:s].cpu().numpy()
np.savetxt('final_samples.csv',posterior_samples[-n_mcmc:]) #Samples with burnin removed
np.savetxt('final_lps.csv',mcmc_lps[-n_mcmc:])




# %%
posterior_samples = (mcmc_params[-n_mcmc:]).cpu().numpy()

# %%
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
titles = ['Colonization rate (/h)','Replication rate (/h)','Capacity','Death rate (/h)']
for i in range(4):
    ax[i].hist(posterior_samples[:, i], bins=30, density=True, alpha=0.7)
    ax[i].set_xlabel(titles[i])
    ax[i].axvline(gt[i].item(),color='r')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
[axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) for axi in ax]
[axi.ticklabel_format(style='sci', axis='x', scilimits=(0, 0)) for axi in ax]

ax[0].set_ylabel('Density')
fig.tight_layout()
plt.savefig('histograms.png',dpi=600)
plt.show()

# %%
plt.plot(mcmc_lps)
plt.axhline(lp_gt.item(),color='r')
plt.ylabel('Log posterior')
ly = max(np.max(mcmc_lps),lp_gt+1)
plt.ylim(ly-100,ly)

plt.savefig('logpost.png',dpi=600)

# %%



