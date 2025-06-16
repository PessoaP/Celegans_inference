import torch
import pandas as pd
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib import ticker

torch.manual_seed(42)

from dataclass import dataset
import mcmc

## Checking that GPU is available ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device selected:", device)

if torch.cuda.is_available():
    print("CUDA is available. Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is NOT available. Running on CPU.")

"""$$
\begin{array}{|c|l|c|c|}
\hline
 & \text{Reaction} & \text{Stoichiometry} & \text{Rate} \\
\hline
\text{Colonization} & \emptyset \to E & +1 & \alpha \\
\hline
\text{Replication} & E \to 2E & +1 & \mu E \\
\hline
\text{Competition} & E \to \emptyset & -1 & \left(\frac{\mu}{K}\right) E^2 \\
\hline
\text{Expulsion} & E \to \emptyset & -1 & dE \\
\hline
\end{array}
$$

"""

# Loading the synthetic dataset
dseed = 0
df = pd.read_csv('synthetic_data/synth_seed{}.csv'.format(dseed))
df

#Loading the ground truth

gt = pd.read_csv('synthetic_data/gt_map.csv', header=None, delim_whitespace=True).to_numpy()
gt = torch.tensor(gt[gt[:,0]==dseed][0,1:]).float()
gt

# The dataset need to be send to the `dataset` class
Ts     = df['Time'].to_numpy()
counts = df['Counts'].to_numpy()
dils   = df['Dilution'].to_numpy()
data = dataset(Ts,counts,dils)
print('It should run on: ',data.device)


logposterior = lambda x: mcmc.logposterior(x,data)
#logposterior = lambda x: data.loglike(x)

lp_gt = logposterior(gt.to(data.device))
lp_gt

params = mcmc.prior.sample()

params = data.ode_initialization(params)
lp = logposterior(params)
print ('Ground truth log posterior', lp)

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
plt.savefig('Initialization_result.png',dpi=600)

initial_guess=params

n_burn1=200
n_burn2=0
n_mcmc=2000


dim = 4
params = initial_guess.clone().detach()#.double()
lp = logposterior(params)

mcmc_params = []
mcmc_lps = []

L = 1e-2 *torch.tensor((1,1,1,5),device=data.device)  # scalar proposal for burn1
#cov_init = 1e-4*torch.eye(dim).to(data.device)

total_steps = n_burn1 + n_burn2 + n_mcmc

greedy = False
for s in tqdm(range(total_steps)):
    greedy = s < n_burn1

    # ---- Step and store ----
    params, lp = mcmc.next_MCMC_sample(logposterior,params, lp, L=L, greedy=greedy)
    mcmc_params.append(params*1)
    mcmc_lps.append(lp.item())

    if (s + 1) % 10 == 0:
        print(f"[{s+1}] lp = {lp.item():.3f}")
        print(params.cpu())
        print('If it is running on GPU it should say CUDA:', params.device )

        #It will save the full chain so far
        posterior_samples = np.array([s.cpu().numpy() for s in mcmc_params])
        np.savetxt('samples_so_far.csv',posterior_samples)

posterior_samples = torch.stack(mcmc_params[-n_mcmc:]).cpu().numpy()

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
plt.show()

plt.plot(mcmc_lps)
plt.axhline(lp_gt.item(),color='r')
plt.ylabel('Log posterior')

