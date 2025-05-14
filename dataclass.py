import torch
import repop
import simulator
from numpy import log
from tqdm import tqdm

def simulate_for_likelihood(params, times, Nsamples=2**15):
    """Simulate population trajectories up to each time in `times`."""
    
    t,E = simulator.sample(params, N=Nsamples, T=times[0])
    simulations = [E.int()]
    for T in times[1:]:
        delta_t, E = simulator.sample(params, E_initial=1*E, N=Nsamples, T=T-t)
        t += delta_t
        simulations.append(E.int())
    return simulations

class dataset():
    def __init__(self, ts, counts, dils, cutoff=300, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
        # Convert counts and dilutions to tensors and send to device
        self.counts = torch.tensor(counts.reshape(-1, 1))
        self.dils = torch.tensor(dils.reshape(-1, 1))
        self.Ts = torch.tensor(ts.reshape(-1, 1))

        # Define upper bound for total number of cells
        self.Nmax = 2 * (counts * dils).max() + 1
        self.n = torch.arange(self.Nmax)

        self.ndatapoints = self.counts.size(0)
        self.cutoff = cutoff

        # Compute log p(k, phi | n ) via REPOP
        # Before here is better done on CPU. 
        self.lpkdil_n = repop.get_lpkdil_n(self.counts, self.dils, self.n, 
                                           cutoff, self.Nmax).to(device) 
        #only now we start on GPU
        

        # Unique measurement times and row-to-timepoint mapping
        self.times, self.T_index = torch.unique(torch.tensor(ts), return_inverse=True)
        self.T_index = self.T_index.to(device)
        print('Dataset loaded successfully.')

    def lpkdil_ns(self, ns, reduce=False, concat=False):
        """
        Selects (and optionally reduces) log p(k | n, phi) across relevant n for each time t.

        Args:
            ns (list of 1D tensors): For each t, n-values to retain.
            reduce (bool): If True, compute log-mean per datapoint across selected n's.
            concat (bool): If True, concatenate output across timepoints.

        Returns:
            Either a list of tensors (one per t) or a single concatenated tensor.
        """
        list_lpkdil_ns = []

        for t in range(len(self.times)):
            mask = (self.T_index == t)
            lpk = self.lpkdil_n[mask]  # (num_datapoints_t, Nmax)

            n_indices = ns[t]#.to(self.device)

            # Fill with -inf initially
            lpk_condensed = torch.full(
                (lpk.size(0), len(n_indices)),
                float('-inf'),
                dtype=lpk.dtype,
                device=self.device
            )

            valid_mask = n_indices < self.Nmax
            valid_n = n_indices[valid_mask]

            if valid_n.numel() > 0:
                lpk_condensed[:, valid_mask] = lpk[:, valid_n]

            if reduce:
                lpk_condensed = torch.logsumexp(lpk_condensed, dim=1) - log(len(n_indices))

            list_lpkdil_ns.append(lpk_condensed)

        if reduce and concat:
            return torch.cat(list_lpkdil_ns, dim=0)

        return list_lpkdil_ns

    def loglike(self, value, Nsamples=2**15):
        """
        Simulate `ns` using current times and compute total log-likelihood (log-sum over samples).

        Args:
            value: parameters to pass to simulator
            Nsamples: number of samples per timepoint

        Returns:
            Scalar log-likelihood estimate via log-sum-exp over ns.
        """
        ns = simulate_for_likelihood(value, self.times, Nsamples)
        log_probs = self.lpkdil_ns(ns, reduce=True, concat=True)
        #del ns
        return torch.sum(log_probs, dim=0)

    def ode_initialization(self, init=None):
        """
        Initialize parameters for ODE fitting via gradient descent.
        """
        # If no initialization is provided, start from a zero vector (log(1) = 0)
        if init is None:
            init = torch.tensor((1/24,1e-2,
                                 target[self.Ts==self.Ts.max()].median(),
                                 1e-4))

        lparams = torch.log(init.data).to('cpu')

        target = self.counts * self.dils

        # Optimizer and loss function
        lparams.requires_grad_()
        optimizer = torch.optim.Adam([lparams], lr=.01)
        l2 = lambda x: torch.sqrt((x * x).sum())  # L2 norm
        loss_hist = []

        print('Initializing using mass-action similarity')

        for it in tqdm(range(500)):
            optimizer.zero_grad()

            # Simulate ODE with current parameters
            n_ode = simulator.integrate_mass_action(torch.exp(lparams), self.Ts, dt=0.1)

            scaledtime = self.Ts/self.Ts.min()
            # Compute loss (normalized L2 relative error)
            loss = l2((target/n_ode-1)/(scaledtime**2)) *self.Ts.min()/ self.ndatapoints
            #print(target/n_ode)
            

            # Check for valid loss before applying backward
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

                loss_hist.append(loss.item())

                if (it + 1) % 10 == 0:
                    gradient_norm = l2(lparams.grad).item()
                    print('gradient:', gradient_norm, loss_hist[-1]  )

                    if gradient_norm<1e-5:
                        break
        # Return parameters in original (non-log) space
        return torch.exp(lparams).detach().to(self.device)