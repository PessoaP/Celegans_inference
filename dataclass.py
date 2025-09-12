import torch
import repop
import simulator
from tqdm import tqdm

def simulate_for_likelihood(params, times, Nsamples=2**15):
    """Simulate population trajectories up to each time in `times`."""
    
    times = times.to(params.device)
    t,E = simulator.sample(params, N=Nsamples, T=times[0])
    simulations = [E.int()]
    for T in times[1:]:
        delta_t, E = simulator.sample(params, E_initial=1*E, N=Nsamples, T=T-t)
        t += delta_t
        simulations.append(E.int())
    return simulations

class TimeSeriesInferenceDataset():
    def __init__(self, ts, counts, dils, cutoff=300, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
        # Always process these first on CPU
        cpu = torch.device('cpu')
        def to_cpu_tensor(arr):
            if isinstance(arr, torch.Tensor):
                return arr.reshape(-1, 1).clone().detach().to(cpu)
            return torch.tensor(arr, device=cpu).reshape(-1, 1)

        self.counts = to_cpu_tensor(counts)
        self.dils   = to_cpu_tensor(dils)
        self.Ts     = to_cpu_tensor(ts)

        self.Nmax = int(2 * (self.counts * self.dils).max().item() + 1)
        self.n = torch.arange(self.Nmax, device=cpu)

        self.ndatapoints = self.counts.size(0)
        self.cutoff = cutoff

        # Now call REPOP on CPU
        self.lpkdil_n = repop.get_lpkdil_n(self.counts, self.dils, self.n, 
                                           cutoff, self.Nmax).to(self.device) 
        # Only here do we switch to the chosen device (GPU or CPU)

        # Move times and index to device after REPOP
        self.Ts = self.Ts.to(self.device)
        self.counts = self.counts.to(self.device)
        self.dils = self.dils.to(self.device)
        self.n = self.n.to(self.device)

        self.times, self.T_index = torch.unique(self.Ts, return_inverse=True)
        self.T_index = self.T_index.to(self.device)
        self.times = self.times.to(self.device)
        print(f'Dataset loaded: {self.ndatapoints} points, {len(self.times)} timepoints.')

        
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
        dataset = self  # for clarity
        lpkdil_list = []
        for index in range(len(dataset.times)):
            mask = (dataset.T_index == index).reshape(-1)
            lpkdil_n = dataset.lpkdil_n[mask]
            n_ind = ns[index]
            N_samples = len(n_ind)
            lpkdil_ns = lpkdil_n[:,n_ind[n_ind<lpkdil_n.shape[-1]]]
            if reduce:
                lpkdil_theta = torch.logsumexp(lpkdil_ns,axis=1) - torch.log(torch.tensor(N_samples))
                lpkdil_list.append(lpkdil_theta)
            else:
                lpkdil_list.append(lpkdil_ns)
        if concat:
            return torch.cat(lpkdil_list, dim=0)
        else:   
            return lpkdil_list

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
        self.last_simulated_ns = ns  # so we can save the summary statistics of the trajectories later
        log_probs = self.lpkdil_ns(ns, reduce=True, concat=True)
        return torch.sum(log_probs)

    def ode_initialization(self, init=None):
        """
        Initialize parameters for ODE fitting via gradient descent (on CPU).
        """
        # Always move everything to CPU for ODE initialization
        target = (self.counts * self.dils).cpu()
        Ts_cpu = self.Ts.cpu().float()
        ndatapoints = self.ndatapoints  # This should be an int already

        # Start from init or reasonable guess (ensure CPU)
        if init is None:
            init = torch.tensor([1/24, 1e-2, float(target[Ts_cpu == Ts_cpu.max()].median()), 1e-4], dtype=torch.float32, device='cpu')
        else:
            init = init.detach().cpu().float()

        lparams = torch.log(init).clone().detach().requires_grad_()

        optimizer = torch.optim.Adam([lparams], lr=.01)
        l2 = lambda x: torch.sqrt((x * x).sum())
        loss_hist = []

        print('Initializing using mass-action similarity (CPU)')
        for it in tqdm(range(500)):
            optimizer.zero_grad()

            # Simulate ODE with current parameters (all CPU)
            n_ode = simulator.integrate_mass_action(torch.exp(lparams), Ts_cpu, dt=0.1, device='cpu')

            scaledtime = Ts_cpu / Ts_cpu.min()
            loss = l2((target / n_ode - 1) / (scaledtime ** 2)) * Ts_cpu.min() / ndatapoints

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                loss_hist.append(loss.item())
                if (it + 1) % 10 == 0:
                    gradient_norm = l2(lparams.grad).item()
                    print('gradient:', gradient_norm, loss_hist[-1])
                    if gradient_norm < 1e-5:
                        break

        # Return parameters in original (non-log) space, **move to desired device**
        return torch.exp(lparams).detach().to(self.device)
