import torch
import repop
from recovery_tests.grid_TL.TL_with_inferred_rho import simulator_TL
from tqdm import tqdm


def simulate_for_likelihood(params, times):
    """
    Simulate ONCE up to max(times) and record E snapshots at each requested time.
    """
    N = int(params.shape[1])  # number of worms simulated

    device = params.device
    times = times.to(device).reshape(-1)
    times = torch.sort(times).values  # optional but safe

    return simulator_TL.sample_record(params=params, record_times=times, N=N, device=device)


class TimeSeriesInferenceDataset():
    def __init__(self, ts, counts, dils, kappa_samples, cutoff=300, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
        def to_device_tensor(arr):
            if isinstance(arr, torch.Tensor):
                return arr.reshape(-1, 1).clone().detach().to(device)
            return torch.tensor(arr, device=device).reshape(-1, 1)

        self.counts = to_device_tensor(counts)
        self.dils   = to_device_tensor(dils)
        self.Ts     = to_device_tensor(ts)

        # --- Store fixed capacity samples (Day-9 learned) ---
        if not isinstance(kappa_samples, torch.Tensor):
            kappa_samples = torch.tensor(kappa_samples)
        self.kappa_samples = kappa_samples.to(self.device).reshape(-1).float()  # each individual worm's capacity for simulations

        self.Nmax = int(2 * (self.counts * self.dils).max().item() + 1)
        self.n = torch.arange(self.Nmax, device=device)

        self.ndatapoints = self.counts.size(0)
        self.cutoff = cutoff

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

        
    def lpkdil_ns(self, ns, reduce=True, concat=True):
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
            #mask_cpu = mask.to("cpu")                 # CPU mask for indexing lpkdil_n (which is on CPU)
            lpkdil_n = dataset.lpkdil_n[mask]    

            n_ind_full = ns[index].to(device=lpkdil_n.device, dtype=torch.long)
            M = n_ind_full.numel()

            valid = (n_ind_full >= 0) & (n_ind_full < lpkdil_n.shape[-1])

            if valid.sum() == 0:
                raise RuntimeError(
                    f"No valid n within support at time index {index}. "
                    f"Max n in ns = {int(n_ind_full.max())}, Nmax = {lpkdil_n.shape[-1]}."
                )

            # Allocate (ndata_at_time, M)
            lpkdil_ns = torch.full(
                (lpkdil_n.shape[0], M),
                -torch.inf,
                device=lpkdil_n.device,
                dtype=lpkdil_n.dtype,
            )

            # fill only valid Monte Carlo samples (i.e., those within [0, Nmax))
            lpkdil_ns[:, valid] = lpkdil_n[:, n_ind_full[valid]]

            log_N_samples = torch.log(
                torch.tensor(M, device=lpkdil_ns.device, dtype=lpkdil_ns.dtype)
            )

            if reduce:
                lpkdil_theta = torch.logsumexp(lpkdil_ns,axis=1) - log_N_samples
                lpkdil_list.append(lpkdil_theta)
            else:
                lpkdil_list.append(lpkdil_ns)
        if concat:
            return torch.cat(lpkdil_list, dim=0)
        else:   
            return lpkdil_list
    
    def _build_params_with_capacity(self, theta_phys):
        """
        theta_phys: (4,) = [alpha, mu, d, rho]
        Returns: params (5, Nsamples) = [alpha, mu, k, d, rho]
        """
        theta_phys = theta_phys.to(self.device).float().reshape(-1)
        if theta_phys.numel() != 4:
            raise ValueError(
                f"theta_phys must have 4 entries [alpha, mu, d, rho]; got {tuple(theta_phys.shape)}"
            )

        alpha, mu, d, rho = theta_phys[0], theta_phys[1], theta_phys[2], theta_phys[3]

        # optional sanity (I recommend)
        if torch.any(rho < 0):
            raise ValueError("rho must be >= 0 (otherwise alpha_eff can go negative).")

        k_vec = self.kappa_samples.to(self.device).float().reshape(-1)
        Nsamples = k_vec.numel()

        alpha_vec = alpha.expand(Nsamples)
        mu_vec    = mu.expand(Nsamples)
        d_vec     = d.expand(Nsamples)
        rho_vec   = rho.expand(Nsamples)

        params = torch.stack([alpha_vec, mu_vec, k_vec, d_vec, rho_vec], dim=0)
        return params

    def loglike(self, theta_phys):
        """
        theta_phys: (4,) = [alpha, mu, d, rho]
        """
        theta_phys = theta_phys.to(self.device).float().reshape(-1)
        if theta_phys.numel() != 4:
            raise ValueError(f"theta_phys must have 4 entries [alpha, mu, d, rho]; got {tuple(theta_phys.shape)}")

        params = self._build_params_with_capacity(theta_phys)  # (5, Nsamples)

        ns = simulate_for_likelihood(params, self.times)
        log_probs = self.lpkdil_ns(ns, reduce=True, concat=True)
        return torch.sum(log_probs)
