import torch
import repop
from utils import simulator
from tqdm import tqdm

class TimeSeriesInferenceDataset():
    def __init__(self, ts, counts, dils, kappa_samples, cutoff=300, rho=1.0, t_switch=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
        def to_device_tensor(arr):
            if isinstance(arr, torch.Tensor):
                return arr.reshape(-1, 1).clone().detach().to(device)
            return torch.tensor(arr, device=device).reshape(-1, 1)

        self.counts = to_device_tensor(counts)
        self.dils   = to_device_tensor(dils)
        self.Ts     = to_device_tensor(ts)
        self.rho = float(rho)
        self.t_switch = None if t_switch is None else float(t_switch)

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

        self.times, inv = torch.unique(self.Ts, return_inverse=True)
        self.times, perm = torch.sort(self.times)
        invperm = torch.empty_like(perm)
        invperm[perm] = torch.arange(perm.numel(), device=self.device)
        self.T_index = invperm[inv].to(self.device)
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
    
    def _build_params_with_capacity(self, theta_phys, rho_scale=1.0):
        '''
        Build per-trajectory parameter matrix with fixed per-worm capacity samples.

        Args:
            theta_phys: torch.Tensor shape (3,) = [alpha, mu, d]   (NO k here)
            rho_scale: factor multiplying alpha after t_switch

        Requires:
            self.kappa_samples is a torch.Tensor of shape (Nsamples,)
            (already drawn via stratified sampling)
        Returns:
            params: torch.Tensor shape (4, Nsamples) in simulator order [alpha, mu, k, d]
        '''
        theta_phys = theta_phys.to(self.device).float().reshape(-1)
        if theta_phys.numel() != 3:
            raise ValueError("theta_phys must be [alpha, mu, d]")

        alpha, mu, d = theta_phys[0], theta_phys[1], theta_phys[2]
        alpha = alpha * float(rho_scale)

        k_vec = self.kappa_samples.to(self.device).float().reshape(-1)
        Nsamples = k_vec.numel()

        alpha_vec = alpha.expand(Nsamples)
        mu_vec    = mu.expand(Nsamples)
        d_vec     = d.expand(Nsamples)

        params = torch.stack([alpha_vec, mu_vec, k_vec, d_vec], dim=0)
        return params


    def loglike(self, theta_phys):
        """
        Args:
            theta_phys: torch.Tensor shape (3,) = [alpha, mu, d]
                (kinetics only; capacity samples are fixed in self.kappa_samples)

        Requires:
            self.kappa_samples: torch.Tensor shape (Nsamples,)
            self.times: torch.Tensor shape (n_times,)

        Returns:
            Scalar log-likelihood estimate
        """
        ns = self.simulate_for_likelihood(theta_phys)
        log_probs = self.lpkdil_ns(ns, reduce=True, concat=True)
        return torch.sum(log_probs)


    def simulate_for_likelihood(self, theta_phys):
        times = self.times.to(self.device).reshape(-1)
        N = int(self.kappa_samples.numel())
        device = self.device

        # No switch
        if (self.t_switch is None) or (self.rho == 1.0):
            params = self._build_params_with_capacity(theta_phys, rho_scale=1.0)
            return simulator.sample_record(params=params, record_times=times, N=N, device=device)

        t0 = float(self.t_switch)

        mask_pre  = times < t0
        times_pre  = times[mask_pre]
        times_post = times[~mask_pre]   # >= t0

        # Stage 1: base alpha
        params_pre = self._build_params_with_capacity(theta_phys, rho_scale=1.0)

        snapshots_pre = []
        if times_pre.numel() > 0:
            snapshots_pre = simulator.sample_record(params=params_pre, record_times=times_pre, N=N, device=device)
        
        if times_post.numel() == 0:
            return snapshots_pre
        # state at switch
        _, E_split = simulator.sample(params=params_pre, E_initial=None, T=t0, N=N, device=device)

        # Stage 2: scaled alpha
        params_post = self._build_params_with_capacity(theta_phys, rho_scale=self.rho)

        snapshots_post = []
        if times_post.numel() > 0:
            shifted = (times_post - t0).to(device=device, dtype=torch.float32)
            snapshots_post = simulator.sample_record(
                params=params_post, record_times=shifted, N=N, E_initial=E_split, device=device
            )

        # Stitch
        out = []
        i_pre = 0
        i_post = 0
        for is_pre in mask_pre.tolist():
            if is_pre:
                out.append(snapshots_pre[i_pre]); i_pre += 1
            else:
                out.append(snapshots_post[i_post]); i_post += 1
        return out
