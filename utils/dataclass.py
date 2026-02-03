import torch
import repop
from utils import simulator
from tqdm import tqdm


def simulate_for_likelihood(params, times):
    """
    Simulate ONCE up to max(times) and record E snapshots at each requested time.
    """
    N = int(params.shape[1])  # number of worms simulated

    device = params.device
    times = times.to(device).reshape(-1)
    times = torch.sort(times).values  # optional but safe

    return simulator.sample_record(params=params, record_times=times, N=N, device=device)


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
        Build per-trajectory parameter matrix with fixed per-worm capacity samples.

        Args:
            theta_phys: torch.Tensor shape (3,) = [alpha, mu, d]   (NO k here)

        Requires:
            self.kappa_samples is a torch.Tensor of shape (Nsamples,)
            (already drawn via stratified sampling)
        Returns:
            params: torch.Tensor shape (4, Nsamples) in simulator order [alpha, mu, k, d]
        """
        # --- Check theta is presented properly ---
        theta_phys = theta_phys.to(self.device).float().reshape(-1)
        if theta_phys.numel() != 3:
            raise ValueError(f"theta_phys must have 3 entries [alpha, mu, d]; got shape {tuple(theta_phys.shape)}")

        alpha, mu, d = theta_phys[0], theta_phys[1], theta_phys[2]
        k_vec = self.kappa_samples.to(self.device).float().reshape(-1)  # vector of capacities must be loaded in self.kappa_samples
        Nsamples = k_vec.numel() # Nsamples: number of trajectories ("worms")

        # --- broadcast kinetics to (Nsamples,) ---
        alpha_vec = alpha.expand(Nsamples)
        mu_vec    = mu.expand(Nsamples)
        d_vec     = d.expand(Nsamples)

        # --- assemble (4, Nsamples) ---
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
        # --- sanitize theta ---
        theta_phys = theta_phys.to(self.device).float().reshape(-1)
        if theta_phys.numel() != 3:
            raise ValueError(f"theta_phys must have 3 entries [alpha, mu, d]; got shape {tuple(theta_phys.shape)}")

        # --- build per-worm params (4, Nsamples) using fixed kappa samples ---
        params = self._build_params_with_capacity(theta_phys)  # (4, Nsamples)
        Nsamples = params.shape[1]

        # --- simulate trajectories ---
        ns = simulate_for_likelihood(params, self.times) 
        # self.last_simulated_ns = ns -- don't save ns on self

        # --- score simulated ns against observed plate-count+dilution data ---
        log_probs = self.lpkdil_ns(ns, reduce=True, concat=True) # Does the calculation (logmeanexp / logsumexp - log N).

        return torch.sum(log_probs)



    def ode_initialization(self, init=None, iters=300, lr=0.03):
        """
        Rough CPU ODE init for prior centering. Returns (alpha, mu, d).
        """
        target = (self.counts * self.dils).detach().cpu().float().reshape(-1)
        Ts     = self.Ts.detach().cpu().float().reshape(-1)

        # --- initial guesses
        if init is None:
            alpha0 = torch.tensor(1/24, dtype=torch.float32)
            r0     = torch.tensor(1/6,  dtype=torch.float32)   # mu-d ~ 0.166/h as a starting point
            d0     = torch.tensor(0.1,  dtype=torch.float32)
        else:
            init = init.detach().cpu().float()
            alpha0 = init[0]
            mu0    = init[1]
            d0     = init[2]
            r0     = (mu0 - d0).clamp_min(1e-3)

        # optimize in log space for positivity
        lalpha = torch.log(alpha0.clamp_min(1e-12)).requires_grad_()
        lr_    = torch.log(r0.clamp_min(1e-12)).requires_grad_()
        ld     = torch.log(d0.clamp_min(1e-12)).requires_grad_()

        opt = torch.optim.Adam([lalpha, lr_, ld], lr=lr)

        # time weights: downweight earliest points (more stochastic colonization)
        # e.g. weight ~ (t / t_max)^2
        tmax = Ts.max().clamp_min(1e-6)
        w = (Ts / tmax) ** 2
        w = w / w.mean().clamp_min(1e-12)

        eps = 1.0 

        best = None
        best_loss = float("inf")

        k_fixed = torch.median(self.kappa_samples.detach().cpu()).clamp_min(1.0)

        for it in tqdm(range(iters), desc="ODE init", leave=False):
            opt.zero_grad()

            alpha = torch.exp(lalpha)
            r     = torch.exp(lr_)          # r = mu - d
            d     = torch.exp(ld)
            mu    = r + d

            theta_ode = torch.stack([alpha, mu, k_fixed, d])  # (4,)
            n_ode = simulator.integrate_mass_action(theta_ode, Ts, dt=0.1, device='cpu').reshape(-1)
            n_ode = n_ode.clamp_min(eps)

            # relative error, weighted toward later times
            rel = (target / n_ode - 1.0)
            loss = torch.sqrt((w * rel * rel).mean())

            if torch.isfinite(loss):
                loss.backward()
                opt.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = torch.stack([alpha, mu, d]).detach().clone()

        if best is None:
            best = torch.stack([alpha.detach(), (r+d).detach(), d.detach()])

        return best.to(self.device)

