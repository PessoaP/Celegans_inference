import torch
import repop
from utils import simulator
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
        
        def to_device_tensor(arr):
            if isinstance(arr, torch.Tensor):
                return arr.reshape(-1, 1).clone().detach().to(device)
            return torch.tensor(arr, device=device).reshape(-1, 1)

        self.counts = to_device_tensor(counts)
        self.dils   = to_device_tensor(dils)
        self.Ts     = to_device_tensor(ts)

        self.Nmax = int(2 * (self.counts * self.dils).max().item() + 1)
        self.n = torch.arange(self.Nmax, device=device)

        self.ndatapoints = self.counts.size(0)
        self.cutoff = cutoff

        # Now call REPOP on CPU
        self.lpkdil_n = repop.get_lpkdil_n(self.counts.cpu(), self.dils.cpu(), self.n.cpu(),
                                        cutoff, self.Nmax).cpu()

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
            mask_cpu = mask.to("cpu")                 # CPU mask for indexing lpkdil_n (which is on CPU)
            lpkdil_n = dataset.lpkdil_n[mask_cpu]    

            n_ind_full = ns[index].to(device=lpkdil_n.device, dtype=torch.long)
            M = n_ind_full.numel()

            valid = (n_ind_full >= 0) & (n_ind_full < lpkdil_n.shape[-1])

            if valid.sum() == 0:
                raise RuntimeError(
                    f"No valid n within support at time index {index}. "
                    f"Max n in ns = {int(n_ind_full.max())}, Nmax = {lpkdil_n.shape[-1]}."
                )

            # lpkdil_ns now has shape (ndata_at_time, M)
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


    def log_uniform(shape, low, high, device=None, dtype=torch.float32):
        """
        Sample x ~ LogUnif(low, high) elementwise.
        low/high must be > 0.
        """
        low  = float(low)
        high = float(high)
        u = torch.rand(shape, device=device, dtype=dtype)
        return torch.exp(math.log(low) + (math.log(high) - math.log(low)) * u)

    def prior_initialization(self, init=None, rng=None,
                            alpha_range=(1e-4, 1.0),   # per hour
                            mu_range=(1e-4, 2.0),      # per hour
                            d_range=(1e-4, 2.0),       # per hour
                            k_range=None,              # set from data if None
                            k_multiplier_range=(1.0, 1e3),
                            require_mu_gt_d=True):
        """
        Initialize theta from a (log-)uniform prior rather than ODE fitting.
        Returns theta = [alpha, mu, k, d] on self.device.
        """

        device = self.device
        dtype  = torch.float32

        # If user provides init, just return it.
        if init is not None:
            init = init.detach().to(device=device, dtype=dtype)
            return init

        # data-driven scale for k, but NOT fixed: just sets a plausible range
        target = (self.counts * self.dils).detach().to(device='cpu', dtype=dtype).reshape(-1)
        max_obs = float(target.max().clamp_min(1.0).item())

        if k_range is None:
            # Let k float widely around max observed.
            # k in [max_obs * m_low, max_obs * m_high]
            m_low, m_high = k_multiplier_range
            k_low  = max_obs * float(m_low)
            k_high = max_obs * float(m_high)
        else:
            k_low, k_high = k_range

        # sample parameters
        alpha = log_uniform((), alpha_range[0], alpha_range[1], device=device, dtype=dtype)
        mu    = log_uniform((), mu_range[0],    mu_range[1],    device=device, dtype=dtype)
        d     = log_uniform((), d_range[0],     d_range[1],     device=device, dtype=dtype)
        k     = log_uniform((), k_low,          k_high,         device=device, dtype=dtype)

        if require_mu_gt_d:
            # enforce mu > d without weird truncation artifacts: resample d a few times
            for _ in range(10):
                if (mu > d).item():
                    break
                d = log_uniform((), d_range[0], d_range[1], device=device, dtype=dtype)
            # last resort: clamp
            d = torch.minimum(d, mu * 0.99)

        theta = torch.stack([alpha, mu, k, d])
        return theta


    def ode_initialization(self, init=None, cap_multiplier=1.2, iters=300, lr=0.03):
        """
        Rough CPU ODE init for prior centering.
        Fix k to cap_multiplier * max observed reconstructed count.
        Fit alpha, r=(mu-d), and d with r>0, d>0. Then mu = r + d.
        """
        # --- CPU data
        target = (self.counts * self.dils).detach().cpu().float().reshape(-1)
        Ts     = self.Ts.detach().cpu().float().reshape(-1)

        # --- fixed capacity
        k_fixed = (cap_multiplier * target.max()).clamp_min(1.0)

        # --- initial guesses
        if init is None:
            alpha0 = torch.tensor(1/24, dtype=torch.float32)
            r0     = torch.tensor(1/6,  dtype=torch.float32)   # mu-d ~ 0.166/h as a starting point
            d0     = torch.tensor(0.1,  dtype=torch.float32)
        else:
            init = init.detach().cpu().float()
            alpha0 = init[0]
            mu0    = init[1]
            d0     = init[3]
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

        eps = 1.0  # floor for n_ode in reconstruction scale

        best = None
        best_loss = float("inf")

        for it in tqdm(range(iters), desc="ODE init (k fixed)", leave=False):
            opt.zero_grad()

            alpha = torch.exp(lalpha)
            r     = torch.exp(lr_)          # r = mu - dFalse
            d     = torch.exp(ld)
            mu    = r + d
            k     = k_fixed

            theta = torch.stack([alpha, mu, k, d])  # (4,)

            n_ode = simulator.integrate_mass_action(theta, Ts, dt=0.1, device='cpu').reshape(-1)
            n_ode = n_ode.clamp_min(eps)

            # relative error, weighted toward later times
            rel = (target / n_ode - 1.0)
            loss = torch.sqrt((w * rel * rel).mean())

            if torch.isfinite(loss):
                loss.backward()
                opt.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = theta.detach().clone()

        if best is None:
            best = torch.stack([alpha.detach(), (r+d).detach(), k_fixed.detach(), d.detach()])

        return best.to(self.device)


   

# Old ODE initialization method (disabled)
    # def ode_initialization(self, init=None):
    #     """
    #     Initialize parameters for ODE fitting via gradient descent (on CPU).
    #     This is just a rough initialization to get parameters into a reasonable range for the prior.
    #     """
    #     # Always move everything to CPU for ODE initialization
    #     target = (self.counts * self.dils).cpu()
    #     Ts_cpu = self.Ts.cpu().float()
    #     ndatapoints = self.ndatapoints  # This should be an int already

    #     # Start from init or reasonable guess (ensure CPU)
    #     if init is None:
    #         init = torch.tensor([1/24, 1e-2, float(target[Ts_cpu == Ts_cpu.max()].median()), 1e-4], dtype=torch.float32, device='cpu')
    #     else:
    #         init = init.detach().cpu().float()

    #     lparams = tor           gradient_norm = l2(lparams.grad).item()
    #                 print('gradient:', gradient_norm, loss_hist[-1])
    #                 if gradient_norm < 1e-5:
    #                     break

    #     # Return parameters in original (non-log) space, **move to desired device**
    #     return torch.exp(lparams).detach().to(self.device)ch.log(init).clone().detach().requires_grad_()

    #     optimizer = torch.optim.Adam([lparams], lr=.01)
    #     l2 = lambda x: torch.sqrt((x * x).sum())
    #     loss_hist = []

    #     print('Initializing using mass-action similarity (CPU)')
    #                 gradient_norm = l2(lparams.grad).item()
    #                 print('gradient:', gradient_norm, loss_hist[-1])
    #                 if gradient_norm < 1e-5:
    #                     break

    #     # Return parameters in original (non-log) space, **move to desired device**
    #     return torch.exp(lparams).detach().to(self.device)   # Simulate ODE with current parameters (all CPU)
    #         n_ode = simulator.integrate_mass_action(torch.exp(lparams), Ts_cpu, dt=0.1, device='cpu')

    #         scaledtime = Ts_cpu / Ts_cpu.min()
    #         loss = l2((target / n_ode - 1) / (scaledtime ** 2)) * Ts_cpu.min() / ndatapoints

    #         if not (torch.isnan(loss) or torch.isinf(loss)):
    #             loss.backward()
    #             optimizer.step()
    #             loss_hist.append(loss.item())
    #             if (it + 1) % 10 == 0:for it in tqdm(range(500)):
    #         optimizer.zero_grad()

    #                 gradient_norm = l2(lparams.grad).item()
    #                 print('gradient:', gradient_norm, loss_hist[-1])
    #                 if gradient_norm < 1e-5:
    #                     break

    #     # Return parameters in original (non-log) space, **move to desired device**
    #     return torch.exp(lparams).detach().to(self.device)   # Simulate ODE with current parameters (all CPU)
    #         n_ode = simulator.integrate_mass_action(torch.exp(lparams), Ts_cpu, dt=0.1, device='cpu')

    #         scaledtime = Ts_cpu / Ts_cpu.min()
    #         loss = l2((target / n_ode - 1) / (scaledtime ** 2)) * Ts_cpu.min() / ndatapoints

    #         if not (torch.isnan(loss) or torch.isinf(loss)):
    #             loss.backward()
    #             optimizer.step()
    #             loss_hist.append(loss.item())
    #             if (it + 1) % 10 == 0:
         
