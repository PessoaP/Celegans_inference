import torch
import repop
import simulator

def simulate_for_likelihood(params, times, Nsamples=2**10):
        # Simulate trajectories for each unique measurement time in `data`.

    simulations = []
    t = 0
    for T in times:
        delta_t, E = simulator.sample(params, N=Nsamples, T=T - t)
        t += delta_t
        simulations.append(E.int())
    return simulations

class dataset():
    def __init__(self, ts, counts, dils, cutoff=300, 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        
        # These calculations stay on GPU
        # Convert counts and dilutions to column tensors for consistency
        self.counts = torch.tensor(counts.reshape(-1, 1))
        self.dils = torch.tensor(dils.reshape(-1, 1))
        
        # Define upper bound for total possible number of cells
        self.Nmax = 2 * (counts * dils).max() + 1
        # self.Nmax = 1e6
        # print(self.Nmax)
        self.n = torch.arange(self.Nmax)  # Latent total cell counts

        self.ndatapoints = self.counts.size(0)
        self.cutoff = cutoff

        # Compute log p(k | n, phi) for all datapoints
        self.lpkdil_n = repop.get_lpkdil_n(self.counts, self.dils, self.n, 
                                           cutoff, self.Nmax).to(device)
        #Now we move to GPU, if available

        # Unique timepoints and index mapping for grouping
        self.times, self.T_index = torch.unique(
            torch.tensor(ts), return_inverse=True
        )
        self.T_index = self.T_index.to(device)

        print('Dataset loaded succesfully')


    def lpkdil_ns(self, ns, reduce=False, concat=False):
        """
        Selects (and optionally reduces) log p(k | n, phi) across relevant n for each time t.

        Args:
            ns (list of 1D tensors): For each t, indices of n-values to retain.
            reduce (bool): If True, compute log-mean over selected columns (logsumexp - log N).
            concat (bool): If True and reduce is True, concatenate outputs into a single tensor.
            return_indices (bool): If True and concat is True, also return original row indices.

        Returns:
            If reduce=False or concat=False:
                list of tensors (one per time)
            If reduce=True and concat=True and return_indices=False:
                Tensor of shape (total_datapoints,)
            If reduce=True and concat=True and return_indices=True:
                Tuple: (Tensor, Tensor) = (values, row_indices)
        """
        list_lpkdil_ns = []

        for t in range(len(self.times)):
            mask = (self.T_index == t)

            lpk = self.lpkdil_n[mask]  # (num_datapoints_t, Nmax)
            n_indices = ns[t].to(self.device)
            assert torch.max(n_indices) < self.Nmax, f"ns[{t}] has out-of-bound indices."

            lpk_condensed = lpk[:, n_indices]  # (num_datapoints_t, len(ns[t]))
            
            # Indices of n-values to extract for this time
            n_indices = ns[t].to(self.device)

            # Create output tensor filled with -inf (log(0))
            lpk_condensed = torch.full(
                (lpk.size(0), len(n_indices)),
                float('-inf'),
                dtype=lpk.dtype,
                device=self.device
            )

            # Identify which indices are valid
            valid_mask = n_indices < self.Nmax
            valid_n = n_indices[valid_mask]

            # Fill in valid columns
            if valid_n.numel() > 0:
                lpk_condensed[:, valid_mask] = lpk[:, valid_n]

            if reduce:
                lpk_condensed = (
                    torch.logsumexp(lpk_condensed, dim=1)
                    - torch.log(torch.tensor(len(n_indices), dtype=torch.float32, device=self.device))
                )

            list_lpkdil_ns.append(lpk_condensed)

        if reduce and concat:
            values = torch.cat(list_lpkdil_ns, dim=0)
            return values

        return list_lpkdil_ns
    
    def loglike(self,value,Nsamples = 2**15):
        ns = simulate_for_likelihood(value,self.times,Nsamples)
        return torch.logsumexp(self.lpkdil_ns(ns,reduce=True,concat=True),axis=0)
