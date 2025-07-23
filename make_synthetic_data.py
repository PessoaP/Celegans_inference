import os
import torch
import numpy as np
import pandas as pd

from simulator import sample

class SyntheticSimulator:
    """
    Simulates counts from a stochastic growth model at different times,
    applies dilution, and exports results to CSV.
    """
    def __init__(self, params, device=None):
        self.params = params
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs("synthetic_data", exist_ok=True)
        
    def sample_n(self, size, T):
        print(f"Sampling {size} trajectories at times {T[:5]}... (showing first 5)")
        _, E = sample(self.params, T=T, N=size, device=self.device)
        return E

    def sample_data(self, size=75, Ts=torch.tensor([1, 3, 5, 7, 9])):
        print(f"Simulating for {size} worms per day, days = {Ts.tolist()}")
        Ts = Ts.clone().to(self.device).float()
        T_batch = Ts.repeat_interleave(size) * 24  # Convert days to hours
        print(f"T_batch shape: {T_batch.shape}, values: {T_batch[:10]}")
        n_sam = self.sample_n(size=T_batch.numel(), T=T_batch).cpu()
        print(f"n_sam: min={n_sam.min()}, max={n_sam.max()}, dtype={n_sam.dtype}")

    def sample_data(self, size=75, Ts=torch.tensor([1, 3, 5, 7, 9])):
        """
        Simulate and dilute counts for multiple timepoints (wide format).
        """
        Ts = Ts.clone().to(self.device).float()
        T_batch = Ts.repeat_interleave(size) * 24  # Convert days to hours
        n_sam = self.sample_n(size=T_batch.numel(), T=T_batch).cpu()

        # Serial dilution steps (vectorized)
        n1 = torch.distributions.Binomial(total_count=n_sam.float(), probs=10/200).sample()
        n2 = torch.distributions.Binomial(total_count=n1.float(), probs=10/100).sample()
        n3 = torch.distributions.Binomial(total_count=n2.float(), probs=10/100).sample()
        c1 = torch.distributions.Binomial(total_count=n1.float(), probs=90/100).sample()
        c2 = torch.distributions.Binomial(total_count=n2.float(), probs=90/100).sample()
        c3 = torch.distributions.Binomial(total_count=n3.float(), probs=90/100).sample()

        n_timepoints = len(Ts)
        df = pd.DataFrame({
            'Worm #': np.tile(np.arange(1, size + 1), n_timepoints),
            'CFU_22': c1.int().cpu().numpy(),
            'CFU_222': c2.int().cpu().numpy(),
            'CFU_2222': c3.int().cpu().numpy(),
            'Day': np.repeat(Ts.cpu().numpy(), size)
        })
        return df

    def sample_save(self, size=75, Ts=torch.tensor([1, 3, 5, 7, 9]), filename="synthetic_data/synthetic_data.csv"):
        print("Beginning synthetic data generation and save...")
        df = self.sample_data(size=size, Ts=Ts)
        print(f"Saving to {filename}...")
        df.to_csv(filename, index=False)
        print(f"Synthetic data saved to {filename}")

# --------- Main Script Usage ---------

if __name__ == "__main__":
    # Example parameter vector: adjust as needed
    params = torch.tensor([1/20, 1/4, 1e5, 0.1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim = SyntheticSimulator(params=params, device=device)

    # Simulate for 75 worms at each of days 1,3,5,7,9
    sim.sample_save(size=75, Ts=torch.tensor([1, 3, 5, 7, 9]))
