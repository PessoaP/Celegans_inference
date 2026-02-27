import os, sys
import subprocess

import repop
import torch
import numpy as np

from utils.load_and_clean_real_data import load_and_pool_real_data

from repop.utils import Igaussmix_loglike 

# Set up functions to include only the rightmost n Gaussians that REPOP predicts 
def get_rightmost_components_from_dt(dt, top_n=4, renormalize=True):
    m, s, r = dt.ev

    # sort by mean (ascending)
    idx = torch.argsort(m)
    print(dt.ev)
    print(idx)

    if len(r) < top_n:
        raise ValueError(f"Only {len(r)} components available")

    # take the rightmost top_n
    idx = idx[-top_n:]

    mN, sN, rN = m[idx], s[idx], r[idx]

    if renormalize:
        rN = rN / rN.sum()

    return mN, sN, rN

def reconstruct_top_components(dt, top_n=4, narray=None, cpu=True, **kwargs):
    if narray is None:
        x = dt.n
    else:
        x = narray * 1.

    m, s, r = get_rightmost_components_from_dt(dt, top_n=top_n, **kwargs)

    p = torch.exp(Igaussmix_loglike(x, m, s, r))

    if cpu:
        x, p = x.cpu(), p.cpu()

    return x, p

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
J = 128
Nsamples = 2**20

# Load data from only Day 9 from selected real data CSVs
capacity_calibrating_datasets = ["real_data/Exp_4_live_highpH.csv","real_data/Exp_1_live_lowpH.csv"]
repop_save_files = ["capacity_day9_repop_precalibration_highpH.npz","capacity_day9_repop_precalibration_lowpH.npz"]
kappa_files = ["kappa_samples_stratified_highpH.npz","kappa_samples_stratified_lowpH.npz"]

for filepath, repop_save_file,kappa_file in zip(capacity_calibrating_datasets, repop_save_files,kappa_files):
    print(f"Processing {filepath} ")
    df_day9 = load_and_pool_real_data(
        filepaths=[filepath],  # load in selected CSVs
        days=9, # filter to only include Day 9 data from those datasets
        cutoff=300,
        verbose=True,
    )

    # Make sure that Day 9 data actually loaded & exists
    if len(df_day9) == 0:
        raise ValueError("No Day 9 rows after filtering/cleaning.")

    counts = df_day9["Counts"].to_numpy()
    dils   = df_day9["Dilution"].to_numpy()


    # Run REPOP on Day 9 data to extract p(K)
    # REPOP expects counts/dils aligned per observation
    dt = repop.dataset(counts=counts, dils=dils, cutoff=300)

    # Fit the mixture (required before reconstruction)
    dt.evaluate(components=repop.weak_limit, observe=True,component_cut=0) 

    print(dt.ev)

    # Use just the top 4 Gaussians in the mixture as the true "capacity" distribution
    x, p = reconstruct_top_components(dt, top_n=4, cpu=True, renormalize=True)

    # Make them 1D numpy arrays for saving / sampling
    x = x.reshape(-1).detach().cpu().numpy()
    p = p.reshape(-1).detach().cpu().numpy()
    p = p / p.sum()

    np.savez(repop_save_file, kappa=x, p=p)


    sample_script = os.path.join(PROJECT_ROOT, "Celegans_inference/utils/capacity_sampling_stratified.py")
    print("Running:", sample_script)
    subprocess.run([
        sys.executable, sample_script,
        "--capacity_npz", os.path.join(PROJECT_ROOT,"Celegans_inference/"+repop_save_file),
        "--J", str(J),
        "--Nsamples", str(Nsamples),
        "--seed", "0",
        "--out", os.path.join(PROJECT_ROOT,"Celegans_inference/"+kappa_file),
    ], check=True, cwd=PROJECT_ROOT)