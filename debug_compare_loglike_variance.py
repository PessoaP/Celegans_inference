# figure out if loglike(theta) is actually the culprit of high variance
# or if it's something else in the logposterior calculation
import os, sys, time
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from dataclass import TimeSeriesInferenceDataset
from utils.load_and_clean_real_data import load_and_clean_real_data
from utils import mcmc

def main(path, repeats=8, Nsamples=2**15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); np.random.seed(0)
    print("Device:", device)

    df = load_and_clean_real_data(path, cutoff=300)
    ts = torch.tensor(df["Day"].values * 24)
    counts = torch.tensor(df["Counts"].values)
    dils = torch.tensor(df["Dilution"].values)
    dataset = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)

    prior, init_guess = mcmc.make_prior_from_initial_guess(dataset, frac_error=0.5, device=device)
    logpost_u = mcmc.make_logposterior_u(dataset, prior, debug=False)

    theta = torch.tensor([1/20, 0.25, 2e5, 0.1], dtype=torch.float32, device=device)
    u = mcmc.to_u(theta)

    print("theta:", theta.detach().cpu().numpy())
    print("Nsamples:", Nsamples, "repeats:", repeats)

    # ---- repeat loglike(theta) ----
    lls = []
    for i in range(repeats):
        t0=time.time()
        ll = dataset.loglike(theta, Nsamples=Nsamples)
        lls.append(float(ll))
        print(f"loglike run {i:02d}: {lls[-1]:.6f} (dt={time.time()-t0:.2f}s)")
    lls=np.array(lls)
    print("loglike std:", lls.std(), "range:", lls.max()-lls.min())

    # ---- repeat components at same u ----
    lps=[]
    comps=[]
    for i in range(repeats):
        theta2 = mcmc.to_theta(u)
        lp = float(prior.log_prob(theta2))
        ll = float(dataset.loglike(theta2, Nsamples=Nsamples))
        jac = float(mcmc.logabsdet_J_exp(u))
        tot = lp + ll + jac
        comps.append((lp,ll,jac,tot))
        lps.append(tot)
        print(f"comp run {i:02d}: lp={lp:.3f} ll={ll:.3f} jac={jac:.3f} total={tot:.3f}")
    lps=np.array(lps)
    print("total std:", lps.std(), "range:", lps.max()-lps.min())

if __name__=="__main__":
    main(sys.argv[1])
