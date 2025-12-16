# tests for dataclass.py

import pytest
import torch
import repop
from utils import simulator
from utils.dataclass import TimeSeriesInferenceDataset, simulate_for_likelihood



def test_simulate_for_likelihood_shapes_and_nonneg():
    device = torch.device("cuda")
    torch.manual_seed(0)

    params = torch.tensor([1/20, 1/4, 1e5, 0.1], dtype=torch.float32, device=device)
    times = torch.tensor([24.0, 48.0, 72.0], dtype=torch.float32, device=device)
    Nsamples = 256  # keep small for test speed

    sims = simulate_for_likelihood(params, times, Nsamples=Nsamples)

    assert len(sims) == len(times)
    for E in sims:
        assert isinstance(E, torch.Tensor)
        assert E.shape == (Nsamples,)
        assert torch.all(E >= 0)


def test_dataset_init_basic(monkeypatch):
    device = torch.device("cuda")

    counts = torch.tensor([10., 20., 30.])
    dils   = torch.tensor([1., 2., 3.])   # arbitrary
    ts     = torch.tensor([24., 48., 24.])

    # Fake get_lpkdil_n with a deterministic small tensor
    def fake_get_lpkdil_n(counts_, dils_, n_, cutoff, Nmax):
        # Expect shape: (ndatapoints, Nmax)
        ndatapoints = counts_.shape[0]
        return torch.arange(ndatapoints * Nmax, dtype=torch.float32).reshape(ndatapoints, Nmax)

    monkeypatch.setattr(repop, "get_lpkdil_n", fake_get_lpkdil_n)

    ds = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=300, device=device)

    assert ds.counts.shape == (3, 1)
    assert ds.dils.shape   == (3, 1)
    assert ds.Ts.shape     == (3, 1)
    assert ds.ndatapoints == 3
    assert ds.lpkdil_n.shape[0] == 3
    assert ds.n.shape[0] == ds.Nmax
    # unique times should be [24, 48] in some order
    assert set(ds.times.view(-1).tolist()) == {24.0, 48.0}

def test_lpkdil_ns_selection_and_reduce(monkeypatch):
    device = torch.device("cpu")

    # Create a minimal fake dataset object
    counts = torch.tensor([1., 1.])
    dils   = torch.tensor([1., 1.])
    ts     = torch.tensor([24., 48.])

    def fake_get_lpkdil_n(counts_, dils_, n_, cutoff, Nmax):
        # 2 datapoints, Nmax = 4 -> 2x4 table
        # row i, col j = i*10 + j for easy inspection
        return torch.tensor([[0., 1., 2., 3.],
                             [10., 11., 12., 13.]], dtype=torch.float32)

    monkeypatch.setattr(repop, "get_lpkdil_n", fake_get_lpkdil_n)

    ds = TimeSeriesInferenceDataset(ts, counts, dils, cutoff=4, device=device)

    # Suppose we want:
    # at time index 0: use n indices [1, 3]
    # at time index 1: use n indices [0, 2, 3]
    ns = [
        torch.tensor([1, 3]),
        torch.tensor([0, 2, 3]),
    ]

    # reduce=False: we should get the raw slices
    out = ds.lpkdil_ns(ns, reduce=False, concat=False)
    assert len(out) == 2
    # First time: row 0, cols 1 and 3 -> [1, 3]
    assert torch.allclose(out[0], torch.tensor([[1., 3.]]))
    # Second time: row 1, cols 0,2,3 -> [[10, 12, 13]]
    assert torch.allclose(out[1], torch.tensor([[10., 12., 13.]]))

    # reduce=True, concat=True: log-mean over selected ns, then concatenated
    out_red = ds.lpkdil_ns(ns, reduce=True, concat=True)
    # shape should be (2,) : one per datapoint
    assert out_red.shape == (2,)

def test_loglike_calls_structurally(monkeypatch):
    device = torch.device("cuda")

    counts = torch.tensor([5., 5.])
    dils   = torch.tensor([1., 1.])
    ts     = torch.tensor([24., 24.])

    def fake_get_lpkdil_n(counts_, dils_, n_, cutoff, Nmax):
        # 2 datapoints, Nmax = 3
        return torch.zeros((2, 3), dtype=torch.float32)

    monkeypatch.setattr(repop, "get_lpkdil_n", fake_get_lpkdil_n)

    ds = TimeSeriesInferenceDataset(ts, counts, dils, device=device)

    # Monkeypatch lpkdil_ns to something simple and inspectable
    def fake_lpkdil_ns(ns, reduce=False, concat=False):
        # pretend every ns list gives log_prob  -1 for each datapoint
        # If reduce=True, concat=True -> want a (ndatapoints,) vector of -1
        assert reduce is True
        assert concat is True
        return torch.full((ds.ndatapoints,), -1.0, device=device)

    monkeypatch.setattr(ds, "lpkdil_ns", fake_lpkdil_ns)

    params = torch.tensor([1/20, 1/4, 1e5, 0.1], dtype=torch.float32, device=device)
    val = ds.loglike(params, Nsamples=16)

    # 2 datapoints, each -1 -> sum = -2
    assert torch.is_tensor(val)
    assert torch.allclose(val, torch.tensor(-2.0, device=device))


def test_ode_initialization_returns_params_on_device(monkeypatch):
    device = torch.device("cuda")

    counts = torch.tensor([10., 20., 30.])
    dils   = torch.tensor([1., 1., 1.])
    ts     = torch.tensor([24., 48., 72.])

    def fake_get_lpkdil_n(counts_, dils_, n_, cutoff, Nmax):
        return torch.zeros((counts_.shape[0], Nmax))

    monkeypatch.setattr(repop, "get_lpkdil_n", fake_get_lpkdil_n)

    ds = TimeSeriesInferenceDataset(ts, counts, dils, device=device)

    init_params = ds.ode_initialization()
    assert init_params.shape == (4,)
    assert init_params.device == device
    assert torch.all(init_params > 0)

def test_ode_initialization_returns_params_on_device(monkeypatch):
    device = torch.device("cpu")

    counts = torch.tensor([10., 20., 30.])
    dils   = torch.tensor([1., 1., 1.])
    ts     = torch.tensor([24., 48., 72.])

    def fake_get_lpkdil_n(counts_, dils_, n_, cutoff, Nmax):
        return torch.zeros((counts_.shape[0], Nmax))

    monkeypatch.setattr(repop, "get_lpkdil_n", fake_get_lpkdil_n)

    ds = TimeSeriesInferenceDataset(ts, counts, dils, device=device)

    init_params = ds.ode_initialization()
    assert init_params.shape == (4,)
    assert init_params.device == device
    assert torch.all(init_params > 0)
