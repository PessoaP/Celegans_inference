# unit_tests/repop_units.py
import torch
import repop

def test_gaussmix_loglike_shapes_and_finiteness():
    torch.manual_seed(0)
    n = torch.linspace(-3, 3, 11)              # (11,)
    mus = torch.tensor([-1.0, 1.0])            # (2,)
    sigs = torch.tensor([0.5, 0.5])            # (2,)
    rhos = torch.tensor([0.4, 0.6])            # (2,)

    logp = repop.Igaussmix_loglike(n, mus, sigs, rhos)

    assert logp.shape == (11,)
    assert torch.isfinite(logp).all()

def test_gaussmix_normalization_discrete():
    torch.manual_seed(0)
    xs = torch.linspace(-10, 10, 4001)
    mus = torch.tensor([-1.0, 1.0])
    sigs = torch.tensor([0.8, 0.8])
    rhos = torch.tensor([0.3, 0.7])

    logp = repop.Igaussmix_loglike(xs, mus, sigs, rhos)
    p = torch.exp(logp)
    total_mass = p.sum().item()
    assert abs(total_mass - 1.0) < 1e-6


