# tests/test_mcmc_logposterior.py

import torch
import numpy as np
import pytest

from utils import mcmc 


class DummyPrior:
    """Simple prior with fixed log_prob."""
    def __init__(self, value):
        # value should be a Python float or 0-D tensor
        self.value = torch.tensor(float(value), dtype=torch.float32)

    def log_prob(self, theta):
        # ignore theta, just return fixed value
        return self.value


class DummyData:
    """Simple dataset with fixed log-likelihood."""
    def __init__(self, value):
        self.value = torch.tensor(float(value), dtype=torch.float32)

    def loglike(self, theta):
        # ignore theta, just return fixed value
        return self.value


def test_logposterior_u_happy_path(capsys):
    """
    Check that:
      logposterior_u(u) = lp + ll + sum(u)
    and that debug=True prints the components.
    """
    prior = DummyPrior(value=-5.0)
    data = DummyData(value=-3.0)

    # use the debug version
    logposterior_u = mcmc.make_logposterior_u(data, prior, debug=True)

    # choose u so theta = exp(u) is all ones
    u = torch.zeros(4)  # theta = [1,1,1,1], jacobian = sum(u) = 0

    total = logposterior_u(u)

    # check numeric value: lp + ll + jacobian
    lp = prior.log_prob(torch.exp(u))
    ll = data.loglike(torch.exp(u))
    jac = mcmc.logabsdet_J_exp(u)

    expected = lp + ll + jac
    assert torch.isclose(total, expected)

    # check that debug output contains components
    captured = capsys.readouterr()
    out = captured.out
    assert "lp (prior):" in out
    assert "ll (likelihood):" in out
    assert "jacobian:" in out
    assert "total:" in out


def test_logposterior_u_prior_minus_inf():
    """
    If prior.log_prob returns -inf, logposterior_u should return -inf.
    """
    prior = DummyPrior(value=-float("inf"))
    data = DummyData(value=-3.0)

    logposterior_u = mcmc.make_logposterior_u(data, prior, debug=False)
    u = torch.zeros(4)

    total = logposterior_u(u)
    assert not torch.isfinite(total)
    assert total.item() == -float("inf")


def test_logposterior_u_likelihood_minus_inf():
    """
    If data.loglike returns -inf, logposterior_u should return -inf.
    """
    prior = DummyPrior(value=-5.0)
    data = DummyData(value=-float("inf"))

    logposterior_u = mcmc.make_logposterior_u(data, prior, debug=False)
    u = torch.zeros(4)

    total = logposterior_u(u)
    assert not torch.isfinite(total)
    assert total.item() == -float("inf")
