import torch
import pytest
import repop  

def _toy_inputs(device="cpu"):
    # Two datapoints with different dilutions, small counts
    counts = torch.tensor([5, 12], dtype=torch.long, device=device).reshape(-1, 1)  # (2,1)
    dils   = torch.tensor([10.0, 200.0], dtype=torch.float32, device=device).reshape(-1, 1)  # (2,1)

    Nmax = 501
    n = torch.arange(Nmax, device=device, dtype=torch.long)  # (Nmax,)
    cutoff = 300
    return counts, dils, n, cutoff, Nmax

def test_get_lpkdil_n_no_nans_and_has_finite_mass():
    counts, dils, n, cutoff, Nmax = _toy_inputs()
    out = repop.get_lpkdil_n(counts, dils, n, cutoff=cutoff, Nmax=Nmax)

    assert out.ndim == 2
    assert out.shape[0] == counts.shape[0]
    assert out.shape[1] == n.numel()

    # NaNs are never OK
    assert not torch.isnan(out).any(), "NaNs in get_lpkdil_n output"

    # Each row should have at least one finite entry
    finite_per_row = torch.isfinite(out).any(dim=1)
    assert finite_per_row.all(), "Some datapoints have no finite likelihood over n support"


def test_get_lpkdil_n_cutoff_minus1_matches_counts_loglike():
    counts, dils, n, _, Nmax = _toy_inputs()
    out_no_corr = repop.get_lpkdil_n(counts, dils, n, cutoff=-1, Nmax=Nmax)
    base = repop.counts_loglike(counts, n, dils)

    assert out_no_corr.shape == base.shape
    assert torch.allclose(out_no_corr, base, atol=0.0, rtol=0.0), \
        "cutoff=-1 branch must equal counts_loglike exactly"

def test_get_lpkdil_n_reasonable_magnitude_smoke():
    """
    This catches the +1e20 style bug.
    A true log pmf/log-likelihood table should not contain astronomically large magnitudes.
    If your function returns a *score* that can be positive, tune the threshold upward,
    but it should never be ~1e20.
    """
    counts, dils, n, cutoff, Nmax = _toy_inputs()
    out = repop.get_lpkdil_n(counts, dils, n, cutoff=cutoff, Nmax=Nmax)

    mx = out.max().item()

    # These thresholds are intentionally generous.
    # If you *expect* values larger than this, something is off in probabilistic scaling.
    assert mx < 1e6, f"Suspiciously large positive values in lpkdil_n: max={mx}"

def test_get_lpkdil_n_broadcasting_not_degenerate_across_dilutions():
    """
    With different dilutions, the correction terms should usually change the table.
    This catches silent broadcasting mistakes where correction becomes identical across rows.
    """
    counts, dils, n, cutoff, Nmax = _toy_inputs()

    out = repop.get_lpkdil_n(counts, dils, n, cutoff=cutoff, Nmax=Nmax)

    # Compare rows: they should not be identical (unless your model says they can be).
    # Use a small tolerance because values can be close for small n, but not everywhere.
    row0 = out[0]
    row1 = out[1]

    assert not torch.allclose(row0, row1, atol=1e-6, rtol=1e-6), \
        "Rows are identical across very different dilutions—possible broadcasting bug"

@pytest.mark.parametrize("cutoff", [0, 10, 100, 300])
def test_get_lpkdil_n_various_cutoffs_no_nans_and_some_finite(cutoff):
    counts, dils, n, _, Nmax = _toy_inputs()
    out = repop.get_lpkdil_n(counts, dils, n, cutoff=cutoff, Nmax=Nmax)

    # NaNs are always a bug
    assert not torch.isnan(out).any(), f"NaNs in output; stats={finite_stats(out)}"

    # Each datapoint should have at least one finite n in support
    assert torch.isfinite(out).any(dim=1).all(), f"Some rows all -inf; stats={finite_stats(out)}"


def finite_stats(x):
    finite = torch.isfinite(x)
    if finite.any():
        return {
            "min": float(x[finite].min()),
            "max": float(x[finite].max()),
            "finite_frac": float(finite.float().mean()),
            "dtype": x.dtype,
            "shape": tuple(x.shape),
        }
    else:
        return {
            "min": None,
            "max": None,
            "finite_frac": 0.0,
            "dtype": x.dtype,
            "shape": tuple(x.shape),
        }


def test_debug_get_lpkdil_n_components():

    counts = torch.tensor([5, 12], dtype=torch.long).reshape(-1, 1)
    dils   = torch.tensor([10.0, 200.0], dtype=torch.float32).reshape(-1, 1)
    Nmax = 501
    n = torch.arange(Nmax, dtype=torch.long)
    cutoff = 300

    base = repop.counts_loglike(counts, n, dils)
    logZ, lpdil_n = repop.dils_switch(dils, Nmax, cutoff)
    out = repop.get_lpkdil_n(counts, dils, n, cutoff, Nmax)

    print("base stats:", finite_stats(base))
    print("logZ stats:", finite_stats(logZ))
    print("lpdil_n stats:", finite_stats(lpdil_n))
    print("out stats:", finite_stats(out))

    # Locate non-finite entries
    bad = ~torch.isfinite(out)
    if bad.any():
        idx = bad.nonzero()[:10]
        print("First non-finite indices:", idx.tolist())
        for i, j in idx.tolist():
            print(
                f"out[{i},{j}] = {out[i, j].item()}, "
                f"base[{i},{j}] = {base[i, j].item()}"
            )

def test_binomial_loglike_k_gt_n_is_minus_inf():
    k = torch.tensor([5], dtype=torch.long).reshape(-1, 1)
    n = torch.arange(5, dtype=torch.long)  # 0..4
    p = torch.tensor([0.1], dtype=torch.float32).reshape(-1, 1)

    ll = repop.binomial_loglike(k, n, p)  # should broadcast to (1,5)
    assert torch.isneginf(ll).all()

def test_binomial_loglike_no_int64_max_leak():
    k = torch.tensor([12], dtype=torch.long).reshape(-1, 1)
    n = torch.arange(0, 50, dtype=torch.long)
    p = torch.tensor([0.1], dtype=torch.float32).reshape(-1, 1)

    ll = repop.binomial_loglike(k, n, p)
    assert ll.max().item() < 1e6
