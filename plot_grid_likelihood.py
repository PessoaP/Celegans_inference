import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "grid_loglike.csv"  # or "/mnt/data/grid_loglike.csv"

# ----------------------------
# Numerics
# ----------------------------
def logsumexp(a, axis=None):
    """Stable logsumexp that handles -inf nicely."""
    a = np.asarray(a)
    amax = np.nanmax(a, axis=axis, keepdims=True)
    # If all entries are -inf along an axis, amax will be -inf; keep it safe:
    out = amax + np.log(np.nansum(np.exp(a - amax), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out

def normalize_logweights(logw):
    """Return normalized weights from log-weights (softmax), stable."""
    m = np.max(logw)
    w = np.exp(logw - m)
    s = np.sum(w)
    return w / s if s > 0 else w

# ----------------------------
# Load + helpers
# ----------------------------
def load_grid(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    required = {"alpha", "mu", "omega", "loglike"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    # Ensure numeric
    for c in ["alpha", "mu", "omega", "loglike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows that are fully NaN in loglike (keep -inf!)
    df = df.dropna(subset=["loglike"])
    return df

def nearest_in_grid(values, x):
    values = np.asarray(values)
    return values[np.argmin(np.abs(values - x))]

def pivot_2d(df, x, y, z, agg="logsumexp"):
    """
    Make a 2D grid over (x,y) by marginalizing over z.
    agg:
      - 'logsumexp': marginalize/integrate out z in log space (default)
      - 'max': take maximum over z (profile likelihood)
      - 'meanexp': average in probability space then log (log(mean(exp)))
    """
    # Group by x,y and aggregate over z
    g = df.groupby([x, y])["loglike"]

    if agg == "logsumexp":
        vals = g.apply(lambda s: logsumexp(s.values))
    elif agg == "max":
        vals = g.max()
    elif agg == "meanexp":
        vals = g.apply(lambda s: (logsumexp(s.values) - np.log(len(s))))
    else:
        raise ValueError("agg must be one of: 'logsumexp', 'max', 'meanexp'")

    grid = vals.unstack(y)  # rows: x, cols: y
    # Sort axes numerically
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    return grid

# ----------------------------
# Plotting
# ----------------------------
def plot_heatmap(grid2d, x_label, y_label, title=None, cbar_label="log (marginal likelihood)"):
    """
    grid2d: DataFrame indexed by x, columns by y, values are log-something.
    """
    x_vals = grid2d.index.values
    y_vals = grid2d.columns.values
    Z = grid2d.values

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    # Use pcolormesh with explicit edges (nice for non-uniform grids too)
    # Build edges by midpoints; fallback to simple spacing
    def edges(vals):
        vals = np.asarray(vals)
        if len(vals) == 1:
            return np.array([vals[0] - 0.5, vals[0] + 0.5])
        mids = (vals[1:] + vals[:-1]) / 2
        left = vals[0] - (mids[0] - vals[0])
        right = vals[-1] + (vals[-1] - mids[-1])
        return np.concatenate([[left], mids, [right]])

    xe = edges(x_vals)
    ye = edges(y_vals)

    pcm = ax.pcolormesh(xe, ye, Z.T, shading="auto")  # transpose so y is vertical
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    return fig, ax

def plot_1d_marginal(df, var, marginalize_over=("alpha", "mu", "omega"), mode="posterior"):
    """
    mode:
      - 'logmarg': plots log marginal (up to constant)
      - 'posterior': plots normalized marginal posterior over grid points (uniform prior on grid)
    """
    others = [v for v in marginalize_over if v != var]
    g = df.groupby(var)["loglike"].apply(lambda s: logsumexp(s.values))
    g = g.sort_index()
    x = g.index.values
    logm = g.values

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    if mode == "logmarg":
        ax.plot(x, logm, marker="o")
        ax.set_ylabel("log marginal (up to const.)")
    elif mode == "posterior":
        p = normalize_logweights(logm)
        ax.plot(x, p, marker="o")
        ax.set_ylabel(f"p({var})")
    else:
        raise ValueError("mode must be 'logmarg' or 'posterior'")

    ax.set_xlabel(var)
    ax.set_title(f"1D marginal over {var} (marginalizing {others})")
    plt.tight_layout()
    return fig, ax

def plot_slice(df, fixed, x, y, agg_over=None, agg="max"):
    """
    Plot a 2D slice at fixed values.
    fixed: dict like {'alpha': 0.01} or {'mu': 0.5, 'omega': 0.2}
           values are snapped to nearest grid point present.
    x,y: axes to plot
    agg_over: variable to aggregate over if you still have a remaining dimension
              (e.g., fixed={'alpha':0.01} and x='mu', y='omega' -> agg_over=None)
              (e.g., fixed={'alpha':0.01} and x='mu', y='alpha' not sensible)
    agg: 'max' (profile) or 'logsumexp' (marginal)
    """
    df2 = df.copy()

    # Snap fixed values to nearest in-grid values
    for k, v in fixed.items():
        grid_vals = np.sort(df2[k].unique())
        v_snap = nearest_in_grid(grid_vals, v)
        df2 = df2[df2[k] == v_snap]
        print(f"[slice] fixed {k}={v} -> snapped to {v_snap}")

    if agg_over is None:
        # Expect exactly one value per (x,y); if duplicates remain, use agg
        grid = pivot_2d(df2, x, y, z=None, agg="max")  # z ignored here, but we need pivot
    else:
        # pivot_2d expects a z to marginalize over, so use it directly
        grid = pivot_2d(df2, x, y, z=agg_over, agg=("logsumexp" if agg == "logsumexp" else "max"))

    fig, ax = plot_heatmap(
        grid,
        x_label=x,
        y_label=y,
        title=f"Slice: fixed {fixed} | agg={agg}" + (f" over {agg_over}" if agg_over else "")
    )
    return fig, ax

# Hack: pivot_2d above assumes a z; for exact 2D slices with possible duplicates, do:
def pivot_exact_2d(df, x, y, how="max"):
    g = df.groupby([x, y])["loglike"]
    vals = g.max() if how == "max" else g.apply(lambda s: logsumexp(s.values))
    grid = vals.unstack(y).sort_index(axis=0).sort_index(axis=1)
    return grid

# ----------------------------
# Convenience: pairwise landscapes
# ----------------------------
def pairwise_landscapes(df, agg="logsumexp"):
    """
    Returns dict of 2D grids for:
      (alpha, mu) marginalized over omega
      (alpha, omega) marginalized over mu
      (mu, omega) marginalized over alpha
    """
    grids = {}
    grids[("alpha", "mu")] = pivot_2d(df, "alpha", "mu", "omega", agg=agg)
    grids[("alpha", "omega")] = pivot_2d(df, "alpha", "omega", "mu", agg=agg)
    grids[("mu", "omega")] = pivot_2d(df, "mu", "omega", "alpha", agg=agg)
    return grids

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    df = load_grid(CSV_PATH)

    # Pairwise marginalized landscapes (log-sum-exp marginalization)
    grids = pairwise_landscapes(df, agg="logsumexp")

    fig, ax = plot_heatmap(grids[("mu", "omega")], "mu", "omega",
                           title="log p(data | mu, omega) (marginalized over alpha)")
    plt.show()

    fig, ax = plot_heatmap(grids[("alpha", "mu")], "alpha", "mu",
                           title="log p(data | alpha, mu) (marginalized over omega)")
    plt.show()

    fig, ax = plot_heatmap(grids[("alpha", "omega")], "alpha", "omega",
                           title="log p(data | alpha, omega) (marginalized over mu)")
    plt.show()

    # 1D marginals (normalized posterior over grid points, assuming uniform prior)
    plot_1d_marginal(df, "mu", mode="posterior")
    plt.show()

    plot_1d_marginal(df, "omega", mode="posterior")
    plt.show()

    plot_1d_marginal(df, "alpha", mode="posterior")
    plt.show()

    # Cross-section example: slice at alpha ~ 0.01, show mu-omega plane (profile likelihood)
    alpha_target = 0.01
    df_a = df[df["alpha"] == nearest_in_grid(np.sort(df["alpha"].unique()), alpha_target)]
    grid_slice = pivot_exact_2d(df_a, "mu", "omega", how="max")  # or how="logsumexp"
    plot_heatmap(grid_slice, "mu", "omega", title=f"Slice at alpha≈{alpha_target} (profile over duplicates)")
    plt.show()
