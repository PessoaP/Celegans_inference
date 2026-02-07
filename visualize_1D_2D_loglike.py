import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Saving helper (NEW)
# ============================================================

def _maybe_save(fig, save_dir: str | Path | None = None, fname: str | None = None, dpi: int = 300):
    """
    Save figure to save_dir/fname if both are provided.
    Creates directories automatically.
    """
    if save_dir is None or fname is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / fname, dpi=dpi)


# ============================================================
# Notebook-identical conventions
# ============================================================

def parse_slice_name(slice_name: str):
    parts = slice_name.split("_")
    if len(parts) != 2:
        raise ValueError(f"slice_name must look like 'alpha_mu' (got {slice_name!r})")
    return parts[0], parts[1]


def choose_axes(var1: str, var2: str):
    # notebook convention: x = var2, y = var1
    return var2, var1


# ============================================================
# Non-finite handling (plot-only)
# ============================================================

def _finite_max(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    m = np.isfinite(arr)
    if not np.any(m):
        raise ValueError("All values are non-finite; nothing to plot.")
    return float(np.max(arr[m]))


def _finite_min(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    m = np.isfinite(arr)
    if not np.any(m):
        raise ValueError("All values are non-finite; nothing to plot.")
    return float(np.min(arr[m]))


def _finite_safe_for_plot(Z: np.ndarray, floor: float) -> np.ndarray:
    """
    Matplotlib contour cannot handle non-finite values.
    For display only, replace non-finite with a finite floor below vmin so it saturates.
    Finite values are unchanged.
    """
    Z = np.asarray(Z, dtype=float)
    Zp = Z.copy()
    Zp[~np.isfinite(Zp)] = float(floor)
    return Zp


# ============================================================
# 1D plotting (display window via y-lims only; no clipping)
# ============================================================

def plot_1d_sweep(
    base_dir: str,
    param_name: str,                 # "alpha" / "mu" / "d"
    ll_col: str = "loglike",
    truth: dict | None = None,
    ll_window: float | None = 20.0,  # show [max-ll_window, max+vmax_pad]
    vmax_pad: float = 0.0,           # e.g. 1.0 => max+1
    marker: str = "o",
    figsize=(6, 3.5),
    sort_x: bool = True,
    show: bool = True,
    save_dir: str | Path | None = None,  # NEW
    fname: str | None = None,            # NEW
    dpi: int = 300,                      # NEW
):
    csv_path = os.path.join(base_dir, f"{param_name}_only.csv")
    df = pd.read_csv(csv_path)

    if param_name not in df.columns:
        raise ValueError(f"{csv_path} missing column '{param_name}'. Has: {list(df.columns)}")
    if ll_col not in df.columns:
        raise ValueError(f"{csv_path} missing column '{ll_col}'. Has: {list(df.columns)}")

    x = df[param_name].to_numpy(dtype=float)
    ll = df[ll_col].to_numpy(dtype=float)

    if sort_x:
        idx = np.argsort(x)
        x, ll = x[idx], ll[idx]

    fig = plt.figure(figsize=figsize)
    plt.plot(x, ll, marker=marker)
    plt.xlabel(param_name)
    plt.ylabel(ll_col)
    plt.title(f"1D {ll_col}: {param_name}")

    # display-only window computed from finite values
    llmax = _finite_max(ll)
    if ll_window is not None:
        ymin = llmax - float(ll_window)
        ymax = llmax + float(vmax_pad)
        plt.ylim(ymin, ymax)

    if truth is not None and param_name in truth:
        plt.axvline(truth[param_name], color="k", linestyle="--", linewidth=1, label="ground truth")
        plt.legend()

    plt.tight_layout()

    _maybe_save(fig, save_dir=save_dir, fname=fname, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return x, ll, csv_path


# ============
# 2D plotting 
# ============

def plot_2d_grid_slice(
    base_dir: str,
    slice_name: str,                 # "alpha_mu", "alpha_d", "mu_d"
    held_var: str,
    truth: dict,
    ll_col: str = "loglike",
    contour_deltas=(0.5, 1.0, 2.0, 4.0),
    ll_window: float = 20.0,         # show [max-ll_window, max+vmax_pad]
    vmax_pad: float = 0.0,           # e.g. 1.0 => max+1
    cmap=None,
    figsize=(6, 5),
    show: bool = True,
    save_dir: str | Path | None = None,  # NEW
    fname: str | None = None,            # NEW
    dpi: int = 300,                      # NEW
):
    var1, var2 = parse_slice_name(slice_name)
    xvar, yvar = choose_axes(var1, var2)

    csv_path = os.path.join(base_dir, f"{slice_name}.csv")
    df = pd.read_csv(csv_path)

    for col in (xvar, yvar, ll_col):
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing column '{col}'. Has: {list(df.columns)}")

    xs = np.sort(df[xvar].unique())
    ys = np.sort(df[yvar].unique())
    XN, YN = len(xs), len(ys)

    LL = np.full((YN, XN), np.nan, dtype=float)
    x_to_j = {x: j for j, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}

    for _, row in df.iterrows():
        i = y_to_i[row[yvar]]
        j = x_to_j[row[xvar]]
        LL[i, j] = float(row[ll_col])

    finite = np.isfinite(LL)
    if not np.any(finite):
        raise ValueError(f"All {ll_col} values are non-finite in {csv_path}")

    LLmax = float(np.max(LL[finite]))
    vmin = LLmax - float(ll_window)
    vmax = LLmax + float(vmax_pad)

    # Plot-safe copy for imshow/contour (keeps finite unchanged; prevents crashes)
    LL_plot = _finite_safe_for_plot(LL, floor=vmin - 1.0)

    held_text = ""
    if held_var is not None and held_var in truth:
        held_text = f" ({held_var} fixed = {truth[held_var]:g})"
    elif held_var is not None:
        held_text = f" ({held_var} fixed)"

    fig = plt.figure(figsize=figsize)

    im = plt.imshow(
        LL_plot,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, label=ll_col)
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    plt.title(f"2D {ll_col}: {yvar}–{xvar} (shown [{vmin:.1f}, {vmax:.1f}]){held_text}")

    # Contours at max - delta (NOTEBOOK STYLE)
    deltas = np.array(contour_deltas, dtype=float)
    levels = np.sort(LLmax - deltas)

    plt.contour(
        xs, ys, LL_plot,
        levels=levels,
        colors="white",
        linewidths=1.2,
    )

    if (xvar in truth) and (yvar in truth):
        plt.scatter(truth[xvar], truth[yvar], color="red", marker="x", s=80, label="ground truth")
        plt.legend()

    plt.tight_layout()

    # NEW: save
    _maybe_save(fig, save_dir=save_dir, fname=fname, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return xs, ys, LL, csv_path


# ============================================================
# Make the 2x3 panel that combines 1D and 2D slices
# ============================================================

def make_combined_panel(
    base_dir: str,
    truth: dict,
    ll_window: float = 20.0,
    vmax_pad: float = 1.0,  # max + 1
    contour_deltas=(0.5, 1.0, 2.0, 4.0),
    figsize=(12, 6),
    save_dir: str | Path | None = None,  # NEW (preferred over save_path)
    fname: str | None = None,            # NEW
    dpi: int = 300,                      # NEW
):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- 1D row
    for j, p in enumerate(["alpha", "mu", "d"]):
        ax = fig.add_subplot(gs[0, j])

        csv_path = os.path.join(base_dir, f"{p}_only.csv")
        df = pd.read_csv(csv_path)
        x = df[p].to_numpy(dtype=float)
        ll = df["loglike"].to_numpy(dtype=float)
        idx = np.argsort(x)
        x, ll = x[idx], ll[idx]

        ax.plot(x, ll, marker="o")
        ax.set_xlabel(p)
        ax.set_ylabel("loglike")
        ax.set_title(f"{p} only")

        llmax = _finite_max(ll)
        ax.set_ylim(llmax - ll_window, llmax + vmax_pad)

        if p in truth:
            ax.axvline(truth[p], color="k", linestyle="--", linewidth=1)

    # --- 2D row
    slices = [("alpha_mu", "d"), ("alpha_d", "mu"), ("mu_d", "alpha")]

    for j, (slice_name, held) in enumerate(slices):
        ax = fig.add_subplot(gs[1, j])

        var1, var2 = parse_slice_name(slice_name)
        xvar, yvar = choose_axes(var1, var2)

        csv_path = os.path.join(base_dir, f"{slice_name}.csv")
        df = pd.read_csv(csv_path)

        xs = np.sort(df[xvar].unique())
        ys = np.sort(df[yvar].unique())

        LL = np.full((len(ys), len(xs)), np.nan, dtype=float)
        x_to_j = {x: jj for jj, x in enumerate(xs)}
        y_to_i = {y: ii for ii, y in enumerate(ys)}

        for _, row in df.iterrows():
            LL[y_to_i[row[yvar]], x_to_j[row[xvar]]] = float(row["loglike"])

        finite = np.isfinite(LL)
        if not np.any(finite):
            raise ValueError(f"All loglike values are non-finite in {csv_path}")
        LLmax = float(np.max(LL[finite]))

        vmin = LLmax - ll_window
        vmax = LLmax + vmax_pad
        LL_plot = _finite_safe_for_plot(LL, floor=vmin - 1.0)

        im = ax.imshow(
            LL_plot,
            origin="lower",
            aspect="auto",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label="loglike", pad=0.02, fraction=0.046)

        levels = np.sort(LLmax - np.array(contour_deltas, dtype=float))
        ax.contour(xs, ys, LL_plot, levels=levels, colors="white", linewidths=1.2)

        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)

        held_text = f" ({held} fixed = {truth[held]:g})" if held in truth else f" ({held} fixed)"
        ax.set_title(f"{slice_name}{held_text}")

        if (xvar in truth) and (yvar in truth):
            ax.scatter(truth[xvar], truth[yvar], color="red", marker="x", s=60)

    _maybe_save(fig, save_dir=save_dir, fname=fname, dpi=dpi)

    return fig


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    base = "recovery_tests/grid_regular"  # <-- change
    truth = {"alpha": 0.05, "mu": 0.5, "d": 0.12}

    # Choose a subfolder under figures/
    SAVE_DIR = Path("figures") / "grid_synth_regularfeed"

    # --- Notebook-identical single figures (now saving) ---
    plot_1d_sweep(base, "alpha", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="alpha_1d.png", show=True)
    plot_1d_sweep(base, "mu", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="mu_1d.png", show=True)
    plot_1d_sweep(base, "d", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="d_1d.png", show=True)

    plot_2d_grid_slice(base, "alpha_mu", held_var="d", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="alpha_mu_2d.png", show=True)
    plot_2d_grid_slice(base, "alpha_d", held_var="mu", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="alpha_d_2d.png", show=True)
    plot_2d_grid_slice(base, "mu_d", held_var="alpha", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="mu_d_2d.png", show=True)

    # --- Combined Panel ---
    fig = make_combined_panel(
        base,
        truth,
        ll_window=20,
        vmax_pad=1.0,
        save_dir=SAVE_DIR,
        fname="1D_2D_slices_likelihood_combined.png",
    )
    plt.show()
