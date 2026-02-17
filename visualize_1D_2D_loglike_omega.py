import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Pretty parameter labels
# ============================================================

VAR_LABELS = {
    "alpha": r"$\alpha$",
    "mu":    r"$\mu$",
    "omega": r"$\omega$",
}

def lab(v: str) -> str:
    return VAR_LABELS.get(v, v)


# ============================================================
# Helper Functions
# ============================================================

def _maybe_save(fig, save_dir: str | Path | None = None, fname: str | None = None, dpi: int = 300):
    if save_dir is None or fname is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / fname, dpi=dpi)


def parse_slice_name(slice_name: str):
    parts = slice_name.split("_")
    if len(parts) != 2:
        raise ValueError(f"slice_name must look like 'alpha_mu' (got {slice_name!r})")
    return parts[0], parts[1]


def choose_axes(var1: str, var2: str):
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


def _finite_safe_for_plot(Z: np.ndarray, floor: float) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    Zp = Z.copy()
    Zp[~np.isfinite(Zp)] = float(floor)
    return Zp


# ============================================================
# 1D plotting 
# ============================================================

def plot_1d_sweep(
    base_dir: str,
    param_name: str,
    ll_col: str = "loglike",
    truth: dict | None = None,
    ll_window: float | None = 20.0,
    vmax_pad: float = 0.0,
    marker: str = "o",
    figsize=(6, 3.5),
    sort_x: bool = True,
    show: bool = True,
    save_dir: str | Path | None = None,
    fname: str | None = None,
    dpi: int = 300,
):
    csv_path = os.path.join(base_dir, f"{param_name}_only.csv")
    df = pd.read_csv(csv_path)

    x = df[param_name].to_numpy(dtype=float)
    ll = df[ll_col].to_numpy(dtype=float)

    if sort_x:
        idx = np.argsort(x)
        x, ll = x[idx], ll[idx]

    fig = plt.figure(figsize=figsize)
    plt.plot(x, ll, marker=marker)
    plt.xlabel(lab(param_name))
    plt.ylabel("log-likelihood")
    plt.title(f"1D log-likelihood: {lab(param_name)}")

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


# ============================================================
# 2D plotting 
# ============================================================

def plot_2d_grid_slice(
    base_dir: str,
    slice_name: str,
    held_var: str,
    truth: dict,
    ll_col: str = "loglike",
    contour_deltas=(0.5, 1.0, 2.0, 4.0),
    ll_window: float = 20.0,
    vmax_pad: float = 0.0,
    cmap=None,
    figsize=(6, 5),
    show: bool = True,
    save_dir: str | Path | None = None,
    fname: str | None = None,
    dpi: int = 300,
):
    var1, var2 = parse_slice_name(slice_name)
    xvar, yvar = choose_axes(var1, var2)

    csv_path = os.path.join(base_dir, f"{slice_name}.csv")
    df = pd.read_csv(csv_path)

    xs = np.sort(df[xvar].unique())
    ys = np.sort(df[yvar].unique())

    LL = np.full((len(ys), len(xs)), np.nan, dtype=float)
    x_to_j = {x: j for j, x in enumerate(xs)}
    y_to_i = {y: i for i, y in enumerate(ys)}

    for _, row in df.iterrows():
        LL[y_to_i[row[yvar]], x_to_j[row[xvar]]] = float(row[ll_col])

    finite = np.isfinite(LL)
    LLmax = float(np.max(LL[finite]))

    vmin = LLmax - float(ll_window)
    vmax = LLmax + float(vmax_pad)

    LL_plot = _finite_safe_for_plot(LL, floor=vmin - 1.0)

    held_text = ""
    if held_var is not None:
        if held_var in truth:
            held_text = f" ({lab(held_var)} fixed = {truth[held_var]:g})"
        else:
            held_text = f" ({lab(held_var)} fixed)"

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

    plt.colorbar(im, label="log-likelihood")
    plt.xlabel(lab(xvar))
    plt.ylabel(lab(yvar))
    plt.title(f"2D log-likelihood: {lab(yvar)}–{lab(xvar)}{held_text}")

    levels = np.sort(LLmax - np.array(contour_deltas, dtype=float))
    plt.contour(xs, ys, LL_plot, levels=levels, colors="white", linewidths=1.2)

    if (xvar in truth) and (yvar in truth):
        plt.scatter(truth[xvar], truth[yvar], color="red", marker="x", s=80, label="ground truth")
        plt.legend()

    plt.tight_layout()
    _maybe_save(fig, save_dir=save_dir, fname=fname, dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return xs, ys, LL, csv_path


# ============================================================
# Combined Panel
# ============================================================

def make_combined_panel(
    base_dir: str,
    truth: dict,
    ll_window: float = 20.0,
    vmax_pad: float = 1.0,
    contour_deltas=(0.5, 1.0, 2.0, 4.0),
    figsize=(12, 6),
    save_dir: str | Path | None = None,
    fname: str | None = None,
    dpi: int = 300,
):
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # --- 1D row
    for j, p in enumerate(["alpha", "mu", "omega"]):
        ax = fig.add_subplot(gs[0, j])
        df = pd.read_csv(os.path.join(base_dir, f"{p}_only.csv"))

        x = df[p].to_numpy(dtype=float)
        ll = df["loglike"].to_numpy(dtype=float)
        idx = np.argsort(x)
        x, ll = x[idx], ll[idx]

        ax.plot(x, ll, marker="o")
        ax.set_xlabel(lab(p))
        ax.set_ylabel("log-likelihood")
        ax.set_title(f"{lab(p)}")

        llmax = _finite_max(ll)
        ax.set_ylim(llmax - ll_window, llmax + vmax_pad)

        if p in truth:
            ax.axvline(truth[p], color="k", linestyle="--", linewidth=1)

    # --- 2D row
    slices = [("alpha_mu", "omega"), ("alpha_omega", "mu"), ("mu_omega", "alpha")]

    for j, (slice_name, held) in enumerate(slices):
        ax = fig.add_subplot(gs[1, j])

        var1, var2 = parse_slice_name(slice_name)
        xvar, yvar = choose_axes(var1, var2)

        df = pd.read_csv(os.path.join(base_dir, f"{slice_name}.csv"))

        xs = np.sort(df[xvar].unique())
        ys = np.sort(df[yvar].unique())

        LL = np.full((len(ys), len(xs)), np.nan, dtype=float)
        x_to_j = {x: jj for jj, x in enumerate(xs)}
        y_to_i = {y: ii for ii, y in enumerate(ys)}

        for _, row in df.iterrows():
            LL[y_to_i[row[yvar]], x_to_j[row[xvar]]] = float(row["loglike"])

        LLmax = float(np.max(LL[np.isfinite(LL)]))

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
        fig.colorbar(im, ax=ax, label="log-likelihood", pad=0.02, fraction=0.046)

        levels = np.sort(LLmax - np.array(contour_deltas, dtype=float))
        ax.contour(xs, ys, LL_plot, levels=levels, colors="white", linewidths=1.2)

        ax.set_xlabel(lab(xvar))
        ax.set_ylabel(lab(yvar))

        held_text = f" ({lab(held)} fixed = {truth[held]:g})" if held in truth else f" ({lab(held)} fixed)"
        ax.set_title(f"{lab(yvar)}–{lab(xvar)}{held_text}")

        if (xvar in truth) and (yvar in truth):
            ax.scatter(truth[xvar], truth[yvar], color="red", marker="x", s=60)

    _maybe_save(fig, save_dir=save_dir, fname=fname, dpi=dpi)
    return fig



# ============================================================
# Running
# ============================================================

if __name__ == "__main__":
    base = "recovery_tests/grid_TL_omega"  # <-- change per dataset
    truth = {"alpha": 0.05, "mu": 0.5, "omega": 0.24}

    # Choose a subfolder under figures/
    SAVE_DIR = Path("figures") / "grid_synth_TL_omega"

    # --- Notebook-identical single figures (now saving) ---
    plot_1d_sweep(base, "alpha", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="alpha_1d.png", show=True)
    plot_1d_sweep(base, "mu", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="mu_1d.png", show=True)
    plot_1d_sweep(base, "omega", truth=truth, ll_window=20, vmax_pad=1.0,
                 save_dir=SAVE_DIR, fname="d_1d.png", show=True)

    plot_2d_grid_slice(base, "alpha_mu", held_var="omega", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="alpha_mu_2d.png", show=True)
    plot_2d_grid_slice(base, "alpha_omega", held_var="mu", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="alpha_omega_2d.png", show=True)
    plot_2d_grid_slice(base, "mu_omega", held_var="alpha", truth=truth, ll_window=20, vmax_pad=1.0,
                       save_dir=SAVE_DIR, fname="mu_omega_2d.png", show=True)

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
