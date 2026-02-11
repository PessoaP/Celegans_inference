#!/usr/bin/env python3
"""
visualize_marginal_likelihoods_fullgrid.py

Given ONE CSV containing a full Cartesian grid of parameter values and a log-likelihood column,
compute and visualize:

  1) 1D marginal log-likelihood curves for each parameter:
       log p(x) = logsumexp_{others} loglike

  2) 2D marginal log-likelihood heatmaps for each parameter pair:
       log p(x, y) = logsumexp_{others} loglike

Outputs (by default):
  figures/marginals/
    <stem>__marginals_1D.png/.svg          (panel figure)
    <stem>__marginals_2D.png/.svg          (panel figure)
    <stem>__marginal_1D__<var>.png/.svg    (individual)
    <stem>__marginal_2D__<var1>_<var2>.png/.svg (individual)

Notes:
- Uses logsumexp (stable), treats -inf correctly.
- For plotting, marginals are shifted so max = 0 (purely for readability).
  This preserves relative structure while making color/axes interpretable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# numerics: stable logsumexp
# -----------------------------
def logsumexp_np(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        # all -inf (or all nan/inf) -> return -inf
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


# -----------------------------
# IO / utils
# -----------------------------
def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def parse_vars(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def sorted_unique(vals: np.ndarray) -> np.ndarray:
    return np.sort(np.unique(np.asarray(vals, dtype=float)))


def safe_shift_to_zero(arr: np.ndarray) -> np.ndarray:
    """Shift so max finite is 0; keep -inf as -inf."""
    arr = np.asarray(arr, dtype=float)
    m = np.max(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else -np.inf
    if not np.isfinite(m):
        return arr
    out = arr - m
    return out


def finite_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    """Replace non-finite with a floor for imshow."""
    out = np.array(arr, dtype=float, copy=True)
    out[~np.isfinite(out)] = floor
    return out


# -----------------------------
# marginalization
# -----------------------------
def marginal_1d(df: pd.DataFrame, var: str, vars_all: List[str], llcol: str) -> pd.DataFrame:
    keep = [var]
    drop = [v for v in vars_all if v not in keep]
    if var not in df.columns:
        raise ValueError(f"Missing variable column {var!r}")

    g = df.groupby(keep, sort=False)[llcol].apply(lambda x: logsumexp_np(x.to_numpy()))
    out = g.reset_index(name="log_marg")
    out = out.sort_values(var)
    out["log_marg_shift"] = safe_shift_to_zero(out["log_marg"].to_numpy())
    return out


def marginal_2d(df: pd.DataFrame, var1: str, var2: str, vars_all: List[str], llcol: str) -> pd.DataFrame:
    keep = [var1, var2]
    for v in keep:
        if v not in df.columns:
            raise ValueError(f"Missing variable column {v!r}")

    g = df.groupby(keep, sort=False)[llcol].apply(lambda x: logsumexp_np(x.to_numpy()))
    out = g.reset_index(name="log_marg")
    out = out.sort_values([var1, var2])
    out["log_marg_shift"] = safe_shift_to_zero(out["log_marg"].to_numpy())
    return out


def pivot_2d(m2: pd.DataFrame, xvar: str, yvar: str, valcol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted_unique(m2[xvar].to_numpy())
    ys = sorted_unique(m2[yvar].to_numpy())

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {x: j for j, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}

    for _, r in m2.iterrows():
        Z[yi[float(r[yvar])], xi[float(r[xvar])]] = float(r[valcol])
    return xs, ys, Z


# -----------------------------
# plotting
# -----------------------------
def plot_1d_curve(m1: pd.DataFrame, var: str, outdir: str, stem: str) -> None:
    x = m1[var].to_numpy(dtype=float)
    y = m1["log_marg_shift"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, lw=2)
    ax.set_xlabel(var)
    ax.set_ylabel("log marginal (shifted; max=0)")
    ax.set_title(f"1D marginal over {var}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(f"{outdir}/{stem}__marginal_1D__{var}.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginal_1D__{var}.svg")
    plt.close(fig)


def plot_2d_heatmap(m2: pd.DataFrame, xvar: str, yvar: str, outdir: str, stem: str, ll_window: float) -> None:
    xs, ys, Z = pivot_2d(m2, xvar, yvar, "log_marg_shift")

    # display window: keep values >= -ll_window (since max is 0 after shift)
    vmin = -float(ll_window)
    Zp = finite_floor(Z, vmin - 1.0)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(
        Zp,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        vmin=vmin,
        vmax=0.0,
    )
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"2D marginal: ({yvar}, {xvar})  [shifted; max=0]")
    cb = fig.colorbar(im, ax=ax, label="log marginal (shifted)")
    fig.tight_layout()

    fig.savefig(f"{outdir}/{stem}__marginal_2D__{yvar}_{xvar}.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginal_2D__{yvar}_{xvar}.svg")
    plt.close(fig)


def plot_panel_1d(marg1d: Dict[str, pd.DataFrame], vars_all: List[str], outdir: str, stem: str) -> None:
    n = len(vars_all)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for k, v in enumerate(vars_all):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        m1 = marg1d[v]
        ax.plot(m1[v].to_numpy(dtype=float), m1["log_marg_shift"].to_numpy(dtype=float), lw=2)
        ax.set_title(f"{v}")
        ax.set_xlabel(v)
        ax.set_ylabel("log marg (max=0)")
        ax.grid(True, alpha=0.25)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle("1D marginal log-likelihoods (logsumexp over other vars)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(f"{outdir}/{stem}__marginals_1D.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginals_1D.svg")
    plt.close(fig)


def plot_panel_2d(marg2d: Dict[Tuple[str, str], pd.DataFrame], pairs: List[Tuple[str, str]],
                  outdir: str, stem: str, ll_window: float) -> None:
    n = len(pairs)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)

    for k, (v1, v2) in enumerate(pairs):
        # we’ll plot x=v2, y=v1 to match your earlier convention
        yvar, xvar = v1, v2
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        m2 = marg2d[(v1, v2)]
        xs, ys, Z = pivot_2d(m2, xvar, yvar, "log_marg_shift")

        vmin = -float(ll_window)
        Zp = finite_floor(Z, vmin - 1.0)

        im = ax.imshow(
            Zp,
            origin="lower",
            aspect="auto",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=vmin,
            vmax=0.0,
        )
        ax.set_title(f"{yvar} vs {xvar}")
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle("2D marginal log-likelihood heatmaps (logsumexp over remaining vars)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(f"{outdir}/{stem}__marginals_2D.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginals_2D.svg")
    plt.close(fig)


# -----------------------------
# main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV file containing full grid and loglike column.")
    p.add_argument("--vars", default="alpha,mu,omega",
                   help="Comma-separated variable columns to marginalize over (default: alpha,mu,omega).")
    p.add_argument("--loglike-col", default="loglike", help="Column name for log-likelihood (default: loglike).")
    p.add_argument("--outdir", default="figures/marginals", help="Output directory (default: figures/marginals).")
    p.add_argument("--y-window", type=float, default=None,
                    help="If set, restrict 1D marginal y-axis to [-y_window, 0] (after shifting max to 0).")
    p.add_argument("--ll-window", type=float, default=20.0,
                   help="For 2D heatmaps, show values down to -ll-window (after shifting max to 0).")
    p.add_argument("--no-individual", action="store_true",
                   help="Only write the combined panel figures (skip per-var/per-pair figures).")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    vars_all = parse_vars(args.vars)
    llcol = args.loglike_col

    df = pd.read_csv(csv_path)
    missing = [v for v in vars_all + [llcol] if v not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. CSV has: {list(df.columns)}")

    # Keep only what we need (faster, cleaner)
    df = df[vars_all + [llcol]].copy()

    # Force numeric (defensive)
    for v in vars_all:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df[llcol] = pd.to_numeric(df[llcol], errors="coerce")

    # Drop rows with missing parameter values (loglike can be -inf; that's fine)
    df = df.dropna(subset=vars_all)

    ensure_outdir(args.outdir)
    stem = csv_path.stem

    # --- 1D marginals
    marg1d: Dict[str, pd.DataFrame] = {}
    for v in vars_all:
        m1 = marginal_1d(df, v, vars_all, llcol)
        marg1d[v] = m1
        if not args.no_individual:
            plot_1d_curve(m1, v, args.outdir, stem)

    plot_panel_1d(marg1d, vars_all, args.outdir, stem)

    # --- 2D marginals (all unordered pairs)
    pairs: List[Tuple[str, str]] = []
    marg2d: Dict[Tuple[str, str], pd.DataFrame] = {}
    for i in range(len(vars_all)):
        for j in range(i + 1, len(vars_all)):
            v1, v2 = vars_all[i], vars_all[j]
            pairs.append((v1, v2))
            m2 = marginal_2d(df, v1, v2, vars_all, llcol)
            marg2d[(v1, v2)] = m2
            if not args.no_individual:
                # By convention, save as y_x in filename to match your other scripts
                plot_2d_heatmap(m2, xvar=v2, yvar=v1, outdir=args.outdir, stem=stem, ll_window=args.ll_window)

    plot_panel_2d(marg2d, pairs, args.outdir, stem, ll_window=args.ll_window)

    print(f"[done] wrote figures to: {args.outdir}/")


if __name__ == "__main__":
    main()


# How to call:
# python visualize_marginal_likelihoods_fullgrid.py \
#   --csv recovery_tests/grid_Exp1/grid_likelihood.csv \
#   --vars alpha,mu,omega \
#   --ll-window 20
