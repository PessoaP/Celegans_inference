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
    <stem>__marginals_1D.png/.svg               (panel figure)
    <stem>__marginals_2D.png/.svg               (panel figure)
    <stem>__marginal_1D__<var>.png/.svg         (individual)
    <stem>__marginal_2D__<var1>_<var2>.png/.svg (individual)

Notes:
- Uses logsumexp (stable), treats -inf correctly.
- Plots are in ABSOLUTE log-marginal units (no max-shifting).
- Default display window:
    1D: y in [ymax - 20, ymax + 1]
    2D: colorbar in [zmax - 20, zmax]
- Optional truth overlay:
    --truth "alpha=0.01,mu=0.04,omega=0.1"
  draws vertical dashed truth lines on 1D and an 'x' marker on 2D.
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


def max_finite(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else -np.inf


def finite_floor(arr: np.ndarray, floor: float) -> np.ndarray:
    """Replace non-finite with a floor for imshow."""
    out = np.array(arr, dtype=float, copy=True)
    out[~np.isfinite(out)] = floor
    return out


def parse_truth(s: str | None) -> Dict[str, float]:
    """
    Parse comma-separated name=value pairs into a dict.
    Example: "alpha=0.01,mu=0.04,omega=0.1"
    """
    if s is None:
        return {}
    s = s.strip()
    if not s:
        return {}
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--truth entries must be name=value (got {part!r})")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty truth key in {part!r}")
        out[k] = float(v)
    return out


# -----------------------------
# marginalization
# -----------------------------
def marginal_1d(df: pd.DataFrame, var: str, vars_all: List[str], llcol: str) -> pd.DataFrame:
    if var not in df.columns:
        raise ValueError(f"Missing variable column {var!r}")

    g = df.groupby([var], sort=False)[llcol].apply(lambda x: logsumexp_np(x.to_numpy()))
    out = g.reset_index(name="log_marg")
    out = out.sort_values(var)
    return out


def marginal_2d(df: pd.DataFrame, var1: str, var2: str, vars_all: List[str], llcol: str) -> pd.DataFrame:
    for v in (var1, var2):
        if v not in df.columns:
            raise ValueError(f"Missing variable column {v!r}")

    g = df.groupby([var1, var2], sort=False)[llcol].apply(lambda x: logsumexp_np(x.to_numpy()))
    out = g.reset_index(name="log_marg")
    out = out.sort_values([var1, var2])
    return out


def pivot_2d(m2: pd.DataFrame, xvar: str, yvar: str, valcol: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = sorted_unique(m2[xvar].to_numpy())
    ys = sorted_unique(m2[yvar].to_numpy())

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {float(x): j for j, x in enumerate(xs)}
    yi = {float(y): i for i, y in enumerate(ys)}

    for _, r in m2.iterrows():
        Z[yi[float(r[yvar])], xi[float(r[xvar])]] = float(r[valcol])
    return xs, ys, Z


# -----------------------------
# plotting
# -----------------------------
def set_top_window_1d(ax: plt.Axes, y: np.ndarray, y_window: float, y_pad: float) -> None:
    ymax = max_finite(y)
    if np.isfinite(ymax):
        ax.set_ylim(ymax - float(y_window), ymax + float(y_pad))


def plot_1d_curve(
    m1: pd.DataFrame,
    var: str,
    outdir: str,
    stem: str,
    y_window: float = 20.0,
    y_pad: float = 1.0,
    truth: Dict[str, float] | None = None,
) -> None:
    x = m1[var].to_numpy(dtype=float)
    y = m1["log_marg"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(x, y, lw=2)

    truth = truth or {}
    if var in truth and np.isfinite(truth[var]):
        ax.axvline(truth[var], ls="--", lw=1.8)

    set_top_window_1d(ax, y, y_window=y_window, y_pad=y_pad)

    ax.set_xlabel(var)
    ax.set_ylabel("log marginal")
    ax.set_title(f"1D marginal over {var} (absolute)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{outdir}/{stem}__marginal_1D__{var}.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginal_1D__{var}.svg")
    plt.close(fig)


def plot_2d_heatmap(
    m2: pd.DataFrame,
    xvar: str,
    yvar: str,
    outdir: str,
    stem: str,
    ll_window: float = 20.0,
    truth: Dict[str, float] | None = None,
) -> None:
    xs, ys, Z = pivot_2d(m2, xvar, yvar, "log_marg")

    zmax = max_finite(Z)
    vmax = zmax
    vmin = zmax - float(ll_window) if np.isfinite(zmax) else -float(ll_window)

    Zp = finite_floor(Z, vmin - 1.0)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(
        Zp,
        origin="lower",
        aspect="auto",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        vmin=vmin,
        vmax=vmax,
    )

    truth = truth or {}
    if (xvar in truth) and (yvar in truth):
        ax.plot([truth[xvar]], [truth[yvar]], marker="x", ms=9, mew=2)

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    ax.set_title(f"2D marginal: ({yvar}, {xvar})  [absolute; top {ll_window:g}]")
    fig.colorbar(im, ax=ax, label="log marginal")

    fig.tight_layout()
    fig.savefig(f"{outdir}/{stem}__marginal_2D__{yvar}_{xvar}.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginal_2D__{yvar}_{xvar}.svg")
    plt.close(fig)


def plot_panel_1d(
    marg1d: Dict[str, pd.DataFrame],
    vars_all: List[str],
    outdir: str,
    stem: str,
    y_window: float = 20.0,
    y_pad: float = 1.0,
    truth: Dict[str, float] | None = None,
) -> None:
    n = len(vars_all)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    truth = truth or {}

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for k, v in enumerate(vars_all):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        m1 = marg1d[v]
        x = m1[v].to_numpy(dtype=float)
        y = m1["log_marg"].to_numpy(dtype=float)

        ax.plot(x, y, lw=2)

        if v in truth and np.isfinite(truth[v]):
            ax.axvline(truth[v], ls="--", lw=1.6)

        set_top_window_1d(ax, y, y_window=y_window, y_pad=y_pad)

        ax.set_title(f"{v}")
        ax.set_xlabel(v)
        ax.set_ylabel("log marg")
        ax.grid(True, alpha=0.25)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle("1D marginal log-likelihoods (logsumexp over other vars)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(f"{outdir}/{stem}__marginals_1D.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginals_1D.svg")
    plt.close(fig)


def plot_panel_2d(
    marg2d: Dict[Tuple[str, str], pd.DataFrame],
    pairs: List[Tuple[str, str]],
    outdir: str,
    stem: str,
    ll_window: float = 20.0,
    truth: Dict[str, float] | None = None,
) -> None:
    n = len(pairs)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    truth = truth or {}

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)

    for k, (v1, v2) in enumerate(pairs):
        # plot x=v2, y=v1 to match your earlier convention
        yvar, xvar = v1, v2
        r, c = divmod(k, ncols)
        ax = axes[r, c]

        m2 = marg2d[(v1, v2)]
        xs, ys, Z = pivot_2d(m2, xvar, yvar, "log_marg")

        zmax = max_finite(Z)
        vmax = zmax
        vmin = zmax - float(ll_window) if np.isfinite(zmax) else -float(ll_window)

        Zp = finite_floor(Z, vmin - 1.0)

        im = ax.imshow(
            Zp,
            origin="lower",
            aspect="auto",
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            vmin=vmin,
            vmax=vmax,
        )

        if (xvar in truth) and (yvar in truth):
            ax.plot([truth[xvar]], [truth[yvar]], marker="x", ms=8, mew=2)

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
    p.add_argument(
        "--vars",
        default="alpha,mu,omega",
        help="Comma-separated variable columns to marginalize over (default: alpha,mu,omega).",
    )
    p.add_argument("--loglike-col", default="loglike", help="Column name for log-likelihood (default: loglike).")
    p.add_argument("--outdir", default="figures/marginals", help="Output directory (default: figures/marginals).")

    # Defaults match your requested range behavior:
    # 1D: [max - 20, max + 1]
    # 2D: [max - 20, max]
    p.add_argument(
        "--y-window",
        type=float,
        default=20.0,
        help="1D y-axis lower window: show [max - y_window, max + y_pad] (default: 20).",
    )
    p.add_argument(
        "--y-pad",
        type=float,
        default=1.0,
        help="1D y-axis upper padding above max (default: 1).",
    )
    p.add_argument(
        "--ll-window",
        type=float,
        default=20.0,
        help="2D color window: show [max - ll_window, max] in absolute log units (default: 20).",
    )

    p.add_argument(
        "--truth",
        default=None,
        help='Optional ground truth as "name=value,name=value". Example: --truth "alpha=0.01,mu=0.04,omega=0.1"',
    )

    p.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the combined panel figures (skip per-var/per-pair figures).",
    )
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

    truth = parse_truth(args.truth)

    # --- 1D marginals
    marg1d: Dict[str, pd.DataFrame] = {}
    for v in vars_all:
        m1 = marginal_1d(df, v, vars_all, llcol)
        marg1d[v] = m1
        if not args.no_individual:
            plot_1d_curve(
                m1, v, args.outdir, stem,
                y_window=args.y_window,
                y_pad=args.y_pad,
                truth=truth,
            )

    plot_panel_1d(
        marg1d, vars_all, args.outdir, stem,
        y_window=args.y_window,
        y_pad=args.y_pad,
        truth=truth,
    )

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
                plot_2d_heatmap(
                    m2,
                    xvar=v2,
                    yvar=v1,
                    outdir=args.outdir,
                    stem=stem,
                    ll_window=args.ll_window,
                    truth=truth,
                )

    plot_panel_2d(
        marg2d, pairs, args.outdir, stem,
        ll_window=args.ll_window,
        truth=truth,
    )

    print(f"[done] wrote figures to: {args.outdir}/")


if __name__ == "__main__":
    main()


# How to call:
# python visualize_marginal_likelihoods_fullgrid.py \
#   --csv recovery_tests/grid_Exp1/grid_loglike_Exp1_full.csv \
#   --vars alpha,mu,omega \
#   --ll-window 30
#
# With ground truth:
# python visualize_marginal_likelihoods_fullgrid.py \
#   --csv recovery_tests/grid_TL_seed10/grid_likelihood.csv \
#   --vars alpha,mu,omega \
#   --truth "alpha=0.05,mu=0.5,omega=0.24"
