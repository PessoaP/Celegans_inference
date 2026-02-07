#!/usr/bin/env python3
"""
visualize_2D_grid_slices.py

2D log-likelihood slices (mu–omega) along alpha from a grid-search CSV.

Input CSV must contain:
    alpha, mu, omega, loglike
(extra columns are ignored)

Outputs (fixed outdir):
    figures/grid_3d/
        <csvstem>__slice_alpha_<value>.png/.svg
        <csvstem>__slices_all.png/.svg

Default behavior (real data):
- show top-N loglike window (default 20)
- white contour lines
- mark slice-wise MLE (gold dot)

Optional (synthetic):
- overlay ground truth with --truth ... --show-truth
"""

from __future__ import annotations
import argparse
import os
from typing import Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# Headless-safe backend (won't break interactive desktop use)
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# helpers
# ============================================================

OUTDIR = "figures/grid_3d"


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


def finite_safe(arr: np.ndarray, floor: float) -> np.ndarray:
    out = np.array(arr, dtype=float, copy=True)
    out[~np.isfinite(out)] = floor
    return out


def snap(values: np.ndarray, target: float) -> float:
    return float(values[np.argmin(np.abs(values - target))])


def parse_truth(s: Optional[str]) -> Optional[Dict[str, float]]:
    if s is None:
        return None
    out: Dict[str, float] = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


# ============================================================
# grid construction
# ============================================================

def build_slice(df: pd.DataFrame, alpha_value: float, *, snap_alpha: bool = True):
    if "alpha" not in df.columns:
        raise ValueError("CSV missing required column 'alpha'.")

    alphas = np.sort(df["alpha"].unique())
    a = snap(alphas, alpha_value) if snap_alpha else float(alpha_value)
    sdf = df[df["alpha"] == a]

    if sdf.empty:
        raise ValueError(f"No rows at alpha={a}")

    for col in ("omega", "mu", "loglike"):
        if col not in sdf.columns:
            raise ValueError(f"CSV missing required column {col!r}.")

    xs = np.sort(sdf["omega"].unique())
    ys = np.sort(sdf["mu"].unique())

    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)
    xi = {x: j for j, x in enumerate(xs)}
    yi = {y: i for i, y in enumerate(ys)}

    for _, r in sdf.iterrows():
        Z[yi[r["mu"]], xi[r["omega"]]] = float(r["loglike"])

    finite = np.isfinite(Z)
    if not np.any(finite):
        raise ValueError(f"All loglike non-finite at alpha={a}")

    llmax = float(Z[finite].max())
    i, j = np.argwhere(Z == llmax)[0]

    return dict(
        alpha=a,
        xs=xs,
        ys=ys,
        Z=Z,
        llmax=llmax,
        mle_mu=float(ys[i]),
        mle_omega=float(xs[j]),
    )


# ============================================================
# plotting
# ============================================================

def plot_slice(
    df: pd.DataFrame,
    alpha_value: float,
    *,
    ll_window: float,
    contour_deltas,
    mark_mle: bool = True,
    truth=None,
    show_truth: bool = False,
    snap_alpha: bool = True,
    save: bool = True,
    prefix: str = "",
):
    S = build_slice(df, alpha_value, snap_alpha=snap_alpha)

    vmin = S["llmax"] - ll_window
    Zp = finite_safe(S["Z"], vmin - 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        Zp,
        origin="lower",
        aspect="auto",
        extent=[S["xs"].min(), S["xs"].max(), S["ys"].min(), S["ys"].max()],
        vmin=vmin,
        vmax=S["llmax"],
    )

    levels = np.sort(S["llmax"] - np.asarray(contour_deltas, dtype=float))
    ax.contour(S["xs"], S["ys"], Zp, levels=levels, colors="white", linewidths=1.2)

    ax.set_xlabel("omega")
    ax.set_ylabel("mu")
    ax.set_title(f"loglike slice at alpha={S['alpha']:.4g}")

    if mark_mle:
        ax.scatter(
            S["mle_omega"], S["mle_mu"],
            s=60, c="gold", edgecolor="black",
            linewidth=0.6, label="slice MLE"
        )

    if show_truth and truth is not None and ("omega" in truth and "mu" in truth):
        ax.scatter(
            truth["omega"], truth["mu"],
            c="red", marker="x", s=80, label="ground truth"
        )

    if mark_mle or (show_truth and truth is not None):
        ax.legend()

    fig.colorbar(im, ax=ax, label="loglike")
    fig.tight_layout()

    if save:
        ensure_outdir()
        stem = f"{prefix}slice_alpha_{S['alpha']:.4g}" if prefix else f"slice_alpha_{S['alpha']:.4g}"
        fig.savefig(f"{OUTDIR}/{stem}.png", dpi=300)
        fig.savefig(f"{OUTDIR}/{stem}.svg")

    plt.close(fig)


def plot_all_slices(
    df: pd.DataFrame,
    *,
    ll_window: float,
    contour_deltas,
    ncols: int,
    mark_mle: bool,
    truth,
    show_truth: bool,
    snap_alpha: bool,
    prefix: str = "",
):
    alphas = np.sort(df["alpha"].unique())
    n = len(alphas)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    for k, a in enumerate(alphas):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        S = build_slice(df, float(a), snap_alpha=snap_alpha)

        vmin = S["llmax"] - ll_window
        Zp = finite_safe(S["Z"], vmin - 1)

        im = ax.imshow(
            Zp,
            origin="lower",
            aspect="auto",
            extent=[S["xs"].min(), S["xs"].max(), S["ys"].min(), S["ys"].max()],
            vmin=vmin,
            vmax=S["llmax"],
        )

        levels = np.sort(S["llmax"] - np.asarray(contour_deltas, dtype=float))
        ax.contour(S["xs"], S["ys"], Zp, levels=levels, colors="white", linewidths=1.2)

        if mark_mle:
            ax.scatter(S["mle_omega"], S["mle_mu"], c="gold", s=35, edgecolor="black", linewidth=0.5)

        # show truth only on matching alpha slice (if truth alpha provided)
        if show_truth and truth is not None and ("omega" in truth and "mu" in truth):
            if "alpha" not in truth:
                ax.scatter(truth["omega"], truth["mu"], c="red", marker="x", s=50)
            else:
                a_truth = snap(alphas, truth["alpha"]) if snap_alpha else float(truth["alpha"])
                if np.isclose(S["alpha"], a_truth):
                    ax.scatter(truth["omega"], truth["mu"], c="red", marker="x", s=50)

        ax.set_title(f"α={S['alpha']:.3g}")
        ax.set_xlabel("omega")
        ax.set_ylabel("mu")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.tight_layout()
    ensure_outdir()
    stem = f"{prefix}slices_all" if prefix else "slices_all"
    fig.savefig(f"{OUTDIR}/{stem}.png", dpi=300)
    fig.savefig(f"{OUTDIR}/{stem}.svg")
    plt.close(fig)


# ============================================================
# main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="Single CSV to plot.")
    p.add_argument("--dir", default=None, help="Directory of CSVs to plot (runs all matching --pattern).")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern inside --dir (default: *.csv)")
    p.add_argument("--ll-window", type=float, default=20.0)
    p.add_argument("--contours", default="0.5,1,2,4")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--no-mle", action="store_true")
    p.add_argument("--truth", default=None)
    p.add_argument("--show-truth", action="store_true")
    p.add_argument("--no-snap-alpha", action="store_true")
    args = p.parse_args()

    # Decide which CSV(s) to run
    csv_paths = []
    if args.dir is not None:
        d = Path(args.dir)
        csv_paths = sorted(str(pth) for pth in d.glob(args.pattern))
    elif args.csv is not None:
        csv_paths = [args.csv]
    else:
        raise ValueError("Provide either --csv <file.csv> or --dir <folder>")

    if len(csv_paths) == 0:
        raise ValueError("No CSV files matched. Check --dir/--pattern.")

    deltas = [float(x.strip()) for x in args.contours.split(",") if x.strip() != ""]
    truth = parse_truth(args.truth)
    ensure_outdir()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        prefix = Path(csv_path).stem + "__"  # prevents overwrites in a fixed OUTDIR

        if args.alpha is not None:
            plot_slice(
                df, args.alpha,
                ll_window=args.ll_window,
                contour_deltas=deltas,
                mark_mle=not args.no_mle,
                truth=truth,
                show_truth=args.show_truth,
                snap_alpha=not args.no_snap_alpha,
                prefix=prefix,
            )
        else:
            plot_all_slices(
                df,
                ll_window=args.ll_window,
                contour_deltas=deltas,
                ncols=args.ncols,
                mark_mle=not args.no_mle,
                truth=truth,
                show_truth=args.show_truth,
                snap_alpha=not args.no_snap_alpha,
                prefix=prefix,
            )


if __name__ == "__main__":
    main()
