#!/usr/bin/env python3
"""
visualize_3D_highlikelihood_merged.py

Merge multiple grid-search CSVs into ONE 3D high-likelihood point cloud and
color-code points by relative likelihood.

Inputs (same style as your 2D script):
  --csv <file.csv>          OR
  --dir <folder> [--pattern "*.csv"]

Expected columns:
  alpha, mu, omega, loglike

Output:
  figures/grid_3d/
    merged__3D_highlike_delta_<Δ>.png
    merged__3D_highlike_delta_<Δ>.svg

Color:
  color value = clip(loglike - global_max, -delta, 0)
  (so 0 is best, -delta is the threshold boundary)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTDIR = "figures/grid_3d"


def ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def load_one(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    need = {"alpha", "mu", "omega", "loglike"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}. Has {list(df.columns)}")
    df = df[["alpha", "mu", "omega", "loglike"]].copy()
    df["d"] = df["alpha"] * df["omega"]
    df["source"] = Path(csv_path).stem
    return df


def gather_csvs(csv_paths: List[str]) -> pd.DataFrame:
    parts = [load_one(p) for p in csv_paths]
    return pd.concat(parts, ignore_index=True)


def merged_highlike_plot(
    df: pd.DataFrame,
    *,
    delta: float,
    out_stem: str,
    max_points: int | None,
    elev: float,
    azim: float,
    s: float,
    alpha_scatter: float,
    per_file_max: bool,
) -> Tuple[int, int]:
    """
    Returns (kept_points, total_finite_points)
    """
    LL = df["loglike"].to_numpy(dtype=float)
    finite = np.isfinite(LL)
    if not np.any(finite):
        raise ValueError("All loglike values are non-finite; cannot plot.")

    # Decide thresholding reference
    if per_file_max:
        # Compute max within each source, then keep points >= max(source) - delta
        df2 = df.loc[finite].copy()
        df2["llmax_src"] = df2.groupby("source")["loglike"].transform("max")
        keep = df2["loglike"] >= (df2["llmax_src"] - float(delta))

        sub = df2.loc[keep].copy()
        # For coloring: relative to each source max, clipped
        rel = np.clip(sub["loglike"].to_numpy() - sub["llmax_src"].to_numpy(), -float(delta), 0.0)
        title_max = "per-file max"
        total_finite = len(df2)

    else:
        # Global max across all sources
        LLmax = float(np.max(LL[finite]))
        keep = finite & (LL >= LLmax - float(delta))
        sub = df.loc[keep].copy()
        rel = np.clip(sub["loglike"].to_numpy() - LLmax, -float(delta), 0.0)
        title_max = "global max"
        total_finite = int(np.sum(finite))

    kept = len(sub)
    if kept == 0:
        raise ValueError(f"No points satisfy the threshold. Try larger --delta.")

    # Optional downsample AFTER thresholding (keeps color semantics)
    if max_points is not None and kept > max_points:
        sub = sub.sample(n=max_points, random_state=0)
        # recompute rel for sampled rows
        if per_file_max:
            rel = np.clip(sub["loglike"].to_numpy() - sub["llmax_src"].to_numpy(), -float(delta), 0.0)
        else:
            # we can recover LLmax from rel+loglike, but easiest: recompute from full finite set
            LLmax = float(np.max(LL[finite]))
            rel = np.clip(sub["loglike"].to_numpy() - LLmax, -float(delta), 0.0)
        kept = len(sub)

    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        sub["alpha"].to_numpy(),
        sub["mu"].to_numpy(),
        sub["d"].to_numpy(),       # <-- derived quantity from mu*omega
        c=rel,
        s=s,
        alpha=alpha_scatter,
        cmap="viridis",
        linewidths=0.0,
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("mu")
    ax.set_zlabel("d")
    ax.view_init(elev=elev, azim=azim)

    ax.set_title(
        f"Merged high-likelihood cloud ({title_max})\n"
        f"kept {kept} of {total_finite} finite points; "
        f"colored by loglike - max clipped to [-{delta:g}, 0]"
    )

    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label("relative loglike (loglike - max)")

    fig.tight_layout()

    ensure_outdir()
    fig.savefig(f"{OUTDIR}/{out_stem}.png", dpi=300)
    fig.savefig(f"{OUTDIR}/{out_stem}.svg")
    plt.close(fig)

    return kept, total_finite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="Single CSV to plot.")
    p.add_argument("--dir", default=None, help="Directory of CSVs to plot (merged into one plot).")
    p.add_argument("--pattern", default="*.csv", help="Glob pattern inside --dir (default: *.csv)")

    p.add_argument("--delta", type=float, default=10.0,
                   help="Keep points with loglike >= max - delta (default: 10)")

    p.add_argument("--per-file-max", action="store_true",
                   help="Threshold and color relative to each file's own max, then merge clouds.")

    p.add_argument("--max-points", type=int, default=None,
                   help="Optional cap; downsample kept points to this many.")

    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--s", type=float, default=8.0)
    p.add_argument("--alpha", dest="alpha_scatter", type=float, default=0.6,
                   help="Marker transparency (default: 0.6)")
    args = p.parse_args()

    # Decide which CSV(s) to run
    csv_paths: List[str] = []
    if args.dir is not None:
        d = Path(args.dir)
        csv_paths = sorted(str(pth) for pth in d.glob(args.pattern))
    elif args.csv is not None:
        csv_paths = [args.csv]
    else:
        raise ValueError("Provide either --csv <file.csv> or --dir <folder>")

    if len(csv_paths) == 0:
        raise ValueError("No CSV files matched. Check --dir/--pattern.")

    df = gather_csvs(csv_paths)

    out_stem = f"merged__3D_highlike_delta_{args.delta:g}"
    if args.per_file_max:
        out_stem += "__per_file_max"

    merged_highlike_plot(
        df,
        delta=args.delta,
        out_stem=out_stem,
        max_points=args.max_points,
        elev=args.elev,
        azim=args.azim,
        s=args.s,
        alpha_scatter=args.alpha_scatter,
        per_file_max=args.per_file_max,
    )


if __name__ == "__main__":
    main()
