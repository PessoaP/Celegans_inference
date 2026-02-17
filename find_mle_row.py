#!/usr/bin/env python3
"""
find_mle_row.py

Find the (alpha, mu, omega) at the maximum loglike in a CSV.

Usage:
  python find_mle_row.py --csv path/to/grid.csv
  python find_mle_row.py --csv path/to/grid.csv --llcol loglike --vars alpha,mu,omega
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV.")
    p.add_argument("--llcol", default="loglike", help="Log-likelihood column name (default: loglike).")
    p.add_argument("--vars", default="alpha,mu,omega", help="Comma-separated parameter columns.")
    p.add_argument("--all-ties", action="store_true", help="Print all rows tied for max loglike.")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    vars_all = [v.strip() for v in args.vars.split(",") if v.strip()]
    llcol = args.llcol

    df = pd.read_csv(csv_path)

    missing = [c for c in (vars_all + [llcol]) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. CSV has: {list(df.columns)}")

    # numeric, defensive
    for c in vars_all + [llcol]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ignore non-finite loglike
    ll = df[llcol].to_numpy(dtype=float)
    finite = np.isfinite(ll)
    if not np.any(finite):
        raise ValueError(f"All values in '{llcol}' are non-finite (nan/inf/-inf).")

    df_f = df.loc[finite].copy()
    ll_f = df_f[llcol].to_numpy(dtype=float)

    ll_max = float(np.max(ll_f))

    if args.all_ties:
        best = df_f.loc[df_f[llcol] == ll_max, vars_all + [llcol]].copy()
        best = best.sort_values(vars_all).reset_index(drop=True)
        print(f"max {llcol} = {ll_max}")
        print(best.to_string(index=False))
    else:
        idx = int(np.argmax(ll_f))
        row = df_f.iloc[idx]
        print(f"max {llcol} = {ll_max}")
        for v in vars_all:
            print(f"{v} = {row[v]}")
        print(f"{llcol} = {row[llcol]}")

if __name__ == "__main__":
    main()

# python find_mle_row.py --csv recovery_tests/grid_Exp1/grid_loglike_Exp1_full.csv
