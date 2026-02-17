#!/usr/bin/env python3
"""
visualize_contours_marginals.py

Given ONE CSV containing a full Cartesian grid of parameter values and a log-likelihood column,
compute and visualize:

  1) 1D marginal log-likelihood curves (collapsed over other parameters by log-sum-exp)

  2) 2D marginal log-likelihood heatmaps for each parameter pair

PLUS (topology-forward additions):

  A) Multi-threshold superlevel contours on each 2D marginal heatmap:
       Draw contour lines at levels zmax - delta for delta in --contour-deltas
       (only if those levels fall within the displayed [zmax-ll_window, zmax] window).

  C) Component counting vs threshold on the FULL grid surface (raw loglike):
       For each delta in --topo-deltas, compute number of connected components of
       S(delta) = {theta : loglike(theta) >= max_loglike - delta}
       using grid adjacency (±1 step in one coordinate at a time).

Outputs (by default):
  figures/marginals/
    <stem>__marginals_1D.png/.svg
    <stem>__marginals_2D.png/.svg
    <stem>__marginal_1D__<var>.png/.svg
    <stem>__marginal_2D__<var1>_<var2>.png/.svg
    <stem>__components_vs_delta.png/.svg          (topology C)
    <stem>__components_vs_delta.csv              (topology C)

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
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd

import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# pretty parameter labels
# -----------------------------
VAR_LABELS = {
    "alpha": r"$\alpha$",
    "mu":    r"$\mu$",
    "omega": r"$\omega$",
}

def lab(v: str) -> str:
    """Return mathtext label for known params; otherwise return the raw name."""
    return VAR_LABELS.get(v, v)


# -----------------------------
# numerics: stable logsumexp
# -----------------------------
def logsumexp_np(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    if not np.isfinite(m):
        # all -inf (or all nan/inf) -> return -np.inf
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


def parse_float_list(s: str, *, name: str) -> List[float]:
    """
    Parse comma-separated floats (e.g. "0.5,1,2,5,10,20") into a sorted list.
    """
    s = (s or "").strip()
    if not s:
        return []
    try:
        vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    except ValueError as e:
        raise ValueError(f"Could not parse --{name}={s!r} as comma-separated floats") from e
    # keep order as provided? better: sort increasing
    return sorted(vals)


def fkey(x: float, ndigits: int = 12) -> float:
    """Quantize float for stable dict-keying from CSV grids."""
    return float(np.round(float(x), ndigits))


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
    xi = {fkey(x): j for j, x in enumerate(xs)}
    yi = {fkey(y): i for i, y in enumerate(ys)}

    for _, r in m2.iterrows():
        Z[yi[fkey(r[yvar])], xi[fkey(r[xvar])]] = float(r[valcol])
    return xs, ys, Z


# -----------------------------
# topology (C): components vs delta on full grid
# -----------------------------
class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}
        for x in items:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: int) -> int:
        # path compression
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def n_components(self) -> int:
        roots = {self.find(x) for x in self.parent.keys()}
        return len(roots)


def build_index_maps(df: pd.DataFrame, vars_all: List[str]) -> Tuple[List[np.ndarray], List[Dict[float, int]]]:
    grids = []
    maps = []
    for v in vars_all:
        vals = sorted_unique(df[v].to_numpy(dtype=float))
        grids.append(vals)
        maps.append({fkey(val): i for i, val in enumerate(vals)})
    return grids, maps


def strides_from_shape(shape: List[int]) -> List[int]:
    strides = [1] * len(shape)
    prod = 1
    for d in range(len(shape) - 1, -1, -1):
        strides[d] = prod
        prod *= shape[d]
    return strides


def coords_from_linear(lin: int, shape: List[int], strides: List[int]) -> List[int]:
    coords = [0] * len(shape)
    rem = lin
    for d in range(len(shape)):
        s = strides[d]
        coords[d] = rem // s
        rem = rem % s
    return coords


def linear_from_coords(coords: List[int], strides: List[int]) -> int:
    return int(sum(c * s for c, s in zip(coords, strides)))


def compute_components_vs_delta(
    df: pd.DataFrame,
    vars_all: List[str],
    llcol: str,
    deltas: List[float],
) -> pd.DataFrame:
    """
    Count connected components in S(delta) = {theta : loglike >= max - delta}
    using ±1 adjacency along each grid dimension.
    """
    if not deltas:
        return pd.DataFrame(columns=["delta", "threshold", "n_points", "n_components"])

    grids, maps = build_index_maps(df, vars_all)
    shape = [len(g) for g in grids]
    strides = strides_from_shape(shape)

    ll = df[llcol].to_numpy(dtype=float)
    ll_max = max_finite(ll)
    if not np.isfinite(ll_max):
        # all non-finite: nothing to do
        return pd.DataFrame(
            [{"delta": d, "threshold": np.nan, "n_points": 0, "n_components": 0} for d in deltas]
        )

    # Precompute linear indices for every row (cartesian grid assumption)
    lin_all = np.zeros(len(df), dtype=np.int64)
    for i, v in enumerate(vars_all):
        # map float value -> integer index
        idx = df[v].to_numpy(dtype=float)
        idx_int = np.array([maps[i].get(fkey(x), -1) for x in idx], dtype=np.int64)
        if np.any(idx_int < 0):
            bad = np.where(idx_int < 0)[0][:5]
            raise ValueError(
                f"Value(s) in column {v!r} not found in its own unique grid mapping. "
                f"Examples row indices: {bad.tolist()} (likely float mismatch)"
            )
        lin_all += idx_int * strides[i]

    out_rows = []
    for delta in deltas:
        thr = ll_max - float(delta)
        active_mask = np.isfinite(ll) & (ll >= thr)
        active_lin = lin_all[active_mask]
        n_points = int(active_lin.size)
        if n_points == 0:
            out_rows.append({"delta": float(delta), "threshold": float(thr), "n_points": 0, "n_components": 0})
            continue

        active_set = set(int(x) for x in active_lin.tolist())
        uf = UnionFind(active_set)

        # Union neighbors (only +1 direction per dimension to avoid double work)
        for lin in active_set:
            coords = coords_from_linear(lin, shape, strides)
            for d in range(len(shape)):
                if coords[d] + 1 < shape[d]:
                    nb = coords.copy()
                    nb[d] += 1
                    lin_nb = linear_from_coords(nb, strides)
                    if lin_nb in active_set:
                        uf.union(lin, lin_nb)

        out_rows.append(
            {
                "delta": float(delta),
                "threshold": float(thr),
                "n_points": n_points,
                "n_components": int(uf.n_components()),
            }
        )

    return pd.DataFrame(out_rows)


def plot_components_vs_delta(df_comp: pd.DataFrame, outdir: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(df_comp["delta"].to_numpy(dtype=float), df_comp["n_components"].to_numpy(dtype=float), lw=2, marker="o")
    ax.set_xlabel("delta (log units below max)")
    ax.set_ylabel("# connected components")
    ax.set_title("Topology proxy: components of superlevel set S(delta)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(f"{outdir}/{stem}__components_vs_delta.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__components_vs_delta.svg")
    plt.close(fig)


# -----------------------------
# plotting (marginals + A)
# -----------------------------
def set_top_window_1d(ax: plt.Axes, y: np.ndarray, y_window: float, y_pad: float) -> None:
    ymax = max_finite(y)
    if np.isfinite(ymax):
        ax.set_ylim(ymax - float(y_window), ymax + float(y_pad))


def overlay_superlevel_contours(
    ax: plt.Axes,
    Z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zmax: float,
    ll_window: float,
    contour_deltas: List[float],
) -> None:
    """
    Draw contour lines at levels = zmax - delta for each delta in contour_deltas,
    but only if the level is within the plotted window [zmax-ll_window, zmax].
    """
    if not contour_deltas or not np.isfinite(zmax):
        return
    vmin = zmax - float(ll_window)
    # Only use finite Z for contouring
    Zc = np.array(Z, dtype=float, copy=True)
    Zc[~np.isfinite(Zc)] = np.nan

    levels = []
    for d in contour_deltas:
        lvl = zmax - float(d)
        if (lvl >= vmin) and (lvl <= zmax):
            levels.append(lvl)
    if not levels:
        return

    # Need X/Y grids matching Z shape for contour with proper axes values.
    X, Y = np.meshgrid(xs, ys)
    ax.contour(X, Y, Zc, levels=sorted(levels), linewidths=1.0)


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

    ax.set_xlabel(lab(var))
    ax.set_ylabel("marginal log-likelihood")
    ax.set_title(f"1D marginal over {lab(var)}")
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
    contour_deltas: List[float] | None = None,
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

    # A) superlevel contours on 2D marginals
    overlay_superlevel_contours(
        ax=ax,
        Z=Z,
        xs=xs,
        ys=ys,
        zmax=zmax,
        ll_window=ll_window,
        contour_deltas=contour_deltas or [],
    )

    truth = truth or {}
    if (xvar in truth) and (yvar in truth):
        ax.plot([truth[xvar]], [truth[yvar]], marker="x", ms=9, mew=2)

    ax.set_xlabel(lab(xvar))
    ax.set_ylabel(lab(yvar))
    ax.set_title(f"2D marginal: ({lab(yvar)}, {lab(xvar)})")
    fig.colorbar(im, ax=ax, label="marginal log-likelihood")

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

        ax.set_title(f"{lab(v)}")
        ax.set_xlabel(lab(v))
        ax.set_ylabel("marginal log-likelihood")
        ax.grid(True, alpha=0.25)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle("1D marginal log-likelihoods", y=0.98)
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
    contour_deltas: List[float] | None = None,
) -> None:
    n = len(pairs)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    truth = truth or {}
    contour_deltas = contour_deltas or []

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

        # A) contours
        overlay_superlevel_contours(
            ax=ax,
            Z=Z,
            xs=xs,
            ys=ys,
            zmax=zmax,
            ll_window=ll_window,
            contour_deltas=contour_deltas,
        )

        # truth marker
        if (xvar in truth) and (yvar in truth):
            ax.plot([truth[xvar]], [truth[yvar]], marker="x", ms=8, mew=2)

        ax.set_title(f"{lab(yvar)} vs {lab(xvar)}")
        ax.set_xlabel(lab(xvar))
        ax.set_ylabel(lab(yvar))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    fig.suptitle("2D marginal log-likelihood heatmaps", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(f"{outdir}/{stem}__marginals_2D.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__marginals_2D.svg")
    plt.close(fig)

def plot_corner(
    df: pd.DataFrame,
    vars_all: List[str],
    llcol: str,
    outdir: str,
    stem: str,
    *,
    y_window: float = 20.0,
    y_pad: float = 1.0,
    ll_window: float = 20.0,
    truth: Dict[str, float] | None = None,
    contour_deltas: List[float] | None = None,
) -> None:
    """
    Corner plot:
      - diagonal: 1D marginals
      - lower triangle: 2D marginals (heatmap + optional contours)
      - upper triangle: empty
    Uses absolute log-marg units (no shifting).
    """
    truth = truth or {}
    contour_deltas = contour_deltas or []

    n = len(vars_all)
    if n == 0:
        return

    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.6 * n), squeeze=False)
    im_last = None

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            # Upper triangle empty
            if j > i:
                ax.axis("off")
                continue

            # Diagonal: 1D marginals
            if i == j:
                v = vars_all[i]
                m1 = marginal_1d(df, v, vars_all, llcol)
                x = m1[v].to_numpy(dtype=float)
                y = m1["log_marg"].to_numpy(dtype=float)

                ax.plot(x, y, lw=2)
                if v in truth and np.isfinite(truth[v]):
                    ax.axvline(truth[v], ls="--", lw=1.6)

                set_top_window_1d(ax, y, y_window=y_window, y_pad=y_pad)

                ax.set_title(lab(v))
                ax.grid(True, alpha=0.25)

                # Labels: only left column gets y-label; only bottom row gets x-label
                if j == 0:
                    ax.set_ylabel("marginal log-likelihood")
                else:
                    ax.set_ylabel("")
                if i == n - 1:
                    ax.set_xlabel(lab(v))
                else:
                    ax.set_xlabel("")
                continue

            # Lower triangle: 2D marginals
            yvar = vars_all[i]
            xvar = vars_all[j]

            m2 = marginal_2d(df, yvar, xvar, vars_all, llcol)
            xs, ys, Z = pivot_2d(m2, xvar, yvar, "log_marg")

            zmax = max_finite(Z)
            vmax = zmax
            vmin = zmax - float(ll_window) if np.isfinite(zmax) else -float(ll_window)

            Zp = finite_floor(Z, vmin - 1.0)

            im_last = ax.imshow(
                Zp,
                origin="lower",
                aspect="auto",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                vmin=vmin,
                vmax=vmax,
            )

            # Optional superlevel contours on the 2D marginals (A)
            overlay_superlevel_contours(
                ax=ax,
                Z=Z,
                xs=xs,
                ys=ys,
                zmax=zmax,
                ll_window=ll_window,
                contour_deltas=contour_deltas,
            )

            # Optional truth marker
            if (xvar in truth) and (yvar in truth):
                ax.plot([truth[xvar]], [truth[yvar]], marker="x", ms=8, mew=2)

            ax.grid(False)

            # Labels: only left column gets y-label; only bottom row gets x-label
            if j == 0:
                ax.set_ylabel(lab(yvar))
            else:
                ax.set_ylabel("")
            if i == n - 1:
                ax.set_xlabel(lab(xvar))
            else:
                ax.set_xlabel("")
                
    if im_last is not None:
        cbar = fig.colorbar(im_last, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("marginal log-likelihood")
            
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(f"{outdir}/{stem}__corner.png", dpi=300)
    fig.savefig(f"{outdir}/{stem}__corner.svg")
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

    # A) Contours
    p.add_argument(
        "--contour-deltas",
        default="1,2,5,10,20",
        help="Comma-separated deltas for superlevel contours on 2D marginals (levels = zmax - delta).",
    )
    p.add_argument(
        "--no-contours",
        action="store_true",
        help="Disable superlevel contour overlays on 2D marginals.",
    )

    # C) Topology proxy: components vs delta
    p.add_argument(
        "--topo-deltas",
        default="0.5,1,2,5,10,20",
        help="Comma-separated deltas for component counting on the full grid S(delta).",
    )
    p.add_argument(
        "--no-topology",
        action="store_true",
        help="Disable components-vs-delta computation/plotting.",
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

    contour_deltas = [] if args.no_contours else parse_float_list(args.contour_deltas, name="contour-deltas")
    topo_deltas = parse_float_list(args.topo_deltas, name="topo-deltas")

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
                    contour_deltas=contour_deltas,
                )

    plot_panel_2d(
        marg2d, pairs, args.outdir, stem,
        ll_window=args.ll_window,
        truth=truth,
        contour_deltas=contour_deltas,
    )

    # --- C) Topology proxy on FULL grid: components vs delta
    if (not args.no_topology) and topo_deltas:
        df_comp = compute_components_vs_delta(df=df, vars_all=vars_all, llcol=llcol, deltas=topo_deltas)
        df_comp.to_csv(f"{args.outdir}/{stem}__components_vs_delta.csv", index=False)
        plot_components_vs_delta(df_comp, args.outdir, stem)

    plot_corner(
        df=df,
        vars_all=vars_all,
        llcol=llcol,
        outdir=args.outdir,
        stem=stem,
        y_window=args.y_window,
        y_pad=args.y_pad,
        ll_window=args.ll_window,
        truth=truth,
        contour_deltas=contour_deltas,
    )


    print(f"[done] wrote figures to: {args.outdir}/")


if __name__ == "__main__":
    main()


# How to call:
# python visualize_contours_marginals.py \
#   --csv recovery_tests/grid_TL_seed10/grid_likelihood.csv \
#   --vars alpha,mu,omega \
#   --ll-window 20
#
# With ground truth:
# python visualize_contours_marginals.py \
#   --csv recovery_tests/grid_TL_seed10/grid_likelihood.csv \
#   --vars alpha,mu,omega \
#   --truth "alpha=0.05,mu=0.5,omega=0.24"
#
# With explicit topology deltas / contour deltas:
# python visualize_contours_marginals.py \
#   --csv recovery_tests/grid_TL_seed10/grid_likelihood.csv \
#   --vars alpha,mu,omega \
#   --topo-deltas "0.5,1,2,5,10,20" \
#   --contour-deltas "1,2,5,10,20" \
#   --truth "alpha=0.05,mu=0.5,omega=0.24"

# python visualize_contours_marginals.py \
#   --csv recovery_tests/grid_Exp1/grid_loglike_Exp1_full.csv \
#   --vars alpha,mu,omega \
#   --y-window 100 \
#   --ll-window 100 \
#   --topo-deltas "0.5,1,2,5,10,20,50,100" \
#   --contour-deltas "1,2,5,10,20,50,100"
