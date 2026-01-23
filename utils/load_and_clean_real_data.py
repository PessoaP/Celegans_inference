import pandas as pd
import numpy as np
import os
import glob
from typing import List, Iterable, Optional, Union

def load_and_clean_real_data(
    filepath: str,
    cutoff: int = 300,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    days: Optional[Union[int, Iterable[int]]] = None,
    add_source: bool = False,
):
    df = pd.read_csv(filepath)

    # --- Filter to requested days early (before CFU selection) ---
    if days is not None:
        if isinstance(days, int):
            days_set = {days}
        else:
            days_set = set(days)

        # robust cast in case Day is float/string in CSV
        day_series = pd.to_numeric(df["Day"], errors="coerce").astype("Int64")
        df = df.loc[day_series.isin(days_set)].copy()

        if verbose:
            print(f"[INFO] Day filter {sorted(days_set)} -> {len(df)} rows from {filepath}")

    # Define possible CFU columns and their corresponding dilution factors
    dilution_map = {
        "CFU_2": 2.2,
        "CFU_22": 22,
        "CFU_222": 222,
        "CFU_2222": 2222,
    }

    available_cols = [col for col in dilution_map if col in df.columns]
    available_dils = [dilution_map[col] for col in available_cols]

    if verbose:
        missing = set(dilution_map.keys()) - set(available_cols)
        if missing:
            print(f"[WARN] Missing columns skipped: {', '.join(sorted(missing))}")
        print(f"[INFO] Using dilutions: {available_dils}")

    # Clean and parse CFU columns
    cfu_array = df[available_cols].copy()
    for col in available_cols:
        cfu_array[col] = cfu_array[col].astype(str).str.replace('+', '', regex=False)
        cfu_array[col] = pd.to_numeric(cfu_array[col], errors='coerce')

    cfu_array = cfu_array.to_numpy()
    valid = ~np.isnan(cfu_array)
    under_cutoff = (cfu_array <= cutoff) & valid

    # Select best CFU per row
    selected_idx = []
    for i in range(cfu_array.shape[0]):
        if np.any(under_cutoff[i]):
            idx = np.argmax(under_cutoff[i])  # first under cutoff, least diluted
        elif np.any(valid[i]):
            idx = np.argmax(valid[i])  # fallback: first valid entry
        else:
            selected_idx.append(None)
            continue
        selected_idx.append(idx)

    selected_cts, selected_dils = [], []
    for i, idx in enumerate(selected_idx):
        if idx is not None:
            selected_cts.append(cfu_array[i, idx])
            selected_dils.append(available_dils[idx])
        else:
            selected_cts.append(np.nan)
            selected_dils.append(np.nan)

    # Build result
    df_result = pd.DataFrame({
        "Day": pd.to_numeric(df["Day"], errors="coerce"),
        "Counts": selected_cts,
        "Dilution": selected_dils
    })

    if add_source:
        df_result["SourceFile"] = os.path.basename(filepath)

    total = len(df_result)
    dropped = df_result["Counts"].isna().sum()
    kept = total - dropped
    df_result.dropna(subset=["Counts", "Dilution"], inplace=True)

    df_result["Counts"] = df_result["Counts"].astype(np.int64)
    df_result["Dilution"] = df_result["Dilution"].astype(np.float32)
    df_result["Day"] = df_result["Day"].astype(np.int64)

    if verbose:
        print(f"[INFO] Loaded {total} rows from {filepath}")
        print(f"[INFO] Retained {kept} rows with valid CFU entries.")
        print(f"[INFO] Dropped {dropped} rows with no valid CFU values at any dilution.")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, os.path.basename(filepath).replace(".csv", "_CLEANED.csv"))
        df_result.to_csv(outpath, index=False)
        if verbose:
            print(f"[✓] Cleaned data saved to: {outpath}")

    return df_result


def load_and_pool_real_data(
    filepaths: Optional[List[str]] = None,
    glob_pattern: Optional[str] = None,
    days: Optional[Union[int, Iterable[int]]] = None,
    cutoff: int = 300,
    verbose: bool = True,
):
    if (filepaths is None) == (glob_pattern is None):
        raise ValueError("Provide exactly one of filepaths or glob_pattern.")

    if glob_pattern is not None:
        filepaths = sorted(glob.glob(glob_pattern))
        if len(filepaths) == 0:
            raise FileNotFoundError(f"No files matched glob_pattern={glob_pattern}")

    dfs = []
    for fp in filepaths:
        dfi = load_and_clean_real_data(
            fp, cutoff=cutoff, verbose=verbose, save_dir=None, days=days, add_source=True
        )
        if len(dfi) > 0:
            dfs.append(dfi)

    if len(dfs) == 0:
        return pd.DataFrame(columns=["Day", "Counts", "Dilution", "SourceFile"])

    out = pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"[INFO] Pooled rows: {len(out)} from {len(dfs)} files")
        if days is not None:
            print(f"[INFO] Days pooled: {sorted({int(x) for x in out['Day'].unique()})}")
    return out
