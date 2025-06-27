import pandas as pd
import numpy as np

def load_and_clean_real_data(filepath, cutoff=300, verbose=True):
    df = pd.read_csv(filepath)

    # Define possible CFU columns and their corresponding dilution factors
    dilution_map = {
        "CFU_22": 22,
        "CFU_222": 222,
        "CFU_2222": 2222,
    }

    # Filter to columns that actually exist in the input CSV
    available_cols = [col for col in dilution_map if col in df.columns]
    available_dils = [dilution_map[col] for col in available_cols]

    if verbose:
        missing = set(dilution_map.keys()) - set(available_cols)
        if missing:
            print(f"[WARN] Missing columns skipped: {', '.join(missing)}")
        print(f"[INFO] Using dilutions: {available_dils}")

    # Clean and parse CFU columns
    cfu_array = df[available_cols].copy()
    for col in available_cols:
        cfu_array[col] = cfu_array[col].astype(str).str.replace('+', '', regex=False)
        cfu_array[col] = pd.to_numeric(cfu_array[col], errors='coerce')  # NaN for blanks/invalids

    cfu_array = cfu_array.to_numpy()
    valid = ~np.isnan(cfu_array)
    under_cutoff = (cfu_array <= cutoff) & valid

    # Select best CFU per row
    selected_idx = []
    for i in range(cfu_array.shape[0]):
        if np.any(under_cutoff[i]):
            idx = np.argmax(under_cutoff[i])  # first under cutoff, least diluted
        elif np.any(valid[i]):
            idx = np.argmax(valid[i])         # fallback: first valid entry
        else:
            selected_idx.append(None)
            continue
        selected_idx.append(idx)

    # Build result
    selected_cts, selected_dils = [], []
    for i, idx in enumerate(selected_idx):
        if idx is not None:
            selected_cts.append(cfu_array[i, idx])
            selected_dils.append(available_dils[idx])
        else:
            selected_cts.append(np.nan)
            selected_dils.append(np.nan)

    df_result = pd.DataFrame({
        "Day": df["Day"],
        "Counts": selected_cts,
        "Dilution": selected_dils
    })

    total = len(df_result)
    dropped = df_result["Counts"].isna().sum()
    kept = total - dropped
    df_result.dropna(subset=["Counts", "Dilution"], inplace=True)

    if verbose:
        print(f"[INFO] Loaded {total} rows from {filepath}")
        print(f"[INFO] Retained {kept} rows with valid CFU entries.")
        print(f"[INFO] Dropped {dropped} rows with no valid CFU values at any dilution.")

    return df_result
