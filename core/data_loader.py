import pandas as pd
import numpy as np
import ast
from utils.normalise import normalise_url
import chardet
from utils.parser import parse_embedding

def load_screaming_frog(file):
    """
    Loads Screaming Frog Internal All export, cleans URLs,
    and ensures embeddings are parsed into numeric float arrays.
    """
    df = pd.read_csv(file)

    # --- Detect URL column ---
    url_col = next((c for c in df.columns if c.lower().strip() in ['address', 'url']), None)
    if not url_col:
        raise ValueError(f"No Address/URL column found. Columns: {df.columns.tolist()}")

    df['Address'] = df[url_col].apply(normalise_url)

    # --- Detect Embedding column ---
    embed_col = next((c for c in df.columns if 'embedding' in c.lower()), None)
    if not embed_col:
        raise ValueError(f"No embedding column found. Columns: {df.columns.tolist()}")

    # --- Parse embeddings into np.ndarray(float64) ---
    df['embedding'] = df[embed_col].apply(parse_embedding)
    df = df.dropna(subset=['embedding']).reset_index(drop=True)

    # --- Optional sanity check ---
    if not len(df) or not isinstance(df['embedding'].iloc[0], np.ndarray):
        raise ValueError("Embedding parsing failed – resulting arrays are invalid.")

    return df

def load_gsc(path):
    # Detect encoding first (read small sample)
    raw = path.read()
    result = chardet.detect(raw)
    encoding = result["encoding"] or "utf-8"

    from io import BytesIO
    df = pd.read_csv(BytesIO(raw), encoding=encoding)

    # --- Identify URL column ---
    if "Page" in df.columns:
        page_col = "Page"
    elif "Top pages" in df.columns:
        page_col = "Top pages"
    elif "Top Pages" in df.columns:
        page_col = "Top Pages"
    else:
        raise KeyError("Expected 'Page' or 'Top pages' column in GSC export.")

    df.rename(columns={page_col: "Page"}, inplace=True)
    df["Page"] = df["Page"].apply(normalise_url)

    # --- Clean numeric columns ---
    num_cols = ["Clicks", "Impressions", "Position"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- CTR cleanup (strip '%' and convert to float) ---
    if "CTR" in df.columns:
        df["CTR"] = (
            df["CTR"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
            .fillna(0)
        )

    return df


def merge_data(sf_df, gsc_df):
    # --- 1. Ensure the GSC URL column is standardised ---
    gsc_url_cols = [
        c for c in gsc_df.columns
        if c.strip().lower().replace(" ", "") in ["page", "toppages"]
    ]

    if not gsc_url_cols:
        raise KeyError(
            "GSC file missing URL column. Expected one of: "
            "'Page', 'Top pages', 'Top Pages'."
        )

    gsc_url_col = gsc_url_cols[0]

    if gsc_url_col != "Page":
        print(f"[merge_data] Renaming '{gsc_url_col}' → 'Page'")
        gsc_df = gsc_df.rename(columns={gsc_url_col: "Page"})

    # --- 2. URL normalisation ---
    sf_df["Address"] = sf_df["Address"].astype(str).str.strip().str.lower()
    gsc_df["Page"] = gsc_df["Page"].astype(str).str.strip().str.lower()

    # --- 3. REMOVE invalid SF rows (critical fix) ---
    before = len(sf_df)
    sf_df = sf_df[
        sf_df["Address"].notna() &
        (sf_df["Address"].str.strip() != "") &
        (sf_df["Address"].str.startswith("http"))
    ]
    after = len(sf_df)
    if after < before:
        print(f"[merge_data] Removed {before - after} invalid SF rows (empty or non-URL).")

    # --- 4. Deduplicate SF rows by Address (critical fix) ---
    if sf_df["Address"].duplicated().any():
        dupes = sf_df[sf_df["Address"].duplicated(keep=False)]
        print(f"[merge_data WARNING] {len(dupes)} duplicated SF rows detected. Deduplicating.")
        sf_df = sf_df.drop_duplicates(subset=["Address"], keep="first")

    # --- 5. Perform merge ---
    df = sf_df.merge(gsc_df, left_on="Address", right_on="Page", how="left")

    # --- 6. Diagnostics ---
    total_sf = len(sf_df)
    matched = df["Page"].notna().sum()
    print(f"[merge_data] Matched {matched}/{total_sf} URLs")

    # ------------------------------------------------------------------
    #         7. Normalise GSC metric columns (_x, _y → canonical)
    # ------------------------------------------------------------------
    metrics = ["Clicks", "Impressions", "CTR", "Position"]

    for m in metrics:
        if f"{m}_y" in df.columns:
            df[m] = df[f"{m}_y"]
        elif f"{m}_x" in df.columns:
            df[m] = df[f"{m}_x"]
        elif m not in df.columns:
            df[m] = 0  # created late, but required

        # Drop suffix variants
        for col in [f"{m}_x", f"{m}_y"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

    # --- 8. Safe fill ---
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0)

    return df




def parse_embedding(value):
    try:
        if isinstance(value, str):
            return np.array(ast.literal_eval(value))
    except Exception:
        return np.nan
    return np.nan
