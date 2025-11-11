import pandas as pd
import numpy as np
import ast
from utils.normalise import normalise_url
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
    df = pd.read_csv(path)
    df['Page'] = df['Page'].apply(normalise_url)
    return df

def merge_data(sf_df, gsc_df):
    df = sf_df.merge(gsc_df, left_on='Address', right_on='Page', how='left')
    df['Clicks'] = df['Clicks'].fillna(0)
    df['CTR'] = df['CTR'].fillna(0)
    df['Impressions'] = df['Impressions'].fillna(0)
    return df

def parse_embedding(value):
    try:
        if isinstance(value, str):
            return np.array(ast.literal_eval(value))
    except Exception:
        return np.nan
    return np.nan
