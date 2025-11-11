import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def compute_centroid(df, alpha=0.6, beta=0.3, gamma=0.1):
    """
    Compute the semantic centroid and Structural Drift Index (SDI).

    SDI = Normalised internal prominence × semantic distance from topical centre
    alpha/beta/gamma = weights for content, inlinks, and clicks respectively.
    """
    emb_stack = np.vstack(df["embedding"].values)

    # Weighted centroid (embeddings × prominence)
    weights = (
        alpha
        + beta * (df["Inlinks"] / df["Inlinks"].max())
        + gamma * (df.get("Clicks", 0) / max(df.get("Clicks", 0).max(), 1))
    )
    centroid = np.average(emb_stack, axis=0, weights=weights)
    centroid = centroid / np.linalg.norm(centroid)

    # Semantic distance from centre (cosine distance)
    dot = np.dot(emb_stack, centroid)
    norms = np.linalg.norm(emb_stack, axis=1)
    df["distance_from_centre"] = 1 - (dot / norms)

    # Structural Drift Index (SDI)
    df["SDI"] = (df["Inlinks"] / df["Inlinks"].max()) * df["distance_from_centre"]

    return centroid, df

def add_similarity_metrics(df, centroid):
    embeddings = np.vstack(df['embedding'].values)
    sims = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    df['similarity'] = sims
    df['distance'] = 1 - sims
    return df

def add_internal_authority(df):
    df['IA'] = scale(df['Inlinks'].fillna(0)) * scale(1/(1+df['Crawl Depth'].fillna(0)))
    return df

def add_navboost(df):
    z_ia = zscore(df['IA'])
    z_dist = zscore(df['distance'])
    df['NDI'] = z_ia * z_dist
    p25, p75 = np.percentile(df['IA'], [25, 75]), np.percentile(df['distance'], [25, 75])
    df['NavBoost Category'] = np.select([
        (df['IA'] >= p75[1]) & (df['distance'] >= p75[1]),
        (df['IA'] <= p25[0]) & (df['distance'] <= p25[0]),
        (df['IA'] <= p25[0]) & (df['distance'] >= p75[1])
    ], ["Misaligned Core", "Underlinked Core", "Junk Drift"], default="Healthy Core")
    return df

def scale(series):
    s = series.fillna(0).to_numpy().reshape(-1, 1)
    return MinMaxScaler().fit_transform(s).flatten()

def zscore(series):
    s = series.fillna(0).to_numpy().reshape(-1, 1)
    return StandardScaler().fit_transform(s).flatten()
