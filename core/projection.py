import umap
import numpy as np

def reduce_umap(df, n_neighbors=15, min_dist=0.1, metric="cosine"):
    """
    Reduce embedding dimensionality with UMAP.
    Returns the UMAP reducer and transformed DataFrame.
    """
    embeddings = np.vstack(df["embedding"].values)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    proj = reducer.fit_transform(embeddings)
    df["x"], df["y"] = proj[:, 0], proj[:, 1]
    return df, reducer


def centre_on_centroid(df, centroid, reducer):
    """
    Project the semantic centroid through the same UMAP reducer
    and recenter all points so the centroid is at (0, 0).
    """
    centroid_proj = reducer.transform([centroid])[0]
    df["x_centered"] = df["x"] - centroid_proj[0]
    df["y_centered"] = df["y"] - centroid_proj[1]
    return df, (0.0, 0.0)

