import numpy as np
import pandas as pd

def compute_radial_layout(df, seed=42):
    """
    Convert semantic distance data into polar coordinates.
    r = semantic distance from centroid
    θ = evenly spaced angle or hash-based angle
    """
    np.random.seed(seed)
    
    # Normalise radius (semantic distance)
    df["r"] = df["distance_from_centre"] / df["distance_from_centre"].max()
    
    # Spread angles evenly or pseudo-randomly
    df = df.sort_values("r", ascending=True).reset_index(drop=True)
    df["theta"] = np.linspace(0, 2 * np.pi, len(df), endpoint=False)

    # Convert polar → cartesian (for plotting)
    df["x_radial"] = df["r"] * np.cos(df["theta"])
    df["y_radial"] = df["r"] * np.sin(df["theta"])

    return df
