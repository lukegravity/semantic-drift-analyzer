import numpy as np
import pandas as pd

def get_kpis(df):
    cohesion = df['similarity'].mean()
    focus_drift = (df['distance'] <= df['distance'].quantile(0.5)).sum() / len(df)
    avg_ndi = df['NDI'].mean()
    return {
        "Topical Cohesion": round(cohesion, 3),
        "Focus-Drift Ratio": round(focus_drift, 3),
        "Average NDI": round(avg_ndi, 3)
    }
