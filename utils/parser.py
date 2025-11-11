import ast
import numpy as np

def parse_embedding(val):
    """
    Safely parse embedding text or list to np.ndarray of float64.
    Handles JSON-like strings, pipe-delimited text, and lists.
    Returns empty array for invalid entries.
    """
    if isinstance(val, (list, np.ndarray)):
        return np.array(val, dtype=np.float64)

    if isinstance(val, str):
        val = val.strip()

        # JSON-style list string, e.g. "[0.123, -0.456, 0.789]"
        if val.startswith('[') and val.endswith(']'):
            try:
                parsed = ast.literal_eval(val)
                return np.array(parsed, dtype=np.float64)
            except Exception:
                pass

        # Pipe-delimited string, e.g. "0.123| -0.456| 0.789"
        if '|' in val:
            try:
                parsed = [float(x) for x in val.split('|') if x.strip()]
                return np.array(parsed, dtype=np.float64)
            except Exception:
                pass

    # Fallback → empty vector
    return np.array([], dtype=np.float64)
