import pandas as pd
import numpy as np

def transform(df: pd.DataFrame, method: str = "log") -> dict:
    """
    Applies a transformation to normalize metabolite distributions.
    Default is log transformation (same as MetabolonR).
    """
    if method == "log":
        transformed_df = df.apply(lambda x: np.log(x + 1))
    elif method == "sqrt":
        transformed_df = df.apply(np.sqrt)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'sqrt'.")

    print(f"Transform complete: applied {method} transformation.")
    return {
        "dataframe": transformed_df,
        "method": method
    }