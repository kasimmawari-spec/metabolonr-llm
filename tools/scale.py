import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale(df: pd.DataFrame) -> dict:
    """
    Scales each metabolite to mean=0 and std=1 (z-score scaling).
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)

    print("Scaling complete: all metabolites scaled to mean=0, std=1.")
    return {
        "dataframe": scaled_df
    }