import pandas as pd
from sklearn.impute import KNNImputer

def impute_missing(df: pd.DataFrame, n_neighbors: int = 5) -> dict:
    """
    Fills missing values using KNN imputation.
    Looks at the k most similar samples to estimate missing values.
    """
    missing_before = df.isnull().sum().sum()

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, index=df.index, columns=df.columns)

    missing_after = imputed_df.isnull().sum().sum()

    print(f"Imputation complete: {missing_before} missing values filled, {missing_after} remaining.")
    return {
        "dataframe": imputed_df,
        "missing_before": int(missing_before),
        "missing_after": int(missing_after)
    }