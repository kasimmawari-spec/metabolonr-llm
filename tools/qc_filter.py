import pandas as pd

def qc_filter(df: pd.DataFrame, missing_threshold: float = 0.2) -> dict:
    """
    Removes low-quality metabolites:
    - Those with missingness above the threshold
    - Those with zero variance
    Only processes numeric columns.
    """
    # Keep only numeric columns
    df = df.select_dtypes(include='number')
    original_shape = df.shape

    # Drop metabolites with too many missing values
    missing_rate = df.isnull().mean()
    df = df.loc[:, missing_rate <= missing_threshold]

    # Drop zero-variance metabolites
    zero_var = df.columns[df.var() == 0]
    df = df.drop(columns=zero_var)

    removed = original_shape[1] - df.shape[1]

    print(f"QC filter: removed {removed} metabolites, {df.shape[1]} remaining.")
    return {
        "dataframe": df,
        "original_n_metabolites": original_shape[1],
        "remaining_n_metabolites": df.shape[1],
        "removed_n_metabolites": removed
    }