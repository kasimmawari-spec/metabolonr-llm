import pandas as pd

def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Returns a statistical summary of the metabolomics dataset.
    Only processes numeric columns.
    """
    # Keep only numeric columns
    numeric_df = df.select_dtypes(include='number')

    missing = numeric_df.isnull().sum().sum()
    missing_pct = (missing / (numeric_df.shape[0] * numeric_df.shape[1])) * 100

    summary = {
        "n_samples": numeric_df.shape[0],
        "n_metabolites": numeric_df.shape[1],
        "missing_values": int(missing),
        "missing_percent": round(missing_pct, 2),
        "mean_per_metabolite": numeric_df.mean().round(3).to_dict(),
        "std_per_metabolite": numeric_df.std().round(3).to_dict(),
    }

    print(f"Dataset: {summary['n_samples']} samples, {summary['n_metabolites']} metabolites")
    print(f"Missing values: {summary['missing_values']} ({summary['missing_percent']}%)")
    return summary