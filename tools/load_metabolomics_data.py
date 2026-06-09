import pandas as pd

def load_metabolomics_data(filepath: str) -> dict:
    """
    Loads a metabolomics CSV file into a dataframe.
    Handles the case where rows = metabolites, columns = samples
    by transposing so rows = samples, columns = metabolites.
    """
    df = pd.read_csv(filepath, index_col=0)

    # If more columns than rows, transpose (metabolites x samples -> samples x metabolites)
    if df.shape[1] > df.shape[0]:
        df = df.T

    # Drop any rows that are not numeric sample IDs (e.g. RefMet_name header row)
    df = df[df.index.map(lambda x: str(x).replace('.', '').isdigit())]

    # Force all values to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    df.index = df.index.astype(str)

    result = {
        "shape": df.shape,
        "n_samples": df.shape[0],
        "n_metabolites": df.shape[1],
        "columns": list(df.columns[:5]),
        "index": list(df.index[:5]),
        "dataframe": df
    }

    print(f"Loaded {result['n_samples']} samples and {result['n_metabolites']} metabolites.")
    return result