import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def sources_of_variation(metabolite_df: pd.DataFrame, metadata_df: pd.DataFrame) -> dict:
    results = {}

    # Force everything to numeric, kill anything that isn't
    metabolite_df = metabolite_df.apply(pd.to_numeric, errors='coerce')
    metadata_df = metadata_df.apply(pd.to_numeric, errors='coerce')
    metadata_df = metadata_df.select_dtypes(include='number')

    numeric_cols = metadata_df.columns.tolist()

    if not numeric_cols:
        print("No numeric columns found in metadata.")
        return {"variation_by_variable": {}}

    for variable in numeric_cols:
        y = metadata_df[variable].dropna()
        y = pd.to_numeric(y, errors='coerce').dropna()

        common_samples = metabolite_df.index.intersection(y.index)
        if len(common_samples) < 5:
            continue

        y = y.loc[common_samples]
        r2_scores = []

        for metabolite in metabolite_df.columns:
            try:
                x = pd.to_numeric(metabolite_df.loc[common_samples, metabolite], errors='coerce')
                valid = x.notna() & y.notna()
                if valid.sum() < 5:
                    continue
                x_clean = x[valid].values.reshape(-1, 1)
                y_clean = y[valid].values
                model = LinearRegression()
                model.fit(x_clean, y_clean)
                r2_scores.append(model.score(x_clean, y_clean))
            except Exception:
                continue

        if r2_scores:
            mean_r2 = round(float(np.mean(r2_scores)), 4)
            results[variable] = mean_r2
            print(f"  {variable}: mean R2 = {mean_r2}")

    print("Sources of variation complete.")
    return {"variation_by_variable": results}