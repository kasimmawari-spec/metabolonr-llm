import pandas as pd
import numpy as np
from scipy import stats

def differential_abundance(df: pd.DataFrame, metadata_df: pd.DataFrame,
                           group_column: str, p_adjust_method: str = "bh") -> dict:
    """
    Tests each metabolite for significant differences between two groups.
    """
    # Re-index metadata to numeric-style sample IDs if it still carries mb_sample_id as index
    if 'Sample name' in metadata_df.columns:
        metadata_df = metadata_df.set_index('Sample name')

    df.index = df.index.astype(str)
    metadata_df.index = metadata_df.index.astype(str)

    # Align indices between metabolomics and metadata
    common_samples = df.index.intersection(metadata_df.index)
    if len(common_samples) == 0:
        raise ValueError("No common samples found between metabolomics and metadata. Check that sample IDs match.")

    df = df.loc[common_samples]
    metadata_df = metadata_df.loc[common_samples]

    groups = metadata_df[group_column].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups, found {len(groups)}: {groups}")

    group_a = groups[0]
    group_b = groups[1]

    samples_a = metadata_df[metadata_df[group_column] == group_a].index
    samples_b = metadata_df[metadata_df[group_column] == group_b].index

    results = []

    for metabolite in df.columns:
        vals_a = df.loc[samples_a, metabolite].dropna()
        vals_b = df.loc[samples_b, metabolite].dropna()

        if len(vals_a) < 3 or len(vals_b) < 3:
            continue

        t_stat, p_val = stats.ttest_ind(vals_a, vals_b)
        fold_change = vals_b.mean() - vals_a.mean()

        results.append({
            "metabolite": metabolite,
            "fold_change": round(fold_change, 4),
            "p_value": round(p_val, 4)
        })

    results_df = pd.DataFrame(results)

    if p_adjust_method == "bonferroni":
        results_df["p_adjusted"] = (results_df["p_value"] * len(results_df)).clip(upper=1.0).round(4)
    elif p_adjust_method == "bh":
        n = len(results_df)
        sorted_df = results_df.sort_values("p_value").reset_index(drop=True)
        sorted_df["p_adjusted"] = (sorted_df["p_value"] * n / (sorted_df.index + 1)).clip(upper=1.0).round(4)
        results_df = sorted_df.sort_values("metabolite").reset_index(drop=True)
    else:
        results_df["p_adjusted"] = results_df["p_value"]

    significant = results_df[results_df["p_adjusted"] < 0.05]
    print(f"Differential abundance complete: {len(significant)} significant metabolites (p_adj < 0.05).")

    return {
        "results": results_df,
        "significant": significant,
        "group_a": str(group_a),
        "group_b": str(group_b)
    }