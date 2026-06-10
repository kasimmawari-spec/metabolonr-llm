"""
Runs the full pipeline on data/metabolomics_data.csv and saves the
intermediate outputs as JSON fixtures in tests/fixtures/ for use by
test_pipeline.py.

Pipeline: load -> qc_filter -> impute -> log transform -> scale ->
differential_abundance (group_column=DM_5crit)
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.load_metabolomics_data import load_metabolomics_data
from tools.qc_filter import qc_filter
from tools.impute_missing import impute_missing
from tools.transform import transform
from tools.scale import scale
from tools.differential_abundance import differential_abundance

DATA_PATH = os.path.join(ROOT, "data", "metabolomics_data.csv")
METADATA_PATH = os.path.join(ROOT, "data", "sample_annotation.csv")
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
GROUP_COLUMN = "DM_5crit"


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    # 1. Load
    loaded = load_metabolomics_data(DATA_PATH)
    df = loaded["dataframe"]

    # 2. QC filter
    qc_result = qc_filter(df)
    df = qc_result["dataframe"]
    with open(os.path.join(FIXTURES_DIR, "qc_output.json"), "w") as f:
        json.dump({"kept_metabolites": list(df.columns)}, f, indent=2)

    # 3. Impute missing
    impute_result = impute_missing(df)
    df = impute_result["dataframe"]
    with open(os.path.join(FIXTURES_DIR, "impute_output.json"), "w") as f:
        json.dump({
            "missing_before": impute_result["missing_before"],
            "missing_after": impute_result["missing_after"]
        }, f, indent=2)

    # 4. Log transform
    df = transform(df, "log")["dataframe"]

    # 5. Scale
    df = scale(df)["dataframe"]

    # 6. Differential abundance
    metadata = pd.read_csv(
        METADATA_PATH,
        index_col=0,
        na_values=['-', '.', 'NA', 'N/A', '']
    )
    if 'Sample name' in metadata.columns:
        metadata = metadata.set_index('Sample name')
    metadata.index = metadata.index.astype(str)

    da_result = differential_abundance(df, metadata, GROUP_COLUMN)
    significant = da_result["significant"]
    with open(os.path.join(FIXTURES_DIR, "da_output.json"), "w") as f:
        json.dump({
            "significant": significant[["metabolite", "p_value", "fold_change"]].to_dict(orient="records")
        }, f, indent=2)

    print("Fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
