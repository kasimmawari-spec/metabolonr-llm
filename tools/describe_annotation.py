import pandas as pd


def describe_annotation(metadata_df: pd.DataFrame, max_levels: int = 12) -> dict:
    """
    Summarise the sample annotation file so the agent can choose a grouping
    variable by looking at the data instead of guessing a column name.

    Without this the model has never seen the annotation file. It invents a
    plausible-sounding column ("diabetes", "disease_status"), the lookup fails,
    and the analysis dies. Reading the columns first turns a guess into an
    inference the user can check.

    Returns one entry per column: how many distinct values it has, whether it
    is usable as a two-group comparison, and a few example values.
    """
    if "Sample name" in metadata_df.columns:
        metadata_df = metadata_df.set_index("Sample name")

    columns = []
    for col in metadata_df.columns:
        series = metadata_df[col]
        # '-' and '.' are used as missing-value markers in this annotation format
        clean = series.replace(["-", ".", "", "NA", "N/A"], pd.NA).dropna()
        levels = clean.unique()
        n_levels = len(levels)

        if n_levels == 2:
            kind = "two-group (usable for differential abundance)"
        elif 2 < n_levels <= max_levels:
            kind = f"categorical, {n_levels} levels"
        elif pd.api.types.is_numeric_dtype(pd.to_numeric(clean, errors="coerce")):
            kind = "continuous"
        else:
            kind = f"identifier or free text, {n_levels} distinct values"

        columns.append({
            "column": str(col),
            "kind": kind,
            "n_distinct": int(n_levels),
            "n_present": int(len(clean)),
            "n_missing": int(len(series) - len(clean)),
            "examples": [str(v) for v in levels[:4]],
        })

    two_group = [c["column"] for c in columns if c["n_distinct"] == 2]

    print(f"Annotation described: {len(columns)} columns, "
          f"{len(two_group)} usable as two-group comparisons: {two_group}")

    return {
        "n_samples": int(len(metadata_df)),
        "n_columns": len(columns),
        "two_group_columns": two_group,
        "columns": columns,
    }
