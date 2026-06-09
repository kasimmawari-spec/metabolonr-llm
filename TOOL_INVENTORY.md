# Tool Inventory — Metabolomics LLM Pipeline

This document catalogues all 11 analytical tools available in the pipeline. Each tool is a self-contained Python function in the `tools/` directory and is exposed to the language model agent as a callable tool via `agent.py`.

---

## 1. `load_metabolomics_data`

**File:** `tools/load_metabolomics_data.py`

**Description:** Entry point for the pipeline. Reads a metabolomics CSV file and prepares it as a samples × metabolites DataFrame.

| | Detail |
|---|---|
| **Input** | `filepath: str` — path to a CSV file |
| **Output** | `dataframe: DataFrame`, `n_samples: int`, `n_metabolites: int`, `shape: tuple`, `columns: list`, `index: list` |

**What it does:** Loads the CSV with the first column as the row index. If the file is oriented as metabolites × samples (more columns than rows), it transposes the matrix. Rows whose index is not a numeric sample ID are dropped. All values are coerced to numeric. The index is cast to `str` to ensure consistent alignment with metadata downstream.

---

## 2. `summarize_dataset`

**File:** `tools/summarize_dataset.py`

**Description:** Computes descriptive statistics on the loaded dataset to give an overview of data quality before processing begins.

| | Detail |
|---|---|
| **Input** | `df: DataFrame` — samples × metabolites |
| **Output** | `n_samples: int`, `n_metabolites: int`, `missing_values: int`, `missing_percent: float`, `mean_per_metabolite: dict`, `std_per_metabolite: dict` |

**What it does:** Counts total missing values and calculates the overall missingness percentage. Returns per-metabolite means and standard deviations. Only numeric columns are considered.

---

## 3. `qc_filter`

**File:** `tools/qc_filter.py`

**Description:** Removes metabolites that are too noisy or uninformative to be useful in downstream analysis.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `missing_threshold: float = 0.2` |
| **Output** | `dataframe: DataFrame`, `original_n_metabolites: int`, `remaining_n_metabolites: int`, `removed_n_metabolites: int` |

**What it does:** Drops any metabolite column where more than `missing_threshold` (default 20%) of values are missing. Also drops metabolites with zero variance across all samples, which carry no discriminatory information. Returns the filtered DataFrame alongside counts of what was removed.

---

## 4. `impute_missing`

**File:** `tools/impute_missing.py`

**Description:** Fills in remaining missing values using a machine-learning-based imputation strategy.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `n_neighbors: int = 5` |
| **Output** | `dataframe: DataFrame`, `missing_before: int`, `missing_after: int` |

**What it does:** Applies K-Nearest Neighbours (KNN) imputation via scikit-learn. For each missing entry, the algorithm identifies the `n_neighbors` most similar samples based on non-missing values and estimates the missing value from their mean. Preserves the original sample and metabolite labels.

---

## 5. `transform`

**File:** `tools/transform.py`

**Description:** Applies a mathematical transformation to reduce skewness in metabolite intensity distributions.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `method: str = "log"` — `"log"` or `"sqrt"` |
| **Output** | `dataframe: DataFrame`, `method: str` |

**What it does:** Log transformation (`log(x + 1)`) is the default and mirrors the MetabolonR standard workflow. Square-root transformation is available as an alternative for data that is less skewed. Raises a `ValueError` for unrecognised methods.

---

## 6. `scale`

**File:** `tools/scale.py`

**Description:** Standardises each metabolite so that all features are on a comparable scale before multivariate analysis.

| | Detail |
|---|---|
| **Input** | `df: DataFrame` |
| **Output** | `dataframe: DataFrame` |

**What it does:** Applies z-score (standard) scaling column-wise using scikit-learn's `StandardScaler`, giving each metabolite a mean of 0 and a standard deviation of 1. This prevents high-abundance metabolites from dominating distance-based methods such as PCA and KNN.

---

## 7. `pca`

**File:** `tools/pca.py`

**Description:** Reduces the high-dimensional metabolomics matrix to a small number of principal components for visualisation and exploratory analysis.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `n_components: int = 2` |
| **Output** | `dataframe: DataFrame` (PC scores), `variance_explained: list[float]`, `n_components: int` |

**What it does:** Fits a PCA model on the scaled data and projects each sample onto the requested number of components. Returns the component scores as a DataFrame (rows = samples, columns = PC1, PC2, …) alongside the proportion of total variance explained by each component.

---

## 8. `sources_of_variation`

**File:** `tools/sources_of_variation.py`

**Description:** Quantifies how much of the metabolome-wide variation is statistically attributable to each clinical or technical variable in the metadata.

| | Detail |
|---|---|
| **Input** | `metabolite_df: DataFrame`, `metadata_df: DataFrame` — numeric clinical variables as columns, samples as rows |
| **Output** | `variation_by_variable: dict` — variable name → mean R² |

**What it does:** For each numeric metadata variable, fits a simple linear regression of that variable against each metabolite individually and records the R² score. The mean R² across all metabolites is reported as a proxy for how strongly that variable structures the metabolome. Only numeric metadata columns are used; samples must be present in both DataFrames.

---

## 9. `differential_abundance`

**File:** `tools/differential_abundance.py`

**Description:** Identifies metabolites that are significantly different in concentration between two biological groups.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `metadata_df: DataFrame`, `group_column: str`, `p_adjust_method: str = "bh"` — `"bh"`, `"bonferroni"`, or `"none"` |
| **Output** | `results: DataFrame` (all metabolites), `significant: DataFrame` (p_adj < 0.05), `group_a: str`, `group_b: str` |

**What it does:** Aligns sample indices between the metabolomics data and metadata (handling both `mb_sample_id`-style and numeric-style indices via `set_index('Sample name')` and `astype(str)`). For each metabolite, runs an independent two-sample t-test between the two groups and computes the mean difference (fold change). Multiple testing correction is applied: Benjamini–Hochberg (BH) by default, with Bonferroni as an alternative. Metabolites with an adjusted p-value below 0.05 are returned separately as significant hits.

---

## 10. `pathway_variation`

**File:** `tools/pathway_variation.py`

**Description:** Summarises metabolite variability at the biological pathway level rather than the individual metabolite level.

| | Detail |
|---|---|
| **Input** | `df: DataFrame`, `pathway_map: dict` — e.g. `{"amino acids": ["alanine", "glutamine"], "energy": ["glucose"]}` |
| **Output** | `variance_by_pathway: dict` — pathway name → mean variance |

**What it does:** For each pathway entry in the user-supplied map, subsets the metabolomics DataFrame to the listed metabolites that are actually present, computes the per-metabolite variance, and returns the mean variance across the pathway. Pathways for which none of the listed metabolites exist in the data are silently skipped.

---

## 11. `export_session`

**File:** `tools/export_session.py`

**Description:** Persists the complete analysis session to disk as a structured JSON file for reproducibility and audit purposes.

| | Detail |
|---|---|
| **Input** | `session_log: list` — list of step dicts; `output_dir: str = "logs"` |
| **Output** | `filepath: str`, `n_steps: int` |

**What it does:** Writes a timestamped JSON file to the `logs/` directory (creating it if absent). Each entry in the session log records the tool name, the parameters passed, and a summary of the result. The output file can be used to replay the exact sequence of analysis steps without re-running the language model agent.

---

## Pipeline Order

The tools are typically invoked in the following sequence:

```
load_metabolomics_data
    → summarize_dataset
    → qc_filter
    → impute_missing
    → transform
    → scale
    → pca
    → sources_of_variation
    → differential_abundance
    → pathway_variation
    → export_session
```

All tools accept and return a `dataframe` key so that outputs chain directly into the next step.
