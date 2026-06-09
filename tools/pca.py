import pandas as pd
from sklearn.decomposition import PCA

def pca(df: pd.DataFrame, n_components: int = 2) -> dict:
    """
    Runs Principal Component Analysis on the scaled metabolomics data.
    Reduces many metabolites down to a few components for visualization.
    """
    pca_model = PCA(n_components=n_components)
    components = pca_model.fit_transform(df)

    component_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, index=df.index, columns=component_cols)

    variance_explained = pca_model.explained_variance_ratio_.round(3).tolist()

    print(f"PCA complete: {n_components} components explain {sum(variance_explained)*100:.1f}% of variance.")
    return {
        "dataframe": pca_df,
        "variance_explained": variance_explained,
        "n_components": n_components
    }