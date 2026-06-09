import pandas as pd

def pathway_variation(df: pd.DataFrame, pathway_map: dict) -> dict:
    """
    Groups metabolites by biological pathway and computes
    the mean variance for each pathway.
    Higher variance = that pathway is more variable across samples.
    
    pathway_map: dict like {"amino acids": ["alanine", "glutamine", "serine"],
                             "energy": ["glucose", "lactate"]}
    """
    results = {}

    for pathway, metabolites in pathway_map.items():
        # Only use metabolites that exist in the dataframe
        available = [m for m in metabolites if m in df.columns]
        
        if not available:
            print(f"  {pathway}: no metabolites found in data, skipping.")
            continue

        pathway_variance = df[available].var().mean()
        results[pathway] = round(float(pathway_variance), 4)
        print(f"  {pathway}: mean variance = {results[pathway]}")

    print("Pathway variation complete.")
    return {"variance_by_pathway": results}