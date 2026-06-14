import os
from dotenv import load_dotenv
import anthropic
import pandas as pd
import json

from tools.load_metabolomics_data import load_metabolomics_data
from tools.summarize_dataset import summarize_dataset
from tools.qc_filter import qc_filter
from tools.impute_missing import impute_missing
from tools.transform import transform
from tools.scale import scale
from tools.pca import pca
from tools.export_session import export_session
from tools.sources_of_variation import sources_of_variation
from tools.pathway_variation import pathway_variation
from tools.differential_abundance import differential_abundance

# Default data paths (overridden by app.py when user uploads custom files)
DATA_PATH = "data/metabolomics_data.csv"
ANNOTATION_PATH = "data/sample_annotation.csv"

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

TOOLS = [
    {
        "name": "load_metabolomics_data",
        "description": "Loads a metabolomics CSV file. Call this first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to the CSV file"}
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "summarize_dataset",
        "description": "Returns summary statistics of the loaded dataset.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "qc_filter",
        "description": "Removes low-quality metabolites with too many missing values or zero variance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "missing_threshold": {"type": "number", "description": "Max allowed missing rate (default 0.2)"}
            }
        }
    },
    {
        "name": "impute_missing",
        "description": "Fills missing values using KNN imputation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_neighbors": {"type": "integer", "description": "Number of neighbors for KNN (default 5)"}
            }
        }
    },
    {
        "name": "transform",
        "description": "Applies log or sqrt transformation to normalize the data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "description": "'log' or 'sqrt'"}
            }
        }
    },
    {
        "name": "scale",
        "description": "Scales metabolites to mean=0 and std=1.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "pca",
        "description": "Runs PCA to reduce dimensions and visualize sample variation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_components": {"type": "integer", "description": "Number of PCA components (default 2)"}
            }
        }
    },
    {
        "name": "sources_of_variation",
        "description": "Determines which clinical variables explain the most variation in the metabolomics data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metadata_filepath": {"type": "string", "description": "Path to the metadata CSV file"}
            },
            "required": ["metadata_filepath"]
        }
    },
    {
        "name": "pathway_variation",
        "description": "Groups metabolites by biological pathway and computes variance per pathway.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metadata_filepath": {"type": "string", "description": "Path to the metadata CSV file"}
            },
            "required": ["metadata_filepath"]
        }
    },
    {
        "name": "differential_abundance",
        "description": "Tests each metabolite for significant differences between two groups.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metadata_filepath": {"type": "string", "description": "Path to the metadata CSV file"},
                "group_column": {"type": "string", "description": "Column name in metadata to group by"},
                "p_adjust_method": {"type": "string", "description": "'bh', 'bonferroni', or 'none'"}
            },
            "required": ["metadata_filepath", "group_column"]
        }
    },
    {
        "name": "export_session",
        "description": "Saves the session log to a JSON file for reproducibility.",
        "input_schema": {"type": "object", "properties": {}}
    }
]

state = {"dataframe": None}
session_log = []

def run_tool(tool_name, tool_input):
    if tool_name == "load_metabolomics_data":
        result = load_metabolomics_data(tool_input["filepath"])
        state["dataframe"] = result["dataframe"]
        summary = {"n_samples": result["n_samples"], "n_metabolites": result["n_metabolites"]}

    elif tool_name == "summarize_dataset":
        result = summarize_dataset(state["dataframe"])
        summary = {"missing_percent": result["missing_percent"]}

    elif tool_name == "qc_filter":
        threshold = tool_input.get("missing_threshold", 0.2)
        result = qc_filter(state["dataframe"], threshold)
        state["dataframe"] = result["dataframe"]
        summary = {"removed": result["removed_n_metabolites"]}

    elif tool_name == "impute_missing":
        neighbors = tool_input.get("n_neighbors", 5)
        result = impute_missing(state["dataframe"], neighbors)
        state["dataframe"] = result["dataframe"]
        summary = {"missing_after": result["missing_after"]}

    elif tool_name == "transform":
        method = tool_input.get("method", "log")
        result = transform(state["dataframe"], method)
        state["dataframe"] = result["dataframe"]
        summary = {"method": method}

    elif tool_name == "scale":
        result = scale(state["dataframe"])
        state["dataframe"] = result["dataframe"]
        summary = {}

    elif tool_name == "pca":
        n = tool_input.get("n_components", 2)
        result = pca(state["dataframe"], n)
        summary = {"variance_explained": result["variance_explained"]}

    elif tool_name == "sources_of_variation":
        metadata = pd.read_csv(
            tool_input["metadata_filepath"],
            index_col=0,
            na_values=['-', '.', 'NA', 'N/A', '']
        )
        if 'Sample name' in metadata.columns:
            metadata = metadata.set_index('Sample name')
        metadata.index = metadata.index.astype(str)
        metadata = metadata.select_dtypes(include='number')
        result = sources_of_variation(state["dataframe"], metadata)
        summary = result["variation_by_variable"]

    elif tool_name == "pathway_variation":
        pathway_map = {
            "amino acids": ["alanine", "glutamine", "serine"],
            "energy": ["glucose", "lactate"]
        }
        result = pathway_variation(state["dataframe"], pathway_map)
        summary = result["variance_by_pathway"]

    elif tool_name == "differential_abundance":
        metadata = pd.read_csv(
            tool_input["metadata_filepath"],
            index_col=0,
            na_values=['-', '.', 'NA', 'N/A', '']
        )
        if 'Sample name' in metadata.columns:
            metadata = metadata.set_index('Sample name')
        metadata.index = metadata.index.astype(str)
        group_col = tool_input["group_column"]
        p_method = tool_input.get("p_adjust_method", "bh")
        result = differential_abundance(state["dataframe"], metadata, group_col, p_method)
        summary = {
            "n_significant": len(result["significant"]),
            "top_metabolites": result["results"].head(3).to_dict(orient="records")
        }

    elif tool_name == "export_session":
        result = export_session(session_log)
        summary = {"filepath": result["filepath"]}

    else:
        summary = {"error": f"Unknown tool: {tool_name}"}

    session_log.append({"tool": tool_name, "params": tool_input, "result_summary": summary})
    return json.dumps(summary)


def run_agent(user_message: str):
    print(f"\nUser: {user_message}")
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=f"You are MetabolonR-LLM, a metabolomics analysis agent. When a user asks you to analyze data, you MUST call the available tools to perform the actual analysis - do not respond with text descriptions. A standard pipeline is: load_metabolomics_data → qc_filter → impute_missing → transform → scale → then differential_abundance or other analysis tools. Always call tools; never just describe what you would do. The metabolomics data file is at {DATA_PATH} and the sample annotation file is at {ANNOTATION_PATH}. Always use these exact paths.",
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n-> Calling tool: {block.name} with {block.input}")
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nClaude: {block.text}")
            break


if __name__ == "__main__":
    run_agent("Load data/metabolomics_data.csv, run QC, impute, log transform, scale, then run sources of variation using data/sample_annotation.csv.")