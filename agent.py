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

# Single source of truth for model config — app.py and run_validation.py both use these
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_TOOL_CALLS = 15

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def build_system_prompt() -> str:
    """
    Single source of truth for the system prompt.

    Both the Streamlit app and the validation harness must call this, so the
    system that gets validated is the same system that gets demonstrated.
    Reads DATA_PATH / ANNOTATION_PATH at call time so app.py can override them
    for uploaded files before invoking the agent.
    """
    return (
        "You are MetaboAgent, a metabolomics analysis agent. "
        "When a user asks you to analyze data, you MUST call the available tools "
        "to perform the actual analysis - do not respond with text descriptions. "
        "A standard pipeline is: load_metabolomics_data → qc_filter → impute_missing "
        "→ transform → scale → then differential_abundance or other analysis tools. "
        "Always call tools; never just describe what you would do. "
        f"The metabolomics data file is at {DATA_PATH} and the sample annotation "
        f"file is at {ANNOTATION_PATH}. Always use these exact paths. "
        "IMPORTANT: Even if you have seen results before, you MUST always call the "
        "tools again — never summarize from memory or prior context. "
        "If the request is ambiguous and you cannot responsibly choose an analysis "
        "parameter, call request_clarification instead of guessing. "
        "When you have completed everything the user asked for, call finish_analysis. "
        "Do NOT keep calling tools to fill time, and do not repeat a step you have "
        "already run — call finish_analysis and then give your summary. "
        "CRITICAL: when you report a number, copy it verbatim from the tool output. "
        "Never recompute, rescale, round, convert units or change the sign of a value, "
        "and never relabel a quantity as something the tool did not return (for example "
        "do not present a fold change as log2 unless the tool said so). If you did not "
        "receive a number from a tool, say it is not available rather than supplying one."
    )


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
    },
    {
        # Escape hatch. Under tool_choice={"type": "any"} the model is obliged to emit a
        # tool call every turn, so without this tool it can never ask a question and will
        # always guess a parameter instead. This gives refusal-to-guess a legal,
        # loggable form and makes the ambiguous-prompt condition measurable.
        "name": "request_clarification",
        "description": (
            "Call this INSTEAD of an analysis tool when the request is ambiguous and you "
            "cannot responsibly choose a parameter — for example when no grouping variable "
            "was specified and several columns in the annotation file are plausible. "
            "Do not guess in that situation. State exactly what you need to proceed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The specific question to put to the user"},
                "missing_information": {"type": "string", "description": "What is missing from the request"}
            },
            "required": ["question"]
        }
    },
    {
        # Second escape hatch, same root cause as request_clarification. Forced tool use
        # takes away the model's ability to stop as well as its ability to refuse: with
        # tool_choice={"type": "any"} it must emit a tool call every turn, so it pads the
        # session with redundant calls until MAX_TOOL_CALLS. That padding is the least
        # stable part of the trajectory and would dominate any tool-sequence comparison.
        "name": "finish_analysis",
        "description": (
            "Call this when you have completed everything the user asked for and are ready "
            "to summarise. This ends the session. Do not call any analysis tool again after "
            "this. Never call a tool just to have something to call — if the work is done, "
            "finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "completed": {
                    "type": "string",
                    "description": "One line naming the analysis steps you actually ran"
                }
            },
            "required": ["completed"]
        }
    }
]

state = {"dataframe": None}
session_log = []


def reset_state():
    """Clear all module-level state. Call between validation runs."""
    global state, session_log
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
        state["qc_threshold"] = threshold
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
        state["pca_df"] = result["dataframe"]
        state["pca_variance"] = result["variance_explained"]
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
        state["sov"] = result["variation_by_variable"]
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

        # Keep the full result objects in state so the validation harness can read
        # exact metabolite lists and p-values instead of scraping the text summary.
        state["da_results"] = result["results"]
        state["da_significant"] = result["significant"]
        state["da_group_column"] = group_col
        state["da_p_adjust"] = p_method

        # Sort before taking the head. The results frame comes back ordered by
        # metabolite name, so .head(3) used to hand the model three arbitrary
        # rows, which it then reported to the user as the top hits.
        summary = {
            "n_significant": len(result["significant"]),
            "top_metabolites": (
                result["results"].sort_values("p_adjusted").head(5).to_dict(orient="records")
            )
        }

    elif tool_name == "export_session":
        result = export_session(session_log)
        summary = {"filepath": result["filepath"]}

    elif tool_name == "finish_analysis":
        state["finished"] = True
        summary = {"finished": True, "completed": tool_input.get("completed", "")}

    elif tool_name == "request_clarification":
        state["clarification_requested"] = True
        state["clarification_question"] = tool_input.get("question", "")
        summary = {
            "clarification_requested": True,
            "question": tool_input.get("question", ""),
            "missing_information": tool_input.get("missing_information", "")
        }

    else:
        summary = {"error": f"Unknown tool: {tool_name}"}

    session_log.append({"tool": tool_name, "params": tool_input, "result_summary": summary})
    return json.dumps(summary)


def run_agent(user_message: str, verbose: bool = True):
    """
    Headless agent loop. Mirrors app.py's model configuration exactly: same model,
    same max_tokens, same system prompt, same forced-tool-choice policy, same call
    cap. Returns the final assistant text.
    """
    if verbose:
        print(f"\nUser: {user_message}")

    messages = [{"role": "user", "content": user_message}]
    tool_call_count = 0
    final_text = ""

    while True:
        use_tool_choice = (
            {"type": "any"} if tool_call_count < MAX_TOOL_CALLS else {"type": "auto"}
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=build_system_prompt(),
            tools=TOOLS,
            tool_choice=use_tool_choice,
            messages=messages
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            asked_for_clarification = False
            finished = False

            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    if verbose:
                        print(f"\n-> Calling tool: {block.name} with {block.input}")
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                    if block.name == "request_clarification":
                        asked_for_clarification = True
                        final_text = block.input.get("question", "")
                    if block.name == "finish_analysis":
                        finished = True

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # request_clarification is terminal — the agent is waiting on the user.
            if asked_for_clarification:
                if verbose:
                    print(f"\nMetaboAgent (needs input): {final_text}")
                break

            # finish_analysis is terminal too, but the model still gets one
            # un-forced turn to write its summary.
            if finished:
                closing = client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=build_system_prompt(),
                    tools=TOOLS,
                    tool_choice={"type": "auto"},
                    messages=messages
                )
                for block in closing.content:
                    if hasattr(block, "text"):
                        final_text = block.text
                if verbose:
                    print(f"\nMetaboAgent: {final_text}")
                break

        elif response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    final_text = block.text
            if verbose:
                print(f"\nMetaboAgent: {final_text}")
            break

        else:
            if verbose:
                print(f"\n[stopped: {response.stop_reason}]")
            break

    return final_text


if __name__ == "__main__":
    run_agent(
        "Load data/metabolomics_data.csv, run QC, impute, log transform, scale, "
        "then run sources of variation using data/sample_annotation.csv."
    )