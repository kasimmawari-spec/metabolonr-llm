import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import anthropic
import pandas as pd
import json
import plotly.express as px

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

load_dotenv()
api_key = st.secrets["ANTHROPIC_API_KEY"] if "ANTHROPIC_API_KEY" in st.secrets else os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)

import agent
from agent import TOOLS, run_tool, session_log, state

st.set_page_config(page_title="MetaboAgent", page_icon="🧬", layout="wide")
st.title("🧬 MetaboAgent")
st.caption("An LLM-powered metabolomics analysis pipeline")

DATA_FILES = {
    "Metabolomics data": "data/metabolomics_data.csv",
    "Sample annotation": "data/sample_annotation.csv",
}

EXAMPLE_PROMPTS = [
    "Load data/metabolomics_data.csv, run QC, impute, log transform, scale, then run sources of variation using data/sample_annotation.csv.",
    "Run the full pipeline (QC, impute, transform, scale) and then run differential abundance using DM_5crit from data/sample_annotation.csv.",
    "Load the data and run PCA with 2 components after QC and scaling.",
]


def make_pca_fig(pca_data: dict, color_by: str = None):
    """Rebuild a PCA scatter plot from stored PC data, optionally colored by a clinical variable."""
    var = pca_data["variance_explained"]
    df = pd.DataFrame({"PC1": pca_data["pc1"], "PC2": pca_data["pc2"]})

    color_col = None
    if color_by and color_by in pca_data.get("color_data", {}):
        df[color_by] = pca_data["color_data"][color_by]
        color_col = color_by

    fig = px.scatter(
        df, x="PC1", y="PC2",
        color=color_col,
        title="PCA — Sample Scores",
        labels={
            "PC1": f"PC1 ({var[0]*100:.1f}% variance)",
            "PC2": f"PC2 ({var[1]*100:.1f}% variance)" if len(var) > 1 else "PC2",
        },
        template="simple_white",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return fig


def collect_pca_data() -> dict:
    """Collect PCA scores + annotation columns for coloring."""
    pca_df = state.get("pca_df")
    if pca_df is None:
        return None

    var = state.get("pca_variance", [0, 0])

    # Load annotation for color options
    color_data = {}
    try:
        annot = pd.read_csv(
            agent.ANNOTATION_PATH, index_col=0,
            na_values=['-', '.', 'NA', 'N/A', '']
        )
        if 'Sample name' in annot.columns:
            annot = annot.set_index('Sample name')
        annot.index = annot.index.astype(str)

        # Use categorical + low-cardinality numeric columns
        candidate_cols = list(annot.select_dtypes(include=['object', 'category']).columns)
        for col in annot.select_dtypes(include='number').columns:
            if annot[col].nunique() <= 10:
                candidate_cols.append(col)

        for col in candidate_cols[:8]:
            try:
                vals = annot.loc[pca_df.index, col].astype(str).tolist()
                color_data[col] = vals
            except Exception:
                pass
    except Exception:
        pass

    return {
        "pc1": pca_df["PC1"].tolist(),
        "pc2": pca_df["PC2"].tolist(),
        "variance_explained": var,
        "color_data": color_data,
    }


def render_pca_plot(pca_data: dict, key_suffix: str = ""):
    """Render PCA scatter plot with a color-by dropdown."""
    color_options = ["None"] + list(pca_data.get("color_data", {}).keys())
    color_by = st.selectbox(
        "Color dots by:", color_options,
        key=f"pca_color_{key_suffix}"
    )
    selected = None if color_by == "None" else color_by
    st.plotly_chart(make_pca_fig(pca_data, selected), use_container_width=True)


def format_tool_summary(tool_name: str, summary: dict) -> str:
    if tool_name == "load_metabolomics_data":
        return f"Loaded **{summary.get('n_samples', '?')}** samples and **{summary.get('n_metabolites', '?')}** metabolites."
    elif tool_name == "summarize_dataset":
        return f"Missing data: **{summary.get('missing_percent', '?')}%**"
    elif tool_name == "qc_filter":
        remaining = state["dataframe"].shape[1] if state["dataframe"] is not None else "?"
        return f"QC Filter: removed **{summary.get('removed', '?')}** metabolites, **{remaining}** remaining."
    elif tool_name == "impute_missing":
        return f"Imputation complete: **{summary.get('missing_after', '?')}** missing values remaining."
    elif tool_name == "transform":
        return f"Applied **{summary.get('method', '?')}** transformation."
    elif tool_name == "scale":
        return "Scaled all metabolites to mean=0, std=1."
    elif tool_name == "pca":
        var = summary.get("variance_explained", [])
        pct = f"{sum(var)*100:.1f}%" if var else "?"
        return f"PCA complete. Top components explain **{pct}** of variance."
    elif tool_name == "sources_of_variation":
        return "Computed sources of variation across clinical variables."
    elif tool_name == "pathway_variation":
        return "Computed variance by pathway."
    elif tool_name == "differential_abundance":
        return f"Differential abundance complete: **{summary.get('n_significant', '?')}** significant metabolites found."
    elif tool_name == "export_session":
        return f"Session exported to `{summary.get('filepath', '?')}`."
    elif "error" in summary:
        return f"Error: {summary['error']}"
    return json.dumps(summary)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)

    st.header("📂 Upload Your Data")
    uploaded_data = st.file_uploader(
        "Metabolomics CSV", type=["csv"], key="upload_data",
        help="Your metabolomics abundance matrix (samples × metabolites)"
    )
    uploaded_annot = st.file_uploader(
        "Sample Annotation CSV", type=["csv"], key="upload_annot",
        help="Your sample metadata file with group labels"
    )

    if uploaded_data and uploaded_annot:
        tmp_dir = tempfile.mkdtemp()
        data_path = os.path.join(tmp_dir, "metabolomics_data.csv")
        annot_path = os.path.join(tmp_dir, "sample_annotation.csv")
        with open(data_path, "wb") as f:
            f.write(uploaded_data.getvalue())
        with open(annot_path, "wb") as f:
            f.write(uploaded_annot.getvalue())
        agent.DATA_PATH = data_path
        agent.ANNOTATION_PATH = annot_path
        st.success("✅ Custom data loaded!")
        st.caption(f"Data: {uploaded_data.name}  \nAnnotation: {uploaded_annot.name}")
    else:
        agent.DATA_PATH = "data/metabolomics_data.csv"
        agent.ANNOTATION_PATH = "data/sample_annotation.csv"
        st.info("ℹ️ Using default Progredir dataset, or upload your own above.")

    st.divider()

    st.header("Dataset")
    for label, path in DATA_FILES.items():
        if os.path.exists(path):
            st.markdown(f"✅ **{label}**\n\n`{path}`")
        else:
            st.markdown(f"❌ **{label}**\n\n`{path}` (not found)")

    st.divider()

    st.header("Quick Start")
    for i, example in enumerate(EXAMPLE_PROMPTS):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.queued_prompt = example

    st.divider()

    st.header("Session")
    st.metric("Tool calls made", len(session_log))

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None

if not st.session_state.messages:
    st.info(
        "👋 **Welcome to MetaboAgent**\n\n"
        "This tool runs a metabolomics analysis pipeline using an LLM agent. "
        "Describe what you'd like to do in plain English (e.g. *\"load the data, "
        "run QC and imputation, then run differential abundance\"*) and the agent "
        "will call the appropriate analysis tools step by step.\n\n"
        "Use the **Quick Start** prompts in the sidebar to get going, or type your "
        "own request below."
    )

# Replay chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        pca_data_to_replay = None
        for call in msg.get("tool_calls", []):
            with st.expander(f"🔧 {call['name']}", expanded=False):
                st.markdown(call["summary"])
            if call.get("pca_data"):
                pca_data_to_replay = call["pca_data"]
        if msg["content"]:
            st.markdown(msg["content"])
        if pca_data_to_replay:
            render_pca_plot(pca_data_to_replay, key_suffix=f"replay_{msg_idx}")

chat_prompt = st.chat_input("Tell me what analysis to run...")
prompt = chat_prompt or st.session_state.queued_prompt
st.session_state.queued_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_calls_made = []
        messages = [{"role": "user", "content": prompt}]

        with st.spinner("Running analysis..."):
            pca_data_collected = None
            tool_call_count = 0
            MAX_TOOL_CALLS = 15

            while True:
                use_tool_choice = {"type": "any"} if tool_call_count < MAX_TOOL_CALLS else {"type": "auto"}

                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=(
                        f"You are MetaboAgent, a metabolomics analysis agent. "
                        f"When a user asks you to analyze data, you MUST call the available tools "
                        f"to perform the actual analysis - do not respond with text descriptions. "
                        f"A standard pipeline is: load_metabolomics_data → qc_filter → impute_missing "
                        f"→ transform → scale → then differential_abundance or other analysis tools. "
                        f"Always call tools; never just describe what you would do. "
                        f"The metabolomics data file is at {agent.DATA_PATH} and the sample annotation "
                        f"file is at {agent.ANNOTATION_PATH}. Always use these exact paths. "
                        f"IMPORTANT: Even if you have seen results before, you MUST always call the tools again — never summarize from memory or prior context."
                    ),
                    tools=TOOLS,
                    tool_choice=use_tool_choice,
                    messages=messages
                )

                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            tool_call_count += 1
                            with st.status(f"Running {block.name}...", expanded=False) as status:
                                result = run_tool(block.name, block.input)
                                result_dict = json.loads(result)
                                summary = format_tool_summary(block.name, result_dict)
                                status.update(label=f"{block.name}", state="complete")
                                st.markdown(summary)

                            # Store PCA variance for annotation loading
                            if block.name == "pca":
                                state["pca_variance"] = result_dict.get("variance_explained", [0, 0])

                            pca_data = None
                            if block.name == "pca" and state.get("pca_df") is not None:
                                pca_data = collect_pca_data()
                                pca_data_collected = pca_data

                            tool_calls_made.append({
                                "name": block.name,
                                "summary": summary,
                                "pca_data": pca_data,
                            })
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                elif response.stop_reason == "end_turn":
                    final_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            final_text = block.text
                    st.markdown(final_text)
                    if pca_data_collected:
                        render_pca_plot(pca_data_collected, key_suffix="live")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_text,
                        "tool_calls": tool_calls_made,
                    })
                    break