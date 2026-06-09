import streamlit as st
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

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Copy TOOLS and run_tool from agent.py
from agent import TOOLS, run_tool, session_log, state

st.set_page_config(page_title="MetabolonR-LLM", page_icon="🧬", layout="wide")
st.title("🧬 MetabolonR-LLM")
st.caption("An LLM-powered metabolomics analysis pipeline")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Tell me what analysis to run..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Running analysis..."):
            messages = [{"role": "user", "content": prompt}]

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    tools=TOOLS,
                    messages=messages
                )

                if response.stop_reason == "tool_use":
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            st.info(f"Running: **{block.name}**...")
                            result = run_tool(block.name, block.input)
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
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                    break