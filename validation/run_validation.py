"""
Run each prompt 3 times through the agent and save structured session logs.
Produces 60 log files in validation/logs/prompt_XX_run_Y.json.

Usage:
    python validation/run_validation.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root so `import agent` resolves correctly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import agent  # noqa: E402  (module-level state: agent.state, agent.session_log)

LOG_DIR = Path(__file__).parent / "logs"
PROMPTS_FILE = Path(__file__).parent / "prompts.json"


def reset_agent():
    agent.state["dataframe"] = None
    agent.session_log.clear()


def extract_diff_abund_params(session_log):
    """Return group_column and p_adjust_method if differential_abundance was called."""
    for entry in session_log:
        if entry["tool"] == "differential_abundance":
            return {
                "group_column": entry["params"].get("group_column"),
                "p_adjust_method": entry["params"].get("p_adjust_method", "bh"),
            }
    return None


def extract_significant_metabolites(session_log):
    """Return top metabolite names from the differential_abundance result summary."""
    for entry in session_log:
        if entry["tool"] == "differential_abundance":
            top = entry["result_summary"].get("top_metabolites", [])
            return [r["metabolite"] for r in top if "metabolite" in r]
    return []


def run_single(prompt_obj, run_id):
    reset_agent()

    error = None
    try:
        agent.run_agent(prompt_obj["prompt"])
    except Exception as exc:
        error = str(exc)

    log_snapshot = list(agent.session_log)

    return {
        "prompt_id": prompt_obj["id"],
        "run_id": run_id,
        "category": prompt_obj["category"],
        "prompt": prompt_obj["prompt"],
        "timestamp": datetime.now().isoformat(),
        "tool_sequence": [e["tool"] for e in log_snapshot],
        "tool_calls": log_snapshot,
        "diff_abund_params": extract_diff_abund_params(log_snapshot),
        "significant_metabolites": extract_significant_metabolites(log_snapshot),
        "asked_clarification": len(log_snapshot) == 0,
        "error": error,
    }


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    total = len(prompts) * 3
    done = 0

    for prompt_obj in prompts:
        pid = prompt_obj["id"]
        for run_id in range(1, 4):
            print(f"\n{'=' * 60}")
            print(f"Prompt {pid:02d} / Run {run_id}  [{prompt_obj['category']}]")
            print(f"{'=' * 60}")

            record = run_single(prompt_obj, run_id)

            out_path = LOG_DIR / f"prompt_{pid:02d}_run_{run_id}.json"
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2, default=str)

            done += 1
            n_tools = len(record["tool_sequence"])
            clarified = record["asked_clarification"]
            print(
                f"  Saved: {out_path.name}  "
                f"| tools called: {n_tools}  "
                f"| asked_clarification: {clarified}  "
                f"| ({done}/{total})"
            )


if __name__ == "__main__":
    main()
