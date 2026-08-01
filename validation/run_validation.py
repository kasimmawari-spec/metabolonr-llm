"""
Run each prompt 3 times through the agent and save structured session logs.
Produces one log file per session in validation/logs/prompt_XX_run_Y.json.

Usage:
    python validation/run_validation.py                # full run
    python validation/run_validation.py --dry-run      # no API calls, print the plan
    python validation/run_validation.py --prompt 12    # single prompt, all repeats
    python validation/run_validation.py --runs 1       # fewer repeats
"""

import argparse
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


def extract_diff_abund_params(session_log):
    """group_column and p_adjust_method actually passed to differential_abundance."""
    for entry in session_log:
        if entry["tool"] == "differential_abundance":
            return {
                "group_column": entry["params"].get("group_column"),
                "p_adjust_method": entry["params"].get("p_adjust_method", "bh"),
            }
    return None


def extract_significant(state):
    """Full significant-metabolite list, read from agent state rather than the
    3-row text summary in the session log."""
    sig = state.get("da_significant")
    if sig is None:
        return []
    if hasattr(sig, "columns") and "metabolite" in getattr(sig, "columns", []):
        return sorted(str(m) for m in sig["metabolite"].tolist())
    if hasattr(sig, "tolist"):
        return sorted(str(m) for m in sig.tolist())
    return sorted(str(m) for m in sig)


def extract_pvalues(state):
    """metabolite -> adjusted p, so two runs can be compared numerically and not
    just as set membership."""
    res = state.get("da_results")
    if res is None or not hasattr(res, "columns"):
        return {}
    if not {"metabolite", "p_adjusted"} <= set(res.columns):
        return {}
    return {str(r.metabolite): float(r.p_adjusted) for r in res.itertuples()}


def run_single(prompt_obj, run_id):
    agent.reset_state()          # full reset: dataframe, DA results, clarification flags

    error = None
    try:
        agent.run_agent(prompt_obj["prompt"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    log = list(agent.session_log)
    state = agent.state

    return {
        "prompt_id": prompt_obj["id"],
        "run_id": run_id,
        "category": prompt_obj["category"],
        "prompt": prompt_obj["prompt"],
        "timestamp": datetime.now().isoformat(),
        "model": agent.MODEL,
        "tool_sequence": [e["tool"] for e in log],
        "tool_calls": log,
        "diff_abund_params": extract_diff_abund_params(log),
        "significant_metabolites": extract_significant(state),
        "n_significant": len(extract_significant(state)),
        "p_adjusted": extract_pvalues(state),
        "asked_clarification": any(e["tool"] == "request_clarification" for e in log),
        "finished_cleanly": any(e["tool"] == "finish_analysis" for e in log),
        "hit_call_cap": len(log) >= agent.MAX_TOOL_CALLS,
        "clarification_question": state.get("clarification_question", ""),
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no API calls; print the plan")
    ap.add_argument("--prompt", type=int, default=None, help="run a single prompt id")
    ap.add_argument("--runs", type=int, default=3, help="repeats per prompt (default 3)")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(PROMPTS_FILE))
    if args.prompt is not None:
        prompts = [p for p in prompts if p["id"] == args.prompt]
        if not prompts:
            sys.exit(f"No prompt with id {args.prompt}")

    total = len(prompts) * args.runs

    if args.dry_run:
        print(f"Would run {total} sessions with model {agent.MODEL}:")
        for p in prompts:
            print(f"  [{p['category']:>13}] prompt {p['id']:02d} × {args.runs}")
        return

    done = 0
    for prompt_obj in prompts:
        pid = prompt_obj["id"]
        for run_id in range(1, args.runs + 1):
            print(f"\n{'=' * 62}")
            print(f"Prompt {pid:02d} / Run {run_id}  [{prompt_obj['category']}]")
            print(f"{'=' * 62}")

            record = run_single(prompt_obj, run_id)

            out_path = LOG_DIR / f"prompt_{pid:02d}_run_{run_id}.json"
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2, default=str)

            done += 1
            print(
                f"  Saved {out_path.name} | tools: {len(record['tool_sequence'])} "
                f"| significant: {record['n_significant']} "
                f"| clarified: {record['asked_clarification']} "
                f"| error: {record['error']} "
                f"| ({done}/{total})"
            )


if __name__ == "__main__":
    main()
