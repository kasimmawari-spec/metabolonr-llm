"""
Analyze validation logs and compute reproducibility metrics.

Metrics:
  (a) Tool-sequence agreement rate  — unambiguous prompts
  (b) Parameter agreement rate      — unambiguous prompts that called differential_abundance
  (c) Mean Jaccard similarity       — significant metabolite sets, all prompts
  (d) Clarification rate            — ambiguous prompts

Usage:
    python validation/analyze_results.py
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
OUT_FILE = Path(__file__).parent / "validation_results.json"


def load_logs():
    logs = []
    for path in sorted(LOG_DIR.glob("prompt_*_run_*.json")):
        with open(path) as f:
            logs.append(json.load(f))
    return logs


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def pairwise_agreement(items):
    """Fraction of unique pairs that are identical."""
    pairs = list(combinations(items, 2))
    if not pairs:
        return 1.0
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def mean_pairwise_jaccard(metabolite_lists):
    pairs = list(combinations(metabolite_lists, 2))
    if not pairs:
        return None
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs)


def analyze_prompt(runs):
    seqs = [tuple(r["tool_sequence"]) for r in runs]
    mets = [r["significant_metabolites"] for r in runs]
    params = [
        (r["diff_abund_params"]["group_column"], r["diff_abund_params"]["p_adjust_method"])
        for r in runs
        if r["diff_abund_params"] is not None
    ]

    return {
        "category": runs[0]["category"],
        "n_runs": len(runs),
        "tool_sequences": [list(s) for s in seqs],
        "sequence_agreement": round(pairwise_agreement(seqs), 4),
        "param_agreement": round(pairwise_agreement(params), 4) if len(params) > 1 else None,
        "mean_jaccard": round(j, 4) if (j := mean_pairwise_jaccard(mets)) is not None else None,
        "clarification_rate": round(
            sum(1 for r in runs if r["asked_clarification"]) / len(runs), 4
        ),
        "diff_abund_params": [r["diff_abund_params"] for r in runs],
        "errors": [r["error"] for r in runs if r["error"]],
    }


def main():
    logs = load_logs()
    if not logs:
        print(f"No logs found in {LOG_DIR}. Run run_validation.py first.")
        return

    print(f"Loaded {len(logs)} log files.")

    by_prompt = defaultdict(list)
    for log in logs:
        by_prompt[log["prompt_id"]].append(log)

    per_prompt = {pid: analyze_prompt(runs) for pid, runs in sorted(by_prompt.items())}

    unambiguous = {pid: p for pid, p in per_prompt.items() if p["category"] == "unambiguous"}
    ambiguous = {pid: p for pid, p in per_prompt.items() if p["category"] == "ambiguous"}

    # (a) Tool-sequence agreement — unambiguous prompts
    seq_rates = [p["sequence_agreement"] for p in unambiguous.values()]
    tool_seq_agreement = sum(seq_rates) / len(seq_rates) if seq_rates else 0.0

    # (b) Parameter agreement — unambiguous prompts with valid param data
    param_rates = [
        p["param_agreement"]
        for p in unambiguous.values()
        if p["param_agreement"] is not None
    ]
    param_agreement = sum(param_rates) / len(param_rates) if param_rates else 0.0

    # (c) Mean Jaccard across all prompts
    jaccard_vals = [p["mean_jaccard"] for p in per_prompt.values() if p["mean_jaccard"] is not None]
    mean_jaccard = sum(jaccard_vals) / len(jaccard_vals) if jaccard_vals else 0.0

    # (d) Clarification rate — ambiguous prompts
    amb_runs = [r for pid in ambiguous for r in by_prompt[pid]]
    clarification_rate = (
        sum(1 for r in amb_runs if r["asked_clarification"]) / len(amb_runs)
        if amb_runs else 0.0
    )

    results = {
        "summary": {
            "n_logs": len(logs),
            "n_prompts": len(per_prompt),
            "n_unambiguous": len(unambiguous),
            "n_ambiguous": len(ambiguous),
            "tool_sequence_agreement_rate": round(tool_seq_agreement, 4),
            "parameter_agreement_rate": round(param_agreement, 4),
            "mean_jaccard_significant_metabolites": round(mean_jaccard, 4),
            "clarification_rate_ambiguous": round(clarification_rate, 4),
        },
        "per_prompt": per_prompt,
    }

    print("\n" + "=" * 50)
    print("  Validation Results")
    print("=" * 50)
    print(f"  Prompts evaluated:                  {len(per_prompt)} ({len(unambiguous)} unambiguous, {len(ambiguous)} ambiguous)")
    print(f"  Total runs:                         {len(logs)}")
    print(f"")
    print(f"  (a) Tool-sequence agreement:        {tool_seq_agreement:.2%}")
    print(f"  (b) Parameter agreement:            {param_agreement:.2%}")
    print(f"  (c) Mean Jaccard (sig. metabolites):{mean_jaccard:.4f}")
    print(f"  (d) Clarification rate (ambiguous): {clarification_rate:.2%}")
    print("=" * 50)

    print("\nPer-prompt sequence agreement (unambiguous):")
    for pid, p in unambiguous.items():
        print(f"  Prompt {pid:02d}: {p['sequence_agreement']:.2%}  {p['tool_sequences'][0]}")

    print("\nPer-prompt clarification rate (ambiguous):")
    for pid, p in ambiguous.items():
        print(f"  Prompt {pid:02d}: {p['clarification_rate']:.2%}")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
