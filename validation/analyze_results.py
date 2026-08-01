"""
Analyse validation logs.

The key distinction — and the reason a single pooled Jaccard was uninterpretable
— is between:

  TIER 1  DETERMINISM      the same prompt, repeated.
                           The engine is deterministic, so anything short of an
                           identical tool sequence, identical parameters and an
                           identical significant set is a real failure.
                           Expected: 1.00 on every measure.

  TIER 2  ROBUSTNESS       different paraphrases of the same intent.
                           Below 1.00 is expected and informative: it measures
                           how much the wording of a request changes the answer.

  TIER 3  REFUSAL          ambiguous prompts. The agent should call
                           request_clarification rather than guess a parameter.

  TIER 4  FALSE PREMISE    a prompt asserting group values that do not exist.
                           Either an error or a clarification is a good outcome;
                           silently analysing the wrong thing is not.

Usage:
    python validation/analyze_results.py
"""

import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
OUT_FILE = Path(__file__).parent / "validation_results.json"


# ------------------------------------------------------------------ helpers
def load_logs():
    logs = []
    for path in sorted(LOG_DIR.glob("prompt_*_run_*.json")):
        logs.append(json.load(open(path)))
    if not logs:
        raise SystemExit(f"No logs found in {LOG_DIR}. Run run_validation.py first.")
    return logs


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def all_equal(items):
    return all(x == items[0] for x in items)


def mean(xs):
    return round(statistics.fmean(xs), 4) if xs else None


def by_prompt(logs):
    d = defaultdict(list)
    for lg in logs:
        d[lg["prompt_id"]].append(lg)
    return {k: sorted(v, key=lambda x: x["run_id"]) for k, v in sorted(d.items())}


# ------------------------------------------------- tier 1: same prompt repeated
def tier1_determinism(groups, categories):
    per_prompt, seq_ok, par_ok, set_ok, jac = {}, [], [], [], []

    for pid, runs in groups.items():
        if categories[pid] != "unambiguous" or len(runs) < 2:
            continue
        seqs = [r["tool_sequence"] for r in runs]
        pars = [r["diff_abund_params"] for r in runs]
        sets = [r["significant_metabolites"] for r in runs]
        pairs = [jaccard(a, b) for a, b in combinations(sets, 2)]

        rec = {
            "n_runs": len(runs),
            "identical_tool_sequence": all_equal(seqs),
            "identical_parameters": all_equal(pars),
            "identical_significant_set": all_equal([tuple(s) for s in sets]),
            "mean_jaccard": mean(pairs),
            "n_significant_per_run": [r["n_significant"] for r in runs],
            "errors": [r["error"] for r in runs if r["error"]],
        }
        per_prompt[pid] = rec
        seq_ok.append(rec["identical_tool_sequence"])
        par_ok.append(rec["identical_parameters"])
        set_ok.append(rec["identical_significant_set"])
        jac.extend(pairs)

    n = len(per_prompt)
    return {
        "n_prompts": n,
        "prompts_with_identical_tool_sequence": f"{sum(seq_ok)}/{n}",
        "prompts_with_identical_parameters": f"{sum(par_ok)}/{n}",
        "prompts_with_identical_significant_set": f"{sum(set_ok)}/{n}",
        "mean_within_prompt_jaccard": mean(jac),
        "expected": "all three counts should be n/n and Jaccard 1.0",
        "per_prompt": per_prompt,
    }


# ------------------------------------------- tier 2: paraphrases of one intent
def tier2_robustness(groups, categories):
    reps, seqs, pars = {}, {}, {}
    for pid, runs in groups.items():
        if categories[pid] != "unambiguous":
            continue
        r = runs[0]                                   # run 1 represents the prompt
        reps[pid] = r["significant_metabolites"]
        seqs[pid] = tuple(r["tool_sequence"])
        pars[pid] = json.dumps(r["diff_abund_params"], sort_keys=True)

    ids = sorted(reps)
    pairs = [(a, b) for a, b in combinations(ids, 2)]
    jac = [jaccard(reps[a], reps[b]) for a, b in pairs]
    seq_same = [seqs[a] == seqs[b] for a, b in pairs]
    par_same = [pars[a] == pars[b] for a, b in pairs]

    worst = sorted(zip(jac, pairs))[:5]
    return {
        "n_paraphrases": len(ids),
        "n_pairs": len(pairs),
        "mean_across_paraphrase_jaccard": mean(jac),
        "min_across_paraphrase_jaccard": round(min(jac), 4) if jac else None,
        "pairs_with_same_tool_sequence": f"{sum(seq_same)}/{len(pairs)}",
        "pairs_with_same_parameters": f"{sum(par_same)}/{len(pairs)}",
        "n_significant_by_prompt": {pid: len(reps[pid]) for pid in ids},
        "least_agreeing_pairs": [
            {"prompts": list(p), "jaccard": round(j, 4)} for j, p in worst
        ],
        "note": "below 1.0 is expected here; it measures sensitivity to wording",
    }


# ----------------------------------------------------- tier 3: ambiguous input
def tier3_refusal(groups, categories):
    per_prompt, flags = {}, []
    for pid, runs in groups.items():
        if categories[pid] != "ambiguous":
            continue
        asked = [r["asked_clarification"] for r in runs]
        per_prompt[pid] = {
            "asked_clarification": asked,
            "questions": [r["clarification_question"] for r in runs if r["clarification_question"]],
            "guessed_group_column": [
                r["diff_abund_params"]["group_column"] for r in runs
                if r["diff_abund_params"]
            ],
        }
        flags.extend(asked)
    return {
        "n_sessions": len(flags),
        "clarification_rate": mean([float(f) for f in flags]),
        "sessions_that_asked": f"{sum(flags)}/{len(flags)}",
        "expected": "the agent should ask, not guess, on every ambiguous prompt",
        "per_prompt": per_prompt,
    }


# --------------------------------------------------- tier 4: impossible groups
def tier4_false_premise(groups, categories):
    out = {}
    for pid, runs in groups.items():
        if categories[pid] != "false_premise":
            continue
        out[pid] = [{
            "run": r["run_id"],
            "asked_clarification": r["asked_clarification"],
            "error": r["error"],
            "params": r["diff_abund_params"],
            "n_significant": r["n_significant"],
        } for r in runs]
    return {
        "expected": "error or clarification is acceptable; silently analysing "
                    "non-existent groups is not",
        "per_prompt": out,
    }


# ---------------------------------------------------------------------- main
def main():
    logs = load_logs()
    groups = by_prompt(logs)
    categories = {pid: runs[0]["category"] for pid, runs in groups.items()}

    report = {
        "n_sessions": len(logs),
        "n_prompts": len(groups),
        "model": logs[0].get("model", "unknown"),
        "sessions_with_errors": sum(1 for l in logs if l["error"]),
        "sessions_finished_cleanly": sum(1 for l in logs if l.get("finished_cleanly")),
        "sessions_that_hit_call_cap": sum(1 for l in logs if l.get("hit_call_cap")),
        "median_tool_calls": sorted(len(l["tool_sequence"]) for l in logs)[len(logs) // 2],
        "tier1_determinism_same_prompt": tier1_determinism(groups, categories),
        "tier2_robustness_across_paraphrases": tier2_robustness(groups, categories),
        "tier3_refusal_on_ambiguous": tier3_refusal(groups, categories),
        "tier4_false_premise": tier4_false_premise(groups, categories),
    }

    json.dump(report, open(OUT_FILE, "w"), indent=2, default=str)

    t1 = report["tier1_determinism_same_prompt"]
    t2 = report["tier2_robustness_across_paraphrases"]
    t3 = report["tier3_refusal_on_ambiguous"]
    print(f"\n{'=' * 62}\nVALIDATION SUMMARY  ({report['n_sessions']} sessions, "
          f"{report['sessions_with_errors']} errors)\n{'=' * 62}")
    print(f"\nSESSION SHAPE  — expect clean finishes, few cap hits")
    print(f"  finished via finish_analysis  {report['sessions_finished_cleanly']}/{report['n_sessions']}")
    print(f"  hit the tool-call cap         {report['sessions_that_hit_call_cap']}/{report['n_sessions']}")
    print(f"  median tool calls             {report['median_tool_calls']}")
    print("\nTIER 1  same prompt repeated  — expect everything identical")
    print(f"  identical tool sequence   {t1['prompts_with_identical_tool_sequence']}")
    print(f"  identical parameters      {t1['prompts_with_identical_parameters']}")
    print(f"  identical significant set {t1['prompts_with_identical_significant_set']}")
    print(f"  mean Jaccard              {t1['mean_within_prompt_jaccard']}")
    print("\nTIER 2  across paraphrases    — below 1.0 expected")
    print(f"  mean Jaccard              {t2['mean_across_paraphrase_jaccard']}")
    print(f"  min  Jaccard              {t2['min_across_paraphrase_jaccard']}")
    print(f"  same tool sequence        {t2['pairs_with_same_tool_sequence']}")
    print("\nTIER 3  ambiguous prompts     — expect the agent to ask")
    print(f"  asked for clarification   {t3['sessions_that_asked']}")
    print(f"\nFull report written to {OUT_FILE}")


if __name__ == "__main__":
    main()
