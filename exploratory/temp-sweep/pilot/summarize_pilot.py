"""Tabulate a sampling-pilot run: buckets a/b/c/d per model x temperature with denominators, harness gates,
exact canonical mass, parser cross-checks, and every non-canonical output for hand-checking.

Usage: python summarize_pilot.py runs/<id>   ->  writes runs/<id>/SUMMARY.md (local; runs/ is git-ignored)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sampling_pilot import K, MODELS, N_ANLI, N_STEMS, TEMPS  # noqa: E402

BUCKETS = ["a_harness", "b_noncompliant", "c_valid_wrong", "d_valid_correct"]


def main(run):
    run = Path(run) if Path(run).is_absolute() else HERE / run
    lines = [f"# Sampling pilot summary: `{run.name}`", ""]
    planned_per_model = (N_ANLI + 2 * N_STEMS) * (1 + len(TEMPS) * K)
    table = ["| Model | T | n (planned) | a harness | b non-compliant | c valid wrong | d valid correct | "
             "hit max tokens | mean exact P(invalid) |", "|---|---|---|---|---|---|---|---|---|"]
    gates, noncanon, disagreements = [], [], []
    for model_id, _ in MODELS:
        path = run / f"{model_id.split('/')[1]}.jsonl"
        recs = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if len(recs) != planned_per_model:
            gates.append(f"- {model_id}: {len(recs)} of {planned_per_model} planned records (missing ones count as "
                         "harness failures)")
        for temp in (0.0,) + TEMPS:
            rs = [r for r in recs if r["T"] == temp]
            n_planned = (N_ANLI + 2 * N_STEMS) * (1 if temp == 0.0 else K)
            c = Counter(r["bucket"] for r in rs)
            c["a_harness"] += n_planned - len(rs)
            pv = [1 - r["p_valid"] for r in rs if "p_valid" in r]
            hit = sum(bool(r.get("hit_max_tokens")) for r in rs)
            label = "greedy" if temp == 0.0 else f"{temp}"
            table.append(f"| {model_id.split('/')[1]} | {label} | {n_planned} | " +
                         " | ".join(str(c[b]) for b in BUCKETS) +
                         f" | {hit}/{len(rs)} | {np.mean(pv):.4f} |" if pv else
                         f"| {model_id.split('/')[1]} | {label} | {n_planned} | " +
                         " | ".join(str(c[b]) for b in BUCKETS) + f" | {hit}/{len(rs)} | — |")
        g = [r for r in recs if r["T"] == 0.0 and "mlx_lm_parity" in r]
        h = [r for r in g if "historical_match" in r]
        gates.append(f"- {model_id}: greedy parity with mlx_lm {sum(r['mlx_lm_parity'] for r in g)}/{len(g)}; "
                     f"HaluEval greedy token = historical BENCH commit {sum(r['historical_match'] for r in h)}/{len(h)}")
        for r in recs:
            if r["bucket"] == "b_noncompliant":
                noncanon.append(r)
            if "independent_parser_canonical" in r and r["independent_parser_canonical"] != (r["disposition"] != "non-canonical"):
                disagreements.append(r)
    lines += ["## Buckets per model and temperature", "",
              "Denominators are planned records; missing records are counted as harness failures. Every output is "
              "one token by registered design (`max_new_tokens=1`), so 'hit max tokens' is always n/n and is not "
              "truncation. 'Mean exact P(invalid)' is the probability mass on non-canonical tokens at that T, "
              "averaged over prompts (computed from the distribution, not from the five draws).", ""] + table
    lines += ["", "## Harness gates", ""] + gates
    lines += ["", f"## Parser cross-check: {len(disagreements)} disagreement(s) between the BENCH normalizer and the "
              "independent letters-only parser", ""]
    lines += [f"- {r['model'].split('/')[1]} {r['task']} #{r['sample_idx']} T={r['T']}: {r['raw_text']!r}"
              for r in disagreements[:40]]
    by_tok = defaultdict(list)
    for r in noncanon:
        by_tok[(r["model"].split("/")[1], r["raw_text"])].append(r)
    lines += ["", f"## Non-canonical outputs ({len(noncanon)}), grouped by model and token, with diagnostic "
              "continuations", "", "| Model | Token | Count | T values | Example continuation (greedy, diagnostic) |",
              "|---|---|---|---|---|"]
    for (m, tok), rs in sorted(by_tok.items(), key=lambda kv: -len(kv[1])):
        temps = ",".join(sorted({str(r["T"]) for r in rs}))
        cont = rs[0].get("diagnostic_continuation", "").replace("\n", "\\n").replace("|", "\\|")[:80]
        lines.append(f"| {m} | {tok!r} | {len(rs)} | {temps} | `{cont}` |")
    (run / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
