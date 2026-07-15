# Codex handoff — Amendment A4 + Phase-4 launch readiness

**Authored:** 2026-07-14  
**Codex execution status:** not run by Codex  
**Readiness verdict:** code-authoring complete; Phase 4 remains deliberately BLOCKED until the
executor runs the A4 restamp and the final-manifest Phase-3 smoke re-audit below.

## What A4 changes

- `stage_b/bench_spec.py` is the dependency-free runtime source of truth for
  `SPEC_VERSION`.
- `run_bench.py` imports that constant, includes the leaf in future extension-manifest
  construction, and does not restate the value.
- `analyze_universality.py` imports the same object as `ACCEPTED_SPEC_VERSION`; both registered
  input gates remain strict equality checks.
- Preregistration §9 appends A4 without editing A1/A2/A3. It records the bundled E3 regeneration
  and the still-blocking smoke provenance consequence.
- `restamp_manifest_a4.py` requires the exact post-A2/A3 manifest, refuses a second A4, verifies
  unchanged frozen files, appends A4, re-stamps the three changed existing files, adds
  `stage_b/bench_spec.py` as NEW, preserves the preceding restamp provenance, and does not touch
  `SMOKE_SUMMARY.json`.

Historical restamp scripts necessarily contain historical version strings. Therefore the useful
single-literal assertion is over runtime source files, excluding `restamp_manifest_*.py`; claiming
that a raw grep over every provenance script returns one occurrence would be false.

## Static Phase-4 readiness findings

### Confirmatory cohort and planned denominators

Static result: **fail closed; no LOMO escape**.

- The cohort literal contains exactly ten model slugs. The confirmatory tasks are exactly
  `halueval_qa`, `anli_r1_rep`, and `triviaqa_paired_rep`: 3 × 10 = **30 cells**.
- `score_endpoints()` constructs A1 from all ten planned HaluEval-QA slugs and each B1 endpoint
  from all twenty planned family-B slug/task pairs. It does not derive denominators from the
  number of successful profiles.
- Behavioral, commitment, abort, and control failures contribute no deployable pass. A genuinely
  unrun or errored cell yields `pass=null` / `blocked_by_unrun=true` for the endpoint, which is
  stricter than silently treating the reduced observed set as a denominator: it cannot produce a
  PASS. The executor must not report a verdict until every planned confirmatory cell is terminal.
- Registered A2 separately fixes its planned denominator at ten, emits a failed row for every
  unusable holdout, and returns abort=FAIL if fewer than three training models remain.
- The full six-task command emits 60 cell records, but only the 30 A/B cells feed confirmatory A1
  and B1. Family C remains descriptive.

### Stem-cluster confirmatory path

Static result: **the grouped confirmatory gates cannot silently become row gates on a gate-clean
registered input**.

1. The launch data gate requires `stem_id` on every BENCH row. For grouped tasks it hard-fails
   unless every stem is an exact two-row `{0,1}` pair; thus a grouped registered file cannot have
   one unique stem per row.
2. Both ACE and readout collection set `require_stem_ids=True`. Their source labels/stems are
   checked against `sample_idx`, and merge rejects one-arm presence or any cross-arm stem drift.
3. `run_cell()` rejects a merge with no stems, calibrates both `row` and `cluster`, and persists
   the ordered stem vector and its digest into the matrix/profile provenance.
4. Resume requires both units, the ordered-stem digest, the planned sample denominator, and
   control provenance.
5. A1 reads the HaluEval-QA `cluster` geometric result; B1-valid reads the TriviaQA-paired
   `cluster` result. Their row results remain only the separately named procedural/descriptive
   paths. The selector's registered unique-stem reduction to the sealed row algorithm is
   reachable for ungrouped ANLI, not for a grouped file that passed the pairing gate.

### Abort, provenance, resume, and A3 portability guards

Static result: **launch checks precede smoke/strict work and violations do not become warnings**.

- Confirmatory supply is enforced mechanically by the exact 1000-row gate plus exact paired-stem
  structure (500 stems for grouped confirmatory tasks). A task gate failure becomes ABORT for its
  planned cells; reduced-n substitution is absent. The work order records the HaluEval-QA supply
  precondition as cleared, but Codex did not rerun that gate.
- Extension-manifest/spec/file/runtime/T0 hashes and parity-report existence, digest, cohort, and
  disposition consistency raise before data gating or model work. Model snapshot drift produces
  ABORT cells. Nuance required by frozen §8.1: an honestly recorded parity sentinel failure is not
  itself an abort; it is stamped as a feature-version delta. Inconsistent, missing, or hash-drifted
  parity provenance is a hard error. The registered manifest currently attests all ten parity
  checks passed; Codex did not reverify that attestation.
- Exclusion references are matched by the exact expected SHA-256 multiset, and ANLI/TriviaQA Arrow
  inputs resolve via `$CONFLUENCE_HF_CACHE`, the platform Hugging Face datasets cache, then the
  recorded builder path. All resolved bytes must match the frozen source hashes. This validation
  is called by `gate_tasks()` on both smoke and strict paths.
- Strict launch checks `SMOKE_SUMMARY.json` spec, final extension-manifest digest, and the complete
  task→data-digest map before entering the cell loop.
- `--resume` revalidates spec, model/task, data digest, seed, nboot, strict row/drop denominator,
  extension-manifest digest, fusion key, module hashes, model snapshot, row+cluster endpoints,
  control seeds/digest/disposition, feature parity, matrix presence/size, panel digest, ordered
  stems, and commitment audit. Drift raises a terminal error and cannot be counted as deployable.

### E3 artifact contract

Static result: every emitted E3 budget record is constructed with `label_budget` and
`subsample_unit`, plus both deployability fractions. This was inspected in source only; the
regeneration, the supplied 15/18 expectation, and ANLI identity were **not run by Codex**.

## Executor sequence — paste in order

All commands below are executor commands and were **not run by Codex**.

### 1. Compile/import review, then machine-restamp A4

```bash
cd /Users/msrk/Documents/commit-confluence

./confluence python -m py_compile \
  stage_b/bench_spec.py \
  stage_b/run_bench.py \
  stage_b/analyze_universality.py \
  stage_b/restamp_manifest_a4.py

./confluence python -c "import sys; sys.path.insert(0, 'stage_b'); import bench_spec, analyze_universality; assert analyze_universality.ACCEPTED_SPEC_VERSION is bench_spec.SPEC_VERSION"

rg -n -F 'SPEC_VERSION = ' \
  stage_b --glob '*.py' --glob '!restamp_manifest_*.py'

./confluence python stage_b/restamp_manifest_a4.py --executor "Claude Code / MK"
```

Expected grep result: one assignment in `stage_b/bench_spec.py`. Stop if the restamp does not
report Phase 4 still blocked.

### 2. FIRST pipeline action: re-run all 60 smokes against the final manifest (O2)

This intentionally performs the Phase-3 smoke work again with the frozen harness; it does not
blindly edit the manifest digest in the old summary.

```bash
cd /Users/msrk/Documents/commit-confluence
set -o pipefail

./confluence bench \
  --phase smoke \
  --nboot 2000 \
  --out-dir stage_b/profiles_bench \
  --task halueval_qa=stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl \
  --task anli_r1_rep=stage_b/data_bench/anli_r1_rep_seed20260711_n1000.jsonl \
  --task triviaqa_paired_rep=stage_b/data_bench/triviaqa_paired_rep_seed20260711_n1000.jsonl \
  --task anli_r2=stage_b/data_bench/anli_r2_seed20260711_n1000.jsonl \
  --task halueval_dialogue=stage_b/data_bench/halueval_dialogue_seed20260711_n1000.jsonl \
  --task halueval_summarization=stage_b/data_bench/halueval_summarization_seed20260711_n1000.jsonl \
  2>&1 | tee stage_b/profiles_bench/smoke_phase3_a4.log
```

Seed `20260711` is a non-overridable harness constant; nboot is explicitly fixed above at 2000.
Stop unless all 60 planned smoke tags have a terminal PASS/BEHAVIORAL-FAIL/ABORT disposition and
the rewritten `SMOKE_SUMMARY.json` attests the final extension-manifest and all six task hashes.

### 3. Regenerate and check the registered sealed E3 artifact once (O3)

```bash
cd /Users/msrk/Documents/commit-confluence

./confluence analyze \
  --profiles-dir stage_b/profiles \
  --out stage_b/universality.json

./confluence python - <<'PY'
import json
import subprocess

new = json.load(open("stage_b/universality.json"))["E3_label_efficiency"]
old = json.loads(subprocess.check_output(
    ["git", "show", "bc6e2be:stage_b/universality.json"], text=True
))["E3_label_efficiency"]

for deployment, budgets in new.items():
    for budget in budgets.values():
        assert budget.get("subsample_unit") in {"row", "stem"}, (deployment, budget)

anli = sorted(key for key in new if key.endswith("|anli_r1"))
assert anli and all(new[key] == old[key] for key in anli), "ANLI E3 drifted"

excluded = {
    "Llama-3.1-8B-Instruct-4bit|anli_r1",
    "gemma-3-4b-it-4bit|anli_r1",
}
at_150 = [budgets["150"] for key, budgets in new.items() if key not in excluded]
assert len(at_150) == 18, len(at_150)
both = sum(row["frac_deployable_full"] >= 0.8 and
           row["frac_deployable_geom"] >= 0.8 for row in at_150)
assert both == 15, both
print("E3 expectation met: ANLI bit-identical; 15/18 at 150 labels on both endpoints")
PY
```

Stop and do not commit/propagate the regenerated artifact if either assertion fails.

### 4. Detached strict Phase-4 run — exact registered invocation

The raw trace streams to the run log. Launch and completion lifecycle lines also append to the
vault log without flooding that Markdown file with the full model trace.

```bash
cd /Users/msrk/Documents/commit-confluence

export RUN_LOG=/Users/msrk/Documents/commit-confluence/stage_b/profiles_bench/strict_phase4_a4.log
export EXIT_FILE=/Users/msrk/Documents/commit-confluence/stage_b/profiles_bench/strict_phase4_a4.exit
export VAULT_LOG=/Users/msrk/Documents/the_GOAT/wiki/log.md

printf '\n%s — BENCH Phase 4 launch: seed=20260711 nboot=2000 resume=true; 30 confirmatory + 30 descriptive cells.\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" \
  | tee -a "$RUN_LOG" "$VAULT_LOG"

nohup zsh -c '
  ./confluence bench \
    --phase strict \
    --resume \
    --nboot 2000 \
    --out-dir stage_b/profiles_bench \
    --task halueval_qa=stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl \
    --task anli_r1_rep=stage_b/data_bench/anli_r1_rep_seed20260711_n1000.jsonl \
    --task triviaqa_paired_rep=stage_b/data_bench/triviaqa_paired_rep_seed20260711_n1000.jsonl \
    --task anli_r2=stage_b/data_bench/anli_r2_seed20260711_n1000.jsonl \
    --task halueval_dialogue=stage_b/data_bench/halueval_dialogue_seed20260711_n1000.jsonl \
    --task halueval_summarization=stage_b/data_bench/halueval_summarization_seed20260711_n1000.jsonl \
    >> "$RUN_LOG" 2>&1
  status=$?
  printf "%s\n" "$status" > "$EXIT_FILE"
  printf "\n%s — BENCH Phase 4 process exited status=%s; inspect SUMMARY/profile provenance before analysis.\n" "$(date "+%Y-%m-%dT%H:%M:%S%z")" "$status" \
    | tee -a "$RUN_LOG" "$VAULT_LOG"
  exit "$status"
' >/dev/null 2>&1 &

printf '%s\n' "$!" | tee stage_b/profiles_bench/strict_phase4_a4.pid
```

Do not treat process exit zero alone as the verdict. Require `strict_phase4_a4.exit == 0`, then
inspect the final summary for all 60 planned cell records, exactly 30 A/B confirmatory records,
no unrun/error confirmatory cell, fixed denominators 10/20, and intact per-cell provenance.

### 5. Post-run descriptive analysis and registered A2 scoring

Run only after Phase 4 has completed and the cohort checks above pass.

```bash
cd /Users/msrk/Documents/commit-confluence

./confluence analyze \
  --profiles-dir stage_b/profiles_bench \
  --out stage_b/profiles_bench/UNIVERSALITY.json

./confluence analyze \
  --profiles-dir stage_b/profiles_bench \
  --bench-a2 \
  --skip lomo,transfer,labeleff \
  --out stage_b/profiles_bench/A2_REGISTERED.json
```

## Executor verification still required

None of the following was run by Codex: Python parse/import/compile; restamp or hash production;
the runtime-literal check; frozen-surface comparisons; doctor; sealed endpoint verification;
data/parity/exclusion gates; smoke forwards/re-audit; E3 regeneration or numerical assertions;
model forwards; strict resume; summary/profile validation; BENCH analyses; A2 scoring; paper or
vault propagation.

Before believing or publishing the result, the executor must additionally confirm the protected
surfaces (`sealed_selector.py`, `confluence_calibrator.py`, and all frozen Phase-2 data manifests)
stayed byte-unchanged; the pre-restamp extension manifest matched the registered A2/A3 state and
then received only the machine-produced A4 delta; `./confluence doctor` reports READY; and
`./confluence verify` still reports geometric 18/20 PASS and full 18/20 FAIL. Any unregistered
movement is a stop condition.

The pre-existing untracked `profiles_bench/GATE_FAILURES.json`, `SUMMARY.json`, and
`strict_phase4.log` were not edited or treated as registered results by Codex.
