# CODEX WORK ORDER — BENCH Phase 1 (harness build)

**Spec of record:** `stage_b/PRE_REGISTRATION_BENCH.md` v1.2 — read it in full first.
Where this work order and the pre-reg disagree, the pre-reg wins.
**Codex constraint:** write/audit only. Author code and static checks; run NOTHING
(no tests, no model inference, no dataset downloads, no builders). Every verification
that requires execution is listed under "Executor handoff" and will be run by
Claude Code or MK.

## Hard boundaries

- ❌ Do NOT edit: any T0 module (enumerated in pre-reg §1), `fusion_signs.json`
  (sha256 frozen at `92b5468b…5372`), `run_seal.py`, the sealed `stage_b/profiles/`,
  `PRE_REGISTRATION_BENCH.md` itself, or anything in `t0-morphology-furnace`.
- ⚠️ `confluence_calibrator.py` (Phase-0 baseline sha256 `f5527916…3bbb`): edits limited
  to exactly the four §7.4 items — stem persistence, cluster resampling, paired-design
  null permutation, canonical fusion task key. Keep the diff minimal and reviewable;
  the row path must remain byte-behavior-identical when no `stem_id` is supplied
  (the §1 parity sentinel will verify this by execution).
- 🧾 New files use the frozen names: `generate_bench_data.py`, `run_bench.py`
  (both in `stage_b/`). No siblings, no extra helper files outside the §1 manifest set
  (if one proves unavoidable, stop and say so — it needs an Amendments entry first).

## Deliverables (build order)

1. **`confluence_calibrator.py` edits (§7.4 — do this first; everything depends on it):**
   - Persist per-row `stem_id` from the JSONL through loaders, ACE/readout alignment,
     `merge_matrices`, and the `.npz`; equality-check against source metadata; hard
     error on mismatch or missing ids for grouped tasks.
   - Stem-cluster selector per the exact §5 algorithm (stems sampled with replacement;
     in-bag multiplicity; OOB = rows of absent stems; in-bag selection/sign-lock, OOB
     evaluation; CI = 2.5th pct over nboot=2000). Ungrouped ⇒ stem=row ⇒ must reduce to
     the sealed procedure identically.
   - Paired-design null permutation per §4 (within-stem label swap p=0.5; global row
     permutation for ungrouped; RandomState(20260711+90210+k), k∈{0,1,2}).
   - Accept a canonical fusion-sign task key separately from the reported BENCH task key
     (§1 alias table: `anli_r1_rep`→`anli_r1`, `triviaqa_paired_rep`→`triviaqa_paired`;
     new tasks → frozen modal fallbacks). Replace the `sealed_selector="imported, not
     modified"` provenance string with an honest resampling-unit-specific value for
     cluster results.
2. **`stage_b/generate_bench_data.py` (§7.1, §3):** builders
   `build_halueval_{qa,dialogue,summarization}` (pinned-commit GitHub raw download +
   sha256 record, byte-frozen §3.1 templates, paired emission, `stem_id`,
   2048-token common-intersection filter per §4), `build_anli_generic` (frozen local
   Arrow artifacts of §3.2 by fingerprint+sha256, `train_r1` / split-stratified
   `dev_r2+test_r2`, enumerated exclusion union), `build_triviaqa_rep` (§3.3, sealed
   builder logic verbatim apart from quota + exclusion union). Sampling must implement
   §3.4 exactly (RNG classes, orders, quotas, emission order) and write the §3.4-5
   manifest records. Reuse sealed template/label conventions byte-identically.
3. **`check_fresh_data.py` generic path (§7.2):** exclusion-union assertion, length-cap,
   one-token-cue, stem-cap ≤2, `stem_id` presence; keep schema/balance/intra-dup.
   Do not break the existing `--task {anli,triviaqa}` sealed path.
4. **`stage_b/run_bench.py` (§7.3):** (task→file) pairs; `profiles_bench/`; §5 endpoint
   semantics (terminal statuses BEHAVIORAL-FAIL / COMMITMENT-FAIL / CONTROL-FAIL[unit] /
   ABORT scored as failures over fixed denominators; unrun blocks scoring); stratified
   16-row smoke mode (§4); full-cell commitment audit with the frozen YES/NO
   normalization; K=3 controls per §4 with per-unit disposition; fusion alias stamping;
   resume validation comparing every §7.3 field (bootstrap_unit, ordered stem_id vector
   sha256, canonical fusion key, panel sha256, `bench/1.2`, commitment tally, control
   seeds/results, manifest sha256) — mismatch = terminal ERROR, recompute.
   Include an `--emit-manifest` step that writes `profiles_bench/EXTENSION_MANIFEST.json`
   per §1 (file sha256 set + runtime version tuple + pre/post calibrator diff summary).
5. **`analyze_universality.py` (§7.5):** registered A2 path implementing the exact §5
   estimator (within-model `_rank01`, pooled-training single global sign via
   `_score_candidate`, `auroc_fixed` holdout, strict >0.55, planned denominator always
   /10, <3 usable training models → abort=FAIL, full stamping), separate from the
   descriptive `fixed_cell_max_survival` landscape.
6. **Static self-review note** (append to this file or a short `BUILD_NOTES_BENCH.md`):
   what you changed, the calibrator diff summary, any spec ambiguity you resolved and
   how, and the exact commands the executor should run — marked "not run by Codex."

## Executor handoff (NOT Codex — Claude Code / MK will run)

- 🔬 Ten-model frozen-row parity sentinel (§1) on pre-existing seal data/matrices.
- 🧪 Unit-level checks + a limit-8 lenient dry run of each builder/gate.
- 🔏 `EXTENSION_MANIFEST.json` recording + calibrator-diff inspection vs the Phase-0 hash.
- Phases 2–5 (data build, smokes, strict cells, analysis).

## Gate reminder

Phase order is binding: MK sign-off (Phase 0) precedes Phase 2 data draws. Building this
code is Phase-1 work; no dataset file may be generated, downloaded, or sampled by Codex
under any circumstances.
