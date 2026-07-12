# BENCH Phase 1 build notes

**Status:** authored by Codex, 2026-07-12. `PRE_REGISTRATION_BENCH.md` v1.2 is the
controlling specification. Codex ran static file/diff/hash inspection only; no Python,
test, builder, dataset, model, harness, or manifest command was run. The executor should
still run every handoff command below in order.

## Authored changes

- `confluence_calibrator.py`: optional source-row `stem_id` persistence through both
  extraction arms and merge; exact first-occurrence-ordered stem bootstrap; paired
  within-stem null swaps; explicit row/cluster endpoint and control records; frozen
  replication-task fusion aliases. With no stems and the default row unit, selection still
  calls the imported sealed selector directly.
- `generate_bench_data.py`: frozen HaluEval templates/source commit, exact Arrow hashes,
  exclusion union, cohort-wide tokenizer length intersection, registered RNG/order rules,
  paired stems, manifests, and an explicitly non-registered `--preview-limit 8` path.
- `check_fresh_data.py`: preserved legacy callable/CLI behavior and added the BENCH union,
  exact balance, cue, stem, pair, cohort-token-count, and length gates.
- `run_bench.py`: extension manifest, synthetic Phase-1 checks, ten-model frozen-row parity,
  manifest attestation, model/hash gates, stratified smokes, full commitment audit,
  row/cluster controls, fixed-denominator terminal-state scoring, resume validation, and
  source-manifest verification.
- `analyze_universality.py`: registered A2 fixed-cell LOMO path, separate from the existing
  multiplicity-prone landscape.

## Calibrator diff scope

The Phase-0 sha256 frozen by the pre-reg is
`f55279162eb15a3806a0698e1540e898a13f70dbdc9e99820a969bb2bf563bbb`.
The authored diff is restricted to §7.4 plumbing/statistics: stem metadata, cluster
resampling, paired nulls, canonical fusion task keys, and the metadata needed to deliver
the §4/§7.3 commitment audit without a second model forward (`gen_token_ids`). The executor
must inspect the actual diff and record the post-edit hash in `EXTENSION_MANIFEST.json`.

## Static interpretation choices

- `stem_id` is accepted at top level (the builders' canonical form) or under `meta` for
  compatibility; partial presence is a hard error.
- The frozen commitment normalizer strips leading/trailing whitespace, case-folds, then
  removes trailing ASCII or Unicode-category punctuation. It accepts only `yes` or `no`.
- A missing required smoke is `UNRUN` and blocks endpoint scoring. A completed behavioral,
  commitment, control, source, or model-drift failure is priced inside the registered
  denominator according to its endpoint/unit.
- The parity sentinel must complete for all ten models. A mismatch is not silently waived
  and is not a global stop under pre-reg §1/§8.1: the executor attestation records the
  affected slugs, and every affected BENCH profile carries the frozen feature-version-delta
  caveat. T0 hash or resolved-model drift remains an abort.
- The work order requests limit-8 builder checks, while the pre-reg forbids BENCH draws
  before the phase gate. The preview path is authored but must not be executed until MK/Claude
  confirms the governing phase permits those draws. Preview manifests say `preview=true` and
  cannot pass the registered n=1000 launcher gate.

## Executor handoff — not run by Codex

Run from `/Users/msrk/Documents/commit-confluence` in the T0 environment. Stop on an
incomplete sentinel, code/hash failure, or unit-check failure. A completed byte-parity
mismatch is inspected and attested as a feature-version delta per pre-reg §1/§8.1; it is
not mislabeled as a parity pass.

```bash
git diff --check
python -m py_compile confluence_calibrator.py stage_b/generate_bench_data.py stage_b/check_fresh_data.py stage_b/run_bench.py stage_b/analyze_universality.py
python stage_b/run_bench.py --phase1-unit-checks
git diff -- confluence_calibrator.py
python stage_b/run_bench.py --out-dir stage_b/profiles_bench --emit-manifest
python stage_b/run_bench.py --out-dir stage_b/profiles_bench --parity-sentinel
python stage_b/run_bench.py --out-dir stage_b/profiles_bench --attest-phase1 --diff-reviewed
```

After the governing phase permits real source access, exercise every builder at the
non-registered limit and gate each emitted file with `--expect-n 8 --length-cap 2048
--require-stem-ids`, passing all four enumerated `--sealed` references:

```bash
python stage_b/generate_bench_data.py --out-dir stage_b/data_bench_preview --raw-dir stage_b/data_bench_preview/raw_halueval --preview-limit 8
python stage_b/check_fresh_data.py --fresh stage_b/data_bench_preview/halueval_qa_seed20260711_n8.jsonl --task halueval_qa --expect-n 8 --length-cap 2048 --require-stem-ids --sealed "${CONFLUENCE_T0_REPO:-$HOME/Documents/t0-morphology-furnace}/experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl" --sealed "${CONFLUENCE_T0_REPO:-$HOME/Documents/t0-morphology-furnace}/experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl" --sealed stage_b/data/anli_R1_seed20260612_n200.jsonl --sealed stage_b/data/triviaqa_paired_seed20260612_n200.jsonl
```

Repeat the gate command for the other five preview JSONLs using their registered task keys.
The executor should add focused synthetic/adversarial checks for: duplicated/missing stems,
one-arm stem mismatch, unique-stem row parity, paired multiplicity, wrong fusion aliases,
noncanonical commitment tokens, stale manifest/runtime/model/data/panel/stem/control fields,
and missing-smoke `UNRUN` behavior.

After Phase 2 produces and gates the full confirmatory files, the confirmatory Phase-3/4
commands are:

```bash
python stage_b/run_bench.py --phase smoke --task halueval_qa=stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl --task anli_r1_rep=stage_b/data_bench/anli_r1_rep_seed20260711_n1000.jsonl --task triviaqa_paired_rep=stage_b/data_bench/triviaqa_paired_rep_seed20260711_n1000.jsonl
python stage_b/run_bench.py --phase strict --resume --task halueval_qa=stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl --task anli_r1_rep=stage_b/data_bench/anli_r1_rep_seed20260711_n1000.jsonl --task triviaqa_paired_rep=stage_b/data_bench/triviaqa_paired_rep_seed20260711_n1000.jsonl
python stage_b/analyze_universality.py --profiles-dir stage_b/profiles_bench --bench-a2 --skip lomo,transfer,labeleff --out stage_b/profiles_bench/A2_REGISTERED.json
```

Add C-family `--task` arguments only after reading their actual filenames/n from the data
manifests; the launcher preserves A/B-before-C order regardless of CLI order. Phases 2–5
remain executor-owned and were not started by Codex.
