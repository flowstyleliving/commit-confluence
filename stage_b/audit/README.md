# Executed end-to-end audit (v2, fail-closed — 2026-08-12)

Spot verification of the sealed seed-20260612 run by *execution*, not inspection —
complementing the read-only reviews and the statistical re-derivation in
`stage_b/verify_endpoints.py`. v2 hardens the v1 scripts after a gpt-5.6-sol
adversarial audit found them fail-open, provenance-unbound, and loosely specified
(v1 preserved in git history at `e8e4db2`; its executed results are restated below).

## Scripts

- `trace_cc_example.py` — fail-closed hand-trace (numpy/json/hashlib only; no
  calibrator, no sklearn). Verifies the data-file sha256 against the profile stamp
  and full matrix↔jsonl label alignment, derives the winner column from the
  profile's exact winner string, recomputes its full-sample AUROC via a from-scratch
  Mann-Whitney formula, and REQUIRES agreement with the profile (4-dp + sign).
  Nonzero exit on any failed check. Non-gating exhibit: the most-flagged example
  (for Qwen2.5-7B/anli_r1: jsonl row 168, gold-entailed — the rank-1 flag is a
  false positive; a ranking endpoint at AUROC ~0.79 claims no per-example verdicts).
- `verify_cell.py` — fail-closed re-extraction verifier. Re-runs real MLX forwards
  on selected rows and requires float64 BYTE equality with the committed matrix,
  with asserted comparison counts, exact panel-triple identity, finiteness, model
  snapshot-revision match (always fatal on drift), and per-module code-hash
  comparison against the profile's recorded `module_hashes` (fatal unless
  `--acknowledge-code-drift`, which records the drift loudly instead). Writes a
  machine-readable JSON run record to `runs/` for every run, pass or fail.

## Scope honesty

These are *sparse spot checks plus one stored-column arithmetic trace* — they
demonstrate reproducibility and artifact integrity for the compared values; they do
not re-derive the nested-OOB selection, cover all 20 cells, re-extract fusion
columns, or detect a bug shared by the original and audit executions (both use the
production extraction functions). Readout rows are prefix-only because the readout
extractor derives per-row RNG from the run row index (`seed + 100_003 + i`).

## Provenance note (resolved 2026-08-12)

The sealed profiles' `module_hashes` match the **vendored seal-time copies** in
`vendor/t0_core/`, not the current `t0-morphology-furnace` checkout: the tag→seal
delta is the additive, opt-in KV-tension code (in-place edits present 2026-06-08 →
reverted 2026-07-25; documented in the vault log). Re-extraction under the *tag*
version reproduced sealed values bit-exactly, empirically confirming the delta was
extraction-inert for the sealed panel on all compared values. To re-extract under
byte-exact seal-time code, run with `CONFLUENCE_T0_REPO=<repo>/vendor/t0_core`.
`confluence_calibrator.py` itself has post-seal (BENCH) drift and requires the
explicit acknowledgment flag.

## Re-run

    PYTHONDONTWRITEBYTECODE=1 CONFLUENCE_T0_REPO=$PWD/vendor/t0_core \
      <t0-repo>/.venv/bin/python stage_b/audit/verify_cell.py \
      --task anli_r1 --model Qwen2.5-7B-Instruct-4bit \
      --ace-rows 168,21,0,100 --readout-prefix 4 --acknowledge-code-drift

`PYTHONDONTWRITEBYTECODE=1` avoids writing `__pycache__` into the sealed
dependency tree. Executed run records live in `runs/` (JSON, one per run).

## Executed results

- v1 (2026-08-12, fail-open scripts, `e8e4db2`, operator-witnessed): Qwen2.5-7B/anli_r1
  attention rows {168,21,0,100} 84/84 exact; Llama-3.2-3B/triviaqa_paired attention
  rows {7,42,123,199} 84/84 exact + readout prefix rows 0-3 24/24 exact (mlx_lm 0.29.1).
- v2: see the JSON records in `runs/`.
