# Spot-check audit, fail-closed (v2.1 — 2026-08-12/13)

Sparse spot verification of the sealed seed-20260612 run by *execution* —
re-running real model forwards on sampled rows plus one calibrator-independent
arithmetic trace. Complements the read-only reviews and the statistical
re-derivation in `stage_b/verify_endpoints.py`. Hardened across two gpt-5.6-sol
adversarial audit rounds (fail-open v1 at `e8e4db2`, v2 at `7edafee`; both
preserved in git history).

## Scripts

- `trace_cc_example.py` — fail-closed hand-trace (numpy/json/hashlib only; no
  calibrator, no sklearn). Verifies the data-file sha256 against the profile stamp
  and full matrix↔jsonl label alignment, derives the winner column from the
  profile's exact winner string, recomputes its full-sample AUROC via a from-scratch
  Mann-Whitney formula, and REQUIRES agreement with the profile (4-dp + sign).
  Nonzero exit on any failed check. Only matrix-stored winners (Attention/Readout)
  can be traced; fusion winners fail resolution (fail-closed). Non-gating exhibit:
  the most-flagged example (for Qwen2.5-7B/anli_r1: jsonl row 168, gold-entailed —
  the rank-1 flag is a false positive; a ranking endpoint at AUROC ~0.79 claims no
  per-example verdicts).
- `verify_cell.py` — fail-closed re-extraction verifier. Requires float64
  byte-identical agreement with the committed matrix (source arrays asserted
  float64 and finite first), with asserted comparison counts against the 21+6
  protocol constants, exact panel-triple identity and uniqueness, complete
  sample_idx permutation, matrix-meta↔profile agreement, distinct nonnegative row
  args, model snapshot-revision match (always fatal on drift), and per-module code
  hashes vs the profile's `module_hashes` — drift is fatal unless waived
  HASH-PINNED via `--acknowledge-code-drift MODULE=SHA256` (the waiver forgives
  one specific known state of one file, never "whatever that module now is";
  recorded, never blanket).
  Every run that reaches or crashes out of the check flow writes a JSON record to
  `runs/` (verdict PASS / PASS_WITH_ACKNOWLEDGED_CODE_DRIFT / FAIL / ERROR); a run
  killed externally leaves no record. Records self-bind: they carry sha256 of the
  verifier file itself, the profile, the matrix npz, and the data jsonl, plus git
  heads (dirty-state excludes `runs/` outputs).

## Scope honesty

These are *sparse spot checks plus one stored-column arithmetic trace*. They
demonstrate numerical reproduction of the sampled values under recorded
conditions; they do not re-derive the nested-OOB selection, cover all 20 cells,
re-extract fusion columns, or detect a bug shared by the original and audit
executions (both use the production extraction functions). Readout rows are
prefix-only because the readout extractor derives per-row RNG from the run row
index (`seed + 100_003 + i`).

Known open limitations of the provenance layer: the model-weight check compares
the resolved HF snapshot *revision pointer* under the default cache layout — it
does not hash weight shards, and a non-default HF cache location could diverge
from what MLX actually loads; dependency *versions* (numpy/mlx/mlx_lm) are
recorded but installed libraries are not hashed; run records carry no
cryptographic attestation, so they are transcripts, not proofs of execution.
`module_hashes()` itself is supplied by `confluence_calibrator.py`, which is one
of the drift-acknowledged modules — an irreducible bootstrapping caveat unless the
hashing is reimplemented independently.

## Provenance note (reconstructed and spot-corroborated — 2026-08-12/13)

The sealed profiles' `module_hashes` match the **vendored seal-time copies** in
`vendor/t0_core/`, not the current `t0-morphology-furnace` checkout. Precisely:
the current t0 checkout's sealed *root* modules are byte-identical to the
`t0-ace-sealed-2026-05-26` tag, while the readout modules
(`exploratory/shadow-ambiguity/`) postdate that tag in committed history; the
seal-time state additionally carried uncommitted, additive, opt-in KV-tension
edits (present 2026-06-08 → reverted 2026-07-25; preserved as a patch and in the
vendored copies; documented in the vault log). Corroboration is by execution:
re-extraction under the *current t0 checkout* (v1, operator-witnessed) and under
the *vendored seal-time copies* (v2/v2.1, recorded in `runs/`) both reproduce the
sealed values byte-identically on all compared cells, consistent with the
KV-tension delta being extraction-inert for the sealed panel. This is
corroboration on sampled values, not a formal proof of full-run provenance.

## Re-run

    PYTHONDONTWRITEBYTECODE=1 CONFLUENCE_T0_REPO=$PWD/vendor/t0_core \
      <t0-repo>/.venv/bin/python stage_b/audit/verify_cell.py \
      --task anli_r1 --model Qwen2.5-7B-Instruct-4bit \
      --ace-rows 168,21,0,100 --readout-prefix 4 \
      --acknowledge-code-drift confluence_calibrator.py=c79009a3adaf57c6

(The pinned hash is the sha256 prefix of the post-seal BENCH-era
`confluence_calibrator.py`; recompute with `shasum -a 256` if it has moved again —
a moved hash is exactly what the pin exists to surface.)

`PYTHONDONTWRITEBYTECODE=1` avoids writing `__pycache__` into the dependency
tree. Executed run records live in `runs/` (JSON, one per run).

## Executed results

- v1 (2026-08-12, fail-open scripts, `e8e4db2`, operator-witnessed): Qwen2.5-7B/anli_r1
  attention rows {168,21,0,100} 84/84 exact; Llama-3.2-3B/triviaqa_paired attention
  rows {7,42,123,199} 84/84 exact + readout prefix rows 0-3 24/24 exact (mlx_lm 0.29.1,
  current t0 checkout).
- v2 (2026-08-13, `7edafee`, vendored seal-time code): both cells ACE 84/84 +
  readout 24/24 byte-identical; records in `runs/` (generated pre-commit from a
  dirty tree — see their git state; superseded by the v2.1 records).
- v2.1: records in `runs/` with self-binding hashes, generated from the committed
  verifier.
