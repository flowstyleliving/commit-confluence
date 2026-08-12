# Executed end-to-end audit (2026-08-12)

Three scripts that verify the sealed seed-20260612 run's chain by *execution*, not
inspection — complementing the three prior read-only reviews and the statistical
re-derivation in `stage_b/verify_endpoints.py`.

- `trace_cc_example.py` — hand-trace of Qwen2.5-7B / anli_r1. Checks the data-file
  sha256 against the profile stamp and the matrix/jsonl label alignment, then
  recomputes the winner cell's full-sample AUROC from the raw score column with a
  from-scratch Mann-Whitney formula (numpy only — no calibrator, no sklearn).
  Result as run: 0.7896, sign -1 — exact match to the committed profile. Also
  surfaces the cell's most-flagged example (row 168), which is gold-entailed:
  the rank-1 flag is a false positive, illustrating what a ranking endpoint at
  AUROC ~0.79 does and does not claim.
- `spot_reextract.py` — fresh MLX forward passes on 4 saved prompts
  (Qwen2.5-7B / anli_r1, rows 168/21/0/100) vs the stored matrix.
  Result as run: 84/84 attention-panel values bit-exact (worst |diff| = 0.0).
- `spot_reextract_cell2.py` — second cell, different family + task
  (Llama-3.2-3B / triviaqa_paired): attention on scattered rows 7/42/123/199
  (84/84 bit-exact) plus the fresh readout pass on prefix rows 0-3 (24/24
  bit-exact). Between the two cells all 27 stored matrix columns are exercised.
  Method note: readout per-row RNG is derived from the run row index
  (`seed + 100_003 + i`), so only a *prefix* subset is RNG-aligned with the
  full run; scattered rows are valid only for the deterministic attention capture.

Re-run (extraction scripts need the sealed dependency repo + its venv; the trace
script needs only numpy):

    PYTHONDONTWRITEBYTECODE=1 CONFLUENCE_T0_REPO=<t0-repo> \
        <t0-repo>/.venv/bin/python stage_b/audit/spot_reextract.py

`PYTHONDONTWRITEBYTECODE=1` keeps the sealed t0 working tree byte-identical
(no `__pycache__` writes); it was verified pristine via `git status` before and
after both original runs (mlx_lm 0.29.1).
