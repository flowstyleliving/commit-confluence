# Stage B build + cross-check result (2026-06-10)

## What was built

`confluence_calibrator.py` (repo root) - the unified dispatcher, by **composition**:
- imports the sealed `_nested_bootstrap_oob_auroc` / `_score_candidate` / `_cell_label` from
  `t0-morphology-furnace/pri_calibrator.py` **read-only** (no edits to the frozen ACE/T0 core);
- `load_readout_matrix()` pulls per-sample `{fisher_eff_rank, spectral_entropy, neg_shadow_logvol_r1,
  null_ratio_post_rank1, surprise, p_max}` + labels from an RPV comprehensive run (null_ratio sourced
  here only, R3);
- `run_selection()` hands a merged score matrix to the sealed nested-OOB selector and returns the
  honest OOB CI + winner + per-cell full-sample marginals;
- `calibrate_cell()` emits BOTH endpoints (PRIMARY incl. confidence; SECONDARY geometric-only).

The ACE attention-pass collection (12 `ATTENTION_PANEL_T0` cells) is the remaining wire-in for the
fresh run; the selection/merge/emit spine is done and verified.

## Cross-check (no model run; sealed rows only) - PASS

`stage_b/cross_check.py` -> `stage_b/out_cross_check.json`.

| cell | PRIMARY (incl conf) | GEOMETRIC-only | fisher marginal mine / theirs |
|---|---|---|---|
| Qwen3-8B / anli | fisher_eff_rank lo=0.77 Y | fisher_eff_rank lo=0.77 Y | 0.8508 / 0.8508 |
| Mistral-7B / anli | null_ratio lo=0.67 Y | null_ratio lo=0.67 Y | 0.7703 / 0.7703 |
| Phi-3.5 / anli | null_ratio lo=0.73 Y | null_ratio lo=0.73 Y | 0.7411 / 0.7411 |
| Qwen2.5-7B / anli | null_ratio lo=0.74 Y | null_ratio lo=0.74 Y | 0.6787 / 0.6787 |
| Llama-3.1-8B / triviaqa | fisher_eff_rank lo=0.71 Y | fisher_eff_rank lo=0.76 Y | 0.8756 / 0.8756 |
| gemma-3-4b / anli | surprise lo=0.45 **n** | spectral_entropy lo=0.41 **n** | 0.5583 / 0.5583 |

Assertions (all PASS):
- Qwen3-8B/anli geom winner is RPV (`fisher_eff_rank`), not null_ratio - the backstop relationship.
- Phi-3.5/anli geom winner is `null_ratio` - v3 wins where it is strongest.
- gemma-3-4b/anli geometric-only NOT deployable - the documented gap.
- gemma-3-4b/anli NOT deployable even WITH confidence under nested-OOB - see finding below.
- all `fisher_eff_rank` full-sample marginals match the comprehensive run within 0.05 (exact to 4 dp).

## Finding: nested-OOB is stricter than the Stage-A marginal screen

Stage A judged deployability with a single-cell marginal bootstrap (surprise CI_lo 0.55 -> deployable
on gemma-3-4b/anli). The Stage B standard - nested-OOB, which re-selects the panel inside each in-bag
resample and scores out-of-bag - gives the selected winner CI_lo 0.45 on that cell. The gap is the
in-sample-vs-OOB selection penalty the methodology exists to expose. Consequence: `gemma-3-4b/anli`
is reclassified from "surprise-backstopped" to a **genuine orphan under honest selection** - the
allowed <= 1/20 PRIMARY failure. On the other 5 cross-checked cells the nested-OOB CI_lo sits well
above 0.50 (0.67-0.77), so the strictness bites only at the already-weak margin.

## Status

- [x] selector imported read-only; merge + dual-endpoint emit; marginal reproduction exact.
- [ ] ACE attention-pass collection wired into the fresh run.
- [ ] fresh-seed sealed run over the 10-model cohort (compute step).
