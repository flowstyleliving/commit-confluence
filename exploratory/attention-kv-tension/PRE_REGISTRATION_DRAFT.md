# Attention KV-Tension Pre-Registration Draft

**Status:** `[OPEN - IMPLEMENTED CONTRACTS, NOT RUN]`  
**Date:** 2026-06-09  
**Lane:** ACE follow-up / `W_u`-free attention morphology

## Question

Does the ACE attention signal live in disagreement among query heads, or in
disagreement among the shared key-value groups those query heads read from?

Existing ACE already exposes:

- `js`: disagreement across all query heads.
- `js_kv_groups`: collapse query heads that share a KV group, then measure
  disagreement across KV groups.

This lane adds the missing decomposition: tension **inside** each shared KV
group versus tension **between** KV groups.

## New cells

Implemented behind `pri_calibrator.py --attention-kv-tension`; not part of the
sealed ACE default panel.

| Metric | Meaning |
|---|---|
| `js_within_kv_groups` | mean JS-radius among query heads that share each KV group |
| `js_within_kv_groups_no_bos` | same, after dropping BOS and renormalizing |
| `js_kv_tension_gap` | raw `js` minus collapsed `js_kv_groups` |
| `js_kv_tension_ratio` | `js_kv_groups / (js_within_kv_groups + eps)` for GQA only |

Pinned convention: when `n_q == n_kv` (MHA), there is no within-group degree of
freedom. `js_within_kv_groups` returns `0.0`; `js_kv_tension_ratio` is undefined
and returns no score.

## Stage 0 - sealed-profile audit

Read-only audit of the existing t0 sealed ACE profiles:

- `experiments/t0-sealed/2026-05-26/profiles/anli/*.profile.json`
- `experiments/t0-sealed/2026-05-26/profiles/triviaqa/*.profile.json`

Question: does collapsed `js_kv_groups` already beat raw `js` often enough to
justify implementing the decomposition?

Result:

| Quantity | Value |
|---|---:|
| layer cells audited | 54 |
| mean absolute-orientation delta, `js_kv_groups - js` | -0.013 |
| cells with delta >= +0.03 | 13/54 |
| cells with delta >= +0.05 | 8/54 |

Largest positive pockets:

| Task | Model | Layer | Delta | Raw `js` | `js_kv_groups` |
|---|---|---|---:|---:|---:|
| TriviaQA | Llama-3.2-3B | mid | +0.1448 | 0.5644 | 0.7092 |
| TriviaQA | Qwen3-8B | final | +0.1104 | 0.6476 | 0.7580 |
| TriviaQA | Qwen3-8B | last_minus_1 | +0.1100 | 0.5616 | 0.6716 |
| TriviaQA | Phi-4-mini | last_minus_1 | +0.0952 | 0.6240 | 0.7192 |
| TriviaQA | Phi-4-mini | mid | +0.0768 | 0.7624 | 0.8392 |
| ANLI | Qwen2.5-7B | last_minus_1 | +0.0675 | 0.6281 | 0.6956 |

Verdict: warm but scoped. The mean is slightly negative, so this is not a
universal improvement claim. The pockets are large enough to justify the
within/between decomposition as an exploratory panel.

## Stage 1 - implementation contracts

Fast tests added in `tests/test_attention_cells.py`:

- identical heads -> zero disagreement
- heads fight inside each KV group -> high `within`, low `between`
- KV groups differ while heads inside each group agree -> high ratio
- MHA has no within-group degree of freedom
- BOS-driven within-group tension is removed by the no-BOS variant

Verification:

```bash
.venv/bin/python -m pytest tests/test_attention_cells.py -q -m "not slow"
```

Current result: `57 passed, 2 deselected`.

## Stage 2 - smoke run

First smoke target: Qwen2.5-7B on ANLI R1, because the sealed-profile audit
showed a `last_minus_1` `js_kv_groups` pocket there.

Smoke command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --data experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl \
  --out exploratory/attention-kv-tension/smoke/Qwen2.5-7B-Instruct-4bit.profile.json \
  --task-label kv_tension_smoke_anli_r1_20260609 \
  --t0-commit \
  --attention-kv-tension \
  --attention-only \
  --n-bootstrap 200
```

Smoke is a plumbing check only. It reuses sealed data and a lower bootstrap
count, so it cannot validate the claim.

## Stage 3 - pilot gate

If smoke is sane, run a fresh pilot on:

- Qwen2.5-7B
- Qwen3-8B
- Mistral-7B
- Gemma-3-4B
- Phi-4-mini or Phi-3.5-mini as the MHA/control-ish contrast

Promotion bar:

- one of the KV-tension cells adds at least `+0.03` AUROC over the best existing
  ACE routing comparator on at least `2/5` models, and
- the result is OOB-clean (`CI_lo > 0.50`, no severe coverage warning), and
- shuffled-label control is flat.

Falsification bar:

- no KV-tension cell beats the existing ACE routing comparator by `+0.02` on any
  OOB-clean model, or
- all apparent wins collapse to BOS/sink artifacts under the no-BOS variant.

## Live read

This is a better handle than last-query V-norms because it tests structure, not
loudness. Expected outcomes:

- `js_within_kv_groups` wins: local query-head committee tension.
- `js_kv_groups` / high ratio wins: memory-group split.
- all fail: existing ACE routing cells were already enough.
