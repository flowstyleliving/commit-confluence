# PRE-REGISTRATION — Per-layer depth curve of attention-morphology separation (2026-08-16)

**Status: EXPLORATORY / DESCRIPTIVE lane. Frozen before any extraction runs.**
This is not a promotion gate and touches no sealed claim. All cells are Modal/torch,
NON-byte-comparable — never pooled with sealed/byte-comparable cells.

## Question

The three-rung panel (mid = N//2, N−2, N−1) showed (a) Qwen ≥32B separation ≈chance at
mid, peaking at N−2, dipping at N−1; (b) Llama-3.3-70B attention mild-at-mid, fading,
signal post-stack; (c) rung flatness/instability at 3–14B (`wiki/results/depth-marginals-2026-08-16`).
Three rungs cannot answer:
1. **Peak placement law** — does the peak track **absolute** offset from the end
   (N − ℓ* ≈ const) or **relative** depth (ℓ*/N ≈ const)? (The transfer question:
   a relative law would generalize to unprobed model sizes.)
2. **Rise shape** — is the climb from chance to peak **gradual** or a **cliff**?
3. **Early layers** — does any block below 0.4N separate at all? (Never measured.)

## Vehicles and data (frozen)

- Models (all nf4, single A100-80GB, `attn_implementation="eager"`, chat template):
  Qwen2.5-7B-Instruct (N=28), Qwen2.5-32B-Instruct (N=64), Qwen2.5-72B-Instruct (N=80);
  **negative-control family:** Llama-3.3-70B-Instruct (N=80).
- Tasks (existing volume files, reused verbatim):
  - `data/anli_r1_n200.jsonl` — sha256 `57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b`
  - `data/halueval_qa_n200.jsonl` — sha256 `a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0`
    (stress-panel manifest, seed 20260612, 100/100 balanced)
- Grid: 4 models × 2 tasks = 8 extraction cells, n=200 each.

## Measurement (frozen)

Per prompt: one t=0 forward on the chat-templated prompt (`use_cache=False`,
`output_attentions=True`). For **every decoder block ℓ**, read the last-query attention
row `attentions[ℓ][0, :, −1, :]` and compute, via the **sealed kernel**
`pri_calibrator._compute_attention_score` (per-layer weights presented under tag
`final`; `n_kv_heads` from config; `v_norm_captures=None`):

- `js_no_bos` — **primary metric** (BOS-sink-controlled inter-head JS radius)
- `js`, `js_kv_groups`, `bos_mass` — secondary, descriptive

Output: per (model, task) an npz of scores `[n_kept, N_layers, 4]` + labels + commit
token ids. **The extractor prints gates and progress only — no AUROC of any kind.**

Faithfulness gates (all fail-closed, mirroring the June torch runs):
- rows 0–1: o_proj reconstruction cos ≥ 0.999 at the final block; commit token is YES/NO
- whole run: YES/NO commit rate ≥ 0.5; `max_dropped = 0` (any per-row failure aborts)
- attention tensor shape guards per block (H = num_attention_heads, T×T, non-None)

## Statistics (frozen; local, code committed before any results are inspected)

Per (model, task, metric, block): **sign-free AUROC** `a = max(auc, 1−auc)` (sign is a
free parameter per block, consistent with the three-rung marginals methodology).
- Bootstrap: 1000 row-resamples, seed **20260816**; sign refit inside each resample.
- Shuffled-label envelope: 200 label permutations, seed **20260816**, same sign-free
  statistic per block → per-block 97.5th-percentile envelope. (The envelope inherits
  the max-side selection bias, so clearing it is meaningful.)
- **Qualifying peak** (per model, task; primary metric only): ℓ* = argmax block with
  AUROC ≥ 0.65 AND above the shuffled envelope. If no block qualifies → "no peak".
  Bootstrap distribution of ℓ* reported as median + [5%, 95%].

## Endpoints and decision rules (frozen)

**E1 — peak placement law** (primary; Qwen trio, per task; requires a qualifying peak
in ≥2 of the 3 sizes; medians of the bootstrap ℓ* distributions):
- **ABSOLUTE-supported**: span across sizes of median(N − ℓ*) ≤ 2 blocks AND span of
  median(ℓ*/N) ≥ 0.03.
- **RELATIVE-supported**: span of median(ℓ*/N) ≤ 0.015 AND span of median(N − ℓ*) ≥ 3.
- otherwise **UNDECIDED**.
Overall: same outcome on both tasks → that verdict; else MIXED (report per-task).
(Separation check: at N = 28/64/80, "always N−2" gives fraction span 0.046 → ABSOLUTE;
"always the 7B fraction 0.929" gives N−ℓ* ≈ 2/4.5/5.7, span 3.7 → RELATIVE. The rules
cannot both fire.)

**E2 — rise shape** (per qualifying Qwen cell, primary metric): baseline b = median
AUROC over blocks in [0.4N, 0.6N]; rise R = AUROC(ℓ*) − b; J = max adjacent-block jump
on [0.5N, ℓ*]. **CLIFF** if J ≥ 0.15 AND J ≥ 0.5·R; **GRADUAL** if J ≤ 0.08; else MIXED.

**E3 — early layers** (descriptive): any block ℓ < 0.4N with AUROC ≥ 0.60 AND above the
shuffled envelope.

**E4 — terminal dip** (per Qwen cell): fires if ℓ* ≤ N−2 AND AUROC(N−1) ≤ AUROC(ℓ*) − 0.05.

## Predictions (frozen)

- **P1**: every Qwen cell (3 sizes × 2 tasks) has a qualifying peak with ℓ* ≥ N−4.
- **P2**: none — E1 is the open question; the three-rung data is consistent with both laws.
- **P3** (negative control): Llama-3.3-70B has **no qualifying peak** on either task
  (no block reaches 0.75; expected max ≈ 0.6–0.7 in the mid region).
- **P4**: Qwen mid-region [0.4N, 0.6N] median AUROC < 0.60 at 32B and 72B (both tasks).
- **P5**: E4 (terminal dip) fires at 32B and 72B on both tasks.

## Discipline

- This file is committed before any extraction is launched; `depth_score.py` is
  committed before any AUROC is computed or inspected.
- The smoke run (Qwen2.5-7B / anli_r1) may be inspected for **gates, shapes, and
  finiteness only** — never scores.
- No AUROC/curve inspection until all 8 extraction cells are complete; then scoring is
  one execution of `depth_score.py` on all cells at once.
- Misses are reported as written. Amendments require a new dated section here, never an
  in-place edit.

## Cost/infra note

`output_attentions=True` already materializes all blocks' weights in the existing June
extractor; this lane reads them all instead of three, and needs **no commit forward**
(t=0 only) — per-cell cost ≈ half a June extract() cell.
