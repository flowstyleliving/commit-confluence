# Provenance — the GPU (torch) extraction path

## What is here, and why it is a verbatim copy

`modal_app.py` is the **actual** PyTorch extractor that produced the large-model cells discussed
in the companion paper (Qwen2.5-32B/72B, Llama-3.3-70B, and the precision ladder). It is vendored
**byte-identical** from the working repository that ran it:

| | |
|---|---|
| Source | `furnace-guard` @ `0f00233^` (the commit before the guard repo was de-clouded) |
| `modal_app.py` sha256 | `30bef965b305465efe840c307ef4e4f2e20763fbd8e3c62870c5a5b3bb8065dd` |
| Vendored | 2026-07-14, unmodified |

Until 2026-07-14 this directory held a **placeholder** whose `extract()` raised
`NotImplementedError`, while the top-level README described it as the extractor used for the
large-model results. That was wrong, and it is what this file corrects. The real extractor was
never lost — only stranded in a repo that had moved on.

## ⚠️ The calibrator here is NOT the calibrator at repo HEAD

This is the single most important fact on this page.

| | sha256 (first 16) |
|---|---|
| `modal/seal/confluence_calibrator.py` — **what actually ran** | `6142217f7608dc7c` |
| `/confluence_calibrator.py` — repo HEAD, BENCH-amended | `c79009a3adaf57c6` |

The GPU cells were extracted and calibrated during 2026-06-21…06-25. `confluence_calibrator.py`
was **later** amended for the BENCH extension (+268 lines; see `calibrator_diff` in
`stage_b/profiles_bench/EXTENSION_MANIFEST.json`), and that amended version is now load-bearing
for BENCH.

Therefore `modal/seal/` pins the calibrator **version that ran**, not the current one. Repointing
this mount at repo HEAD would silently produce numbers that do not match the published GPU cells.
The duplication is deliberate: it is not a stale copy, it is a *different version*, and it is the
only honest way to keep these results reproducible.

The other six vendored seal modules (`pri_calibrator`, `pri_runtime`, `pri_v2_io_plugins`,
`comprehensive_run`, `diagnose_inter_head_disagreement`, `test_shadow_ambiguity`) **are**
byte-identical to their counterparts under `vendor/t0_core/`. Verified 2026-07-14.

## Comparability — read before citing any number from this path

Every cell produced here is **NON-byte-comparable** to the sealed 18/20:

- **Framework.** torch + HF transformers on NVIDIA, not MLX on Apple silicon.
- **Numerics.** `bitsandbytes` nf4 / int8 / bf16 / fp32, not MLX 4-bit.
- **Capture.** `attn_implementation="eager"`, so ACE reads the model's own softmax weights
  directly rather than the seal's recompute.

These cells are **never pooled** with the sealed or byte-comparable cells. They appear in the
paper as a standalone out-of-stack extension, and the paper says so explicitly
(§"scale and generation close the orphan", and the daggered rows of Table 2).

What they are good for: they close both sealed ANLI orphans as small-model artifacts across two
independent families, and they surface the family-dependent signal *locus* (attention-side for
Qwen, readout-side for Llama). What they are not: evidence that may be averaged into, compared
against, or substituted for a sealed cell.

## Faithfulness gates the extractor enforces on itself

- **`o_proj` reconstruction check** — the captured attention must reconstruct the model's own
  output projection to cos ≈ 1.0, or the run is rejected.
- **Architecture guard** — `modal_app.py:239` *refuses* (raises) on attention modules carrying
  `v_norm` or `use_k_eq_v` rather than emitting silently-wrong value norms. A fail-closed
  refusal, not a placeholder.
- **Commit gate** — `validate()` asserts the committed token is YES/NO before extraction proceeds.
- **D0/D1 split preserved** — the off-by-one fix from the gemma-4 port is carried through here.

## Running it

Requires a Modal account and GPU spend; it is not part of the from-the-repo-alone analysis path.
Nothing in the registered endpoint verification or the descriptive analyses depends on it — those
run from the published matrices with no models and no GPUs. See `modal/README.md`.
