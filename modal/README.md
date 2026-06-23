# commit-confluence on Modal — scaling to larger (NVIDIA) models

## Why
Local MLX caps out around 12–14B on this Mac (the gemma-4-12B run swap-thrashed for ~7.5h).
Modal gives on-demand A100/H100 GPUs to reach 30B / 70B+ comfortably.

## The one architectural fact that drives everything
The sealed pipeline is **MLX** (Apple-Silicon only). Modal runs **NVIDIA** GPUs, so model
forwards must be **PyTorch / HF transformers**. The split that saved us on gemma-4 applies again:

- **Extraction** (model forward + attention capture + hidden states + logits + W_u) — framework
  specific. On Modal this is a *new PyTorch backend*, a sibling of the mlx-lm seal extractor and
  the mlx-vlm gemma-4 extractor.
- **Calibration** (panel sweep, nested-OOB selection, merge, controls) — pure numpy/sklearn,
  **fully portable**. Runs unchanged on Modal or locally on the returned matrix.npz.

### ⚠️ Comparability
Any Modal result is **NON-byte-comparable** to the seal (different framework + dtype, e.g. bf16
or bnb-4bit vs MLX-4bit). Treat every Modal cell like the gemma-4 cell: a standalone exploratory
panel, **never pooled** with the sealed or byte-comparable cells. Report with the caveat inline.

## One-time setup
```bash
pip install modal
modal setup                                                   # opens browser, writes ~/.modal.toml
modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)   # for gated Llama/Gemma
```

## Smoke test (proves infra end-to-end)
```bash
modal run modal/modal_app.py                                  # Qwen2.5-14B default
modal run modal/modal_app.py --model-id meta-llama/Llama-3.3-70B-Instruct
```
Reports n_layers / hidden / vocab / heads / kv_heads / dtype / GPU — and warms the weight cache.

## GPU sizing (bf16; halve roughly for 4-bit)
| Model | bf16 weights | fits on |
|---|---|---|
| 14B | ~28 GB | A100-40GB |
| 30–34B | ~65 GB | A100-80GB |
| 70B | ~140 GB | A100-80GB:2 (2 GPUs) or single 80GB at 4-bit (bitsandbytes) |
Cost ballpark: A100-80GB ≈ a few $/hr on Modal; an n=200×2-task extraction is well under an hour
of GPU once weights are cached — so single-digit dollars per model, not the 7.5h local crawl.

## Extractor port — the real work (in `modal_app.py::extract`, currently a stub)
Mirror `stage_b/gemma4_full_extract.py`. Per tagged layer `{final, mid, last_minus_1}` (indices
from `diagnose_inter_head_disagreement._target_layer_map(n_layers)`), a forward hook on the
attention module recomputes, for the **last query position**:
- `caps[tag]`  = `softmax(q·kᵀ·scale + causal_mask)` → `[H, T]`   (post q/k-norm + RoPE; eager attn)
- `vcaps[tag]` = per-KV-head value L2 norms → `[n_kv, T]`
- `nkv[tag]`   = int

plus logits → surprise / p_max / gid (argmax commit), hidden states `h_prev` (D0 = prompt-last) and
`h_t` (D1 = forward on `[prompt + gid]`) — **keep the D0/D1 off-by-one fix** — and `W_u` rows via
`model.get_output_embeddings().weight` for the support spectrum.

**Validation gate (do before trusting any cell):** recompute o_proj from the captured weights and
check `cos(my_attn_out, model_attn_out) ≈ 1.0`, exactly as `G4Wrap` did (it hit 1.0000). Without
this, the attention statistics can be silently wrong.

Then feed the *existing* `pri_calibrator.ATTENTION_PANEL_T0_WITH_V_NORMS` +
`_compute_attention_score` + `confluence_calibrator.{merge_matrices, calibrate_merged}` (seed
20260612, nboot 2000) — identical to the seal. Emit the same `matrix.npz` schema.

## Prompt format reminder (gemma-4 lesson)
Instruction-tuned models may need their chat template to actually attempt the YES/NO task — a raw
prompt made gemma-4 just continue the question (≈0.37 noise). Use
`tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True)` and sanity-check
the commit token is YES/NO before a full run.
