# gemma-4-12B extension — build spec (option B: mlx-vlm reimplementation)

Status 2026-06-18: feasibility PROVEN; extractor implementation in progress. Parallel venv only;
**no sealed-core edits**; output carries a version+reimpl delta caveat (NOT byte-comparable to the
seal). Pre-reg: `PRE_REGISTRATION_EXT.md` §Phase 2.

## Confirmed (probes)
- `mlx-vlm` loads `gemma4_unified`; `m.language_model` = 48-layer text decoder. (mlx-lm cannot.)
- Forward with `m(x, capture_layer_ids=range(48), return_hidden=True)` → `out.logits (1,T,262144)`
  + `out.hidden_states` = 48 × `(1,T,3840)`. Verified on a chat-templated prompt.
- Unembedding (tied): `lm.logits_from_hidden(h) = embed_tokens.as_linear(h)` then
  `logit_softcap(final_logit_softcapping=30, ·)`. So `W_u` = `embed_tokens` (quantized), softcapped.
- Tokenizer: `apply_chat_template(..., tokenize=True)` returns a **BatchEncoding** → take `["input_ids"]`.
- Head config (confirmed via load): gemma-3-4b = 8 heads / 4 KV / head_dim 256; gemma-3-12b = 16/8/256;
  gemma-4-12b text = per-layer-type (sliding head_dim 256 / global 512; KV 8 sliding / 1 global),
  q_norm/k_norm, `use_k_eq_v` (full layers), partial-RoPE(0.25) proportional, 5:1 sliding:full, KV-sharing.

## Architecture: extraction (gemma4 venv) ⟂ calibration (seal venv)
The gemma-4 extractor only needs to emit a `matrix.npz` in the seal format
(`score_matrix [N,27]`, `labels`, `sample_idx`, `panel`, `meta`). Calibration then runs in the SEAL
venv via the *identical* `confluence_calibrator.calibrate_merged` — so calibration is byte-identical
to the seal; only signal *extraction* is reimplemented. Panel column order MUST match the seal's
`panel` exactly (copy it from a sealed matrix).

## Readout signals (6) — gen_step=1, faithful to `trace_sample`/`trace_pair_features`
Commit instant = **gen_step 1** (generate one token; readout at that step). Per example:
1. **[CORRECTED 2026-06-18 — confirmed `pri_runtime.py:985-1118`; Codex adversarial review caught
   an off-by-one in the WIP]** TWO forwards. Forward A on prompt → `D0 = softmax(logits[last])`;
   `gen_id = argmax D0`; **`surprise = -log D0[gen_id]`** (from the prompt-last dist). Forward B on
   `[prompt + gen_id]` → **`p_t = D1 = softmax(logits[last] of B)`** = the seal's `gen_probs[0]` (the
   POST-commit step-1 dist); **`p_max = max D1`**. `h_prev` = final-layer hidden at prompt-last
   (forward A); `h_t` = final-layer hidden at the `gen_id` position (forward B); `dh = h_t - h_prev`
   handled INTERNALLY by `compute_step`. WIP bug: used `D0` for `p_t`/`p_max`. Validate via mlx-lm
   transitive parity on gemma-3-12b (Codex rec) before trusting.
2. Re-forward `[prompt + gen_id]` with `capture_layer_ids` to get per-layer hidden at the new
   (commit) position; `pinned late layers` = final 25% block (`_pinned_late_layers(n_layers)`).
3. **Logit-lens per pinned layer**: `p_l = softmax(softcap(embed_tokens.as_linear(final_norm(h_l))))`;
   centered softmax-Fisher spectrum `fc_full_spectrum(W_s, p_l)` averaged over the block →
   `fisher_eff_rank`, `spectral_entropy` (normalized), `neg_shadow_logvol_r1` (= −shadow_logvol_post_rank(·,r=1)).
4. **null_ratio_post_rank1** = ‖dh − VᵀV dh‖/‖dh‖, V = top-1 right singular vec of `√p_t · W_u`
   (dh = residual update at commit; match `null_ratio_and_energy`).
Reuse the seal's NUMPY stat fns (reimplement identically, then parity-check): `fisher_eff_rank`,
`fisher_spectral_entropy`, `shadow_logvol_post_rank`, `centered_fisher` (test_shadow_ambiguity);
`null_ratio_and_energy` (pri_runtime). Final RMSNorm γ from `lm.model.norm`.

## ACE signals (21) — attention-weight recompute (the hard, novel-arch part)
Layers probed: `final`, `mid`, `last_minus_1`. 7 stats × 3 layers. Patch each probed layer's
`self_attn.__call__` to ALSO return `A = softmax(scale · q·kᵀ + mask)` using the SAME post-q_norm /
post-k_norm / post-RoPE q,k it builds (and `shared_kv` when `is_kv_shared_layer`). Then feed A (+ value
norms) to the seal stat fns: `_js_radius`, `_js_radius_no_bos`, `_js_radius_kv_groups`,
`_mean_bos_mass`, `_lastq_weighted_v_norm` (+ v_norm_bos, v_norm_max) from
`scripts/diagnose_inter_head_disagreement.py`. Handle: dual head_dim (256/512), per-layer n_kv (GQA
repeat), `use_k_eq_v`, partial-RoPE on full layers, sliding-vs-full mask. **gemma-3 does NOT exercise
these novel bits → the gemma-3-12b parity check won't cover them; eyeball A row-sums≈1, finite.**

## Validation gate (before trusting gemma-4)
Run the SAME extractor on **gemma-3-12b via mlx-vlm**, compare its 27-col matrix to the seal-venv
gemma-3-12b matrix (`profiles_ext/.../gemma-3-12b-it-4bit.matrix.npz`). Require per-column rank
correlation high / AUROC within tolerance. (Cross-impl numerics differ slightly: mlx-vlm vs mlx-lm,
newer mlx — so expect close-but-not-identical; large divergence = bug.)

## Then
Run gemma-4 (anli_r1 + triviaqa_paired, same data files, n=200) → matrix.npz → seal-venv
`calibrate_merged` → `profiles_ext_gemma4/`. Report with the caveat. Update results page + decision
table (generation axis). Predictions already frozen (Phase 2): anli ~50%, trivia ~75% deployable.

## Open risks
- ACE attention recompute correctness on the novel attention (biggest risk; not covered by gemma-3 parity).
- KV-sharing: probed layer may have no own k_proj (keys from a source layer) — must capture the actual keys used.
- dh definition for null_ratio must match `null_ratio_and_energy` exactly.
- Mixed 4/8-bit quant + softcap numerics.
