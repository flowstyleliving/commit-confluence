"""Gemma-4 full ACE + readout extractor — emits 27-col seal-format matrix.npz.

Combines the fixed readout (post-commit p_t) with novel-arch ACE attention capture
for gemma-4's dual-head_dim, KV-sharing, use_k_eq_v, partial-RoPE attention.

Architecture: extraction (gemma4 venv) ⟂ calibration (seal venv).
Only signal EXTRACTION is gemma-4-specific — calibration reuses seal's calibrate_merged.

Usage:
  python gemma4_ace_extract.py <model_id> <task> <data.jsonl> <limit> <out.npz>
  python gemma4_ace_extract.py mlx-community/gemma-4-12B-it-qat-4bit anli_r1 data.jsonl 200 matrix.npz
"""
import sys, os, json
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
T0 = os.environ.get("CONFLUENCE_T0_REPO", os.path.join(REPO_ROOT, "vendor", "t0_core"))
sys.path.insert(0, T0)
sys.path.insert(0, os.path.join(T0, "exploratory/shadow-ambiguity"))
sys.path.insert(0, os.path.join(T0, "scripts"))

import pri_runtime as pipeline
import comprehensive_run as CR
import pri_v2_io_plugins as io_plugins
from diagnose_inter_head_disagreement import (
    _js_radius, _js_radius_no_bos, _js_radius_kv_groups,
    _mean_bos_mass, _attention_entropy, _lastq_weighted_v_norm,
    _mean_v_norm_bos, _mean_v_norm_max,
)
from mlx_vlm import load as vload

EPS = 1e-12


# ── column layout (must match seal panel) ─────────────────────────────
READOUT_KEYS = [
    "null_ratio_post_rank1", "fisher_eff_rank", "spectral_entropy",
    "neg_shadow_logvol_r1", "surprise", "p_max",
]

ACE_TAGS = ["final", "mid", "last_minus_1"]
ACE_STATS = [
    "js_radius", "js_radius_no_bos", "js_radius_kv_groups",
    "bos_mass", "lastq_weighted_v_norm",
    "v_norm_bos", "v_norm_max",
]


# ── ACE attention wrapper (Gemma-4 specific) ──────────────────────────
class Gemma4ACEWrapper:
    """Wraps a Gemma4 Attention module to capture softmax weights + V norms.

    Mirrors the native __call__ EXACTLY for q/k projection, norm, RoPE,
    KV-sharing, dual head_dim, use_k_eq_v, then recomputes A = softmax(scale·q·kᵀ + mask).
    Also captures per-head per-position L2 norm of value vectors.

    Returns native forward output UNCHANGED — cannot perturb generation.
    """

    def __init__(self, orig, weight_captures, v_norm_captures=None):
        self._orig = orig
        self._weights = weight_captures
        self._v_norms = v_norm_captures

    def __getattr__(self, name):
        return getattr(self._orig, name)

    def __call__(self, x, mask=None, cache=None, shared_kv=None, offset=None):
        B, L, _ = x.shape
        n_heads = int(self.n_heads)
        n_kv = int(self.n_kv_heads)
        head_dim = int(self.head_dim)

        # ── queries (exactly native path) ──
        queries = self.q_proj(x).reshape(B, L, n_heads, head_dim)
        queries = self.q_norm(queries)

        # ── offset ──
        if shared_kv is None and cache is not None:
            _offset = mx.array(cache.offset) if hasattr(cache, "offset") else (offset or 0)
        else:
            _offset = offset if offset is not None else 0

        # ── keys for softmax ──
        if shared_kv is not None:
            keys_for_weights = shared_kv[0]  # already normed, RoPE'd, transposed
        else:
            keys = self.k_proj(x).reshape(B, L, n_kv, head_dim)
            keys = self.k_norm(keys)
            keys = keys.transpose(0, 2, 1, 3)  # (B, n_kv, L, head_dim)
            keys_for_weights = self.rope(keys, offset=_offset)

        # ── V norms capture ──
        if self._v_norms is not None:
            if shared_kv is not None:
                # shared_kv[1] is values, already transposed: (B, n_kv, L, head_dim)
                vals = shared_kv[1]  # (B, n_kv, L, head_dim)
            else:
                if self.use_k_eq_v:
                    vals = self.k_proj(x).reshape(B, L, n_kv, head_dim)
                else:
                    vals = self.v_proj(x).reshape(B, L, n_kv, head_dim)
                vals = self.v_norm(vals)
                vals = vals.transpose(0, 2, 1, 3)  # (B, n_kv, L, head_dim)
            vals_f32 = vals.astype(mx.float32)
            v_norms_arr = mx.sqrt(mx.sum(vals_f32 * vals_f32, axis=-1))
            mx.eval(v_norms_arr)
            self._v_norms.append(np.array(v_norms_arr)[0])  # (n_kv, L)

        # ── RoPE queries ──
        queries = queries.transpose(0, 2, 1, 3)  # (B, n_heads, L, head_dim)
        queries = self.rope(queries, offset=_offset)

        # ── GQA repeat ──
        n_repeats = n_heads // n_kv
        if n_repeats > 1:
            keys_expanded = mx.repeat(keys_for_weights, n_repeats, axis=1)
        else:
            keys_expanded = keys_for_weights

        # ── softmax weights (fp32) ──
        q_f32 = queries.astype(mx.float32)
        k_f32 = keys_expanded.astype(mx.float32)
        scores = (q_f32 @ k_f32.transpose(0, 1, 3, 2)) * self.scale

        # Handle mask: mlx-vlm passes string masks ("causal", "sliding") for
        # scaled_dot_product_attention. Build actual mask array for softmax.
        if mask is not None:
            if isinstance(mask, str):
                q_len, kv_len = scores.shape[-2], scores.shape[-1]
                if mask == "causal" or mask == "sliding":
                    cmask = mx.triu(
                        mx.full((q_len, kv_len), float("-inf"), dtype=mx.float32),
                        k=(kv_len - q_len + 1),
                    )
                    scores = scores + cmask
            elif hasattr(mask, "dtype") and mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask.astype(mx.float32)

        weights = mx.softmax(scores, axis=-1, precise=True).astype(mx.float32)
        mx.eval(weights)
        self._weights.append(np.array(weights)[0, :, -1, :])  # (H, T_kv)

        # ── native forward (UNPERTURBED) ──
        return self._orig(x, mask, cache, shared_kv, offset)


# ── helpers ───────────────────────────────────────────────────────────
def final_layer_idx(lm):
    return len(lm.model.layers) - 1


def target_layers(lm):
    n = len(lm.model.layers)
    return {"final": n - 1, "mid": n // 2, "last_minus_1": n - 2}


def forward_capture(lm, ids, layer_idx):
    """Return (logits[last] np float64, final-layer hidden at last pos np float32)."""
    x = mx.array([ids])
    out = lm(x, capture_layer_ids=[layer_idx], return_hidden=True)
    logits = np.asarray(out.logits[0, -1].astype(mx.float32), dtype=np.float64)
    hs = out.hidden_states
    h = hs[0] if isinstance(hs, (list, tuple)) else list(hs.values())[0]
    h_last = np.asarray(h[0, -1].astype(mx.float32), dtype=np.float32)
    return logits, h_last


def ace_stat_row(weights, v_norms, n_kv_heads):
    """Compute 7 ACE stats from attention weights + V norms for one layer."""
    row = {}
    row["js_radius"] = _js_radius(weights)
    row["js_radius_no_bos"] = _js_radius_no_bos(weights)
    row["js_radius_kv_groups"] = _js_radius_kv_groups(weights, n_kv_heads)
    row["bos_mass"] = _mean_bos_mass(weights)
    row["lastq_weighted_v_norm"] = _lastq_weighted_v_norm(weights, v_norms) if v_norms is not None else np.nan
    row["v_norm_bos"] = _mean_v_norm_bos(v_norms) if v_norms is not None else np.nan
    row["v_norm_max"] = _mean_v_norm_max(v_norms) if v_norms is not None else np.nan
    return row


# ── main ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 6:
        print("Usage: python gemma4_ace_extract.py <model_id> <task> <data.jsonl> <limit> <out.npz>")
        return
    model_id, task, data_path = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(sys.argv[4])
    out_npz = sys.argv[5]

    # Load data
    prompts, labels, data_hash = CR._load_calibration_jsonl(str(data_path))
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    # Load model
    print(f"[g4-ace] Loading {model_id} ...", flush=True)
    m, proc = vload(model_id)
    tok = getattr(proc, "tokenizer", proc)
    lm = m.language_model
    layers = lm.model.layers
    fidx = final_layer_idx(lm)
    n_layers = len(layers)
    targets = target_layers(lm)
    print(f"[g4-ace] n_layers={n_layers} final_idx={fidx}", flush=True)

    # Setup readout
    gamma = CR._extract_final_norm_gamma(m) or CR._extract_final_norm_gamma(lm)
    gamma_ok = gamma is not None and (not hasattr(gamma, '__len__') or len(getattr(gamma, 'shape', [])) == 0 or gamma.size > 0)
    proj = pipeline.OutputProjection(lm)
    pri = pipeline.PRIComputer(proj, final_norm_gamma=gamma)
    d_model = int(proj.hidden_size)
    strat = io_plugins.get_prompt_strategy(model_id)
    print(f"[g4-ace] d={d_model} K={CR.K_SUPPORT_DEFAULT} gamma={'ok' if gamma is not None else 'MISSING'}", flush=True)

    # Setup ACE wrappers
    ace_weight_captures = {tag: [] for tag in ACE_TAGS}
    ace_v_norm_captures = {tag: [] for tag in ACE_TAGS}
    originals = {}
    for tag, idx in targets.items():
        layer = layers[idx]
        originals[tag] = layer.self_attn
        layer.self_attn = Gemma4ACEWrapper(
            layer.self_attn,
            ace_weight_captures[tag],
            ace_v_norm_captures[tag],
        )
        n_kv = int(layer.self_attn.n_kv_heads)
        lt = layer.self_attn.layer_type
        print(f"[g4-ace] ACE wrap {tag} layer={idx} type={lt} n_kv={n_kv}", flush=True)

    # Build panel
    ace_cols = []
    for tag in ACE_TAGS:
        for stat in ACE_STATS:
            ace_cols.append(f"{stat}_{tag}")
    panel_cols = READOUT_KEYS + ace_cols
    print(f"[g4-ace] Panel: {len(READOUT_KEYS)} readout + {len(ace_cols)} ACE = {len(panel_cols)} cols", flush=True)

    rows, kept_idx = [], []
    n_prefix_only = 0

    try:
        for i, (prompt, label) in enumerate(zip(prompts, labels)):
            try:
                # Reset captures for this sample
                for tag in ACE_TAGS:
                    ace_weight_captures[tag].clear()
                    ace_v_norm_captures[tag].clear()

                wrapped = strat(prompt, tok)
                enc = tok.apply_chat_template(
                    [{"role": "user", "content": wrapped}],
                    add_generation_prompt=True, tokenize=True
                ) if False else tok(wrapped)["input_ids"]
                ids = [int(t) for t in np.array(enc).reshape(-1)]

                # Forward A: prompt → D0, gen_id, surprise, h_prev
                logits1, h_prev = forward_capture(lm, ids, fidx)
                D0 = np.exp(logits1 - logits1.max()); D0 /= D0.sum()
                gen_id = int(np.argmax(D0))
                surprise = float(-np.log(D0[gen_id] + 1e-300))

                # Forward B: prompt + gen_id → p_t, h_t, ACE captures
                logits2, h_t = forward_capture(lm, ids + [gen_id], fidx)
                p_t = np.exp(logits2 - logits2.max()); p_t /= p_t.sum()
                p_max = float(np.max(p_t))

                # ── Readout ──
                comp = pri.compute_step(
                    h_t=h_t, h_prev=h_prev, p_t=p_t, S_t=surprise, alpha=1.0,
                    topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                    v3_capture_raw=False, v3_capture_centered=False,
                )
                null_ratio = float(comp.get("null_ratio_post_rank1", np.nan))
                spec, sidx, sW = CR._support_spectrum(proj, p_t, d_model, CR.K_SUPPORT_DEFAULT)
                st = CR._spectrum_stats(spec)

                readout_row = {
                    "null_ratio_post_rank1": null_ratio,
                    "fisher_eff_rank": float(st["fisher_eff_rank"]),
                    "spectral_entropy": float(st["spectral_entropy"]),
                    "neg_shadow_logvol_r1": float(st["neg_shadow_logvol_r1"]),
                    "surprise": surprise, "p_max": p_max,
                }

                # ── ACE ──
                ace_row = {}
                all_ace_ok = True
                for tag in ACE_TAGS:
                    wcaps = ace_weight_captures[tag]
                    vcaps = ace_v_norm_captures[tag]
                    layer = layers[targets[tag]]
                    n_kv = int(layer.self_attn.n_kv_heads)

                    # captures[0] = prefix forward; captures[1] = gen_step=1 commit
                    if len(wcaps) < 2 or len(vcaps) < 2:
                        all_ace_ok = False
                        break
                    w = wcaps[1]
                    v = vcaps[1]
                    # Row-sum validation gate (Codex Q7)
                    row_sums = w.sum(axis=1)
                    if not np.allclose(row_sums, 1.0, atol=1e-4):
                        all_ace_ok = False
                        break
                    stats = ace_stat_row(w, v, n_kv)
                    for stat_name, val in stats.items():
                        ace_row[f"{stat_name}_{tag}"] = val
                        if not np.isfinite(val):
                            all_ace_ok = False

                if all_ace_ok and all(np.isfinite(readout_row[k]) for k in READOUT_KEYS):
                    row = {**readout_row, **ace_row, "label": int(label)}
                    rows.append(row)
                    kept_idx.append(i)
                    if i < 8:
                        ace_str = " ".join(
                            f"js_{tag}={ace_row.get(f'js_radius_{tag}', np.nan):.4f}"
                            for tag in ACE_TAGS
                        )
                        print(
                            f"  ex{i} y={label} null={null_ratio:.4f} "
                            f"fer={readout_row['fisher_eff_rank']:.3f} "
                            f"surp={surprise:.3f} pmax={p_max:.3f} | {ace_str}",
                            flush=True,
                        )
                else:
                    n_prefix_only += 1

            except Exception as e:
                import traceback
                print(f"  ex{i} FAIL {type(e).__name__}: {e}", flush=True)
                if i < 3:
                    traceback.print_exc()

        if not rows:
            print("NO ROWS — extraction failed", flush=True)
            return

        # Build matrix
        M = np.array([[r.get(k, np.nan) for k in panel_cols] for r in rows], dtype=np.float64)
        y = np.array([r["label"] for r in rows])
        np.savez(
            out_npz,
            score_matrix=M,
            labels=y,
            sample_idx=np.array(kept_idx),
            panel=json.dumps(panel_cols),
            meta=json.dumps({
                "model": model_id, "task": task, "data_hash": data_hash,
                "ace": True, "non_byte_comparable": True, "version_delta": "gemma4_qat_mlx_vlm",
            }),
        )
        print(f"[g4-ace] WROTE {out_npz}  rows={len(rows)}  prefix_only_dropped={n_prefix_only}", flush=True)

    finally:
        # Restore originals
        for tag, idx in targets.items():
            layers[idx].self_attn = originals[tag]


if __name__ == "__main__":
    main()
