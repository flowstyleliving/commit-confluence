"""gemma-4 (and any mlx-vlm-loadable) READOUT-half extractor — reuses the sealed numpy stat code,
runs the model's OWN native forward (mlx-vlm) for the gen_step=1 captures. See GEMMA4_BUILD_SPEC.md.

Usage:
  python gemma4_readout_extract.py <model_id> <task> <data.jsonl> <limit> <out.npz>
Validation mode (compare to a seal matrix):
  python gemma4_readout_extract.py <model_id> <task> <data.jsonl> <limit> <out.npz> --validate <seal.npz>
"""
import sys, os, json
import numpy as np
import mlx.core as mx

T0 = os.path.expanduser("~/Documents/t0-morphology-furnace")
sys.path.insert(0, T0)
sys.path.insert(0, os.path.join(T0, "exploratory/shadow-ambiguity"))

import pri_runtime as pipeline          # OutputProjection, PRIComputer
import comprehensive_run as CR          # _support_spectrum, _spectrum_stats, K_SUPPORT_DEFAULT, helpers
import pri_v2_io_plugins as io_plugins  # get_prompt_strategy
from mlx_vlm import load as vload

READOUT_KEYS = ["null_ratio_post_rank1", "fisher_eff_rank", "spectral_entropy",
                "neg_shadow_logvol_r1", "surprise", "p_max"]


def final_layer_idx(lm):
    return len(lm.model.layers) - 1


def forward_capture(lm, ids, layer_idx):
    """Return (logits[last] np float64, final-layer hidden at last pos np float32)."""
    x = mx.array([ids])
    out = lm(x, capture_layer_ids=[layer_idx], return_hidden=True)
    logits = np.asarray(out.logits[0, -1].astype(mx.float32), dtype=np.float64)
    hs = out.hidden_states
    h = hs[0] if isinstance(hs, (list, tuple)) else list(hs.values())[0]
    h_last = np.asarray(h[0, -1].astype(mx.float32), dtype=np.float32)
    return logits, h_last


def main():
    model_id, task, data_path, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    out_npz = sys.argv[5]
    validate = sys.argv[6] == "--validate" if len(sys.argv) > 6 else False
    seal_npz = sys.argv[7] if validate else None

    prompts, labels, data_hash = CR._load_calibration_jsonl(str(data_path))
    if limit:
        prompts, labels = prompts[:limit], labels[:limit]

    m, proc = vload(model_id)
    tok = getattr(proc, "tokenizer", proc)
    lm = m.language_model
    fidx = final_layer_idx(lm)
    gamma = CR._extract_final_norm_gamma(m) or CR._extract_final_norm_gamma(lm)
    proj = pipeline.OutputProjection(lm)
    pri = pipeline.PRIComputer(proj, final_norm_gamma=gamma)
    d_model = int(proj.hidden_size)
    strat = io_plugins.get_prompt_strategy(model_id)
    print(f"[g4-readout] model={model_id} final_idx={fidx} d={d_model} K={CR.K_SUPPORT_DEFAULT} "
          f"gamma={'ok' if gamma is not None else 'MISSING'}", flush=True)

    rows, kept_idx = [], []
    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        try:
            wrapped = strat(prompt, tok)
            enc = tok.apply_chat_template([{"role": "user", "content": wrapped}],
                                          add_generation_prompt=True, tokenize=True) \
                if False else tok(wrapped)["input_ids"]
            ids = [int(t) for t in np.array(enc).reshape(-1)]
            # forward 1: prompt -> D0 (pre-commit), gen_id, surprise, h_prev
            logits1, h_prev = forward_capture(lm, ids, fidx)
            D0 = np.exp(logits1 - logits1.max()); D0 /= D0.sum()
            gen_id = int(np.argmax(D0))
            surprise = float(-np.log(D0[gen_id] + 1e-300))
            # forward 2: prompt + gen_id -> D1 (post-commit, p_t), h_t, p_max
            logits2, h_t = forward_capture(lm, ids + [gen_id], fidx)
            p_t = np.exp(logits2 - logits2.max()); p_t /= p_t.sum()
            p_max = float(np.max(p_t))
            comp = pri.compute_step(h_t=h_t, h_prev=h_prev, p_t=p_t, S_t=surprise, alpha=1.0,
                                    topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                                    v3_capture_raw=False, v3_capture_centered=False)
            null_ratio = float(comp.get("null_ratio_post_rank1", np.nan))
            spec, sidx, sW = CR._support_spectrum(proj, p_t, d_model, CR.K_SUPPORT_DEFAULT)
            st = CR._spectrum_stats(spec)
            row = {"null_ratio_post_rank1": null_ratio,
                   "fisher_eff_rank": float(st["fisher_eff_rank"]),
                   "spectral_entropy": float(st["spectral_entropy"]),
                   "neg_shadow_logvol_r1": float(st["neg_shadow_logvol_r1"]),
                   "surprise": surprise, "p_max": p_max, "label": int(label)}
            if all(np.isfinite(row[k]) for k in READOUT_KEYS):
                rows.append(row); kept_idx.append(i)
                if i < 8:
                    print(f"  ex{i} y={label} null={null_ratio:.4f} fer={row['fisher_eff_rank']:.3f} "
                          f"se={row['spectral_entropy']:.3f} nsl={row['neg_shadow_logvol_r1']:.3f} "
                          f"surp={surprise:.3f} pmax={p_max:.3f}", flush=True)
        except Exception as e:
            import traceback; print(f"  ex{i} FAIL {type(e).__name__}: {e}", flush=True)
            if i < 3: traceback.print_exc()

    if not rows:
        print("NO ROWS — extraction failed"); return
    M = np.array([[r[k] for k in READOUT_KEYS] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows])
    np.savez(out_npz, score_matrix=M, labels=y, sample_idx=np.array(kept_idx),
             panel=json.dumps(READOUT_KEYS), meta=json.dumps({"model": model_id, "task": task,
             "data_hash": data_hash, "readout_only": True}))
    print(f"WROTE {out_npz}  rows={len(rows)}", flush=True)

    if validate and os.path.exists(seal_npz):
        z = np.load(seal_npz, allow_pickle=True)
        panel = json.loads(str(z["panel"])) if "panel" in z.files else None
        sm = z["score_matrix"]
        print("\n=== PARITY vs seal ===")
        for k in READOUT_KEYS:
            j = next((idx for idx, name in enumerate(panel) if k in str(name)), None)
            if j is None:
                print(f"  {k}: not in seal panel"); continue
            seal_col = sm[kept_idx, j].astype(np.float64)
            mine = M[:, READOUT_KEYS.index(k)]
            if np.std(seal_col) < 1e-12 or np.std(mine) < 1e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(seal_col, mine)[0, 1])
            print(f"  {k:24s} corr={corr:+.3f}  seal[:3]={np.round(seal_col[:3],3)} mine[:3]={np.round(mine[:3],3)}")
    print("READOUT_EXTRACT_DONE")


if __name__ == "__main__":
    main()
