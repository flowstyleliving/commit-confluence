"""gemma-4 ACE capture (the novel-arch piece). Wraps the probed layers' attention with a
gemma-4-signature wrapper that recomputes the last-query softmax weights (H,T) + per-KV-head value
norms (n_kv,T) in the SEAL caps format, then scores the 21 ACE cells via the seal's own
_compute_attention_score (identical stats). Self-validation: reapply o_proj to (weights@values) and
match the model's real attention output. Run in the gemma4 venv (mlx-vlm). See GEMMA4_BUILD_SPEC.md.

Usage: python gemma4_ace_capture.py <model_id> <data.jsonl> <limit>   (sanity/limit mode)
"""
import sys, os, json
import numpy as np
import mlx.core as mx

T0 = os.path.expanduser("~/Documents/t0-morphology-furnace")
sys.path.insert(0, T0); sys.path.insert(0, os.path.join(T0, "exploratory/shadow-ambiguity"))
sys.path.insert(0, os.path.expanduser("~/Documents/commit-confluence"))
sys.path.insert(0, os.path.expanduser("~/Documents/commit-confluence/stage_b"))

import pri_calibrator as SEAL
from diagnose_inter_head_disagreement import _target_layer_map
import comprehensive_run as CR
from mlx_vlm import load as vload

PANEL = SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS


class G4Wrap:
    """Module-shaped wrapper matching gemma-4 attn signature (x, mask, cache, shared_kv, offset)."""
    def __init__(self, attn, store, tag, validate=False):
        object.__setattr__(self, "_attn", attn)
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_validate", validate)

    def __getattr__(self, n):
        return getattr(object.__getattribute__(self, "_attn"), n)

    def __call__(self, *args, **kw):
        a = self._attn
        x = args[0]
        shared_kv = args[3] if len(args) > 3 else kw.get("shared_kv")
        offset = args[4] if len(args) > 4 else kw.get("offset")
        off = offset if offset is not None else 0
        try:
            B, L, _ = x.shape
            q = a.q_proj(x).reshape(B, L, a.n_heads, -1)
            q = a.q_norm(q).transpose(0, 2, 1, 3)
            q = a.rope(q, offset=off)
            if shared_kv is not None:
                kt, vt = shared_kv                       # already post-norm/rope, [B,n_kv,L,d]
            else:
                k = a.k_proj(x).reshape(B, L, a.n_kv_heads, -1)
                v = k if getattr(a, "use_k_eq_v", False) else a.v_proj(x).reshape(B, L, a.n_kv_heads, -1)
                kt = a.rope(a.k_norm(k).transpose(0, 2, 1, 3), offset=off)
                vt = a.v_norm(v).transpose(0, 2, 1, 3)
            nrep = a.n_heads // kt.shape[1]
            kk = mx.repeat(kt, nrep, axis=1)
            scores = (q.astype(mx.float32) @ kk.astype(mx.float32).transpose(0, 1, 3, 2)) * a.scale
            w = mx.softmax(scores, axis=-1)              # [B,H,L,T]
            w_last = np.array(w[0, :, -1, :])            # (H, T)
            vn = np.array(mx.linalg.norm(vt[0].astype(mx.float32), axis=-1))  # (n_kv, T)
            self._store["caps"].setdefault(self._tag, []).append(w_last)
            self._store["vcaps"].setdefault(self._tag, []).append(vn)
            self._store["nkv"][self._tag] = int(kt.shape[1])
            if self._validate and self._tag not in self._store["valdone"]:
                vv = mx.repeat(vt, nrep, axis=1)
                ctx = (w.astype(mx.float32) @ vv.astype(mx.float32)).transpose(0, 2, 1, 3).reshape(B, L, -1)
                myout = np.array(a.o_proj(ctx.astype(x.dtype))[0, -1].astype(mx.float32))
                self._store["myout"][self._tag] = myout
        except Exception as e:
            import traceback
            self._store["errs"].append(f"{self._tag}: {type(e).__name__}: {e}\n{traceback.format_exc()[:500]}")
        out = a(*args, **kw)
        if self._validate and self._tag in self._store.get("myout", {}) and self._tag not in self._store["valdone"]:
            real = out[0] if isinstance(out, tuple) else out
            self._store["realout"][self._tag] = np.array(real[0, -1].astype(mx.float32))
            self._store["valdone"].add(self._tag)
        return out


def main():
    model_id, data_path, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
    prompts, labels, _ = CR._load_calibration_jsonl(str(data_path))
    prompts, labels = prompts[:limit], labels[:limit]
    m, proc = vload(model_id)
    tok = getattr(proc, "tokenizer", proc)
    lm = m.language_model
    layers = lm.model.layers
    tags = _target_layer_map(len(layers))
    print(f"[g4-ace] {model_id} n_layers={len(layers)} tags={tags}", flush=True)
    store = {"caps": {}, "vcaps": {}, "nkv": {}, "errs": [], "myout": {}, "realout": {}, "valdone": set()}
    for tag, idx in tags.items():
        layers[idx].self_attn = G4Wrap(layers[idx].self_attn, store, tag, validate=True)

    for i, (prompt, label) in enumerate(zip(prompts, labels)):
        for k in ("caps", "vcaps"):
            store[k].clear()
        enc = tok(prompt)["input_ids"]
        ids = [int(t) for t in np.array(enc).reshape(-1)]
        _ = lm(mx.array([ids]))
        if i == 0:
            for tag in tags:
                w = store["caps"].get(tag, [None])[0]
                vn = store["vcaps"].get(tag, [None])[0]
                if w is not None:
                    print(f"  [{tag}] w={w.shape} rowsum={float(w.sum(-1).mean()):.4f} "
                          f"finite={np.isfinite(w).all()} vn={vn.shape} nkv={store['nkv'][tag]}", flush=True)
                if tag in store["myout"]:
                    mo, ro = store["myout"][tag], store["realout"][tag]
                    cos = float(np.dot(mo, ro) / (np.linalg.norm(mo) * np.linalg.norm(ro) + 1e-9))
                    print(f"  [{tag}] o_proj match cos(my, model)={cos:.4f}  maxabs={np.max(np.abs(mo-ro)):.3f}", flush=True)
            # score the 21 ACE cells for example 0
            ncells, nfinite = 0, 0
            for cell in PANEL:
                sc = SEAL._compute_attention_score(cell, store["caps"], store["nkv"],
                                                   v_norm_captures=store["vcaps"])
                ncells += 1
                if sc is not None and np.isfinite(sc):
                    nfinite += 1
            print(f"  ACE cells: {nfinite}/{ncells} finite", flush=True)
    if store["errs"]:
        print("ERRORS:", store["errs"][0], flush=True)
    print("G4_ACE_SANITY_DONE", flush=True)


if __name__ == "__main__":
    main()
