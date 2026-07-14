"""Full gemma-4 commit-confluence extractor: validated ACE capture (cos=1.0 vs model) + p_t-fixed
readout → 27-col matrix in seal panel order → seal calibrate_merged. Run in the gemma4 venv.
Usage: python gemma4_full_extract.py <model_id> <task> <data.jsonl> <out.npz> <seal_ref.npz>
"""
import sys, os, json
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
T0 = os.environ.get("CONFLUENCE_T0_REPO", os.path.join(REPO_ROOT, "vendor", "t0_core"))
sys.path.insert(0, T0); sys.path.insert(0, os.path.join(T0, "exploratory/shadow-ambiguity"))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import pri_calibrator as SEAL
import pri_runtime as PIPE
import comprehensive_run as CR
import confluence_calibrator as CC
import pri_v2_io_plugins as io_plugins
from diagnose_inter_head_disagreement import _target_layer_map
from gemma4_ace_capture import G4Wrap
from mlx_vlm import load as vload

MODEL, TASK, DATA, NPZ, SEAL_REF = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
ACE_PANEL = SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS
ref = np.load(SEAL_REF, allow_pickle=True)
full_panel = json.loads(str(ref["panel"])) if isinstance(ref["panel"].tolist(), str) else [str(x) for x in ref["panel"]]
readout_panel = full_panel[len(ACE_PANEL):]
RO_KEYS = ["null_ratio_post_rank1", "fisher_eff_rank", "spectral_entropy", "neg_shadow_logvol_r1",
           "surprise", "p_max"]


def metric_of(cell):
    s = str(cell)
    for m in RO_KEYS:
        if m in s:
            return m
    return None


prompts, labels, dh = CR._load_calibration_jsonl(DATA)
m, proc = vload(MODEL)
tok = getattr(proc, "tokenizer", proc)
lm = m.language_model
layers = lm.model.layers
tags = _target_layer_map(len(layers))
fidx = len(layers) - 1
gamma = CR._extract_final_norm_gamma(m) or CR._extract_final_norm_gamma(lm)
proj = PIPE.OutputProjection(lm)
pri = PIPE.PRIComputer(proj, final_norm_gamma=gamma)
dmodel = int(proj.hidden_size)
# gemma-4-it does NOT perform the YES/NO task under raw_passthrough (the io-plugin
# default for gemma-*): on a raw prompt it just continues the question text (commits
# " The"/" Adam"), so every confluence signal is noise (~0.37 AUROC, full panel fails too).
# It requires its own chat template to actually commit to YES/NO (verified: g4_diag.py).
# This makes gemma-4 even less protocol-comparable to the raw-prompt seal cells (gemma-3-12b
# tolerated raw_passthrough and passed) — already covered by the standing non-byte-comparable
# caveat; using the template is the only way to extract a meaningful task-commitment signal.
def strat(p, _tok):
    return _tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
store = {"caps": {}, "vcaps": {}, "nkv": {}, "errs": [], "myout": {}, "realout": {}, "valdone": set()}
for tag, idx in tags.items():
    layers[idx].self_attn = G4Wrap(layers[idx].self_attn, store, tag, validate=False)
print(f"[g4-full] {MODEL} {TASK} layers={len(layers)} tags={tags} d={dmodel} "
      f"ace_cells={len(ACE_PANEL)} ro_cells={len(readout_panel)}", flush=True)


def fwd(ids):
    out = lm(mx.array([ids]), capture_layer_ids=[fidx], return_hidden=True)
    logits = np.asarray(out.logits[0, -1].astype(mx.float32), dtype=np.float64)
    hs = out.hidden_states
    h = hs[0] if isinstance(hs, (list, tuple)) else list(hs.values())[0]
    return logits, np.asarray(h[0, -1].astype(mx.float32), dtype=np.float32)


ace_rows, ro_rows, keep = [], [], []
for i, (p, y) in enumerate(zip(prompts, labels)):
    try:
        enc = tok(strat(p, tok))["input_ids"]
        ids = [int(t) for t in np.array(enc).reshape(-1)]
        for k in ("caps", "vcaps"):
            store[k].clear()
        logitsA, h_prev = fwd(ids)                       # forward A: prompt -> D0 + h_prev + ACE caps
        pA = np.exp(logitsA - logitsA.max()); pA /= pA.sum()
        gid = int(np.argmax(pA)); surprise = float(-np.log(pA[gid] + 1e-300))
        ace_row = []
        for cell in ACE_PANEL:
            sc = SEAL._compute_attention_score(cell, store["caps"], store["nkv"],
                                               v_norm_captures=store["vcaps"])
            ace_row.append(float(sc) if (sc is not None and np.isfinite(sc)) else 0.0)
        logitsB, h_t = fwd(ids + [gid])                  # forward B: +commit -> D1 + h_t
        pB = np.exp(logitsB - logitsB.max()); pB /= pB.sum()
        p_t, p_max = pB, float(pB.max())
        comp = pri.compute_step(h_t=h_t, h_prev=h_prev, p_t=p_t, S_t=surprise, alpha=1.0,
                                topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                                v3_capture_raw=False, v3_capture_centered=False)
        spec, _, _ = CR._support_spectrum(proj, p_t, dmodel, CR.K_SUPPORT_DEFAULT)
        st = CR._spectrum_stats(spec)
        myro = {"null_ratio_post_rank1": float(comp.get("null_ratio_post_rank1", np.nan)),
                "fisher_eff_rank": float(st["fisher_eff_rank"]),
                "spectral_entropy": float(st["spectral_entropy"]),
                "neg_shadow_logvol_r1": float(st["neg_shadow_logvol_r1"]),
                "surprise": surprise, "p_max": p_max}
        ro_row = [myro[metric_of(c)] for c in readout_panel]
        if all(np.isfinite(v) for v in ro_row):
            ace_rows.append(ace_row); ro_rows.append(ro_row); keep.append(i)
    except Exception as e:
        import traceback
        print(f"ex{i} FAIL {type(e).__name__}: {e}", flush=True)
        if i < 2:
            traceback.print_exc()
    if i % 25 == 0:
        print(f"  {i}/{len(prompts)} kept={len(keep)}", flush=True)

yk = np.array([int(labels[i]) for i in keep])
ace_d = {"sample_idx": np.array(keep), "labels": yk, "score_matrix": np.array(ace_rows),
         "panel": list(ACE_PANEL), "slug": MODEL.split("/")[-1], "data_hash": dh}
ro_d = {"sample_idx": np.array(keep), "labels": yk, "score_matrix": np.array(ro_rows),
        "panel": list(readout_panel), "data_hash": dh}
mm = CC.merge_matrices(ace_d, ro_d, max_dropped=0)
np.savez(NPZ, score_matrix=mm["score_matrix"], labels=mm["labels"],
         sample_idx=mm["sample_idx"], panel=json.dumps([str(c) for c in mm["panel"]]),
         meta=json.dumps({"model": MODEL, "task": TASK, "n": mm["n_aligned"]}))
prof = CC.calibrate_merged(mm, n_bootstrap=2000, seed=20260612, model_id=MODEL, benchmark=TASK)
ge = prof.get("secondary_geometric_only", {}); pr = prof.get("primary_full_panel", {})
print("G4_FULL_RESULT")
print(json.dumps({"task": TASK, "n_aligned": mm["n_aligned"],
                  "geom_winner": ge.get("winner"), "geom_ci_lo": ge.get("oob_auroc_ci_lo"),
                  "geom_deployable": ge.get("deployable"),
                  "primary_winner": pr.get("winner"), "primary_ci_lo": pr.get("oob_auroc_ci_lo"),
                  "primary_deployable": pr.get("deployable"),
                  "controls_pass": prof.get("controls", {}).get("pass")}, indent=2, default=str))
