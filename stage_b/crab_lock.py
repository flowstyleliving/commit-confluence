"""Crab-lock: starve gemma-3-12b's ACE attention signals to the gemma-3-4b head budget
(8 query heads / 4 KV-groups) WITHOUT touching the model, then recalibrate ANLI. Tests the
head-resolution hypothesis: does the 0.71 deployable signal collapse back toward the 0.40 orphan
when the morphology statistics lose head resolution?

Runtime monkeypatch of pri_calibrator._compute_panel_scores_for_sample (slices captured attention +
value tensors on the head axis); readout/null_ratio/RPV/confidence + calibration are untouched.
No sealed-core file is edited. Run in the SEAL venv (t0 .venv, mlx-lm 0.29.1)."""
import sys, os, json
os.chdir(os.path.expanduser("~/Documents/commit-confluence"))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, "stage_b")
sys.path.insert(0, os.path.expanduser("~/Documents/t0-morphology-furnace"))
os.makedirs("/tmp/crab", exist_ok=True)

import pri_calibrator as SEAL

H, KV, NFULL = 8, 4, 16          # 4b budget: 8 query heads / 4 KV-groups (12b has 16/8)
_orig = SEAL._compute_panel_scores_for_sample
_seen = {"printed": False}


def _slice(d):
    if not d:
        return d
    out = {}
    for tag, lst in d.items():
        nl = []
        for t in lst:
            a = t
            if hasattr(a, "shape"):
                if not _seen["printed"]:
                    print(f"[crab] cap tag={tag} shape={tuple(a.shape)}", flush=True)
                for ax, s in enumerate(a.shape):       # slice whichever axis == head count
                    if s == NFULL:
                        idx = [slice(None)] * a.ndim
                        idx[ax] = slice(0, H)
                        a = a[tuple(idx)]
                        break
            nl.append(a)
        out[tag] = nl
    _seen["printed"] = True
    return out


def _patched(*args, attention_captures=None, attention_n_kv_heads=None,
             attention_v_norm_captures=None, **kw):
    attention_captures = _slice(attention_captures)
    attention_v_norm_captures = _slice(attention_v_norm_captures)
    if isinstance(attention_n_kv_heads, dict):
        attention_n_kv_heads = {k: KV for k in attention_n_kv_heads}
    elif attention_n_kv_heads is not None:
        attention_n_kv_heads = KV
    return _orig(*args, attention_captures=attention_captures,
                 attention_n_kv_heads=attention_n_kv_heads,
                 attention_v_norm_captures=attention_v_norm_captures, **kw)


SEAL._compute_panel_scores_for_sample = _patched
print(f"[crab] patched ACE to {H} heads / {KV} KV-groups (gemma-3-4b budget)", flush=True)

from run_seal import run_cell

prof = run_cell("mlx-community/gemma-3-12b-it-4bit", "anli_r1",
                "stage_b/data/anli_R1_seed20260612_n200.jsonl",
                seed=20260612, nboot=2000, limit=None,
                npz_path="/tmp/crab/g3-12b.anli.subset.npz", strict=True)

ge = prof.get("secondary_geometric_only", {})
pr = prof.get("primary_full_panel", {})
print("CRAB_LOCK_RESULT")
print(json.dumps({
    "subset_geom_winner": ge.get("winner"),
    "subset_geom_ci_lo": ge.get("oob_auroc_ci_lo"),
    "subset_geom_deployable": ge.get("deployable"),
    "subset_primary_ci_lo": pr.get("oob_auroc_ci_lo"),
    "controls_pass": prof.get("controls", {}).get("pass"),
    "n_aligned": prof.get("n_aligned"),
    "REFERENCE_full_12b_ci_lo": 0.709,
    "REFERENCE_orphan_4b_ci_lo": 0.403,
    "H_heads": H, "KV_groups": KV,
}, indent=2))
