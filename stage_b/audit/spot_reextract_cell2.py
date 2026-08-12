"""Spot re-extraction, cell 2: Llama-3.2-3B / triviaqa_paired (different family + task).

Two arms:
  A) ACE attention (21 cells) on scattered rows {7, 42, 123, 199} via a subset file
     (deterministic capture, no RNG).
  B) Fresh readout (6 cells: RPV/null_ratio/surprise/p_max) on PREFIX rows 0-3 via
     limit=4 (per-row RNG is derived from the run row index i, so only a prefix
     subset reproduces the full run's draws).
Together: all 27 stored matrix columns exercised. Read-only; writes only to a temp dir.
"""
import json, os, sys, tempfile
import numpy as np

CC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CC_ROOT)
import confluence_calibrator as CC

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
TASK = "triviaqa_paired"
JSONL = f"{CC_ROOT}/stage_b/data/triviaqa_paired_seed20260612_n200.jsonl"
NPZ = f"{CC_ROOT}/stage_b/profiles/{TASK}/Llama-3.2-3B-Instruct-4bit.matrix.npz"
ACE_ROWS = [7, 42, 123, 199]
SCRATCH = tempfile.mkdtemp(prefix="cc_audit_")

m = np.load(NPZ, allow_pickle=True)
stored, sidx = m["score_matrix"], m["sample_idx"]
stored_panel = json.loads(str(m["panel"]))
pos = {int(v): i for i, v in enumerate(sidx)}
col_of = {(str(c[1]), str(c[2])): j for j, c in enumerate(stored_panel)}

def compare(fresh_mat, fresh_panel, family, jsonl_rows, tag):
    worst, n_exact, n_total = 0.0, 0, 0
    for k, jrow in enumerate(jsonl_rows):
        srow = stored[pos[jrow]]
        for a, c in enumerate(fresh_panel):
            f_v = fresh_mat[k, a]
            s_v = srow[col_of[(family, str(c[2]))]]
            d = abs(f_v - s_v)
            worst = max(worst, d)
            n_total += 1
            n_exact += int(f_v == s_v)
            if d > 0:
                print(f"  [{tag}] row {jrow} {c[2]}: fresh={f_v!r} stored={s_v!r} diff={d:.3e}")
    print(f"[{tag}] {n_exact}/{n_total} bit-exact; worst |diff| = {worst:.3e}")
    return worst

# ── A) ACE attention on scattered rows ────────────────────────────────────
rows = [json.loads(l) for l in open(JSONL)]
subset = os.path.join(SCRATCH, "spot_subset_cell2.jsonl")
with open(subset, "w") as f:
    for r in ACE_ROWS:
        f.write(json.dumps(rows[r]) + "\n")
print("[spot2] ACE pass: loading Llama-3.2-3B, 4 forwards ...", flush=True)
ace = CC.collect_ace_matrix(MODEL, subset, seed=20260612, max_new_tokens=1)
w_ace = compare(ace["score_matrix"], ace["panel"], "Attention", ACE_ROWS, "ACE")

# ── B) readout on prefix rows 0-3 ─────────────────────────────────────────
print("[spot2] readout pass: prefix limit=4 ...", flush=True)
ro = CC.collect_readout_matrix_fresh(MODEL, TASK, JSONL, seed=20260612, limit=4)
ro_rows = [int(i) for i in ro["sample_idx"]]
w_ro = compare(ro["score_matrix"], ro["panel"], "Readout", ro_rows, "READOUT")

worst = max(w_ace, w_ro)
print(f"\n[spot2] VERDICT: {'REPRODUCED' if worst == 0.0 else ('close: ' + repr(worst))}")
