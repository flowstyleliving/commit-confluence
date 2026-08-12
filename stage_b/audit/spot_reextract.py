"""Spot re-extraction: re-run the REAL model forward pass on 4 saved prompts and
compare every attention-panel value against the sealed matrix on disk.

Rows chosen: 168 (the #1-flagged false positive), 21 (least flagged),
0 (first row), 100 (arbitrary mid). Runs read-only; writes only to a temp dir.
"""
import json, os, sys, tempfile
import numpy as np

CC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CC_ROOT)
import confluence_calibrator as CC

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
JSONL = f"{CC_ROOT}/stage_b/data/anli_R1_seed20260612_n200.jsonl"
NPZ = f"{CC_ROOT}/stage_b/profiles/anli_r1/Qwen2.5-7B-Instruct-4bit.matrix.npz"
ROWS = [168, 21, 0, 100]
SCRATCH = tempfile.mkdtemp(prefix="cc_audit_")

print("[spot] SEAL source:", CC.SEAL_SOURCE, flush=True)

# subset jsonl with just the 4 rows, in ROWS order
rows = [json.loads(l) for l in open(JSONL)]
subset = os.path.join(SCRATCH, "spot_subset.jsonl")
with open(subset, "w") as f:
    for r in ROWS:
        f.write(json.dumps(rows[r]) + "\n")

print("[spot] loading model + running 4 forward passes ...", flush=True)
ace = CC.collect_ace_matrix(MODEL, subset, seed=20260612, max_new_tokens=1)
fresh, fresh_panel = ace["score_matrix"], ace["panel"]

# stored values
m = np.load(NPZ, allow_pickle=True)
stored, sidx = m["score_matrix"], m["sample_idx"]
stored_panel = json.loads(str(m["panel"]))
pos = {int(v): i for i, v in enumerate(sidx)}  # jsonl row -> stored matrix row

labels = [str(c[2]) for c in fresh_panel]
print(f"\n[spot] comparing {len(labels)} attention cells x {len(ROWS)} prompts")
col_of = {str(c[2]): j for j, c in enumerate(stored_panel) if c[1] == "Attention"}

worst = 0.0
n_exact = n_total = 0
for k, jrow in enumerate(ROWS):
    srow = stored[pos[jrow]]
    for a, lab in enumerate(labels):
        f_v, s_v = fresh[k, a], srow[col_of[lab]]
        d = abs(f_v - s_v)
        worst = max(worst, d)
        n_total += 1
        n_exact += int(f_v == s_v)
        if lab == "final_bos_mass":
            print(f"  row {jrow:>3} final_bos_mass: fresh={f_v:.9f} stored={s_v:.9f} "
                  f"{'EXACT' if f_v == s_v else f'diff={d:.2e}'}")

print(f"\n[spot] {n_exact}/{n_total} values bit-exact; worst |diff| = {worst:.3e}")
print("[spot] VERDICT:", "REPRODUCED" if worst < 1e-6 else "MISMATCH - investigate")
