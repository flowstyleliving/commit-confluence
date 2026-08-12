"""End-to-end trace of one commit-confluence seal cell: Qwen2.5-7B / anli_r1.

Independent check: uses ONLY numpy + json. No calibrator import, no sklearn.
AUROC is recomputed from scratch via the Mann-Whitney rank formula.
"""
import json
import hashlib
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stage_b/
JSONL = f"{BASE}/data/anli_R1_seed20260612_n200.jsonl"
NPZ = f"{BASE}/profiles/anli_r1/Qwen2.5-7B-Instruct-4bit.matrix.npz"
PROF = f"{BASE}/profiles/anli_r1/Qwen2.5-7B-Instruct-4bit.profile.json"

# ── 1. data file: sha256 must match what the profile stamped ──────────────
sha = hashlib.sha256(open(JSONL, "rb").read()).hexdigest()
prof = json.load(open(PROF))
print("data sha256 matches profile stamp:", sha == prof["data_file_sha256"])

rows = [json.loads(l) for l in open(JSONL)]
print("jsonl rows:", len(rows), "| class balance:", sum(r["label"] for r in rows), "of", len(rows), "are label=1 (contradiction/NO)")

# ── 2. matrix: alignment back to the jsonl ────────────────────────────────
m = np.load(NPZ, allow_pickle=True)
scores, labels, sidx = m["score_matrix"], m["labels"], m["sample_idx"]
panel = json.loads(str(m["panel"]))
jsonl_labels = np.array([rows[i]["label"] for i in sidx])
print("matrix labels == jsonl labels at sample_idx:", bool(np.array_equal(labels, jsonl_labels)))

# ── 3. find the winner column and recompute its AUROC from scratch ────────
# panel entries are [family, column, label]; winner = attention[final_bos_mass] @ step 0
col = next(j for j, c in enumerate(panel) if "final_bos_mass" in str(c))
print("winner column index:", col, "| panel entry:", panel[col])
s = scores[:, col]

def auroc_mannwhitney(y, x):
    """AUROC = P(score(pos) > score(neg)), ties count half. Pure numpy."""
    pos, neg = x[y == 1], x[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

auc_raw = auroc_mannwhitney(labels, s)
auc_dir = max(auc_raw, 1 - auc_raw)
sign = 1 if auc_raw >= 0.5 else -1
w = prof["primary_full_panel"]["winner_marginal"]
print(f"recomputed: auroc={auc_dir:.4f} sign={sign:+d}  |  profile says: auroc={w['auroc']} sign={w['sign']}")

# ── 4. what direction means what ──────────────────────────────────────────
mu0, mu1 = s[labels == 0].mean(), s[labels == 1].mean()
print(f"\nmean final_bos_mass | entailed(label=0): {mu0:.4f} | contradiction(label=1): {mu1:.4f}")
print("=> sign -1: LOWER bos_mass scores as MORE hallucination-like (contradiction)")

# ── 5. concrete examples ──────────────────────────────────────────────────
flag = -s  # sign-adjusted: higher = more flagged
order = np.argsort(-flag)

def show(i, title):
    r = rows[sidx[i]]
    hyp = r["prompt"].split("Hypothesis:")[1].split("\nAnswer:")[0].strip()
    print(f"\n--- {title} ---")
    print(f"jsonl row {sidx[i]} | final_bos_mass = {s[i]:.6f} | gold label = {r['label']} "
          f"({'contradiction/NO' if r['label'] else 'entailed/YES'})")
    print(f"hypothesis: {hyp[:160]}")

show(order[0], "MOST flagged as hallucination (lowest bos_mass)")
# strongest false positive: most-flagged example whose gold label says it is TRUE
fp = next(i for i in order if labels[i] == 0)
rank_fp = int(np.where(order == fp)[0][0]) + 1
show(fp, f"strongest FALSE POSITIVE (flag rank {rank_fp}/200): called hallucination, actually true")
show(order[-1], "LEAST flagged (highest bos_mass)")

# error census at the balanced-threshold operating point (top-100 flagged = predicted contradiction)
pred = np.zeros(200, dtype=int); pred[order[:100]] = 1
tp = int(((pred == 1) & (labels == 1)).sum()); fp_n = int(((pred == 1) & (labels == 0)).sum())
print(f"\nif we flagged the top-100: {tp} true hallucinations caught, {fp_n} true statements wrongly flagged")
