"""Fail-closed end-to-end hand-trace of one commit-confluence seal cell.

Independent check: uses ONLY numpy + json + hashlib. No calibrator import, no sklearn.
AUROC is recomputed from scratch via the Mann-Whitney rank formula and REQUIRED to
match the committed profile (4-dp, the profile's own precision) with the same sign.
The winner cell is derived from the profile's exact winner string (no substring
matching). Any failed check exits nonzero. v2 after the 2026-08-12 gpt-5.6-sol audit;
the fail-open v1 is preserved in git history at e8e4db2.

Scope limit: only winners stored in the 27-column matrix (Attention/Readout cells)
can be traced. Fusion winners are computed downstream of the matrix and will fail
resolution here - fail-closed, but this CLI is NOT generic over all cells.

Also prints (non-gating) the cell's most-flagged example - for the sealed
Qwen2.5-7B/anli_r1 cell this is jsonl row 168, which is gold-entailed: the rank-1
flag is a false positive, illustrating what a ranking endpoint at AUROC ~0.79 does
and does not claim.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # stage_b/
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def cell_label(triple):
    """Format a panel triple exactly as profile winner strings do."""
    step, family, metric = triple
    if family == "Attention":
        return f"attention[{metric}] @ step {step}"
    return f"{family} {metric} @ step {step}"


def auroc_mannwhitney(y, x):
    """AUROC = P(score(pos) > score(neg)), ties count half. Pure numpy."""
    pos, neg = x[y == 1], x[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="anli_r1")
    ap.add_argument("--model", default="Qwen2.5-7B-Instruct-4bit")
    a = ap.parse_args()

    prof_path = f"{BASE}/profiles/{a.task}/{a.model}.profile.json"
    npz_path = f"{BASE}/profiles/{a.task}/{a.model}.matrix.npz"
    prof = json.load(open(prof_path))
    jsonl = os.path.join(os.path.dirname(BASE), prof["data_path"])
    print(f"[trace] {a.model}/{a.task}")

    # ── 1. data file: sha256 must match what the profile stamped ──────────
    sha = hashlib.sha256(open(jsonl, "rb").read()).hexdigest()
    check("data sha256 matches profile stamp", sha == prof["data_file_sha256"])

    rows = [json.loads(l) for l in open(jsonl)]
    n1 = sum(r["label"] for r in rows)
    print(f"  jsonl rows: {len(rows)} | {n1}/{len(rows)} label=1")

    # ── 2. matrix: alignment back to the jsonl ────────────────────────────
    m = np.load(npz_path, allow_pickle=False)
    scores, labels, sidx = m["score_matrix"], m["labels"], m["sample_idx"]
    panel = [tuple(c) for c in json.loads(str(m["panel"]))]
    check("matrix labels == jsonl labels at sample_idx (all rows)",
          len(sidx) == len(rows)
          and all(int(labels[i]) == int(rows[int(sidx[i])]["label"]) for i in range(len(sidx))))

    # ── 3. derive the winner column from the profile's EXACT winner string ─
    winner = prof["primary_full_panel"]["winner"]
    matches = [j for j, c in enumerate(panel) if cell_label(c) == winner]
    check(f"winner '{winner}' resolves to exactly one panel column",
          len(matches) == 1, f"matches={matches}")
    if len(matches) != 1:
        sys.exit(1)
    col = matches[0]
    s = scores[:, col]
    check("winner column all finite", bool(np.isfinite(s).all()))

    # ── 4. recompute its AUROC from scratch; REQUIRE profile agreement ────
    auc_raw = auroc_mannwhitney(labels, s)
    auc_dir, sign = max(auc_raw, 1 - auc_raw), (1 if auc_raw >= 0.5 else -1)
    w = prof["primary_full_panel"]["winner_marginal"]
    check("recomputed AUROC matches profile at its 4-dp precision",
          round(float(auc_dir), 4) == w["auroc"], f"recomputed={auc_dir:.6f} profile={w['auroc']}")
    check("recomputed sign matches profile", sign == w["sign"], f"{sign} vs {w['sign']}")

    # ── 5. non-gating exhibits: direction + the most-flagged example ──────
    mu0, mu1 = s[labels == 0].mean(), s[labels == 1].mean()
    print(f"  class means: label0={mu0:.4f} label1={mu1:.4f} "
          f"(sign {sign:+d}: {'lower' if sign < 0 else 'higher'} score = more label-1-like)")
    flag = -s if sign < 0 else s
    top = int(np.argsort(-flag)[0])
    r = rows[int(sidx[top])]
    print(f"  most-flagged: jsonl row {int(sidx[top])} score={s[top]:.6f} "
          f"gold label={r['label']}"
          + (" <- rank-1 flag is a FALSE POSITIVE" if r["label"] == 0 else ""))

    ok = not FAILURES
    print(f"[trace] VERDICT: {'PASS' if ok else 'FAIL ' + str(FAILURES)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
