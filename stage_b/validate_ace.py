#!/usr/bin/env python3
"""
S2 validation: prove collect_ace_matrix (import-replicated collection) reproduces the SEALED
ACE run. Run full n=200 ACE on Qwen2.5-7B/anli (sealed winner = final_v_norm_lastq_weighted,
so this stresses the v-norm path added by the S1 fix) and compare ALL 21 cell full-sample
AUROCs to the sealed profile's candidate_panel. Match within tolerance => wiring faithful.

Matches sealed provenance: max_new_tokens=1, calibration_seed=20260512, same data file.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC
SEAL = CC.SEAL

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DATA = os.path.join(CC.T0_REPO, "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl")
PROFILE = os.path.join(CC.T0_REPO, "experiments/t0-sealed/2026-05-26/profiles/anli/Qwen2.5-7B-Instruct-4bit.profile.json")
HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 0.01

def main():
    prof = json.load(open(PROFILE))
    sealed = {e["rank_label"]: e for e in prof["calibration_stats"]["candidate_panel"]}
    sealed_hash = prof.get("task", {}).get("data_hash_sha256")

    print(f"[validate] collecting ACE on {MODEL} full n=200 (max_new_tokens=1)...", flush=True)
    ace = CC.collect_ace_matrix(MODEL, DATA, seed=20260512, max_new_tokens=1)  # sealed provenance
    sm, y, panel = ace["score_matrix"], ace["labels"], ace["panel"]
    print(f"[validate] data_hash mine={ace['data_hash'][:16]} sealed={(sealed_hash or '')[:16]} "
          f"match={ace['data_hash']==sealed_hash}")

    rows, max_diff, winner_ok = [], 0.0, None
    winner_label = prof["detector"]["metric"]["label"]
    for j, cell in enumerate(panel):
        lab = cell[2]
        auc, sign, n_eval = SEAL._score_candidate(sm[:, j], y)
        s = sealed.get(lab, {})
        sealed_auc = s.get("auroc")
        diff = abs(auc - sealed_auc) if (np.isfinite(auc) and sealed_auc is not None) else float("nan")
        if np.isfinite(diff):
            max_diff = max(max_diff, diff)
        if lab == winner_label:
            winner_ok = bool(np.isfinite(diff) and diff < TOL)
        rows.append({"cell": lab, "mine": None if not np.isfinite(auc) else round(float(auc), 4),
                     "sealed": sealed_auc, "diff": None if not np.isfinite(diff) else round(diff, 4),
                     "is_winner": lab == winner_label})

    rows.sort(key=lambda r: (-(r["diff"] or -1)))
    print(f"\n{'cell':<34} {'mine':>8} {'sealed':>8} {'diff':>8}  win")
    for r in rows:
        print(f"{r['cell']:<34} {str(r['mine']):>8} {str(r['sealed']):>8} {str(r['diff']):>8}"
              f"  {'<-WIN' if r['is_winner'] else ''}")

    ok = (max_diff < TOL) and (ace["data_hash"] == sealed_hash) and bool(winner_ok)
    print(f"\n[validate] max abs AUROC diff over 21 cells = {max_diff:.4f} (tol {TOL})")
    print(f"[validate] winner '{winner_label}' reproduces: {winner_ok}")
    print(f"[validate] data hash match: {ace['data_hash']==sealed_hash}")
    print(f"\nS2 VALIDATION {'PASS' if ok else 'FAIL'}")

    out = {"model": MODEL, "task": "anli_r1", "n": int(len(y)), "tol": TOL,
           "max_abs_auroc_diff": round(float(max_diff), 5),
           "winner_label": winner_label, "winner_reproduces": winner_ok,
           "data_hash_match": ace["data_hash"] == sealed_hash, "pass": bool(ok), "cells": rows}
    json.dump(out, open(os.path.join(HERE, "out_validate_ace.json"), "w"), indent=1)
    with open(os.path.join(HERE, "VALIDATE_ACE.md"), "w") as f:
        f.write(f"# S2 - ACE reproduction validation ({MODEL.split('/')[-1]} / anli)\n\n")
        f.write(f"Full n=200, max_new_tokens=1, seed 20260512 (sealed provenance).\n\n")
        f.write(f"- max abs AUROC diff over 21 cells: **{max_diff:.5f}** (tol {TOL})\n")
        f.write(f"- winner `{winner_label}` reproduces: **{winner_ok}**\n")
        f.write(f"- data hash match: **{ace['data_hash']==sealed_hash}**\n")
        f.write(f"- **S2 {'PASS' if ok else 'FAIL'}**\n\n")
        f.write("| cell | mine | sealed | diff | winner |\n|---|---:|---:|---:|:--:|\n")
        for r in sorted(rows, key=lambda r: r["cell"]):
            f.write(f"| {r['cell']} | {r['mine']} | {r['sealed']} | {r['diff']} | "
                    f"{'WIN' if r['is_winner'] else ''} |\n")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
