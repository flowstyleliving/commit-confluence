#!/usr/bin/env python3
"""Summarize detector reports: AUROC vs the stated target, and worst-case instrument checks per layer."""
import json
import sys
from pathlib import Path

# Target stated by MK for this check: median OOB AUROC >= 0.55 AND 95% CI lower bound > 0.5.
TARGET_AUROC, TARGET_CI_LO = 0.55, 0.5


def summarize(path):
    r = json.loads(Path(path).read_text())
    out = {"report": str(path), "model": r["provenance"].get("model_id"),
           "snapshot": r["provenance"].get("model_snapshot_sha"), "status": r["status"],
           "error": r.get("error"), "counts": r["counts"],
           "yes_no_commit_rate": r.get("yes_no_commit_rate"), "wall_seconds": r.get("wall_seconds")}
    checks = r["validation"] + r["extraction_checks"]
    layers = {}
    for rec in checks:
        for tag, s in rec.get("layers", {}).items():
            g = s.get("reconstruction", {})
            L = layers.setdefault(tag, {"index": s["index"], "n": 0, "min_cosine": 1.0, "max_relative_l2": 0.0,
                                        "max_row_sum_error": 0.0, "max_causal_leak": 0.0, "all_pass": True})
            L["n"] += 1
            L["min_cosine"] = min(L["min_cosine"], g.get("cosine", float("nan")))
            L["max_relative_l2"] = max(L["max_relative_l2"], g.get("relative_l2", float("nan")))
            L["max_row_sum_error"] = max(L["max_row_sum_error"], s["row_sum_error_max"])
            L["max_causal_leak"] = max(L["max_causal_leak"], s["causal_leak_max"])
            L["all_pass"] &= bool(s.get("pass"))
    parity = [rec["parity"] for rec in checks if "parity" in rec]
    out["layers"] = layers
    if parity:
        out["parity"] = {"n": len(parity),
                         "max_norm_relative_l2": max(p["norm"]["relative_l2"] for p in parity),
                         "max_logit_relative_l2": max(p["logits"]["relative_l2"] for p in parity),
                         "min_logit_cosine": min(p["logits"]["cosine"] for p in parity),
                         "argmax_match_rate": sum(p["argmax_matches"] for p in parity) / len(parity),
                         "logits_scaling": parity[0]["logits_scaling"],
                         "all_pass": all(p["pass"] for p in parity)}
    cal = r.get("calibration")
    if cal:
        for ep in ("primary_full_panel", "secondary_geometric_only"):
            e = cal[ep]
            out[ep] = {k: e[k] for k in ("winner", "oob_auroc_median", "oob_auroc_ci_lo", "oob_auroc_ci_hi",
                                         "winner_stability", "deployable")}
            out[ep]["meets_target"] = bool(e["oob_auroc_median"] >= TARGET_AUROC
                                           and e["oob_auroc_ci_lo"] > TARGET_CI_LO)
        out["shuffled_label_controls_pass"] = cal["controls"]["pass"]
    return out


if __name__ == "__main__":
    print(json.dumps([summarize(p) for p in sys.argv[1:]], indent=2))
