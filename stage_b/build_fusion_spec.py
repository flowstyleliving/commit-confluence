#!/usr/bin/env python3
"""
Build the E4 fusion-cell spec (stage_b/fusion_signs.json) from SEALED-ERA artifacts ONLY.

Pre-registration honesty: the fusion columns are precomputed before the nested-OOB bootstrap,
so their component orientations must NOT be fit on fresh data (that would smuggle full-sample
selection into a column the OOB evaluation cannot undo). Orientations are therefore locked,
per (model, task), from artifacts that PREDATE the fresh seal:

  - ACE component: the modal selected cell across the 18 sealed ACE profiles
    (experiments/t0-sealed/2026-05-26/profiles); its per-(model, task) sign from each profile's
    candidate_panel. Models without a sealed profile (Llama-3.1-8B) get the cohort-modal sign.
  - Readout components (surprise, null_ratio_post_rank1, fisher_eff_rank): full-sample
    _score_candidate sign on the 20260526 RPV comprehensive rows
    (exploratory/shadow-ambiguity/comprehensive_outputs). Missing -> cohort-modal per component.
  - surprise is additionally HARD-CLAMPED to +1 (canonical orientation: higher = riskier);
    if any sealed-era sign disagrees we record the disagreement but keep +1.

This script is committed BEFORE fresh data exists; fusion_signs.json is a frozen input to the
fresh-seed run. Re-running it must be byte-stable (sorted keys, no timestamps).
"""
import json, os, sys, glob, collections

T0 = os.environ.get("CONFLUENCE_T0_REPO", os.path.expanduser("~/Documents/t0-morphology-furnace"))
SEALED_PROFILES = os.path.join(T0, "experiments/t0-sealed/2026-05-26/profiles")
RPV_OUT = os.path.join(T0, "exploratory/shadow-ambiguity/comprehensive_outputs")
sys.path.insert(0, T0)
import numpy as np
import pri_calibrator as SEAL

READOUT_COMPONENTS = ["surprise", "null_ratio_post_rank1", "fisher_eff_rank"]
TASK_ALIAS = {"anli": "anli_r1", "triviaqa": "triviaqa_paired"}


def modal(xs):
    c = collections.Counter(xs)
    top = max(c.values())
    # deterministic tie-break: lexicographically smallest among the most frequent
    return sorted([k for k, v in c.items() if v == top])[0]


def main():
    # ---- ACE: modal winner cell + per-(model, task) sign of that cell ----
    winners, panel_signs = [], {}  # panel_signs[(slug, task)][column_name] = sign
    for f in sorted(glob.glob(os.path.join(SEALED_PROFILES, "*", "*.profile.json"))):
        d = json.load(open(f))
        task = TASK_ALIAS[os.path.basename(os.path.dirname(f))]
        slug = os.path.basename(f).replace(".profile.json", "")
        met = d["detector"]["metric"]
        col = met if isinstance(met, str) else met.get("column_name") or met.get("rank_label")
        winners.append((slug, task, col))
        panel_signs[(slug, task)] = {c["column_name"]: int(c["sign"])
                                     for c in d["calibration_stats"]["candidate_panel"]}
    modal_ace = modal([w[2] for w in winners])
    ace_signs = {}
    for (slug, task), signs in panel_signs.items():
        if modal_ace in signs:
            ace_signs[f"{slug}|{task}"] = signs[modal_ace]
    ace_modal_sign = modal(list(ace_signs.values()))

    # ---- Readout: per-(model, task) full-sample sign on sealed-era RPV rows ----
    ro_signs = {}      # f"{slug}|{task}" -> {component: sign}
    disagreements = []
    for f in sorted(glob.glob(os.path.join(RPV_OUT, "shadow_v2__*__limitall.json"))):
        d = json.load(open(f))
        slug = (d.get("model") or "").split("/")[-1]
        task = d.get("benchmark")
        rows = d.get("rows", [])
        if not slug or not task or not rows:
            continue
        y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
        entry = {}
        for comp in READOUT_COMPONENTS:
            s = np.array([float(r[comp]) if r.get(comp) is not None else np.nan for r in rows])
            auc, sign, _ = SEAL._score_candidate(s, y)
            if comp == "surprise":
                if sign not in (0, 1):
                    disagreements.append({"slug": slug, "task": task, "comp": comp,
                                          "sealed_sign": int(sign), "kept": 1})
                sign = 1  # hard clamp: canonical orientation
            entry[comp] = int(sign) if sign != 0 else 0
        ro_signs[f"{slug}|{task}"] = entry
    ro_modal = {comp: modal([v[comp] for v in ro_signs.values() if v.get(comp, 0) != 0])
                for comp in READOUT_COMPONENTS}
    ro_modal["surprise"] = 1

    spec = {
        "spec_version": "fusion/1.0",
        "built_from": "SEALED-ERA artifacts only (2026-05-26 ACE profiles + 20260526 RPV rows); predates fresh data",
        "ace_component": {"column_name": modal_ace,
                          "winner_tally": dict(collections.Counter([w[2] for w in winners])),
                          "per_cell_sign": dict(sorted(ace_signs.items())),
                          "modal_sign_fallback": ace_modal_sign},
        "readout_components": READOUT_COMPONENTS,
        "readout_per_cell_sign": dict(sorted(ro_signs.items())),
        "readout_modal_sign_fallback": ro_modal,
        "surprise_clamp_disagreements": disagreements,
        "fusion_cells": {
            "fusion_rank_mean_full": {"components": ["ACE_modal"] + READOUT_COMPONENTS,
                                      "endpoints": ["primary"],
                                      "note": "includes surprise -> excluded from geometric endpoint"},
            "fusion_rank_mean_geom": {"components": ["ACE_modal", "null_ratio_post_rank1", "fisher_eff_rank"],
                                      "endpoints": ["primary", "geometric"]},
        },
        "definition": ("per cell: rank-transform each oriented component to (rank-0.5)/n_finite in [0,1] "
                       "over the cell's samples; fusion score = mean of component ranks; "
                       "NaN if ANY component is non-finite for that sample"),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion_signs.json")
    json.dump(spec, open(out, "w"), indent=1, sort_keys=True)
    print(f"modal ACE winner: {modal_ace} (tally {spec['ace_component']['winner_tally']})")
    print(f"ACE per-cell signs: {len(ace_signs)} entries, modal fallback {ace_modal_sign}")
    print(f"readout sign entries: {len(ro_signs)}, modal fallbacks {ro_modal}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
