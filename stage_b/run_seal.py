#!/usr/bin/env python3
"""
Stage B launch harness - the fresh-seed sealed unified-panel run.

For each cohort (model, task): collect ACE (t=0, 21 cells, validated byte-exact) + the FRESH
readout pass (gen_step=1: RPV + null_ratio + surprise + p_max) + merge by sample_idx + run the
sealed nested-OOB dispatcher over the unified 27-cell panel. Emit one CalibrationProfile per cell
and evaluate the two pre-registered endpoints.

Run with the t0 venv. Data files must be FRESH-seed (see stage_b/RUN_README.md); reusing the
sealed 20260526 files makes this a preview, not the registered seal (pass --allow-sealed-data).

    /Users/msrk/Documents/t0-morphology-furnace/.venv/bin/python stage_b/run_seal.py \
        --seed <FRESH> --anli <fresh_anli.jsonl> --triviaqa <fresh_triviaqa.jsonl> \
        --out-dir stage_b/profiles
"""
import sys, os, json, argparse, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC

COHORT = [  # pre-reg cohort P + Llama-3.1-8B (10 models)
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
]
SEALED_DATA = {  # 20260526 sealed-seed files - PREVIEW only (data reuse); fresh files override
    "anli_r1": "/Users/msrk/Documents/t0-morphology-furnace/experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl",
    "triviaqa_paired": "/Users/msrk/Documents/t0-morphology-furnace/experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl",
}


def run_cell(model, benchmark, data_path, seed, nboot, limit):
    ace = CC.collect_ace_matrix(model, data_path, seed=seed, max_new_tokens=1, limit=limit)
    ro = CC.collect_readout_matrix_fresh(model, benchmark, data_path, seed=seed, limit=limit)
    prof = CC.merge_and_calibrate(ace, ro, n_bootstrap=nboot, seed=seed)
    prof["benchmark"] = benchmark
    prof["data_path"] = data_path
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="FRESH analysis/collection seed")
    ap.add_argument("--anli", default=None, help="fresh ANLI R1 jsonl (omit only with --allow-sealed-data)")
    ap.add_argument("--triviaqa", default=None, help="fresh TriviaQA paired jsonl")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "profiles"))
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help=">0 = smoke on first N samples")
    ap.add_argument("--models", default=None, help="comma-substr filter for a smoke subset")
    ap.add_argument("--allow-sealed-data", action="store_true",
                    help="permit the 20260526 sealed files = PREVIEW, not the registered seal")
    a = ap.parse_args()

    tasks = {}
    tasks["anli_r1"] = a.anli or (SEALED_DATA["anli_r1"] if a.allow_sealed_data else None)
    tasks["triviaqa_paired"] = a.triviaqa or (SEALED_DATA["triviaqa_paired"] if a.allow_sealed_data else None)
    missing = [t for t, p in tasks.items() if not p]
    if missing:
        sys.exit(f"ERROR: no data file for {missing}. Provide --anli/--triviaqa (fresh seed) or "
                 f"--allow-sealed-data for a preview. See stage_b/RUN_README.md.")
    is_preview = not (a.anli and a.triviaqa)
    models = [m for m in COHORT if (not a.models or any(s in m for s in a.models.split(",")))]
    os.makedirs(a.out_dir, exist_ok=True)

    cells, errors = [], []
    for model in models:
        for benchmark, data_path in tasks.items():
            tag = f"{model.split('/')[-1]}/{benchmark}"
            print(f"\n===== {tag} =====", flush=True)
            try:
                prof = run_cell(model, benchmark, data_path, a.seed, a.nboot,
                                a.limit if a.limit > 0 else None)
                outp = os.path.join(a.out_dir, benchmark, f"{model.split('/')[-1]}.profile.json")
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                json.dump(prof, open(outp, "w"), indent=1)
                pr, ge = prof["primary_full_panel"], prof["secondary_geometric_only"]
                print(f"[{tag}] PRIMARY {pr['winner']} CI_lo={pr['oob_auroc_ci_lo']} dep={pr['deployable']} "
                      f"| GEOM {ge['winner']} CI_lo={ge['oob_auroc_ci_lo']} dep={ge['deployable']}")
                cells.append({"tag": tag, "model": model, "benchmark": benchmark,
                              "primary": pr, "geometric": ge})
            except Exception as e:
                print(f"[{tag}] ERROR: {e}", flush=True)
                traceback.print_exc()
                errors.append({"tag": tag, "error": str(e)})

    # ---- endpoints ----
    n = len(cells)
    prim_dep = sum(c["primary"]["deployable"] for c in cells)
    geom_dep = sum(c["geometric"]["deployable"] for c in cells)
    # registered thresholds are absolute for the 20-cell cohort; scale otherwise
    prim_bar = 19 if n == 20 else int(np.ceil(0.95 * n))
    geom_bar = 17 if n == 20 else int(np.ceil(0.85 * n))
    from collections import Counter
    winmap = Counter(c["geometric"]["winner"] for c in cells if c["geometric"]["winner"])
    summary = {
        "is_preview": is_preview, "seed": a.seed, "n_cells": n, "n_errors": len(errors),
        "primary_deployable": prim_dep, "primary_bar": prim_bar, "primary_pass": prim_dep >= prim_bar,
        "geometric_deployable": geom_dep, "geometric_bar": geom_bar, "geometric_pass": geom_dep >= geom_bar,
        "geometric_winmap": dict(winmap), "errors": errors,
        "cells": [{"tag": c["tag"], "primary_winner": c["primary"]["winner"],
                   "primary_ci_lo": c["primary"]["oob_auroc_ci_lo"], "primary_dep": c["primary"]["deployable"],
                   "geom_winner": c["geometric"]["winner"], "geom_ci_lo": c["geometric"]["oob_auroc_ci_lo"],
                   "geom_dep": c["geometric"]["deployable"]} for c in cells],
    }
    json.dump(summary, open(os.path.join(a.out_dir, "SUMMARY.json"), "w"), indent=1)
    print(f"\n{'='*60}\n{'PREVIEW' if is_preview else 'SEALED'} run | seed {a.seed} | cells {n} errors {len(errors)}")
    print(f"PRIMARY (full panel):    {prim_dep}/{n} deployable (bar {prim_bar}) -> {'PASS' if summary['primary_pass'] else 'FAIL'}")
    print(f"SECONDARY (geom-only):   {geom_dep}/{n} deployable (bar {geom_bar}) -> {'PASS' if summary['geometric_pass'] else 'FAIL'}")
    print(f"geometric win-map: {dict(winmap)}")
    print(f"summary -> {os.path.join(a.out_dir, 'SUMMARY.json')}")


if __name__ == "__main__":
    main()
