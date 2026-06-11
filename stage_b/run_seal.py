#!/usr/bin/env python3
"""
Stage B launch harness - the fresh-seed sealed unified-panel run.

For each cohort (model, task): collect ACE (t=0, 21 cells, validated byte-exact) + the FRESH
readout pass (gen_step=1: RPV + null_ratio + surprise + p_max) + merge by sample_idx + append
the two pre-registered fusion cells + run the sealed nested-OOB dispatcher over the unified
29-cell panel with per-cell shuffled-label controls. Persists the merged score matrix per cell
(.npz) for the pre-registered descriptive analyses (LOMO / transfer / label-efficiency:
stage_b/analyze_universality.py). Emits one CalibrationProfile per cell and evaluates the two
registered endpoints over the PLANNED cohort (errored/missing cells count as NOT deployable -
amendment A4; the denominator never shrinks).

Run with the t0 venv. Data files must be FRESH-seed and pass stage_b/check_fresh_data.py;
sealed-content files and pilot seeds are refused unless --allow-sealed-data (= preview).

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
# C3: pilot/sealed-era seeds may never stamp a registered run
PILOT_SEEDS = {20260512, 20260526, 20260610, 20260611}


def sealed_content_hashes():
    return {CC._sha256_file(p) for p in SEALED_DATA.values() if os.path.exists(p)}


def run_cell(model, benchmark, data_path, seed, nboot, limit, npz_path, strict):
    ace = CC.collect_ace_matrix(model, data_path, seed=seed, max_new_tokens=1, limit=limit)
    ro = CC.collect_readout_matrix_fresh(model, benchmark, data_path, seed=seed, limit=limit)
    # M4: a registered cell must score every planned sample (max_dropped=0); previews/smokes
    # stay lenient so a stray non-finite row does not abort a sanity check.
    mm = CC.merge_matrices(ace, ro, max_dropped=0 if strict else None)
    # persist the merged matrix BEFORE selection (E1-E3 inputs survive a selection crash)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, score_matrix=mm["score_matrix"], labels=mm["labels"],
                        sample_idx=mm["sample_idx"], panel=json.dumps(mm["panel"]),
                        meta=json.dumps({"model": model, "benchmark": benchmark,
                                         "data_path": str(data_path), "seed": seed,
                                         "ace_data_hash": mm.get("ace_data_hash"),
                                         "readout_data_hash": mm.get("readout_data_hash")}))
    prof = CC.calibrate_merged(mm, n_bootstrap=nboot, seed=seed,
                               model_id=model, benchmark=benchmark)
    prof["data_path"] = str(data_path)
    # M3: stamp the exact data-file bytes + nboot so --resume can validate provenance.
    prof["data_file_sha256"] = CC._sha256_file(data_path)
    prof.setdefault("provenance", {})["n_bootstrap"] = nboot
    return prof


def _validate_resumed_profile(prof, *, model, benchmark, data_path, seed, nboot, npz_path):
    """M3: a resumed profile may only count toward the registered denominator if its provenance
    matches the CURRENT run exactly. Any drift (seed, nboot, model, task, data bytes, code/spec
    hashes, or a missing matrix) raises -> the cell is treated as an error (forces FAIL) rather
    than silently mixing a smoke/preview/old-seed/old-code result into the 20-cell cohort."""
    prov = prof.get("provenance") or {}
    mismatches = []
    if prof.get("model") != model:
        mismatches.append(f"model {prof.get('model')!r} != {model!r}")
    if prof.get("benchmark") != benchmark:
        mismatches.append(f"benchmark {prof.get('benchmark')!r} != {benchmark!r}")
    if prov.get("seed") != seed:
        mismatches.append(f"seed {prov.get('seed')!r} != {seed!r}")
    if prov.get("n_bootstrap") != nboot:
        mismatches.append(f"n_bootstrap {prov.get('n_bootstrap')!r} != {nboot!r}")
    cur_data = CC._sha256_file(data_path)
    if prof.get("data_file_sha256") != cur_data:
        mismatches.append(f"data_file_sha256 {prof.get('data_file_sha256')!r} != {cur_data!r}")
    cur_mods = CC.module_hashes()
    if prov.get("module_hashes") != cur_mods:
        mismatches.append("module_hashes drift (code or fusion_signs.json changed since the profile)")
    if not os.path.exists(npz_path):
        mismatches.append(f"missing matrix npz {npz_path}")
    if mismatches:
        raise RuntimeError("refusing to resume a stale/mismatched profile: " + "; ".join(mismatches)
                           + ". Delete it to recompute, or resume the original registered command.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="FRESH analysis/collection seed")
    ap.add_argument("--anli", default=None, help="fresh ANLI R1 jsonl (omit only with --allow-sealed-data)")
    ap.add_argument("--triviaqa", default=None, help="fresh TriviaQA paired jsonl")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "profiles"))
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help=">0 = smoke on first N samples")
    ap.add_argument("--models", default=None, help="comma-substr filter for a smoke subset")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose profile json already exists (multi-hour run safety)")
    ap.add_argument("--allow-sealed-data", action="store_true",
                    help="permit sealed-content files / pilot seeds = PREVIEW, not the registered seal")
    a = ap.parse_args()

    tasks = {}
    tasks["anli_r1"] = a.anli or (SEALED_DATA["anli_r1"] if a.allow_sealed_data else None)
    tasks["triviaqa_paired"] = a.triviaqa or (SEALED_DATA["triviaqa_paired"] if a.allow_sealed_data else None)
    missing = [t for t, p in tasks.items() if not p]
    if missing:
        sys.exit(f"ERROR: no data file for {missing}. Provide --anli/--triviaqa (fresh seed) or "
                 f"--allow-sealed-data for a preview. See stage_b/RUN_README.md.")

    # ---- C3 provenance guards: CONTENT-based, not path-based ----
    sealed_hashes = sealed_content_hashes()
    sealed_content_used = any(
        os.path.exists(p) and CC._sha256_file(p) in sealed_hashes for p in tasks.values())
    pilot_seed_used = a.seed in PILOT_SEEDS
    if (sealed_content_used or pilot_seed_used) and not a.allow_sealed_data:
        sys.exit("ERROR: sealed-content data file or pilot seed detected "
                 f"(sealed_content={sealed_content_used}, pilot_seed={pilot_seed_used}). "
                 "A registered run needs FRESH data + a fresh seed (run stage_b/check_fresh_data.py "
                 "first). Pass --allow-sealed-data only for an explicit preview.")
    is_preview = sealed_content_used or pilot_seed_used or not (a.anli and a.triviaqa)

    models = [m for m in COHORT if (not a.models or any(s in m for s in a.models.split(",")))]
    planned = [(m, b, p) for m in models for b, p in tasks.items()]
    is_registered_cohort = (len(planned) == 20 and not a.limit and not is_preview)
    # `strict` = this is a real fresh-data run (not a preview, not a smoke). Strict runs enforce
    # the fresh-data gate (M1) AND a zero-drop sample denominator (M4).
    strict = (not is_preview) and (not a.limit)

    # ---- M1: the launch harness ENFORCES the fresh-data gate (A5), not just the docs/CLI ----
    # check_fresh_data.py existed but run_seal never called it: a reserialized sealed file or a
    # partially-overlapping fresh file could launch as a registered cohort. Gate every fresh task
    # file against its sealed counterpart IN-PROCESS before any model forward.
    if strict:
        import check_fresh_data as GATE
        gate_task = {"anli_r1": "anli", "triviaqa_paired": "triviaqa"}
        gate_reports = {}
        for b, p in tasks.items():
            sealed_ref = SEALED_DATA.get(b)
            rep = GATE.run_gate(p, sealed_ref, gate_task[b], expect_n=200)
            gate_reports[b] = rep
            verdict = "PASS" if rep["pass"] else "FAIL"
            print(f"[gate] {b}: {verdict} (n={rep['n']} overlap={rep['n_overlap_with_sealed']} "
                  f"qid_overlap={rep.get('n_qid_overlap_with_sealed')})", flush=True)
        failed = {b: r["hard_failures"] for b, r in gate_reports.items() if not r["pass"]}
        if failed:
            sys.exit(f"ERROR: fresh-data gate FAILED, refusing to launch a registered run: {failed}. "
                     "Fix the data (stage_b/check_fresh_data.py) before launch.")
    os.makedirs(a.out_dir, exist_ok=True)

    results, errors = {}, []
    for model, benchmark, data_path in planned:
        tag = f"{model.split('/')[-1]}/{benchmark}"
        outp = os.path.join(a.out_dir, benchmark, f"{model.split('/')[-1]}.profile.json")
        npzp = os.path.join(a.out_dir, benchmark, f"{model.split('/')[-1]}.matrix.npz")
        print(f"\n===== {tag} =====", flush=True)
        try:
            if a.resume and os.path.exists(outp):
                prof = json.load(open(outp))
                if strict:  # M3: only trust a resumed profile whose provenance matches this run
                    _validate_resumed_profile(prof, model=model, benchmark=benchmark,
                                              data_path=data_path, seed=a.seed, nboot=a.nboot,
                                              npz_path=npzp)
                print(f"[{tag}] resumed from existing profile", flush=True)
            else:
                prof = run_cell(model, benchmark, data_path, a.seed, a.nboot,
                                a.limit if a.limit > 0 else None, npzp, strict)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                json.dump(prof, open(outp, "w"), indent=1)
            pr, ge = prof["primary_full_panel"], prof["secondary_geometric_only"]
            ctrl = (prof.get("controls") or {}).get("pass")
            print(f"[{tag}] PRIMARY {pr['winner']} CI_lo={pr['oob_auroc_ci_lo']} dep={pr['deployable']} "
                  f"| GEOM {ge['winner']} CI_lo={ge['oob_auroc_ci_lo']} dep={ge['deployable']} "
                  f"| controls_pass={ctrl}")
            results[tag] = {"status": "ok", "primary": pr, "geometric": ge, "controls_pass": ctrl}
        except Exception as e:
            print(f"[{tag}] ERROR: {e}", flush=True)
            traceback.print_exc()
            errors.append({"tag": tag, "error": str(e)})
            results[tag] = {"status": "error", "error": str(e)}

    # ---- endpoints over the PLANNED cohort (A4: errors count as NOT deployable) ----
    n_planned = len(planned)
    cell_rows = []
    for model, benchmark, _ in planned:
        tag = f"{model.split('/')[-1]}/{benchmark}"
        r = results.get(tag, {"status": "missing"})
        ok = r.get("status") == "ok"
        cell_rows.append({
            "tag": tag, "status": r.get("status"),
            "primary_winner": r["primary"]["winner"] if ok else None,
            "primary_ci_lo": r["primary"]["oob_auroc_ci_lo"] if ok else None,
            "primary_dep": bool(ok and r["primary"]["deployable"]),
            "geom_winner": r["geometric"]["winner"] if ok else None,
            "geom_ci_lo": r["geometric"]["oob_auroc_ci_lo"] if ok else None,
            "geom_dep": bool(ok and r["geometric"]["deployable"]),
            "controls_pass": r.get("controls_pass") if ok else None,
        })
    prim_dep = sum(c["primary_dep"] for c in cell_rows)
    geom_dep = sum(c["geom_dep"] for c in cell_rows)
    # registered bars are absolute for the 20-cell cohort; scale otherwise (smokes/subsets)
    prim_bar = 19 if n_planned == 20 else int(np.ceil(0.95 * n_planned))
    geom_bar = 17 if n_planned == 20 else int(np.ceil(0.85 * n_planned))
    incomplete = any(c["status"] != "ok" for c in cell_rows)
    control_failures = [c["tag"] for c in cell_rows if c["controls_pass"] is False]
    # M2: an incomplete cohort can never certify a registered PASS. A4 already counts an errored
    # cell as not-deployable, but a crash also means that cell was never EVALUATED - we cannot
    # certify the procedure on partial evidence. `not incomplete` is an explicit pass precondition.
    primary_pass = bool(prim_dep >= prim_bar and not incomplete)
    geometric_pass = bool(geom_dep >= geom_bar and not incomplete)
    from collections import Counter
    winmap = Counter(c["geom_winner"] for c in cell_rows if c["geom_winner"])
    summary = {
        "is_preview": is_preview, "is_registered_cohort": is_registered_cohort,
        "incomplete": incomplete, "seed": a.seed,
        "n_planned": n_planned, "n_ok": sum(c["status"] == "ok" for c in cell_rows),
        "n_errors": len(errors),
        "primary_deployable": prim_dep, "primary_bar": prim_bar, "primary_pass": primary_pass,
        "geometric_deployable": geom_dep, "geometric_bar": geom_bar, "geometric_pass": geometric_pass,
        "control_failures": control_failures,
        "geometric_winmap": dict(winmap), "errors": errors, "cells": cell_rows,
        "module_hashes": CC.module_hashes(),
    }
    json.dump(summary, open(os.path.join(a.out_dir, "SUMMARY.json"), "w"), indent=1)
    kind = "PREVIEW" if is_preview else ("SEALED" if is_registered_cohort else "SUBSET")
    print(f"\n{'='*60}\n{kind} run | seed {a.seed} | planned {n_planned} ok {summary['n_ok']} "
          f"errors {len(errors)}{' | INCOMPLETE' if incomplete else ''}")
    print(f"PRIMARY (full panel):    {prim_dep}/{n_planned} deployable (bar {prim_bar}) -> "
          f"{'PASS' if summary['primary_pass'] else 'FAIL'}")
    print(f"SECONDARY (geom-only):   {geom_dep}/{n_planned} deployable (bar {geom_bar}) -> "
          f"{'PASS' if summary['geometric_pass'] else 'FAIL'}")
    if control_failures:
        print(f"!! shuffled-label CONTROL FAILURES (>=2/3 perms exclude 0.5): {control_failures}")
    print(f"geometric win-map: {dict(winmap)}")
    print(f"summary -> {os.path.join(a.out_dir, 'SUMMARY.json')}")


if __name__ == "__main__":
    main()
