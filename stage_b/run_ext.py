#!/usr/bin/env python3
"""Out-of-sample EXTENSION run — see stage_b/PRE_REGISTRATION_EXT.md (frozen 2026-06-18).

Phase 1: gemma-3-12b-it + Qwen2.5-14B-Instruct x {anli_r1, triviaqa_paired}, seed 20260612,
strict n=200, nboot=2000, via the seal's own run_cell. Writes stage_b/profiles_ext/ ONLY — the
registered seal (stage_b/profiles/) is never touched. Idempotent: skips cells whose profile exists,
skips models not yet in the HF cache. gemma-3-12b/anli runs first (headline orphan-scale question).
"""
import sys, os, json, time
os.chdir(os.path.expanduser("~/Documents/commit-confluence"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "stage_b"))
from run_seal import run_cell

SEED, NBOOT = 20260612, 2000
DATA = {"anli_r1": "stage_b/data/anli_R1_seed20260612_n200.jsonl",
        "triviaqa_paired": "stage_b/data/triviaqa_paired_seed20260612_n200.jsonl"}
# headline cell first
PLAN = [("mlx-community/gemma-3-12b-it-4bit", "anli_r1"),
        ("mlx-community/gemma-3-12b-it-4bit", "triviaqa_paired"),
        ("mlx-community/Qwen2.5-14B-Instruct-4bit", "anli_r1"),
        ("mlx-community/Qwen2.5-14B-Instruct-4bit", "triviaqa_paired")]
OUT = "stage_b/profiles_ext"
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def cached(m):
    tag = m.replace("/", "--")
    try:
        return any(tag in d for d in os.listdir(HUB))
    except Exception:
        return False


def summ(prof):
    g = prof.get("secondary_geometric_only", {})
    p = prof.get("primary_full_panel", {})
    return dict(geom_winner=g.get("winner"), geom_ci_lo=g.get("oob_auroc_ci_lo"),
                geom_dep=g.get("deployable"), prim_winner=p.get("winner"),
                prim_ci_lo=p.get("oob_auroc_ci_lo"), prim_dep=p.get("deployable"),
                controls_pass=prof.get("controls", {}).get("pass"), n=prof.get("n_aligned"))


print("EXT RUN START", flush=True)
for m, t in PLAN:
    slug = m.split("/")[-1]
    d = os.path.join(OUT, t)
    os.makedirs(d, exist_ok=True)
    pj, npz = f"{d}/{slug}.profile.json", f"{d}/{slug}.matrix.npz"
    if os.path.exists(pj):
        print(f"EXIST {slug}/{t} -> skip", flush=True)
        continue
    if not cached(m):
        print(f"SKIP {slug}/{t}: model not in cache yet", flush=True)
        continue
    t0 = time.time()
    print(f"\n===== RUN {slug}/{t} (n=200, nboot={NBOOT}, strict) =====", flush=True)
    try:
        prof = run_cell(m, t, DATA[t], seed=SEED, nboot=NBOOT, limit=None,
                        npz_path=npz, strict=True)
        json.dump(prof, open(pj, "w"), indent=2, default=str)
        s = summ(prof)
        s["secs"] = round(time.time() - t0)
        print(f"DONE {slug}/{t}: {json.dumps(s)}", flush=True)
    except Exception as e:
        import traceback
        print(f"ERROR {slug}/{t}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
print("\nEXT RUN COMPLETE", flush=True)
