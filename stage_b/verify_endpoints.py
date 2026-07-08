#!/usr/bin/env python3
"""verify_endpoints - re-derive the two registered endpoint verdicts from the published matrices.

For every published score matrix (stage_b/profiles/<task>/<slug>.matrix.npz) this script
appends the two pre-registered fusion cells, re-runs the sealed nested-OOB selection for
BOTH pre-registered endpoints at the registered settings (seed 20260612, nboot 2000), and
compares the recomputed winner + OOB CI bounds against the committed .profile.json. It then
re-tallies the endpoint verdicts:

    geometric-only  : deployable >= 17/20  -> registered PASS
    full panel      : deployable >= 19/20  -> registered FAIL (18/20)

The selection RNG is fully seeded, so at the registered settings the recomputation is
byte-exact against the committed profiles. Runs from this repository alone (numpy + scipy +
scikit-learn) - no models, no sealed dependency repo (the vendored sealed_selector.py is
used automatically when t0-morphology-furnace is not importable).

Usage:
  python stage_b/verify_endpoints.py                 # registered settings (a few minutes)
  python stage_b/verify_endpoints.py --nboot 200     # quick smoke: CI bounds shift slightly,
                                                     # exact-match checks are skipped
"""
import sys, os, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import confluence_calibrator as CC

REGISTERED_SEED = 20260612
REGISTERED_NBOOT = 2000
GEOM_BAR, FULL_BAR, N_CELLS = 17, 19, 20


def _parse_panel(raw):
    """Panel entries are JSON triples in the seal npz format; the gemma-4 extension npz
    stores them as stringified python tuples instead - accept both."""
    import ast
    out = []
    for c in json.loads(str(raw)):
        out.append(tuple(ast.literal_eval(c)) if isinstance(c, str) else tuple(c))
    return out


def recompute(npz_path, nboot, seed):
    d = np.load(npz_path, allow_pickle=False)
    M, y = d["score_matrix"], d["labels"]
    panel = _parse_panel(d["panel"])
    meta = json.loads(str(d["meta"]))
    slug = os.path.basename(npz_path).replace(".matrix.npz", "")
    task = os.path.basename(os.path.dirname(npz_path))
    M, panel, _ = CC.append_fusion_columns(M, panel, slug, meta.get("benchmark") or task)
    full = CC.run_selection(M, y, panel, n_bootstrap=nboot, seed=seed)
    geom_keys = {c[2] for c in panel if c[2] not in CC.NON_GEOMETRIC_KEYS}
    geom = CC.run_selection(M, y, panel, n_bootstrap=nboot, seed=seed, restrict_keys=geom_keys)
    return slug, task, full, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles"))
    ap.add_argument("--nboot", type=int, default=REGISTERED_NBOOT)
    ap.add_argument("--seed", type=int, default=REGISTERED_SEED)
    a = ap.parse_args()
    registered_settings = (a.nboot == REGISTERED_NBOOT and a.seed == REGISTERED_SEED)
    print(f"selector source: {CC.SEAL_SOURCE}")
    if not registered_settings:
        print(f"NOTE: non-registered settings (nboot={a.nboot}, seed={a.seed}) - "
              "byte-exact comparison to the committed profiles is skipped.")

    npzs = sorted(glob.glob(os.path.join(a.profiles_dir, "*", "*.matrix.npz")))
    if not npzs:
        sys.exit(f"no matrices under {a.profiles_dir}")
    n_full_dep = n_geom_dep = n_mismatch = n_compared = 0
    for npz in npzs:
        slug, task, full, geom = recompute(npz, a.nboot, a.seed)
        n_full_dep += bool(full["deployable"]); n_geom_dep += bool(geom["deployable"])
        line = (f"{slug}/{task}: geom ci_lo={geom['oob_auroc_ci_lo']:.4f} "
                f"({'dep' if geom['deployable'] else 'NOT dep'}), "
                f"full ci_lo={full['oob_auroc_ci_lo']:.4f} "
                f"({'dep' if full['deployable'] else 'NOT dep'})")
        profp = npz.replace(".matrix.npz", ".profile.json")
        if registered_settings and os.path.exists(profp):
            prof = json.load(open(profp))
            n_compared += 1
            ok = True
            for mine, key in ((full, "primary_full_panel"), (geom, "secondary_geometric_only")):
                ref = prof[key]
                if (mine["winner"] != ref["winner"]
                        or abs(mine["oob_auroc_ci_lo"] - ref["oob_auroc_ci_lo"]) > 1e-12
                        or mine["deployable"] != ref["deployable"]):
                    ok = False
            line += "  [MATCHES profile]" if ok else "  [!! DIFFERS from committed profile]"
            n_mismatch += (not ok)
        print(line)

    print()
    if len(npzs) == N_CELLS:
        print(f"geometric-only endpoint: {n_geom_dep}/{len(npzs)} deployable "
              f"(bar >= {GEOM_BAR}/{N_CELLS}) -> {'PASS' if n_geom_dep >= GEOM_BAR else 'FAIL'}")
        print(f"full-panel endpoint:     {n_full_dep}/{len(npzs)} deployable "
              f"(bar >= {FULL_BAR}/{N_CELLS}) -> {'PASS' if n_full_dep >= FULL_BAR else 'FAIL'}")
    else:
        print(f"deployable: geometric {n_geom_dep}/{len(npzs)}, full {n_full_dep}/{len(npzs)} "
              "(not the registered 20-deployment cohort - no endpoint verdict)")
    if registered_settings and n_compared:
        print(f"profile comparison: {n_compared - n_mismatch}/{n_compared} byte-exact matches")
        if n_mismatch:
            sys.exit(1)


if __name__ == "__main__":
    main()
