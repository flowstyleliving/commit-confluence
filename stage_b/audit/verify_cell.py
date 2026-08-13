#!/usr/bin/env python3
"""Fail-closed spot verifier: re-run REAL model forwards on selected rows of a sealed
cell and require bit-identical (float64 byte-equality) agreement with the committed
matrix, under checked provenance.

Supersedes the v1 fail-open scripts (spot_reextract*.py, removed; see git history at
e8e4db2). v2 hardening after the 2026-08-12 gpt-5.6-sol audit:
  - every check is asserted and the process exits nonzero on ANY discrepancy;
  - expected comparison counts, panel identity (exact triples, no substring matching),
    row identity, dtype, and finiteness are all enforced before equality is judged;
  - equality is float64 BYTE equality (catches +0.0/-0.0; NaN never compares equal);
  - provenance is compared against the profile: per-module code hashes
    (CC.module_hashes() of the actually-imported files vs provenance.module_hashes)
    and the resolved HF model snapshot revision. Code-hash drift is fatal unless
    explicitly acknowledged with --acknowledge-code-drift (recorded, never silent).
    Weight-snapshot drift is always fatal.
  - a machine-readable JSON run record (environment, hashes, comparisons, verdict)
    is written for every run, pass or fail.

Readout rows are PREFIX-ONLY by design: the readout extractor derives per-row RNG from
the run row index (seed + 100_003 + i), so only a prefix subset is RNG-aligned with the
sealed full run. Scattered rows are valid only for the deterministic attention capture.

To re-extract under byte-exact seal-time t0 code, point CONFLUENCE_T0_REPO at the
vendored copy:  CONFLUENCE_T0_REPO=<repo>/vendor/t0_core
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CC_ROOT = os.path.dirname(os.path.dirname(AUDIT_DIR))
sys.path.insert(0, CC_ROOT)
import confluence_calibrator as CC  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_bits(a, b):
    """float64 byte equality; NaN never equal, +0.0 != -0.0."""
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def git_state(path):
    if not os.path.isdir(os.path.join(path, ".git")):
        return None
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "dirty": bool(dirty)}
    except Exception as e:  # git absent/broken: record, don't crash the verifier
        return {"error": str(e)}


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, detail=""):
        self.rows.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail and not ok else ""))

    @property
    def all_ok(self):
        return all(r["ok"] for r in self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. anli_r1, triviaqa_paired")
    ap.add_argument("--model", required=True, help="model slug, e.g. Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--ace-rows", required=True,
                    help="comma-separated jsonl row indices for the attention re-extraction")
    ap.add_argument("--readout-prefix", type=int, default=0,
                    help="N>0: also re-extract the readout pass on prefix rows 0..N-1")
    ap.add_argument("--profiles-dir", default=os.path.join(CC_ROOT, "stage_b", "profiles"))
    ap.add_argument("--acknowledge-code-drift", action="store_true",
                    help="record (instead of fail on) module-hash drift vs the profile; "
                         "weight-snapshot drift is fatal regardless")
    ap.add_argument("--record", default=None, help="run-record JSON path (default: audit/runs/)")
    a = ap.parse_args()
    ace_rows = [int(x) for x in a.ace_rows.split(",")]

    prof_path = os.path.join(a.profiles_dir, a.task, f"{a.model}.profile.json")
    npz_path = os.path.join(a.profiles_dir, a.task, f"{a.model}.matrix.npz")
    prof = json.load(open(prof_path))
    data_path = os.path.join(CC_ROOT, prof["data_path"])
    model_id = prof["model"]
    seed = prof["provenance"]["seed"]

    ck = Checks()
    print(f"[verify_cell] {a.model}/{a.task} | seed {seed} | T0_REPO={CC.T0_REPO} "
          f"| SEAL_SOURCE={CC.SEAL_SOURCE}")

    # ── artifact-side checks ───────────────────────────────────────────────
    ck.add("data_file_sha256 matches profile stamp",
           sha256_file(data_path) == prof["data_file_sha256"])

    m = np.load(npz_path, allow_pickle=False)
    stored, labels, sidx = m["score_matrix"], m["labels"], m["sample_idx"]
    stored_panel = [tuple(c) for c in json.loads(str(m["panel"]))]
    src_rows = [json.loads(l) for l in open(data_path)]
    ck.add("matrix labels == jsonl labels at sample_idx (all rows)",
           len(sidx) == len(src_rows) == prof["n_aligned"]
           and all(int(labels[i]) == int(src_rows[int(sidx[i])]["label"]) for i in range(len(sidx))),
           f"n={len(sidx)} jsonl={len(src_rows)} profile n_aligned={prof['n_aligned']}")
    ck.add("stored matrix dtype float64", stored.dtype == np.float64, str(stored.dtype))
    pos = {int(v): i for i, v in enumerate(sidx)}
    ck.add("requested rows present in matrix", all(r in pos for r in ace_rows))

    # ── model-weight provenance (always fatal on drift) ────────────────────
    snap = CC.model_snapshot_sha(model_id)
    rec_snap = prof["provenance"].get("model_snapshot_sha") or {}
    ck.add("model snapshot revision matches profile",
           bool(snap) and snap.get("resolved_revision") == rec_snap.get("resolved_revision"),
           f"current={snap and snap.get('resolved_revision')} recorded={rec_snap.get('resolved_revision')}")

    # ── ACE re-extraction (scattered rows; deterministic capture) ──────────
    tmp = tempfile.mkdtemp(prefix="cc_verify_")
    subset = os.path.join(tmp, "subset.jsonl")
    src_lines = open(data_path).read().splitlines()
    with open(subset, "w") as f:
        for r in ace_rows:
            f.write(src_lines[r] + "\n")
    ck.add("subset lines byte-identical to source rows",
           open(subset).read().splitlines() == [src_lines[r] for r in ace_rows])

    print(f"[verify_cell] ACE pass: {len(ace_rows)} forwards ...", flush=True)
    ace = CC.collect_ace_matrix(model_id, subset, seed=seed, max_new_tokens=1)
    exp_att = [c for c in stored_panel if c[1] == "Attention"]
    ck.add("fresh ACE panel == stored Attention panel (exact triples, in order)",
           [tuple(c) for c in ace["panel"]] == exp_att,
           f"fresh={len(ace['panel'])} expected={len(exp_att)}")
    ck.add("fresh ACE shape/dtype",
           ace["score_matrix"].shape == (len(ace_rows), len(exp_att))
           and ace["score_matrix"].dtype == np.float64,
           f"{ace['score_matrix'].shape} {ace['score_matrix'].dtype}")
    ck.add("fresh ACE values all finite", bool(np.isfinite(ace["score_matrix"]).all()))

    col = {c: j for j, c in enumerate(stored_panel)}
    n_cmp, n_bad, worst = 0, 0, 0.0
    for k, jrow in enumerate(ace_rows):
        for j, c in enumerate(exp_att):
            f_v, s_v = ace["score_matrix"][k, j], stored[pos[jrow], col[c]]
            n_cmp += 1
            if not (np.isfinite(s_v) and same_bits(f_v, s_v)):
                n_bad += 1
                worst = max(worst, abs(float(f_v) - float(s_v)))
                print(f"  [FAIL] ACE row {jrow} {c[2]}: fresh={f_v!r} stored={s_v!r}")
    exp_n = len(ace_rows) * len(exp_att)
    ck.add(f"ACE comparison count == {exp_n}", n_cmp == exp_n, str(n_cmp))
    ck.add("ACE values byte-identical", n_bad == 0, f"{n_bad} mismatches, worst |diff|={worst:.3e}")

    # ── readout re-extraction (prefix rows only; RNG is row-index derived) ─
    ro_stats = None
    if a.readout_prefix > 0:
        print(f"[verify_cell] readout pass: prefix limit={a.readout_prefix} ...", flush=True)
        ro = CC.collect_readout_matrix_fresh(model_id, a.task, data_path,
                                             seed=seed, limit=a.readout_prefix)
        exp_ro = [c for c in stored_panel if c[1] == "Readout"]
        ck.add("readout sample_idx == exact prefix (no drops)",
               [int(i) for i in ro["sample_idx"]] == list(range(a.readout_prefix)),
               str([int(i) for i in ro["sample_idx"]]))
        ck.add("fresh readout panel == stored Readout panel (exact triples, in order)",
               [tuple(c) for c in ro["panel"]] == exp_ro)
        ck.add("fresh readout values all finite", bool(np.isfinite(ro["score_matrix"]).all()))
        rn, rb, rw = 0, 0, 0.0
        for k in range(a.readout_prefix):
            for j, c in enumerate(exp_ro):
                f_v, s_v = ro["score_matrix"][k, j], stored[pos[k], col[c]]
                rn += 1
                if not (np.isfinite(s_v) and same_bits(f_v, s_v)):
                    rb += 1
                    rw = max(rw, abs(float(f_v) - float(s_v)))
                    print(f"  [FAIL] readout row {k} {c[2]}: fresh={f_v!r} stored={s_v!r}")
        exp_rn = a.readout_prefix * len(exp_ro)
        ck.add(f"readout comparison count == {exp_rn}", rn == exp_rn, str(rn))
        ck.add("readout values byte-identical", rb == 0, f"{rb} mismatches, worst |diff|={rw:.3e}")
        ro_stats = {"n_compared": rn, "n_mismatch": rb, "worst_abs_diff": rw}

    # ── code provenance (after extraction, so hashes reflect imported files) ─
    cur_mods = CC.module_hashes()
    rec_mods = prof["provenance"]["module_hashes"]
    mod_table, drift = {}, []
    for name, rec_h in rec_mods.items():
        cur_h = cur_mods.get(name)
        status = "MATCH" if cur_h == rec_h else ("MISSING" if cur_h is None else "MISMATCH")
        mod_table[name] = {"recorded": rec_h, "current": cur_h, "status": status}
        if status != "MATCH":
            drift.append(name)
    if drift:
        msg = f"module-hash drift vs profile: {drift}"
        if a.acknowledge_code_drift:
            print(f"  [WARN] {msg} (explicitly acknowledged; comparisons above still held)")
        ck.add("module hashes match profile (or drift explicitly acknowledged)",
               a.acknowledge_code_drift, msg)
    else:
        ck.add("module hashes match profile (all recorded modules)", True)

    # ── verdict + machine-readable record ──────────────────────────────────
    verdict = ("PASS" if ck.all_ok and not drift
               else "PASS_WITH_ACKNOWLEDGED_CODE_DRIFT" if ck.all_ok
               else "FAIL")
    record = {
        "schema": "cc-audit-run-record/1",
        "argv": sys.argv,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cell": {"model": model_id, "slug": a.model, "task": a.task, "seed": seed},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mlx_lm": __import__("mlx_lm").__version__,
            "T0_REPO": CC.T0_REPO,
            "seal_source": CC.SEAL_SOURCE,
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "git": {"commit-confluence": git_state(CC_ROOT), "t0_repo": git_state(CC.T0_REPO)},
        },
        "provenance": {"module_hashes": mod_table, "drifted_modules": drift,
                       "code_drift_acknowledged": a.acknowledge_code_drift,
                       "model_snapshot": {"current": snap, "recorded": rec_snap}},
        "comparisons": {
            "ace": {"rows": ace_rows, "n_compared": n_cmp, "n_mismatch": n_bad,
                    "worst_abs_diff": worst},
            "readout": ro_stats,
            "equality_criterion": "float64 byte equality (tobytes)",
        },
        "checks": ck.rows,
        "verdict": verdict,
    }
    if a.record:
        rec_path = a.record
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(os.path.join(AUDIT_DIR, "runs"), exist_ok=True)
        rec_path = os.path.join(AUDIT_DIR, "runs", f"{a.model}__{a.task}__{stamp}.json")
    json.dump(record, open(rec_path, "w"), indent=1)
    print(f"\n[verify_cell] VERDICT: {verdict}  (record -> {rec_path})")
    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()
