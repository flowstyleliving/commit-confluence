#!/usr/bin/env python3
"""Fail-closed spot verifier: re-run REAL model forwards on selected rows of a sealed
cell and require float64 byte-identical agreement with the committed matrix, under
checked provenance.

v2.1 after the second gpt-5.6-sol audit round (v2 at 7edafee, v1 at e8e4db2):
  - a JSON run record is written for every run that reaches OR crashes out of the
    check flow (verdict ERROR on exception; exit 2). A run killed externally still
    leaves no record - that limitation stands.
  - the record binds itself to the artifacts: sha256 of this verifier file, the
    profile, the matrix npz, and the data jsonl are all recorded.
  - --acknowledge-code-drift now takes an explicit comma-separated module allowlist;
    only the named modules may mismatch the profile's recorded hashes. Unlisted
    drift stays fatal. (v2's boolean form was a blanket waiver.)
  - readout gets the same dtype/shape assertions as ACE; panel triples are asserted
    unique; protocol constants (21 attention + 6 readout columns) are asserted;
    sample_idx is asserted a complete permutation; matrix meta (model/benchmark/
    seed/data hashes) is checked against the profile; duplicate or negative row
    arguments are rejected.
  - git dirty-state is computed excluding stage_b/audit/runs/ (run records are
    outputs of this tool, not code under audit).

Known open limitations (deliberate, documented in README): the model-weight check
compares the resolved HF snapshot *revision pointer* under the default cache layout -
it does not hash weight shards, and a non-default HF cache location could diverge
from what MLX loads; dependency coverage records versions (numpy/mlx/mlx_lm) but
does not hash installed libraries; records carry no cryptographic attestation.

Readout rows are PREFIX-ONLY by design: the readout extractor derives per-row RNG
from the run row index (seed + 100_003 + i), so only a prefix subset is RNG-aligned
with the sealed full run. Scattered rows are valid only for the deterministic
attention capture.

To re-extract under the reconstructed seal-time t0 code, point CONFLUENCE_T0_REPO at
the vendored copy:  CONFLUENCE_T0_REPO=<repo>/vendor/t0_core
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
import traceback

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CC_ROOT = os.path.dirname(os.path.dirname(AUDIT_DIR))
sys.path.insert(0, CC_ROOT)
import confluence_calibrator as CC  # noqa: E402

N_ATTENTION_CELLS = 21  # sealed ACE t0 panel with v-norms
N_READOUT_CELLS = 6     # RPV + null_ratio + surprise + p_max


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_bits(a, b):
    """float64 byte equality after np.float64 coercion. Callers must enforce source
    dtype and finiteness separately - this alone does not prove native-byte identity
    of differently-typed sources."""
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def git_state(path):
    if not os.path.isdir(os.path.join(path, ".git")):
        return None
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        lines = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.splitlines()
        # run records are outputs of this tool; their presence must not mark the
        # verifier's own execution as running from a dirty tree
        lines = [l for l in lines if "stage_b/audit/runs/" not in l]
        return {"head": head, "dirty_excluding_run_records": bool(lines),
                "dirty_paths": [l.strip() for l in lines[:20]]}
    except Exception as e:
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
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True, help="model slug, e.g. Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--ace-rows", required=True,
                    help="comma-separated distinct nonnegative jsonl row indices")
    ap.add_argument("--readout-prefix", type=int, default=0,
                    help="N>0: also re-extract the readout pass on prefix rows 0..N-1")
    ap.add_argument("--profiles-dir", default=os.path.join(CC_ROOT, "stage_b", "profiles"))
    ap.add_argument("--acknowledge-code-drift", default="", metavar="MODULES",
                    help="comma-separated module names (e.g. confluence_calibrator.py) whose "
                         "hash drift vs the profile is accepted and recorded; any OTHER drift "
                         "remains fatal. Weight-snapshot drift is always fatal.")
    ap.add_argument("--record", default=None, help="run-record JSON path (default: audit/runs/)")
    a = ap.parse_args()

    ace_rows = [int(x) for x in a.ace_rows.split(",")]
    if len(set(ace_rows)) != len(ace_rows) or any(r < 0 for r in ace_rows):
        ap.error("--ace-rows must be distinct nonnegative indices")
    if a.readout_prefix < 0:
        ap.error("--readout-prefix must be >= 0")
    acked = {s for s in a.acknowledge_code_drift.split(",") if s}

    if a.record:
        rec_path = a.record
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        os.makedirs(os.path.join(AUDIT_DIR, "runs"), exist_ok=True)
        rec_path = os.path.join(AUDIT_DIR, "runs", f"{a.model}__{a.task}__{stamp}.json")

    ck = Checks()
    record = {
        "schema": "cc-audit-run-record/2",
        "argv": sys.argv,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "verdict": "ERROR",
    }

    def finalize(verdict, exit_code):
        record["verdict"] = verdict
        record["checks"] = ck.rows
        json.dump(record, open(rec_path, "w"), indent=1)
        print(f"\n[verify_cell] VERDICT: {verdict}  (record -> {rec_path})")
        sys.exit(exit_code)

    try:
        prof_path = os.path.join(a.profiles_dir, a.task, f"{a.model}.profile.json")
        npz_path = os.path.join(a.profiles_dir, a.task, f"{a.model}.matrix.npz")
        prof = json.load(open(prof_path))
        data_path = os.path.join(CC_ROOT, prof["data_path"])
        model_id = prof["model"]
        seed = prof["provenance"]["seed"]

        record["cell"] = {"model": model_id, "slug": a.model, "task": a.task, "seed": seed}
        record["artifact_hashes"] = {
            "verify_cell.py": sha256_file(os.path.abspath(__file__)),
            "profile_json": sha256_file(prof_path),
            "matrix_npz": sha256_file(npz_path),
            "data_jsonl": sha256_file(data_path),
        }
        print(f"[verify_cell] {a.model}/{a.task} | seed {seed} | T0_REPO={CC.T0_REPO}")

        # ── artifact-side checks ───────────────────────────────────────────
        ck.add("data_file_sha256 matches profile stamp",
               record["artifact_hashes"]["data_jsonl"] == prof["data_file_sha256"])

        m = np.load(npz_path, allow_pickle=False)
        stored, labels, sidx = m["score_matrix"], m["labels"], m["sample_idx"]
        stored_panel = [tuple(c) for c in json.loads(str(m["panel"]))]
        meta = json.loads(str(m["meta"]))
        src_rows = [json.loads(l) for l in open(data_path)]
        n = len(src_rows)

        ck.add("matrix meta matches profile (model/benchmark/seed/data hashes)",
               meta.get("model") == prof["model"] and meta.get("benchmark") == prof["benchmark"]
               and meta.get("seed") == seed
               and meta.get("ace_data_hash") == prof.get("ace_data_hash")
               and meta.get("readout_data_hash") == prof.get("readout_data_hash"),
               json.dumps(meta))
        ck.add("stored panel = 21 Attention + 6 Readout, all triples unique",
               len(stored_panel) == N_ATTENTION_CELLS + N_READOUT_CELLS
               and len(set(stored_panel)) == len(stored_panel)
               and sum(c[1] == "Attention" for c in stored_panel) == N_ATTENTION_CELLS
               and sum(c[1] == "Readout" for c in stored_panel) == N_READOUT_CELLS,
               f"len={len(stored_panel)}")
        ck.add("stored matrix shape/dtype",
               stored.shape == (n, N_ATTENTION_CELLS + N_READOUT_CELLS)
               and stored.dtype == np.float64, f"{stored.shape} {stored.dtype}")
        ck.add("sample_idx is a complete permutation of 0..n-1",
               sorted(int(i) for i in sidx) == list(range(n)))
        ck.add("matrix labels == jsonl labels at sample_idx (all rows)",
               n == prof["n_aligned"]
               and all(int(labels[i]) == int(src_rows[int(sidx[i])]["label"]) for i in range(n)),
               f"n={n} profile n_aligned={prof['n_aligned']}")
        pos = {int(v): i for i, v in enumerate(sidx)}
        ck.add("requested rows in range", all(r < n for r in ace_rows))

        # ── model-weight provenance (always fatal on drift) ────────────────
        snap = CC.model_snapshot_sha(model_id)
        rec_snap = prof["provenance"].get("model_snapshot_sha") or {}
        cur_rev = (snap or {}).get("resolved_revision")
        ck.add("model snapshot revision matches profile (pointer check, not weight hash)",
               cur_rev is not None and cur_rev == rec_snap.get("resolved_revision"),
               f"current={cur_rev} recorded={rec_snap.get('resolved_revision')}")

        # ── ACE re-extraction (scattered rows; deterministic capture) ──────
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
               [tuple(c) for c in ace["panel"]] == exp_att)
        ck.add("fresh ACE shape/dtype",
               ace["score_matrix"].shape == (len(ace_rows), N_ATTENTION_CELLS)
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
        exp_n = len(ace_rows) * N_ATTENTION_CELLS
        ck.add(f"ACE comparison count == {exp_n} ({len(ace_rows)} distinct rows x 21)",
               n_cmp == exp_n, str(n_cmp))
        ck.add("ACE values byte-identical", n_bad == 0,
               f"{n_bad} mismatches, worst |diff|={worst:.3e}")

        # ── readout re-extraction (prefix rows only) ───────────────────────
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
            ck.add("fresh readout shape/dtype",
                   ro["score_matrix"].shape == (a.readout_prefix, N_READOUT_CELLS)
                   and ro["score_matrix"].dtype == np.float64,
                   f"{ro['score_matrix'].shape} {ro['score_matrix'].dtype}")
            ck.add("fresh readout values all finite",
                   bool(np.isfinite(ro["score_matrix"]).all()))
            rn, rb, rw = 0, 0, 0.0
            for k in range(a.readout_prefix):
                for j, c in enumerate(exp_ro):
                    f_v, s_v = ro["score_matrix"][k, j], stored[pos[k], col[c]]
                    rn += 1
                    if not (np.isfinite(s_v) and same_bits(f_v, s_v)):
                        rb += 1
                        rw = max(rw, abs(float(f_v) - float(s_v)))
                        print(f"  [FAIL] readout row {k} {c[2]}: fresh={f_v!r} stored={s_v!r}")
            exp_rn = a.readout_prefix * N_READOUT_CELLS
            ck.add(f"readout comparison count == {exp_rn}", rn == exp_rn, str(rn))
            ck.add("readout values byte-identical", rb == 0,
                   f"{rb} mismatches, worst |diff|={rw:.3e}")
            ro_stats = {"n_compared": rn, "n_mismatch": rb, "worst_abs_diff": rw}

        # ── code provenance (after extraction: hashes reflect imported files) ─
        cur_mods = CC.module_hashes()
        rec_mods = prof["provenance"]["module_hashes"]
        mod_table, drift = {}, []
        for name, rec_h in rec_mods.items():
            cur_h = cur_mods.get(name)
            status = "MATCH" if cur_h == rec_h else ("MISSING" if cur_h is None else "MISMATCH")
            mod_table[name] = {"recorded": rec_h, "current": cur_h, "status": status}
            if status != "MATCH":
                drift.append(name)
        unacked = [d for d in drift if d not in acked]
        if drift:
            print(f"  [WARN] module-hash drift vs profile: {drift} (acknowledged: {sorted(acked)})")
        ck.add("module hashes match profile (or drift within the explicit allowlist)",
               not unacked, f"unacknowledged drift: {unacked}")

        # ── record + verdict ───────────────────────────────────────────────
        try:
            import mlx.core as _mx
            mlx_version = getattr(_mx, "__version__", None)
        except Exception:
            mlx_version = None
        record["environment"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mlx": mlx_version,
            "mlx_lm": __import__("mlx_lm").__version__,
            "T0_REPO": CC.T0_REPO,
            "seal_source": CC.SEAL_SOURCE,
            "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
            "git": {"commit-confluence": git_state(CC_ROOT), "t0_repo": git_state(CC.T0_REPO)},
        }
        record["provenance"] = {
            "module_hashes": mod_table, "drifted_modules": drift,
            "acknowledged_modules": sorted(acked), "unacknowledged_drift": unacked,
            "model_snapshot": {"current": snap, "recorded": rec_snap,
                               "note": "revision-pointer comparison under the default HF cache "
                                       "layout; weight shards are NOT hashed"},
        }
        record["comparisons"] = {
            "ace": {"rows": ace_rows, "n_compared": n_cmp, "n_mismatch": n_bad,
                    "worst_abs_diff": worst},
            "readout": ro_stats,
            "equality_criterion": "float64 byte equality; source arrays asserted float64 "
                                  "and finite before comparison",
        }
        verdict = ("PASS" if ck.all_ok and not drift
                   else "PASS_WITH_ACKNOWLEDGED_CODE_DRIFT" if ck.all_ok
                   else "FAIL")
        finalize(verdict, 0 if verdict != "FAIL" else 1)

    except SystemExit:
        raise
    except Exception:
        record["error"] = traceback.format_exc()
        print(record["error"], file=sys.stderr)
        finalize("ERROR", 2)


if __name__ == "__main__":
    main()
