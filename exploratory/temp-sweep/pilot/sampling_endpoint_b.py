"""A1 standalone behavioural Endpoint B; the original pilot remains unchanged.

Execution belongs to Claude Code / MK after committing the amendment and scripts.
Usage: python pilot/sampling_endpoint_b.py --out runs/<new-id> --amendment-commit <40-hex>
Paths are relative to pilot/. Only its ignored runs/ tree may receive outputs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from sampling_pilot import (HERE, REPO, MODELS, TASKS, TEMPS, K, SEED, MIN_FREE_PCT,
                            free_pct, independent_canonical, sha256, uniforms)

PROTOCOL = "TEMP-v1.3-A1-behavioural/1-pre-output-audit-2026-10-12"
AMENDMENT = HERE.parent / "AMENDMENT_A1_ENDPOINT_B_2026-10-12.md"
NEAR_TIE_MARGIN = 0.15  # fixed post-pilot convention, not a tolerance for historical identity
HARNESS_BUDGET = 0.01
INVALID_FLAG = 0.25
GREEDY_INVALID_FLAG = 0.05
MODEL_TIMEOUT_S = 4 * 60 * 60
UNIFORMS_SHA256 = "d3b9cdb288ff9ecb9746f76edd1dea53ffe28ba4eeccc7d7eba0aee58a31ada6"
BENCH_SHA256 = "4e6ae7dc5fadd1003fb10529e2c26f8a94ec429700d814dc69af5d8121f0dec6"
HIST_SHA256 = [
    "2e60b521e33b219ddc162b4d6a8672b8357f8fa3a339443890b9d27cfe277b94",
    "80c6da24a1d166e866c6fe7e39231fb2b30a97a0341ecfc71a0b5a42bca658ce",
    "89a29c2f2ac0b2b7957ea7d2ab2fdf436cd84a15d0c7ac356149021979b2ed66",
    "0bd53934e48aca43bb5a6bff4f72c6b8f47234ae86b2ac9c4edf73fc41103322",
]
PLAN = [(0.0, 0)] + [(t, d) for t in TEMPS for d in range(K)]


def output_dir(value):
    path = Path(value)
    path = (path if path.is_absolute() else HERE / path).resolve()
    root = (HERE / "runs").resolve()
    assert root.parent == HERE, "runs/ must not point outside pilot/"
    assert root in path.parents, "output must be a new directory below pilot/runs/"
    return path


def load_rows():
    rows = {}
    for task, path, digest, n in TASKS:
        assert sha256(path) == digest, f"data hash mismatch: {task}"
        rs = [dict(json.loads(line), sample_idx=i) for i, line in enumerate(path.read_text().splitlines())]
        assert len(rs) == n, (task, len(rs), n)
        assert all(r["label"] in (0, 1) and isinstance(r["prompt"], str) and r["prompt"] for r in rs)
        if task == "halueval_qa":
            stems = defaultdict(list)
            for r in rs:
                stems[r["stem_id"]].append(r["label"])
            assert len(stems) == 500 and all(sorted(v) == [0, 1] for v in stems.values())
        rows[task] = rs
    return rows


def historical_path(mi):
    return REPO / f"stage_b/profiles_bench/halueval_qa/{MODELS[mi][0].split('/')[1]}.matrix.npz"


def historical_ids(mi, rows):
    path = historical_path(mi)
    assert sha256(path) == HIST_SHA256[mi], "historical matrix hash mismatch"
    with np.load(path, allow_pickle=False) as hist:
        idx = hist["sample_idx"]
        assert idx.shape == (1000,) and np.array_equal(np.sort(idx), np.arange(1000))
        order = np.argsort(idx)
        assert hist["labels"].shape == hist["stem_ids"].shape == hist["gen_token_ids"].shape == (1000,)
        assert np.array_equal(hist["labels"][order], [r["label"] for r in rows])
        assert np.array_equal(hist["stem_ids"][order].astype(str), [str(r["stem_id"]) for r in rows])
        ids = hist["gen_token_ids"][order]
        assert np.issubdtype(ids.dtype, np.integer) and (ids >= 0).all()
        return ids.copy()


def repo_relative_path(path, root=REPO):
    # Relative inputs are repo-relative, never relative to the caller's working directory.
    root = Path(root)
    assert root.is_absolute()
    root = root.resolve()
    return (root / path).resolve().relative_to(root).as_posix()


def file_hashes(files):
    paths = [repo_relative_path(p) for p in files]
    return {p: sha256(REPO / p) for p in paths}


def provenance():
    bench = REPO / "stage_b/run_bench.py"
    assert sha256(bench) == BENCH_SHA256, "BENCH normalizer module hash mismatch"
    files = [AMENDMENT, HERE / "sampling_endpoint_b.py", HERE / "sampling_pilot.py", bench]
    files += [path for _, path, _, _ in TASKS]
    files += [historical_path(mi) for mi in range(len(MODELS))]
    return file_hashes(files)


def committed_provenance(amendment_commit, files):
    assert re.fullmatch(r"[0-9a-f]{40}", amendment_commit or ""), "Claude/MK must supply the pre-run commit"

    def git(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout

    root = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    assert root == REPO.resolve(), "provenance root must be the Git repository root"
    commit = git("rev-parse", "--verify", f"{amendment_commit}^{{commit}}").decode().strip()
    assert commit == amendment_commit, "supply the commit object ID, not a tag object"
    tree = git("rev-parse", "--verify", f"{commit}^{{tree}}").decode().strip()
    for path, digest in files.items():
        committed = git("cat-file", "blob", f"{commit}:{path}")
        assert hashlib.sha256(committed).hexdigest() == digest, f"working tree differs from commit: {path}"
        assert sha256(REPO / path) == digest, f"file changed during launch check: {path}"
    return {"amendment_commit": commit, "amendment_tree": tree}


def greedy_audit(z):
    # Stable sorting preserves ascending token IDs when logits are exactly tied.
    top = np.argsort(-z, kind="stable")[:2]
    assert len(top) == 2 and np.isfinite(z).all()
    margin = float(z[top[0]] - z[top[1]])
    return {"greedy_token_id": int(top[0]), "greedy_top_two_ids": top.tolist(),
            "greedy_top_two_logits": z[top].tolist(), "greedy_logit_margin": margin,
            "near_tie": margin <= NEAR_TIE_MARGIN}


def cell_status(a, b, c, d, greedy=False, anli=False, historical_failures=0,
                infrastructure_failures=None, sampling_verification_failures=0):
    n, observed = a + b + c + d, b + c + d
    assert n > 0 and min(a, b, c, d) >= 0
    assert 0 <= historical_failures <= a
    assert 0 <= sampling_verification_failures <= a
    infrastructure = (max(0, a - historical_failures - sampling_verification_failures)
                      if infrastructure_failures is None else infrastructure_failures)
    assert 0 <= infrastructure <= a
    assert historical_failures + sampling_verification_failures + infrastructure >= a
    reasons = []
    if historical_failures:
        reasons.append("historical_identity")
    if sampling_verification_failures:
        reasons.append("sampling_verification")
    if infrastructure / n > HARNESS_BUDGET:
        reasons.append("harness_budget")
    if not observed:
        reasons.append("no_observed_outcomes")
    # The inherited behavioural bar uses all planned ANLI prompts, not only survivors.
    if greedy and anli and c + d < 190:
        reasons.append("ANLI_canonical_below_190_of_200")
    if sampling_verification_failures:
        return "FAILED: sampling_verification", reasons
    if historical_failures:
        return "FAILED: historical_identity", reasons
    return ("FAILED" if reasons else "PASS"), reasons


def format_flags(b, observed, greedy=False):
    limit = GREEDY_INVALID_FLAG if greedy else INVALID_FLAG
    return ["format_usability"] if observed and b / observed > limit else []


def inverse_cdf(cdf, uniform):
    rank = int(np.searchsorted(cdf, uniform, side="right"))
    token_id = min(rank, len(cdf) - 1)
    return {"token_id": token_id, "vocab_size": len(cdf),
            "cdf_lo": float(cdf[token_id - 1]) if token_id else 0.0,
            "cdf_hi": float(cdf[token_id]), "cdf_clipped": rank == len(cdf)}


def temperature_distribution(z, temp):
    assert temp in TEMPS and z.dtype == np.float64 and np.isfinite(z).all()
    scaled = (z - z.max()) / temp
    weights = np.exp(scaled)
    mass = weights.sum()
    q = weights / mass
    # Log-sum-exp gives -log q without underflow or an arbitrary probability floor.
    logq = scaled - np.log(mass)
    assert np.isfinite(q).all() and abs(q.sum() - 1) <= 1e-12
    assert np.isfinite(logq).all()
    return q, logq


def run_model(mi, out_dir):
    # Cache-only loading keeps the authorized local run from acquiring remote compute or weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["protocol"] == PROTOCOL and manifest["launch_files_sha256"] == provenance()
    assert sha256(out_dir / "uniforms.npy") == UNIFORMS_SHA256
    assert free_pct() >= MIN_FREE_PCT, "insufficient free memory before model load"
    rows, (_, u) = load_rows(), uniforms()
    hist_ids = historical_ids(mi, rows["halueval_qa"])
    model_id, revision = MODELS[mi]
    snap = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    assert snap.is_dir(), f"{model_id}@{revision} not cached; stop, no download"

    import mlx.core as mx
    from mlx_lm import load
    import run_bench as RB

    t0 = time.time()
    model, tok = load(str(snap))
    load_s = time.time() - t0
    V = int(model(mx.array([[tok.bos_token_id or 0]]))[0, -1].shape[-1])
    n_tok = len(tok._tokenizer)
    decoded = tok._tokenizer.batch_decode([[i] for i in range(min(V, n_tok))]) + [""] * max(0, V - n_tok)
    norms = [RB.normalize_commitment(t) for t in decoded]
    canon = np.array([RB.is_canonical_commitment(t) for t in norms])
    yes = np.array([c and "yes".startswith(t) for c, t in zip(canon, norms)])
    no = canon & ~yes
    assert canon.any() and (hist_ids < V).all()
    with (out_dir / f"{model_id.split('/')[1]}.tokens.json").open("x") as f:
        json.dump({"yes": np.flatnonzero(yes).tolist(), "no": np.flatnonzero(no).tolist(),
                   "canonical_decodes": [[int(i), decoded[i], norms[i]] for i in np.flatnonzero(canon)]}, f)

    def record(base, token_id, temp, draw, uv, q=None, logq=None, selection=None):
        disposition = ("YES" if yes[token_id] else "NO") if canon[token_id] else "non-canonical"
        correct = disposition == ("YES" if base["label"] == 0 else "NO")
        bucket = "d_valid_correct" if correct else "c_valid_wrong" if canon[token_id] else "b_noncompliant"
        r = {**base, "T": temp, "sample_index": draw, "uniform": uv, "seed": SEED,
             "token_id": int(token_id), "raw_text": decoded[token_id], "normalized": norms[token_id],
             "disposition": disposition, "sample_error": int(not correct),
             "strict_non_yes": int(disposition != "YES"), "bucket": bucket,
             "token_count": 1, "hit_max_tokens": True, "finish_reason": "length",
             "independent_parser_canonical": independent_canonical(decoded[token_id])}
        if q is not None:
            r.update(p_valid=float(q[canon].sum()), p_yes=float(q[yes].sum()), p_no=float(q[no].sum()),
                     surprise=float(-logq[token_id]))
            r.update(selection)
        if base.get("historical_match") is False:
            r.update(bucket="a_harness", observed_bucket=bucket, historical_identity_failed=True,
                     error="historical_greedy_token_mismatch")
        return r

    n_written, prefix_attempts = 0, 0
    out_path = out_dir / f"{model_id.split('/')[1]}.jsonl"
    with out_path.open("x") as f:
        for task, _, _, _ in TASKS:
            for row in rows[task]:
                base = {"model": model_id, "revision": revision, "task": task, "sample_idx": row["sample_idx"],
                        "stem_id": row.get("stem_id"), "label": row["label"], "row_audit": True}
                fatal = None
                recs = []
                try:
                    prompt = row["prompt"]
                    add = tok.bos_token is None or not prompt.startswith(tok.bos_token)
                    ids = tok.encode(prompt, add_special_tokens=add)
                    base["prompt_tokens"] = len(ids)
                    prefix_attempts += 1
                    z = np.array(model(mx.array([ids]))[0, -1].astype(mx.float32), dtype=np.float64)
                    assert z.shape == (V,) and np.isfinite(z).all(), "non-finite or wrong-shape prefix logits"
                    base.update(greedy_audit(z))
                    greedy = base["greedy_token_id"]
                    base["greedy_canonical"] = bool(canon[greedy])
                    if task == "halueval_qa":
                        historical = int(hist_ids[row["sample_idx"]])
                        historical_disp = ("YES" if yes[historical] else "NO") if canon[historical] else "non-canonical"
                        greedy_disp = ("YES" if yes[greedy] else "NO") if canon[greedy] else "non-canonical"
                        base.update(historical_token_id=historical, historical_match=greedy == historical,
                                    historical_disposition=historical_disp,
                                    historical_disposition_match=historical_disp == greedy_disp)
                    recs = [record(base, greedy, 0.0, 0, None)]
                    for ti, temp in enumerate(TEMPS):
                        q, logq = temperature_distribution(z, temp)
                        cdf = np.cumsum(q)
                        for draw in range(K):
                            uv = float(u[(mi, task)][ti, row["sample_idx"], draw])
                            selection = inverse_cdf(cdf, uv)
                            recs.append(record(base, selection["token_id"], temp, draw, uv, q, logq, selection))
                    assert all(r["greedy_token_id"] == greedy for r in recs), "greedy row audit"
                except Exception as error:
                    fatal = f"{type(error).__name__}: {error}"
                    present = {(r["T"], r["sample_index"]) for r in recs}
                    for r in recs:
                        r.setdefault("observed_bucket", r["bucket"])
                        r.update(bucket="a_harness", infrastructure_failure=True, error=fatal)
                    recs += [{**base, "T": t, "sample_index": d, "seed": SEED, "bucket": "a_harness",
                              "infrastructure_failure": True,
                              "historical_identity_failed": base.get("historical_match") is False,
                              "error": fatal, "traceback": traceback.format_exc()[-800:]}
                             for t, d in PLAN if (t, d) not in present]
                for r in recs:
                    f.write(json.dumps(r, allow_nan=False) + "\n")
                f.flush()
                n_written += len(recs)
                if fatal:
                    # Preserve the failing row and earlier work; summary materializes every unattempted slot.
                    raise RuntimeError(fatal)
    report = {"model": model_id, "records": n_written, "load_s": load_s, "total_s": time.time() - t0,
              "prefix_attempts": prefix_attempts, "vocab_probe_forwards": 1, "post_commit_forwards": 0}
    with (out_dir / f"{model_id.split('/')[1]}.timing.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


def driver(out_dir, amendment_commit):
    assert platform.system() == "Darwin" and platform.machine() == "arm64", "local Apple Silicon only"
    host = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=True).stdout.strip()
    assert host == "Mac16,10", f"expected the registered M4 Mac mini, got {host}"
    launch_files = provenance()
    commit_info = committed_provenance(amendment_commit, launch_files)
    analysis_files = file_hashes([HERE / "summarize_endpoint_b.py"])
    out_dir.mkdir(parents=True, exist_ok=False)
    flat, _ = uniforms()
    np.save(out_dir / "uniforms.npy", flat)
    manifest = {"protocol": PROTOCOL, **commit_info, "launch_files_sha256": launch_files,
                "analysis_files_sha256_at_launch": analysis_files,
                "seed": SEED, "temps": TEMPS, "K": K, "n_anli": 200, "n_halueval_stems": 500,
                "models": MODELS, "near_tie_margin": NEAR_TIE_MARGIN,
                "gates": {"infrastructure_fraction": HARNESS_BUDGET, "historical_mismatches": 0,
                          "sampling_verification_failures": 0,
                          "observed_min": 1, "anli_greedy_canonical_min": 190},
                "format_flags": {"sampled_invalid": INVALID_FLAG, "greedy_invalid": GREEDY_INVALID_FLAG},
                "uniforms_sha256": sha256(out_dir / "uniforms.npy"),
                "data_sha256": {n: d for n, _, d, _ in TASKS}, "historical_sha256": HIST_SHA256,
                "host": host, "platform": platform.platform(), "python": sys.version,
                "started": time.time(), "model_runs": []}
    manifest_path = out_dir / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save()
    try:
        assert manifest["uniforms_sha256"] == UNIFORMS_SHA256, "uniform stream changed"
        assert launch_files == provenance(), "launch files changed after commit verification"
        manifest["packages"] = {p: importlib.metadata.version(p) for p in ("numpy", "mlx", "mlx-lm", "transformers")}
        rows = load_rows()
        for mi in range(len(MODELS)):
            historical_ids(mi, rows["halueval_qa"])
        save()
        for mi, (model_id, _) in enumerate(MODELS):
            pct = free_pct()
            entry = {"model": model_id, "free_memory_pct_before": pct}
            manifest["model_runs"].append(entry)
            save()
            if pct < MIN_FREE_PCT:
                entry["status"] = "stopped: free memory below 30%"
                raise RuntimeError(entry["status"])
            t0 = time.time()
            with (out_dir / f"{model_id.split('/')[1]}.log").open("x") as log:
                try:
                    p = subprocess.run([sys.executable, __file__, "--out", str(out_dir), "--model", str(mi)],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=MODEL_TIMEOUT_S)
                    entry["status"] = f"exit {p.returncode}"
                except subprocess.TimeoutExpired:
                    entry["status"] = f"timeout after {MODEL_TIMEOUT_S} s"
            entry["wall_s"] = time.time() - t0
            save()
            if entry["status"] != "exit 0":
                raise RuntimeError(f"{model_id}: {entry['status']}")
        manifest["execution_status"] = "complete"
    except Exception as error:
        manifest["execution_status"] = "stopped"
        manifest["stop_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished"] = time.time()
        save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--amendment-commit")
    ap.add_argument("--model", type=int, choices=range(len(MODELS)), help=argparse.SUPPRESS)
    args = ap.parse_args()
    out = output_dir(args.out)
    if args.model is None:
        driver(out, args.amendment_commit)
    else:
        run_model(args.model, out)
