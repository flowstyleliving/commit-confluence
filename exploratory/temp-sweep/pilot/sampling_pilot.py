"""Endpoint-(B) sampling pilot for the temperature study: does each model return a valid one-token YES/NO
commit at every temperature, and are harness failures kept apart from model errors?

Follows PRE_REGISTRATION_TEMP.md (v1.3 draft, not frozen) where it can: the four section-2.2 models at their
pinned revisions, raw prompt passthrough, max_new_tokens=1, T in {0.7, 1.0, 1.5} plus greedy, K=5 draws, the
section-2.6 uniform stream from RandomState(20260826) in nested order (generated for the full design so the
pilot's draws equal the ones the full study would use), inverse-CDF sampling over ascending token ID, and the
BENCH commitment normalizer (stage_b/run_bench.py). Pilot subset: the first 20 ANLI rows and the first 10
HaluEval stems (both rows each). Not part of the registered protocol: the exact canonical probability mass per
prompt and T, and a greedy 12-token continuation after each non-canonical draw (diagnostic, for hand-checking).

Usage: python sampling_pilot.py --out runs/<id>            # driver: one subprocess per model
       python sampling_pilot.py --out runs/<id> --model K  # one model (called by the driver)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "stage_b"))
sys.path.insert(0, str(REPO))

MODELS = [  # section 2.2 order, pinned revisions from section 2.1
    ("mlx-community/Qwen3-8B-4bit", "545dc4251c05440727734bcd94334791f6ab0192"),
    ("mlx-community/Llama-3.1-8B-Instruct-4bit", "90215b22ec18e72f623dde2ea7af4097025160e2"),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", "7b2052fd882fe017300d4d42a4eb06a27b816af4"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "a4b8f870474b0eb527f466a03fbc187830d271f5"),
]
TASKS = [  # (name, path, sha256, full row count) from section 2.3
    ("anli_r1", REPO / "stage_b/data/anli_R1_seed20260612_n200.jsonl",
     "57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b", 200),
    ("halueval_qa", REPO / "stage_b/data_bench/halueval_qa_seed20260711_n1000.jsonl",
     "460019e12d25c33e9f842f52bb0e3fd70274256290dc7784101ebaa95b2163e7", 1000),
]
TEMPS, K, SEED = (0.7, 1.0, 1.5), 5, 20260826
N_ANLI, N_STEMS, CONT_TOKENS = 20, 10, 12
MODEL_TIMEOUT_S = 1200
MIN_FREE_PCT = 30  # the Mac mini rebooted from memory pressure earlier today


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows():
    """{task: [row dicts with sample_idx]} for the pilot subset; data hashes must match the pre-registration."""
    out = {}
    for name, path, digest, n in TASKS:
        assert sha256(path) == digest, f"{path} does not match its registered SHA-256"
        rows = [dict(json.loads(line), sample_idx=i) for i, line in enumerate(path.read_text().splitlines())]
        assert len(rows) == n, (name, len(rows))
        if name == "anli_r1":
            out[name] = rows[:N_ANLI]
        else:  # first N_STEMS stems by first occurrence, both rows of each
            stems = list(dict.fromkeys(r["stem_id"] for r in rows))[:N_STEMS]
            out[name] = [r for r in rows if r["stem_id"] in set(stems)]
            assert len(out[name]) == 2 * N_STEMS, len(out[name])
    return out


def uniforms():
    """Section 2.6: one RandomState(20260826) stream in nested order model -> task -> T -> sample_idx -> draw,
    for the full design (all rows), so pilot draws are the full study's draws for the same prompts."""
    flat = np.random.RandomState(SEED).random_sample(sum(len(MODELS) * len(TEMPS) * n * K for *_, n in TASKS))
    # Nested order puts the model outermost, so slice per model, then per task within it.
    per_model = len(TEMPS) * K * sum(n for *_, n in TASKS)
    u = {}
    for mi in range(len(MODELS)):
        block, off = flat[mi * per_model:(mi + 1) * per_model], 0
        for name, _, _, n in TASKS:
            size = len(TEMPS) * n * K
            u[(mi, name)] = block[off:off + size].reshape(len(TEMPS), n, K)
            off += size
    return flat, u


def independent_canonical(text):
    """A second, simpler parser to catch bugs in the BENCH normalizer: letters only, lower-cased."""
    letters = "".join(ch for ch in text.lower() if ch.isalpha())
    return bool(letters) and ("yes".startswith(letters) or "no".startswith(letters))


def run_model(mi, out_dir):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    import run_bench as RB

    model_id, revision = MODELS[mi]
    snap = Path.home() / ".cache/huggingface/hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    assert snap.is_dir(), f"{model_id}@{revision} is not cached; the pilot never downloads"
    rows, (_, u) = load_rows(), uniforms()
    hist = np.load(REPO / f"stage_b/profiles_bench/halueval_qa/{model_id.split('/')[1]}.matrix.npz")
    hist_ids = dict(zip(hist["sample_idx"].tolist(), hist["gen_token_ids"].tolist()))
    out_path = out_dir / f"{model_id.split('/')[1]}.jsonl"
    t0 = time.time()
    model, tok = load(str(snap))
    load_s = time.time() - t0

    # Canonical YES/NO token sets over the whole output vocabulary, with the BENCH normalizer.
    V = int(model(mx.array([[tok.bos_token_id or 0]]))[0, -1].shape[-1])
    n_tok = len(tok._tokenizer)  # the output layer can be wider than the tokenizer; extra slots decode to ""
    decoded = tok._tokenizer.batch_decode([[i] for i in range(min(V, n_tok))]) + [""] * max(0, V - n_tok)
    canon = np.array([RB.is_canonical_commitment(RB.normalize_commitment(t)) for t in decoded])
    is_yes = np.array([c and "yes".startswith(RB.normalize_commitment(t)) for c, t in zip(canon, decoded)])
    is_no = canon & ~is_yes

    def encode(prompt):  # exactly as mlx_lm.stream_generate encodes a string prompt
        add = tok.bos_token is None or not prompt.startswith(tok.bos_token)
        return tok.encode(prompt, add_special_tokens=add)

    def record(base, token_id, temp, draw, u_val, q=None, extra=None):
        text = decoded[token_id]
        norm = RB.normalize_commitment(text)
        ok = RB.is_canonical_commitment(norm)
        disp = ("YES" if is_yes[token_id] else "NO") if ok else "non-canonical"
        correct = (disp == "YES" and base["label"] == 0) or (disp == "NO" and base["label"] == 1)
        r = {**base, "T": temp, "sample_index": draw, "seed": SEED, "uniform": u_val, "token_id": int(token_id),
             "raw_text": text, "normalized": norm, "disposition": disp, "token_count": 1,
             "finish_reason": "length", "hit_max_tokens": True,  # max_new_tokens=1 is the registered design
             "independent_parser_canonical": independent_canonical(text),
             "bucket": "d_valid_correct" if correct else ("c_valid_wrong" if ok else "b_noncompliant")}
        if q is not None:
            r.update(p_valid=float(q[canon].sum()), p_yes=float(q[is_yes].sum()), p_no=float(q[is_no].sum()))
        if not ok:  # diagnostic only: what the model was about to say
            cont = "".join(x.text for x in stream_generate(model, tok, base["_ids"] + [int(token_id)],
                                                            max_tokens=CONT_TOKENS))
            r["diagnostic_continuation"] = text + cont
        if extra:
            r.update(extra)
        return r

    n_written = 0
    with out_path.open("w") as f:
        for task, _, _, _ in TASKS:
            for row in rows[task]:
                base = {"model": model_id, "revision": revision, "task": task, "sample_idx": row["sample_idx"],
                        "stem_id": row.get("stem_id"), "label": row["label"]}
                recs = []
                try:
                    ids = encode(row["prompt"])
                    base["_ids"], base["prompt_tokens"] = ids, len(ids)
                    z = np.array(model(mx.array([ids]))[0, -1].astype(mx.float32), dtype=np.float64)
                    if not np.isfinite(z).all():
                        raise FloatingPointError("non-finite logits")
                    greedy = int(np.argmax(z))  # exact ties resolve to the lowest token ID
                    ref = next(stream_generate(model, tok, row["prompt"], max_tokens=1)).token
                    extra = {"mlx_lm_parity": int(ref) == greedy}
                    if task == "halueval_qa":
                        extra["historical_token_id"] = int(hist_ids[row["sample_idx"]])
                        extra["historical_match"] = int(hist_ids[row["sample_idx"]]) == greedy
                    g = record(base, greedy, 0.0, 0, None, extra=extra)
                    if not extra["mlx_lm_parity"]:
                        g["bucket"], g["error"] = "a_harness", f"greedy {greedy} != mlx_lm {ref}"
                    recs.append(g)
                    for ti, temp in enumerate(TEMPS):
                        q = np.exp((z - z.max()) / temp)
                        q /= q.sum()
                        assert abs(q.sum() - 1) <= 1e-12 and np.isfinite(q).all()
                        cdf = np.cumsum(q)
                        for draw in range(K):
                            uv = float(u[(mi, task)][ti, row["sample_idx"], draw])
                            t_id = min(int(np.searchsorted(cdf, uv, side="right")), V - 1)
                            recs.append(record(base, t_id, temp, draw, uv, q=q))
                except Exception as error:  # keep the denominator: every planned record exists, bucket a
                    msg = f"{type(error).__name__}: {error}"
                    recs = [{**base, "T": temp, "sample_index": d, "seed": SEED, "bucket": "a_harness", "error": msg,
                             "traceback": traceback.format_exc()[-800:]}
                            for temp, d in [(0.0, 0)] + [(t, d) for t in TEMPS for d in range(K)]]
                for r in recs:
                    r.pop("_ids", None)
                    f.write(json.dumps(r) + "\n")
                n_written += len(recs)
                f.flush()
    print(json.dumps({"model": model_id, "records": n_written, "vocab": V, "canonical_tokens": int(canon.sum()),
                      "load_s": round(load_s, 1), "total_s": round(time.time() - t0, 1)}), flush=True)


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))


def driver(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    flat, _ = uniforms()
    np.save(out_dir / "uniforms.npy", flat)  # written and hashed before any model call (section 2.6)
    manifest = {"seed": SEED, "temps": TEMPS, "K": K, "n_anli": N_ANLI, "n_halueval_stems": N_STEMS,
                "uniforms_sha256": sha256(out_dir / "uniforms.npy"), "script_sha256": sha256(__file__),
                "data_sha256": {n: d for n, _, d, _ in TASKS}, "models": MODELS, "started": time.time()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for mi, (model_id, _) in enumerate(MODELS):
        pct = free_pct()
        print(json.dumps({"model": model_id, "free_memory_pct_before": pct}), flush=True)
        if pct < MIN_FREE_PCT:
            print(json.dumps({"model": model_id, "skipped": f"free memory {pct}% < {MIN_FREE_PCT}%"}), flush=True)
            continue
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, "--out", str(out_dir), "--model", str(mi)],
                               timeout=MODEL_TIMEOUT_S)
            status = f"exit {p.returncode}"
        except subprocess.TimeoutExpired:
            status = f"timeout after {MODEL_TIMEOUT_S} s"
        print(json.dumps({"model": model_id, "status": status, "wall_s": round(time.time() - t0, 1)}), flush=True)
    manifest["finished"] = time.time()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", type=int)
    a = ap.parse_args()
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    if a.model is None:
        driver(out)
    else:
        run_model(a.model, out)
