#!/usr/bin/env python3
"""
check_fresh_data - the pre-registered fresh-data gate (amendment A5).

"Fresh seed" is NOT automatically "fresh examples": a fresh draw from the same pool can
overlap the sealed 20260526 examples and quietly dilute the out-of-sample claim. A registered
Stage B run may only use data files that PASS this gate.

HARD checks (any failure -> exit 1, run must not launch):
  1. schema   - every line is JSON with a prompt ("prompt" or "text") and label in {0,1}
  2. n        - expected row count (amendment A1: ANLI 200 rows; TriviaQA 100 pairs = 200 rows)
  3. balance  - |mean(label) - 0.5| <= 0.10 (TriviaQA paired must be exactly 0.50)
  4. intra-dup- zero duplicate prompts within the file
  5. overlap  - ZERO prompt overlap (normalized-text sha256) with the sealed file

SOFT checks (warn only): TriviaQA pairing structure (each "Question:" segment appears exactly
twice - once correct, once wrong); brittle to template drift, so non-fatal.

Legacy usage:
  python stage_b/check_fresh_data.py --fresh <fresh.jsonl> --sealed <sealed.jsonl> \
      --task {anli|triviaqa} [--expect-n 200] [--out report.json]

BENCH v1.2 adds registered task keys, repeatable --sealed references, stem/cue/length
checks, and exact paired structure while preserving the legacy callable and CLI path.
"""
import sys, os, json, argparse, hashlib, collections

BENCH_COHORT = {
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-1.7B-4bit", "mlx-community/Qwen3-8B-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                return None, f"line {ln}: invalid JSON ({e})"
            rows.append((ln, d))
    return rows, None


def prompt_of(d):
    return d.get("prompt") if d.get("prompt") is not None else d.get("text")


def norm_hash(text):
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()


def _question_ids(rows):
    out = set()
    for _, d in rows:
        meta = d.get("meta") or {}
        if meta.get("question_id"):
            out.add(meta["question_id"])
    return out


def _stem_id(d):
    if d.get("stem_id") is not None:
        return str(d["stem_id"])
    value = (d.get("meta") or {}).get("stem_id")
    return None if value is None else str(value)


def _sealed_paths(value):
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [str(value)]
    return [str(path) for path in value]


def run_gate(fresh, sealed, task, expect_n=200):
    """The fresh-data gate as a callable (used by the CLI AND by run_seal.py at launch).
    Returns the report dict; `report['pass']` is the launch-eligibility verdict."""
    class _A:  # lightweight namespace so the body below is unchanged
        pass
    a = _A()
    a.fresh, a.sealed, a.task, a.expect_n, a.out = fresh, sealed, task, expect_n, None
    a.length_cap, a.require_stem_ids = None, False
    return _gate_body(a)


def run_bench_gate(fresh, sealed_paths, task, expect_n, *, length_cap=2048):
    """Registered BENCH gate used in-process by run_bench.py."""
    class _A:
        pass
    a = _A()
    a.fresh, a.sealed, a.task, a.expect_n, a.out = (
        fresh, list(sealed_paths), task, expect_n, None)
    a.length_cap, a.require_stem_ids = length_cap, True
    return _gate_body(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--sealed", required=True, action="append",
                    help="sealed exclusion reference; repeat for BENCH union")
    ap.add_argument("--task", required=True, choices=[
        "anli", "triviaqa", "halueval_qa", "anli_r1_rep",
        "triviaqa_paired_rep", "anli_r2", "halueval_dialogue",
        "halueval_summarization"])
    ap.add_argument("--expect-n", type=int, default=200)
    ap.add_argument("--length-cap", type=int, default=None)
    ap.add_argument("--require-stem-ids", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = _gate_body(a)
    print(json.dumps(report, indent=1))
    print(f"\n{'PASS' if report['pass'] else 'FAIL'} - "
          f"{'fresh file is launch-eligible' if report['pass'] else 'DO NOT LAUNCH; fix the data'}")
    sys.exit(0 if report["pass"] else 1)


def _gate_body(a):
    hard_fail, warns = [], []
    rows, err = load_jsonl(a.fresh)
    if err:
        hard_fail.append(f"schema: {err}")
        rows = []
    labels, hashes, stems, valid_rows = [], [], [], []
    max_wrapped = []
    for ln, d in rows:
        p = prompt_of(d)
        if p is None or not str(p).strip():
            hard_fail.append(f"schema: line {ln} has no prompt/text")
            continue
        if d.get("label") not in (0, 1):
            hard_fail.append(f"schema: line {ln} label={d.get('label')!r} not in {{0,1}}")
            continue
        labels.append(int(d["label"]))
        hashes.append(norm_hash(p))
        stems.append(_stem_id(d))
        max_wrapped.append((d.get("meta") or {}).get("max_wrapped_tokens"))
        valid_rows.append((ln, d))

    n = len(labels)
    if n != a.expect_n:
        hard_fail.append(f"n: {n} rows != expected {a.expect_n}")
    bal = (sum(labels) / n) if n else float("nan")
    bench_task = a.task not in {"anli", "triviaqa"}
    if a.task in {"triviaqa", "triviaqa_paired_rep"} or bench_task:
        if n and abs(bal - 0.5) > 1e-9:
            hard_fail.append(f"balance: BENCH/paired data must be exactly 0.50, got {bal:.4f}")
    elif n and abs(bal - 0.5) > 0.10:
        hard_fail.append(f"balance: |{bal:.4f} - 0.5| > 0.10")

    dup = [h for h, c in collections.Counter(hashes).items() if c > 1]
    if dup:
        hard_fail.append(f"intra-dup: {len(dup)} duplicated prompts within the fresh file")

    sealed_rows, sealed_hashes, serrs = [], set(), []
    for sealed_path in _sealed_paths(a.sealed):
        one_rows, serr = load_jsonl(sealed_path)
        if serr:
            serrs.append(f"{sealed_path}: {serr}")
            continue
        sealed_rows.extend(one_rows)
        sealed_hashes.update(norm_hash(prompt_of(d)) for _, d in one_rows if prompt_of(d))
    if serrs:
        hard_fail.extend(f"sealed file unreadable: {err}" for err in serrs)
    overlap = sorted(set(hashes) & sealed_hashes)
    if overlap:
        hard_fail.append(f"overlap: {len(overlap)} fresh prompts also in the sealed file "
                         f"(must be 0; regenerate excluding sealed examples)")

    # SF1: for TriviaQA, prompt-hash overlap can be 0 while the SAME question reappears with a
    # different injected answer (the generator excludes sealed question_ids; the gate must too).
    qid_overlap = 0
    is_trivia = a.task in {"triviaqa", "triviaqa_paired_rep"}
    if is_trivia and rows and not serrs:
        # SF1-fix-2: require meta.question_id on every row, else the overlap check below is a
        # silently-inert empty-set comparison on a file that legally omits the field.
        n_missing_qid = sum(1 for _, d in rows if not (d.get("meta") or {}).get("question_id"))
        if n_missing_qid:
            hard_fail.append(f"qid-schema: {n_missing_qid} TriviaQA rows lack meta.question_id "
                             f"(required so the sealed-question-id overlap check is meaningful)")
        fresh_qids = _question_ids(rows)
        sealed_qids = _question_ids(sealed_rows)
        qid_overlap = len(fresh_qids & sealed_qids)
        if qid_overlap:
            hard_fail.append(f"qid-overlap: {qid_overlap} fresh question_ids also in the sealed "
                             f"file (must be 0; a reused question is not a fresh example)")

    if is_trivia and rows:
        qcounts = collections.Counter()
        parsed = 0
        for _, d in rows:
            p = str(prompt_of(d) or "")
            if "Question:" in p:
                q = p.split("Question:", 1)[1].split("Proposed answer:", 1)[0].strip()
                qcounts[q] += 1
                parsed += 1
        if parsed < len(rows):
            warns.append(f"pairing: only {parsed}/{len(rows)} prompts parseable for Question segment")
        bad = {q: c for q, c in qcounts.items() if c != 2}
        if bad:
            warns.append(f"pairing: {len(bad)} questions do not appear exactly twice (soft)")

    require_stems = bool(getattr(a, "require_stem_ids", False) or bench_task)
    stem_counts = collections.Counter(stem for stem in stems if stem is not None)
    n_missing_stems = sum(stem is None for stem in stems)
    if require_stems and n_missing_stems:
        hard_fail.append(f"stem-schema: {n_missing_stems} rows lack stem_id")
    over_cap = {stem: count for stem, count in stem_counts.items() if count > 2}
    if over_cap:
        hard_fail.append(f"stem-cap: {len(over_cap)} stems exceed 2 rows")
    grouped_task = a.task in {
        "triviaqa_paired_rep", "halueval_qa", "halueval_dialogue",
        "halueval_summarization"}
    if grouped_task and not n_missing_stems:
        rows_by_stem = collections.defaultdict(list)
        for (_, row), stem in zip(valid_rows, stems):
            rows_by_stem[stem].append(int(row["label"]))
        bad_pairs = {stem: vals for stem, vals in rows_by_stem.items()
                     if len(vals) != 2 or set(vals) != {0, 1}}
        if bad_pairs:
            hard_fail.append(f"pairing: {len(bad_pairs)} stems are not exact {{0,1}} pairs")
    if a.task == "anli_r1_rep":
        bad_split = sum((row.get("meta") or {}).get("source_split") != "train_r1"
                        for _, row in valid_rows)
        if bad_split:
            hard_fail.append(f"anli-r1-source: {bad_split} rows are not train_r1")
    if a.task == "anli_r2":
        allocation = collections.Counter(
            ((row.get("meta") or {}).get("source_split"), int(row["label"]))
            for _, row in valid_rows)
        quota = a.expect_n // 4
        expected_allocation = {("dev_r2", 0): quota, ("dev_r2", 1): quota,
                               ("test_r2", 0): quota, ("test_r2", 1): quota}
        if dict(allocation) != expected_allocation:
            hard_fail.append(
                f"anli-r2-allocation: {dict(allocation)} != {expected_allocation}")

    bad_cues = [ln for (ln, row) in rows
                if not str(prompt_of(row) or "").rstrip().endswith("Answer:")]
    if bench_task and bad_cues:
        hard_fail.append(f"one-token-cue: {len(bad_cues)} prompts do not end in Answer:")
    length_cap = getattr(a, "length_cap", None)
    if length_cap is not None:
        missing_lengths = sum(value is None for value in max_wrapped)
        if missing_lengths:
            hard_fail.append(
                f"length-schema: {missing_lengths} rows lack meta.max_wrapped_tokens")
        too_long = sum(value is not None and int(value) > length_cap for value in max_wrapped)
        if too_long:
            hard_fail.append(f"length-cap: {too_long} rows exceed {length_cap} tokens")
        bad_count_maps = 0
        for _, row in valid_rows:
            meta = row.get("meta") or {}
            counts = meta.get("wrapped_token_counts")
            maximum = meta.get("max_wrapped_tokens")
            if (not isinstance(counts, dict) or set(counts) != BENCH_COHORT
                    or not counts or maximum is None
                    or int(max(counts.values())) != int(maximum)):
                bad_count_maps += 1
        if bad_count_maps:
            hard_fail.append(
                f"length-provenance: {bad_count_maps} rows lack an exact cohort count map/max")

    sealed_paths = _sealed_paths(a.sealed)
    sealed_report = ([os.path.abspath(path) for path in sealed_paths]
                     if bench_task or len(sealed_paths) != 1
                     else os.path.abspath(sealed_paths[0]))
    report = {
        "fresh": os.path.abspath(a.fresh),
        "sealed": sealed_report,
        "task": a.task, "n": n, "label_balance": round(bal, 4) if n else None,
        "n_intra_dup": len(dup), "n_overlap_with_sealed": len(overlap),
        "n_qid_overlap_with_sealed": qid_overlap,
        "n_missing_stem_id": n_missing_stems, "n_stems": len(stem_counts),
        "n_stems_over_cap": len(over_cap), "length_cap": length_cap,
        "fresh_sha256": hashlib.sha256(open(a.fresh, "rb").read()).hexdigest() if os.path.exists(a.fresh) else None,
        "hard_failures": hard_fail, "warnings": warns, "pass": not hard_fail,
    }
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return report


if __name__ == "__main__":
    main()
