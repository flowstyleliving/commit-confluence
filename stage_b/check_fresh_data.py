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

Usage:
  python stage_b/check_fresh_data.py --fresh <fresh.jsonl> --sealed <sealed.jsonl> \
      --task {anli|triviaqa} [--expect-n 200] [--out report.json]
"""
import sys, os, json, argparse, hashlib, collections


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


def run_gate(fresh, sealed, task, expect_n=200):
    """The fresh-data gate as a callable (used by the CLI AND by run_seal.py at launch).
    Returns the report dict; `report['pass']` is the launch-eligibility verdict."""
    class _A:  # lightweight namespace so the body below is unchanged
        pass
    a = _A()
    a.fresh, a.sealed, a.task, a.expect_n, a.out = fresh, sealed, task, expect_n, None
    return _gate_body(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--task", required=True, choices=["anli", "triviaqa"])
    ap.add_argument("--expect-n", type=int, default=200)
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
    labels, hashes = [], []
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

    n = len(labels)
    if n != a.expect_n:
        hard_fail.append(f"n: {n} rows != expected {a.expect_n}")
    bal = (sum(labels) / n) if n else float("nan")
    if a.task == "triviaqa":
        if n and abs(bal - 0.5) > 1e-9:
            hard_fail.append(f"balance: paired TriviaQA must be exactly 0.50, got {bal:.4f}")
    elif n and abs(bal - 0.5) > 0.10:
        hard_fail.append(f"balance: |{bal:.4f} - 0.5| > 0.10")

    dup = [h for h, c in collections.Counter(hashes).items() if c > 1]
    if dup:
        hard_fail.append(f"intra-dup: {len(dup)} duplicated prompts within the fresh file")

    sealed_rows, serr = load_jsonl(a.sealed)
    if serr:
        hard_fail.append(f"sealed file unreadable: {serr}")
        sealed_hashes = set()
    else:
        sealed_hashes = {norm_hash(prompt_of(d)) for _, d in sealed_rows if prompt_of(d)}
    overlap = sorted(set(hashes) & sealed_hashes)
    if overlap:
        hard_fail.append(f"overlap: {len(overlap)} fresh prompts also in the sealed file "
                         f"(must be 0; regenerate excluding sealed examples)")

    # SF1: for TriviaQA, prompt-hash overlap can be 0 while the SAME question reappears with a
    # different injected answer (the generator excludes sealed question_ids; the gate must too).
    qid_overlap = 0
    if a.task == "triviaqa" and rows and not serr:
        fresh_qids = _question_ids(rows)
        sealed_qids = _question_ids(sealed_rows)
        qid_overlap = len(fresh_qids & sealed_qids)
        if qid_overlap:
            hard_fail.append(f"qid-overlap: {qid_overlap} fresh question_ids also in the sealed "
                             f"file (must be 0; a reused question is not a fresh example)")

    if a.task == "triviaqa" and rows:
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

    report = {
        "fresh": os.path.abspath(a.fresh), "sealed": os.path.abspath(a.sealed),
        "task": a.task, "n": n, "label_balance": round(bal, 4) if n else None,
        "n_intra_dup": len(dup), "n_overlap_with_sealed": len(overlap),
        "n_qid_overlap_with_sealed": qid_overlap,
        "fresh_sha256": hashlib.sha256(open(a.fresh, "rb").read()).hexdigest() if os.path.exists(a.fresh) else None,
        "hard_failures": hard_fail, "warnings": warns, "pass": not hard_fail,
    }
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    return report


if __name__ == "__main__":
    main()
