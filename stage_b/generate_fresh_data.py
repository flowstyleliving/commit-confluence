#!/usr/bin/env python3
"""
generate_fresh_data - build the REGISTERED fresh-draw data files (amendments A1 + A5).

Replicates the sealed-era builders exactly (templates, label conventions, sampling
structure) with ONE registered addition: sealed-example EXCLUSION during sampling.
"Fresh seed != fresh examples" - ANLI dev_r1 has only 1000 examples and the sealed
run consumed 200 of them, so a fresh seed alone WOULD collide; exclusion at draw
time is the only way to hit n=200 disjoint rows.

Sources replicated:
  ANLI     - build_anli_jsonl() in PRI_at_commitment/scripts/anli_full_sweep.py
             (facebook/anli dev_r1; drop neutral; balance 100/100; label 0=entail/YES,
              1=contradiction/NO; shuffle rows with the same RandomState)
  TriviaQA - PRI_at_commitment/scripts/generate_triviaqa_paired.py
             (trivia_qa rc.wikipedia validation; paired correct/wrong per question;
              label 0=correct/YES, 1=wrong/NO; cross-sampled wrong answers with
              alias collision guard; schema triviaqa_paired_v1)

Exclusion rule (matches check_fresh_data.py): a candidate is rejected if
sha256(" ".join(prompt.split())) appears in the sealed file's normalized-prompt set.
TriviaQA additionally rejects any question_id present in the sealed file's meta.

This script GENERATES candidates; the registered gate is still stage_b/check_fresh_data.py.

Usage:
  python stage_b/generate_fresh_data.py --seed 20260612 --out-dir stage_b/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

SEALED_DIR = Path(
    "/Users/msrk/Documents/t0-morphology-furnace/experiments/t0-sealed/2026-05-26/data"
)
SEALED_ANLI = SEALED_DIR / "anli_R1_seed20260526_n200.jsonl"
SEALED_TRIVIAQA = SEALED_DIR / "triviaqa_paired_seed20260526_n100.jsonl"

# Byte-identical to anli_full_sweep.py / run_v3_anli.py / anli_smoke.py
ANLI_PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n"
    "\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)

# Byte-identical to generate_triviaqa_paired.py (verified against the sealed manifest)
TRIVIAQA_PROMPT_TEMPLATE = (
    "Instruction: Read the question and proposed answer, then decide whether "
    "the proposed answer is correct. Answer YES if the proposed answer is "
    "correct, NO if the proposed answer is incorrect.\n\n"
    "Question: {question}\nProposed answer: {answer}\nAnswer:"
)


def norm_hash(text: str) -> str:
    # Must match check_fresh_data.norm_hash exactly.
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()


def sealed_prompt_hashes(path: Path) -> set:
    hashes = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompt = d.get("prompt") if d.get("prompt") is not None else d.get("text")
            hashes.add(norm_hash(prompt))
    return hashes


def sealed_question_ids(path: Path) -> set:
    qids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line).get("meta") or {}
            if meta.get("question_id"):
                qids.add(meta["question_id"])
    return qids


def build_anli(seed: int, n_per_class: int, out_path: Path) -> dict:
    from datasets import load_dataset

    excluded = sealed_prompt_hashes(SEALED_ANLI)
    ds = load_dataset("facebook/anli", split="dev_r1")
    rng = np.random.RandomState(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    pos, neg = [], []
    seen, n_excluded = set(), 0
    for idx in order:
        ex = ds[idx]
        if ex["label"] == 1:
            continue  # drop neutral
        prompt = ANLI_PROMPT_TEMPLATE.format(
            premise=ex["premise"], hypothesis=ex["hypothesis"]
        )
        h = norm_hash(prompt)
        if h in excluded:
            n_excluded += 1
            continue
        if h in seen:
            continue
        if ex["label"] == 0 and len(pos) < n_per_class:
            pos.append({"prompt": prompt, "label": 0})
            seen.add(h)
        elif ex["label"] == 2 and len(neg) < n_per_class:
            neg.append({"prompt": prompt, "label": 1})
            seen.add(h)
        if len(pos) == n_per_class and len(neg) == n_per_class:
            break

    if len(pos) < n_per_class or len(neg) < n_per_class:
        raise SystemExit(
            f"ANLI dev_r1 insufficient after sealed-exclusion: pos={len(pos)} "
            f"neg={len(neg)} target={n_per_class} (excluded {n_excluded})"
        )

    rows = pos + neg
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {
        "task": "anli_r1",
        "hf_dataset": "facebook/anli",
        "split": "dev_r1",
        "seed": seed,
        "n_rows": len(rows),
        "n_per_class": n_per_class,
        "n_sealed_excluded_during_sampling": n_excluded,
        "prompt_template": ANLI_PROMPT_TEMPLATE,
        "label_convention": {"0": "entailment - YES", "1": "contradiction - NO"},
        "sealed_exclusion_source": str(SEALED_ANLI),
        "data_hash_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "output": str(out_path.resolve()),
    }


def _canonical(item) -> str:
    return item["answer"]["value"].strip()


def _all_aliases(item) -> set:
    aliases = item["answer"].get("aliases", [])
    value = item["answer"]["value"]
    return {a.lower().strip() for a in aliases} | {value.lower().strip()}


def build_triviaqa(seed: int, n_questions: int, out_path: Path) -> dict:
    from datasets import load_dataset

    excluded_hashes = sealed_prompt_hashes(SEALED_TRIVIAQA)
    excluded_qids = sealed_question_ids(SEALED_TRIVIAQA)
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="validation")
    rng = random.Random(seed)

    pool_size = min(len(ds), n_questions * 10)  # pool_multiplier=10, as sealed
    indices = rng.sample(range(len(ds)), pool_size)
    pool = [ds[i] for i in indices]

    donor_pool = list(pool)
    rng.shuffle(donor_pool)

    pairs, used_qids, used_wrong = [], set(), set()
    n_excluded = 0
    donor_idx = 0
    for item in pool:
        if len(pairs) >= n_questions:
            break
        qid = item["question_id"]
        if qid in used_qids:
            continue
        if qid in excluded_qids:
            n_excluded += 1
            continue

        correct = _canonical(item)
        correct_set = _all_aliases(item)

        wrong = None
        for i in range(len(donor_pool)):
            other = donor_pool[(donor_idx + i) % len(donor_pool)]
            if other["question_id"] == qid:
                continue
            candidate = _canonical(other)
            if candidate.lower().strip() in correct_set:
                continue
            if len(candidate.strip()) < 2:
                continue
            if candidate in used_wrong:
                continue
            wrong = candidate
            donor_idx = (donor_idx + i + 1) % len(donor_pool)
            break
        if wrong is None:
            continue

        question = item["question"].strip()
        prompts = [
            TRIVIAQA_PROMPT_TEMPLATE.format(question=question, answer=correct),
            TRIVIAQA_PROMPT_TEMPLATE.format(question=question, answer=wrong),
        ]
        if any(norm_hash(p) in excluded_hashes for p in prompts):
            n_excluded += 1
            continue

        used_qids.add(qid)
        used_wrong.add(wrong)
        pairs.append(
            {
                "question": question,
                "correct_answer": correct,
                "wrong_answer": wrong,
                "question_id": qid,
            }
        )

    if len(pairs) < n_questions:
        raise SystemExit(
            f"TriviaQA insufficient pairs after sealed-exclusion: {len(pairs)} "
            f"of {n_questions} (excluded {n_excluded}); raise pool multiplier."
        )

    records = []
    for pair in pairs:
        for label, answer_key, kind in (
            (0, "correct_answer", "correct"),
            (1, "wrong_answer", "wrong"),
        ):
            records.append(
                {
                    "prompt": TRIVIAQA_PROMPT_TEMPLATE.format(
                        question=pair["question"], answer=pair[answer_key]
                    ),
                    "label": label,
                    "meta": {
                        "question_id": pair["question_id"],
                        "correct_answer": pair["correct_answer"],
                        "wrong_answer": pair["wrong_answer"],
                        "kind": kind,
                    },
                }
            )
    rng.shuffle(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return {
        "schema_version": "triviaqa_paired_v1",
        "task": "triviaqa_paired",
        "hf_dataset": "trivia_qa",
        "hf_config": "rc.wikipedia",
        "split": "validation",
        "seed": seed,
        "n_questions": len(pairs),
        "n_samples": len(records),
        "n_sealed_excluded_during_sampling": n_excluded,
        "shuffled": True,
        "prompt_template": TRIVIAQA_PROMPT_TEMPLATE,
        "label_convention": {
            "0": "correct_answer - model should answer YES",
            "1": "wrong_answer - model should answer NO (contradiction analog)",
        },
        "sealed_exclusion_source": str(SEALED_TRIVIAQA),
        "data_hash_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "output": str(out_path.resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", default="stage_b/data")
    ap.add_argument("--anli-n-per-class", type=int, default=100)
    ap.add_argument("--triviaqa-n-questions", type=int, default=100)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    anli_path = out_dir / f"anli_R1_seed{a.seed}_n{a.anli_n_per_class * 2}.jsonl"
    tqa_path = out_dir / f"triviaqa_paired_seed{a.seed}_n{a.triviaqa_n_questions * 2}.jsonl"

    print(f"[anli] building n={a.anli_n_per_class * 2} seed={a.seed} ...")
    anli_manifest = build_anli(a.seed, a.anli_n_per_class, anli_path)
    print(f"[anli] wrote {anli_manifest['n_rows']} rows -> {anli_path}")
    print(f"[anli] sealed-excluded during sampling: "
          f"{anli_manifest['n_sealed_excluded_during_sampling']}")

    print(f"[triviaqa] building {a.triviaqa_n_questions} pairs seed={a.seed} ...")
    tqa_manifest = build_triviaqa(a.seed, a.triviaqa_n_questions, tqa_path)
    print(f"[triviaqa] wrote {tqa_manifest['n_samples']} rows "
          f"({tqa_manifest['n_questions']} pairs) -> {tqa_path}")
    print(f"[triviaqa] sealed-excluded during sampling: "
          f"{tqa_manifest['n_sealed_excluded_during_sampling']}")

    for path, manifest in ((anli_path, anli_manifest), (tqa_path, tqa_manifest)):
        mp = path.with_suffix(".manifest.json")
        with mp.open("w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"  manifest: {mp}")

    print("\nNext: gate both files with stage_b/check_fresh_data.py (the registered gate).")


if __name__ == "__main__":
    main()
