#!/usr/bin/env python3
"""Build BENCH v1.2 data exactly as frozen in PRE_REGISTRATION_BENCH.md §3.

This is Phase-1-authored code. Running it is Phase 2 and is intentionally left to the
executor after MK sign-off and the extension-manifest freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


SEED = 20260711
HALU_COMMIT = "b7253db3cdaa0ab2c382f92b26b390109174f77e"
HALU_RAW = {
    "halueval_qa": "qa_data.json",
    "halueval_dialogue": "dialogue_data.json",
    "halueval_summarization": "summarization_data.json",
}
HALU_URL = (
    "https://raw.githubusercontent.com/RUCAIBox/HaluEval/"
    f"{HALU_COMMIT}/data/{{filename}}"
)

ANLI_FINGERPRINT = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
TRIVIA_FINGERPRINT = "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
ARROW_SHA256 = {
    "anli-train_r1.arrow": "b32df9e1ee446fa9d34c6996f788dbce7fbbe9ec682d0672cb340837904ee40a",
    "anli-dev_r2.arrow": "6ff4c3bac8b0ae917cf89dd73cf9966107d5888232d8e423ecde8da8555486fd",
    "anli-test_r2.arrow": "d63398b51f5c29f92b251b1f5b54c9a1a5c9772a1b2a7ed96a047cee0221e655",
    "trivia_qa-validation.arrow": "8e95a5f9ce34a037cc3dd0d2e544961a20470cb6c415f6ab48a1e115ed5a7a90",
}

COHORT = [
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

T0_REPO = Path(os.environ.get(
    "CONFLUENCE_T0_REPO", os.path.expanduser("~/Documents/t0-morphology-furnace")))
SEALED_REFERENCES = [
    T0_REPO / "experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl",
    T0_REPO / "experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl",
    Path(__file__).parent / "data/anli_R1_seed20260612_n200.jsonl",
    Path(__file__).parent / "data/triviaqa_paired_seed20260612_n200.jsonl",
]

ANLI_PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n\n"
    "Premise: {premise}\nHypothesis: {hypothesis}\nAnswer:"
)
TRIVIAQA_PROMPT_TEMPLATE = (
    "Instruction: Read the question and proposed answer, then decide whether "
    "the proposed answer is correct. Answer YES if the proposed answer is "
    "correct, NO if the proposed answer is incorrect.\n\n"
    "Question: {question}\nProposed answer: {answer}\nAnswer:"
)
HALU_TEMPLATES = {
    "halueval_qa": (
        "You are given reference knowledge, a question, and a candidate answer. Decide\n"
        "whether the candidate answer is faithful to the reference knowledge. Answer YES\n"
        "if the candidate answer is supported by the knowledge, NO if it contains\n"
        "hallucinated or fabricated content.\n\n"
        "Knowledge: {knowledge}\nQuestion: {question}\nCandidate answer: {candidate}\n\n"
        "Is the candidate answer faithful to the knowledge? Answer:"
    ),
    "halueval_dialogue": (
        "You are given reference knowledge, a dialogue history, and a candidate response.\n"
        "Decide whether the candidate response is faithful to the reference knowledge.\n"
        "Answer YES if the candidate response is supported by the knowledge, NO if it\n"
        "contains hallucinated or fabricated content.\n\n"
        "Knowledge: {knowledge}\nDialogue history: {dialogue_history}\n"
        "Candidate response: {candidate}\n\n"
        "Is the candidate response faithful to the knowledge? Answer:"
    ),
    "halueval_summarization": (
        "You are given a document and a candidate summary. Decide whether the candidate\n"
        "summary is faithful to the document. Answer YES if the candidate summary is\n"
        "supported by the document, NO if it contains hallucinated or fabricated content.\n\n"
        "Document: {document}\nCandidate summary: {candidate}\n\n"
        "Is the candidate summary faithful to the document? Answer:"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()


def _prompt_of(row: Dict[str, Any]) -> str | None:
    return row.get("prompt") if row.get("prompt") is not None else row.get("text")


def exclusion_union(paths: Sequence[Path]) -> Tuple[set, set]:
    prompt_hashes, question_ids = set(), set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"enumerated exclusion reference missing: {path}")
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = _prompt_of(row)
                if prompt:
                    prompt_hashes.add(norm_hash(prompt))
                qid = (row.get("meta") or {}).get("question_id")
                if qid is not None:
                    question_ids.add(str(qid))
    return prompt_hashes, question_ids


def default_arrow_paths() -> Dict[str, Path]:
    cache = Path(os.environ.get(
        "HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets")))
    anli = cache / "facebook___anli/plain_text/0.0.0" / ANLI_FINGERPRINT
    trivia = cache / "trivia_qa/rc.wikipedia/0.0.0" / TRIVIA_FINGERPRINT
    return {
        "train_r1": anli / "anli-train_r1.arrow",
        "dev_r2": anli / "anli-dev_r2.arrow",
        "test_r2": anli / "anli-test_r2.arrow",
        "triviaqa_validation": trivia / "trivia_qa-validation.arrow",
    }


def load_frozen_arrow(path: Path, expected_sha256: str):
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"frozen Arrow sha256 mismatch: {path}")
    from datasets import Dataset
    return Dataset.from_file(str(path))


def load_cohort_tokenizers():
    """Tokenizer-only local loads: no model weights and no network fallback."""
    import sys
    if str(T0_REPO) not in sys.path:
        sys.path.insert(0, str(T0_REPO))
    import pri_v2_io_plugins as io_plugins
    from mlx_lm.utils import hf_repo_to_path, load_tokenizer

    out = {}
    for model_id in COHORT:
        local_path = hf_repo_to_path(model_id)
        tokenizer = load_tokenizer(local_path)
        out[model_id] = (tokenizer, io_plugins.get_prompt_strategy(model_id))
    return out


def wrapped_token_counts(prompt: str, tokenizers) -> Dict[str, int]:
    counts = {}
    for model_id, (tokenizer, strategy) in tokenizers.items():
        wrapped = strategy(prompt, tokenizer)
        counts[model_id] = int(len(tokenizer.encode(wrapped)))
    return counts


def length_metadata(prompts: Sequence[str], tokenizers, cap: int = 2048):
    per_prompt = [wrapped_token_counts(prompt, tokenizers) for prompt in prompts]
    maximum = max(count for counts in per_prompt for count in counts.values())
    return bool(maximum <= cap), per_prompt, maximum


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return sha256_file(path)


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _parse_halueval(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text().strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"unexpected HaluEval format: {path}")
    return value


def fetch_halueval(task: str, raw_dir: Path) -> Tuple[List[Dict[str, Any]], Path, str]:
    filename = HALU_RAW[task]
    path = raw_dir / HALU_COMMIT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HALU_URL.format(filename=filename)) as response:
        pinned_bytes = response.read()
    if path.exists() and path.read_bytes() != pinned_bytes:
        raise RuntimeError(f"cached HaluEval bytes disagree with pinned commit: {path}")
    path.write_bytes(pinned_bytes)
    digest = sha256_file(path)
    return _parse_halueval(path), path, digest


def _halueval_prompts(task: str, item: Dict[str, Any]) -> Tuple[str, str]:
    template = HALU_TEMPLATES[task]
    if task == "halueval_qa":
        common = {"knowledge": item["knowledge"], "question": item["question"]}
        return (template.format(candidate=item["right_answer"], **common),
                template.format(candidate=item["hallucinated_answer"], **common))
    if task == "halueval_dialogue":
        common = {"knowledge": item["knowledge"],
                  "dialogue_history": item["dialogue_history"]}
        return (template.format(candidate=item["right_response"], **common),
                template.format(candidate=item["hallucinated_response"], **common))
    return (template.format(document=item["document"], candidate=item["right_summary"]),
            template.format(document=item["document"], candidate=item["hallucinated_summary"]))


def build_halueval(task: str, raw_dir: Path, out_path: Path, tokenizers,
                   target_stems: int = 500, min_exploratory_stems: int = 300) -> Dict[str, Any]:
    items, raw_path, raw_sha = fetch_halueval(task, raw_dir)
    eligible, rejected_length = [], 0
    for source_index, item in enumerate(items):
        prompts = _halueval_prompts(task, item)
        ok, counts, maximum = length_metadata(prompts, tokenizers)
        if not ok:
            rejected_length += 1
            continue
        eligible.append((source_index, prompts, counts, maximum))
    if task == "halueval_qa" and len(eligible) < target_stems:
        raise RuntimeError(f"confirmatory HaluEval-QA supply abort: {len(eligible)} < {target_stems}")
    n_stems = min(target_stems, len(eligible))
    if task != "halueval_qa" and n_stems < min_exploratory_stems:
        raise RuntimeError(
            f"exploratory {task} abandoned: {n_stems} admissible stems < {min_exploratory_stems}")
    rng = np.random.RandomState(SEED)
    permutation = rng.permutation(len(eligible)).tolist()
    shuffled_source_ids = [eligible[i][0] for i in permutation]
    selected = [eligible[i] for i in permutation[:n_stems]]
    rows, selected_ids = [], []
    for source_index, prompts, counts, maximum in selected:
        selected_ids.append(source_index)
        for label, prompt, count_map in ((0, prompts[0], counts[0]), (1, prompts[1], counts[1])):
            rows.append({
                "prompt": prompt, "label": label, "stem_id": str(source_index),
                "meta": {"task": task, "source_index": source_index,
                         "wrapped_token_counts": count_map,
                         "max_wrapped_tokens": max(count_map.values()),
                         "stem_max_wrapped_tokens": maximum},
            })
    rng.shuffle(rows)
    data_sha = write_jsonl(out_path, rows)
    return {
        "schema_version": "bench-data/1.2", "task": task, "seed": SEED,
        "source": "RUCAIBox/HaluEval", "source_commit": HALU_COMMIT,
        "raw_file": str(raw_path), "raw_sha256": raw_sha,
        "n_source": len(items), "n_admissible_stems": len(eligible),
        "n_length_rejected": rejected_length, "n_selected_stems": n_stems,
        "n_rows": len(rows), "rng": "numpy.random.RandomState", "rng_seed": SEED,
        "selected_label_counts": {"0": n_stems, "1": n_stems},
        "effective_n_stems": n_stems,
        "candidate_filter_counts": {
            "source_stems": len(items), "length_rejected": rejected_length,
            "admissible_stems": len(eligible), "selected_stems": n_stems,
        },
        "ordered_shuffled_indices": shuffled_source_ids, "selected_source_ids": selected_ids,
        "length_cap": 2048, "prompt_template": HALU_TEMPLATES[task],
        "data_sha256": data_sha, "output": str(out_path.resolve()),
    }


def build_halueval_qa(raw_dir: Path, out_path: Path, tokenizers) -> Dict[str, Any]:
    return build_halueval("halueval_qa", raw_dir, out_path, tokenizers)


def build_halueval_dialogue(raw_dir: Path, out_path: Path, tokenizers) -> Dict[str, Any]:
    return build_halueval("halueval_dialogue", raw_dir, out_path, tokenizers)


def build_halueval_summarization(raw_dir: Path, out_path: Path, tokenizers) -> Dict[str, Any]:
    return build_halueval("halueval_summarization", raw_dir, out_path, tokenizers)


def _anli_record(ex: Dict[str, Any], split: str, source_index: int, tokenizers):
    prompt = ANLI_PROMPT_TEMPLATE.format(premise=ex["premise"], hypothesis=ex["hypothesis"])
    ok, counts, maximum = length_metadata([prompt], tokenizers)
    return prompt, ok, counts[0], maximum, f"{split}:{source_index}"


def build_anli_generic(task: str, arrow_paths: Dict[str, Path], out_path: Path,
                       tokenizers, excluded_hashes: set,
                       preview_limit: int | None = None) -> Dict[str, Any]:
    if task not in {"anli_r1_rep", "anli_r2"}:
        raise ValueError(task)
    split_names = ["train_r1"] if task == "anli_r1_rep" else ["dev_r2", "test_r2"]
    datasets = {split: load_frozen_arrow(
        arrow_paths[split], ARROW_SHA256[arrow_paths[split].name]) for split in split_names}
    rng = np.random.RandomState(SEED)
    selected_by_stratum, shuffled_indices, candidate_filter_counts = {}, {}, {}
    excluded_count = length_rejected = duplicate_count = 0
    seen = set()
    quota = (500 if task == "anli_r1_rep" else 250)
    if preview_limit is not None:
        divisor = 2 if task == "anli_r1_rep" else 4
        if preview_limit % divisor:
            raise ValueError(f"{task} preview limit must be divisible by {divisor}")
        quota = preview_limit // divisor
    for split in split_names:
        ds = datasets[split]
        prepared = {}
        counts_for_split = {"source_rows": len(ds), "neutral": 0, "excluded": 0,
                            "length_rejected": 0, "admissible_before_dedup": 0}
        for source_index in range(len(ds)):
            ex = ds[source_index]
            if int(ex["label"]) == 1:
                counts_for_split["neutral"] += 1
                continue
            prompt, length_ok, counts, maximum, stem_id = _anli_record(
                ex, split, source_index, tokenizers)
            prompt_hash = norm_hash(prompt)
            if prompt_hash in excluded_hashes:
                counts_for_split["excluded"] += 1
                continue
            if not length_ok:
                counts_for_split["length_rejected"] += 1
                continue
            counts_for_split["admissible_before_dedup"] += 1
            prepared[source_index] = (
                ex, prompt, counts, maximum, stem_id, prompt_hash)
        candidate_filter_counts[split] = counts_for_split
        excluded_count += counts_for_split["excluded"]
        length_rejected += counts_for_split["length_rejected"]
        order = list(range(len(ds)))
        rng.shuffle(order)
        shuffled_indices[split] = order
        selected_by_stratum[(split, 0)] = []
        selected_by_stratum[(split, 2)] = []
        for source_index in order:
            if source_index not in prepared:
                continue
            ex, prompt, counts, maximum, stem_id, prompt_hash = prepared[source_index]
            if prompt_hash in seen:
                duplicate_count += 1
                continue
            label = int(ex["label"])
            bucket = selected_by_stratum[(split, label)]
            if len(bucket) >= quota:
                continue
            bucket.append({
                "prompt": prompt, "label": 0 if label == 0 else 1, "stem_id": stem_id,
                "meta": {"task": task, "source_split": split,
                         "source_index": source_index, "wrapped_token_counts": counts,
                         "max_wrapped_tokens": maximum},
            })
            seen.add(prompt_hash)
            if all(len(selected_by_stratum[(split, y)]) == quota for y in (0, 2)):
                break
        for label in (0, 2):
            if len(selected_by_stratum[(split, label)]) != quota:
                raise RuntimeError(
                    f"{task} supply abort: {split}/label{label} "
                    f"{len(selected_by_stratum[(split, label)])} != {quota}")
    ordered_keys = ([("train_r1", 0), ("train_r1", 2)] if task == "anli_r1_rep" else
                    [("dev_r2", 0), ("dev_r2", 2), ("test_r2", 0), ("test_r2", 2)])
    rows = [row for key in ordered_keys for row in selected_by_stratum[key]]
    selected_ids = [row["stem_id"] for row in rows]
    rng.shuffle(rows)
    data_sha = write_jsonl(out_path, rows)
    return {
        "schema_version": "bench-data/1.2", "task": task, "seed": SEED,
        "preview": preview_limit is not None,
        "source": "frozen facebook/anli Arrow", "artifact_fingerprint": ANLI_FINGERPRINT,
        "source_files": {split: {"path": str(arrow_paths[split]),
                                  "sha256": ARROW_SHA256[arrow_paths[split].name]}
                         for split in split_names},
        "n_rows": len(rows), "quota_per_class_per_split": quota,
        "selected_label_counts": {
            "0": sum(int(row["label"]) == 0 for row in rows),
            "1": sum(int(row["label"]) == 1 for row in rows),
        },
        "effective_n_stems": len(rows),
        "n_excluded": excluded_count, "n_length_rejected": length_rejected,
        "n_duplicate_rejected": duplicate_count, "rng": "numpy.random.RandomState",
        "candidate_filter_counts": candidate_filter_counts,
        "rng_seed": SEED, "ordered_shuffled_indices": shuffled_indices,
        "selected_source_ids": selected_ids, "length_cap": 2048,
        "prompt_template": ANLI_PROMPT_TEMPLATE, "data_sha256": data_sha,
        "output": str(out_path.resolve()),
    }


def _canonical(item) -> str:
    return item["answer"]["value"].strip()


def _all_aliases(item) -> set:
    aliases = item["answer"].get("aliases", [])
    value = item["answer"]["value"]
    return {a.lower().strip() for a in aliases} | {value.lower().strip()}


def build_triviaqa_rep(arrow_path: Path, out_path: Path, tokenizers,
                       excluded_hashes: set, excluded_qids: set,
                       target_questions: int = 500) -> Dict[str, Any]:
    ds = load_frozen_arrow(arrow_path, ARROW_SHA256[arrow_path.name])
    all_qids = [str(ds[i]["question_id"]) for i in range(len(ds))]
    if len(all_qids) != len(set(all_qids)):
        raise RuntimeError("TriviaQA duplicate stable question_id in frozen source")
    rng = random.Random(SEED)
    pool_size = min(len(ds), target_questions * 10)
    pool_indices = rng.sample(range(len(ds)), pool_size)
    pool = [ds[i] for i in pool_indices]
    donor_pool = list(pool)
    rng.shuffle(donor_pool)
    donor_order = [str(item["question_id"]) for item in donor_pool]
    pairs, used_qids, used_wrong = [], set(), set()
    donor_idx = qid_excluded = prompt_excluded = length_rejected = no_donor = 0
    for item in pool:
        if len(pairs) >= target_questions:
            break
        qid = str(item["question_id"])
        if qid in used_qids or qid in excluded_qids:
            qid_excluded += int(qid in excluded_qids)
            continue
        correct, correct_set = _canonical(item), _all_aliases(item)
        wrong = None
        for offset in range(len(donor_pool)):
            other = donor_pool[(donor_idx + offset) % len(donor_pool)]
            candidate = _canonical(other)
            if (str(other["question_id"]) == qid or candidate.lower().strip() in correct_set
                    or len(candidate.strip()) < 2 or candidate in used_wrong):
                continue
            wrong = candidate
            donor_idx = (donor_idx + offset + 1) % len(donor_pool)
            break
        if wrong is None:
            no_donor += 1
            continue
        question = item["question"].strip()
        prompts = [TRIVIAQA_PROMPT_TEMPLATE.format(question=question, answer=correct),
                   TRIVIAQA_PROMPT_TEMPLATE.format(question=question, answer=wrong)]
        if any(norm_hash(prompt) in excluded_hashes for prompt in prompts):
            prompt_excluded += 1
            continue
        length_ok, counts, maximum = length_metadata(prompts, tokenizers)
        if not length_ok:
            length_rejected += 1
            continue
        used_qids.add(qid)
        used_wrong.add(wrong)
        pairs.append({"question": question, "correct_answer": correct,
                      "wrong_answer": wrong, "question_id": qid,
                      "token_counts": counts, "maximum": maximum})
    if len(pairs) != target_questions:
        raise RuntimeError(
            f"TriviaQA supply abort: {len(pairs)} pairs != {target_questions}")
    records = []
    for pair in pairs:
        for label, key, kind, counts in (
            (0, "correct_answer", "correct", pair["token_counts"][0]),
            (1, "wrong_answer", "wrong", pair["token_counts"][1]),
        ):
            records.append({
                "prompt": TRIVIAQA_PROMPT_TEMPLATE.format(
                    question=pair["question"], answer=pair[key]),
                "label": label, "stem_id": pair["question_id"],
                "meta": {"task": "triviaqa_paired_rep", "question_id": pair["question_id"],
                         "correct_answer": pair["correct_answer"],
                         "wrong_answer": pair["wrong_answer"], "kind": kind,
                         "wrapped_token_counts": counts,
                         "max_wrapped_tokens": max(counts.values()),
                         "stem_max_wrapped_tokens": pair["maximum"]},
            })
    selected_ids = [pair["question_id"] for pair in pairs]
    rng.shuffle(records)
    data_sha = write_jsonl(out_path, records)
    return {
        "schema_version": "bench-data/1.2", "task": "triviaqa_paired_rep",
        "seed": SEED, "preview": target_questions != 500,
        "source": "frozen trivia_qa rc.wikipedia Arrow",
        "artifact_fingerprint": TRIVIA_FINGERPRINT, "source_file": str(arrow_path),
        "source_sha256": ARROW_SHA256[arrow_path.name], "n_questions": target_questions,
        "n_rows": target_questions * 2, "pool_size": pool_size,
        "selected_label_counts": {"0": target_questions, "1": target_questions},
        "effective_n_stems": target_questions,
        "ordered_shuffled_indices": pool_indices,
        "ordered_donor_question_ids": donor_order,
        "selected_source_ids": selected_ids,
        "n_excluded": qid_excluded + prompt_excluded,
        "n_length_rejected": length_rejected, "rng": "random.Random", "rng_seed": SEED,
        "candidate_filter_counts": {
            "source_rows": len(ds), "pool_rows": len(pool),
            "qid_excluded": qid_excluded, "prompt_excluded": prompt_excluded,
            "length_rejected": length_rejected, "no_usable_donor": no_donor,
            "selected_pairs": len(pairs)},
        "length_cap": 2048, "prompt_template": TRIVIAQA_PROMPT_TEMPLATE,
        "data_sha256": data_sha, "output": str(out_path.resolve()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="stage_b/data_bench")
    ap.add_argument("--raw-dir", default="stage_b/data_bench/raw_halueval")
    ap.add_argument("--tasks", default=",".join([
        "halueval_qa", "anli_r1_rep", "triviaqa_paired_rep", "anli_r2",
        "halueval_dialogue", "halueval_summarization"]))
    ap.add_argument("--train-r1-arrow")
    ap.add_argument("--dev-r2-arrow")
    ap.add_argument("--test-r2-arrow")
    ap.add_argument("--triviaqa-arrow")
    ap.add_argument("--preview-limit", type=int,
                    help="executor-only non-registered row limit; use 8 for dry runs")
    a = ap.parse_args()
    if SEED != 20260711:
        raise RuntimeError("BENCH seed drift")

    defaults = default_arrow_paths()
    arrows = {
        "train_r1": Path(a.train_r1_arrow or defaults["train_r1"]),
        "dev_r2": Path(a.dev_r2_arrow or defaults["dev_r2"]),
        "test_r2": Path(a.test_r2_arrow or defaults["test_r2"]),
        "triviaqa_validation": Path(a.triviaqa_arrow or defaults["triviaqa_validation"]),
    }
    requested = [task for task in a.tasks.split(",") if task]
    unknown = set(requested) - set(HALU_RAW) - {"anli_r1_rep", "anli_r2", "triviaqa_paired_rep"}
    if unknown:
        raise ValueError(f"unknown tasks: {sorted(unknown)}")
    tokenizers = load_cohort_tokenizers()
    excluded_hashes, excluded_qids = exclusion_union(SEALED_REFERENCES)
    out_dir, raw_dir = Path(a.out_dir), Path(a.raw_dir)

    manifests = {}
    if a.preview_limit is not None and (a.preview_limit <= 0 or a.preview_limit % 2):
        raise ValueError("--preview-limit must be a positive even row count")
    for task in requested:
        out_path = out_dir / f"{task}_seed{SEED}_n1000.jsonl"
        if task in HALU_RAW:
            manifest = build_halueval(
                task, raw_dir, out_path, tokenizers,
                target_stems=(a.preview_limit // 2 if a.preview_limit else 500),
                min_exploratory_stems=(0 if a.preview_limit else 300))
            manifest["preview"] = a.preview_limit is not None
        elif task in {"anli_r1_rep", "anli_r2"}:
            manifest = build_anli_generic(
                task, arrows, out_path, tokenizers, excluded_hashes,
                preview_limit=a.preview_limit)
        else:
            manifest = build_triviaqa_rep(
                arrows["triviaqa_validation"], out_path, tokenizers,
                excluded_hashes, excluded_qids,
                target_questions=(a.preview_limit // 2 if a.preview_limit else 500))
        if manifest["n_rows"] != 1000:
            actual_path = out_dir / f"{task}_seed{SEED}_n{manifest['n_rows']}.jsonl"
            out_path.rename(actual_path)
            out_path = actual_path
            manifest["output"] = str(out_path.resolve())
        manifest["exclusion_references"] = [str(path) for path in SEALED_REFERENCES]
        manifest_path = out_path.with_suffix(".manifest.json")
        write_manifest(manifest_path, manifest)
        manifests[task] = {"data": str(out_path), "manifest": str(manifest_path)}
        print(f"[{task}] {manifest['n_rows']} rows -> {out_path}")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
