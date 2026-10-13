"""A1 full-denominator behavioural report. Execution/verification: not run by Codex.

Usage: python pilot/summarize_endpoint_b.py runs/<id>
Raw model files are immutable; derived reports may be rebuilt after an interrupted run.
"""
import json
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sampling_endpoint_b import (PROTOCOL, MODELS, TASKS, TEMPS, K, SEED, PLAN, NEAR_TIE_MARGIN,
                                 HERE, HARNESS_BUDGET, INVALID_FLAG, GREEDY_INVALID_FLAG,
                                 load_rows, output_dir, cell_status, format_flags, uniforms,
                                 provenance, file_hashes, sha256, UNIFORMS_SHA256)

BUCKETS = ("a_harness", "b_noncompliant", "c_valid_wrong", "d_valid_correct")
DERIVED_REPORTS = ("ACCOUNTED_OUTPUTS.jsonl", "PROMPT_AUDIT.jsonl", "CELL_SUMMARY.json", "SUMMARY.md")


def ratio(x, n):
    return x / n if n else None


def fail_record(base, reason, original=None):
    r = dict(original or {})
    if r.get("bucket") != "a_harness" and "bucket" in r:
        r["observed_bucket"] = r["bucket"]
    return {**r, **base, "bucket": "a_harness", "infrastructure_failure": True, "error": reason}


def historical_failure(r):
    return (r.get("historical_identity_failed") is True or r.get("historical_match") is False
            or (r.get("historical_token_id") is not None and r.get("greedy_token_id") is not None
                and r["historical_token_id"] != r["greedy_token_id"]))


def check_sampling(r, uniform):
    assert r["uniform"] == uniform and np.isfinite(uniform) and 0 <= uniform < 1
    token, size = r["token_id"], r["vocab_size"]
    assert type(token) is int and type(size) is int and 0 <= token < size
    lo, hi = r["cdf_lo"], r["cdf_hi"]
    assert np.isfinite([lo, hi]).all() and 0 <= lo <= hi <= 1 + 1e-12
    assert token != 0 or lo == 0
    assert type(r["cdf_clipped"]) is bool
    if r["cdf_clipped"]:
        # Section 2.6 clips searchsorted's V result when the cumulative sum rounds below u.
        assert token == size - 1 and hi <= uniform < 1
    else:
        assert lo <= uniform < hi


def account_records(path, mi, rows, u):
    model, revision = MODELS[mi]
    indexed, issues = defaultdict(list), []
    expected = {(task, row["sample_idx"], t, d) for task, rs in rows.items()
                for row in rs for t, d in PLAN}
    if path.exists():
        with path.open() as f:
            for line_no, line in enumerate(f, 1):
                try:
                    r = json.loads(line)
                    key = (r["task"], r["sample_idx"], r["T"], r["sample_index"])
                    assert key in expected, "unexpected row/draw"
                    indexed[key].append(r)
                except (ValueError, KeyError, TypeError, AssertionError) as error:
                    issues.append({"line": line_no, "error": str(error), "raw": line.rstrip("\n")})
    accounted, audits = [], []
    for task, rs in rows.items():
        for row in rs:
            idx = row["sample_idx"]
            gs = indexed.get((task, idx, 0.0, 0), [])
            g = gs[0] if len(gs) == 1 else {}
            historical = task == "halueval_qa" and any(
                historical_failure(r) for t, d in PLAN for r in indexed.get((task, idx, t, d), []))
            audit = {"model": model, "task": task, "sample_idx": idx, "stem_id": row.get("stem_id")}
            for field in ("greedy_token_id", "greedy_top_two_ids", "greedy_top_two_logits",
                          "greedy_logit_margin", "near_tie", "historical_token_id", "historical_match"):
                audit[field] = g.get(field)
            if historical:
                audit["historical_match"] = False
            audits.append(audit)
            for temp, draw in PLAN:
                key = (task, idx, temp, draw)
                base = {"model": model, "revision": revision, "task": task, "sample_idx": idx,
                        "stem_id": row.get("stem_id"), "label": row["label"], "T": temp,
                        "sample_index": draw, "seed": SEED}
                copies = indexed.get(key, [])
                if not copies:
                    missing = fail_record(base, "missing/unattempted record")
                    if historical:
                        missing["historical_identity_failed"] = True
                    accounted.append(missing)
                    continue
                r = copies[0]
                reason = None
                if len(copies) != 1:
                    reason = "duplicate planned key; all copies retained in raw file"
                elif issues:
                    reason = "unassignable raw records; model file cannot be trusted"
                elif any(r.get(k) != v for k, v in base.items()):
                    reason = "source/model/seed alignment defect"
                elif r.get("bucket") not in BUCKETS:
                    reason = "unknown bucket"
                # Quarantined draws still have to account for their inverse-CDF selection.
                sampling_failure = None
                if temp != 0 and (r.get("bucket") != "a_harness" or any(
                        k in r for k in ("token_id", "uniform", "cdf_lo", "cdf_hi", "p_valid", "surprise"))):
                    try:
                        check_sampling(r, float(u[(mi, task)][TEMPS.index(temp), idx, draw]))
                    except (KeyError, TypeError, ValueError, AssertionError) as error:
                        sampling_failure = f"inverse-CDF audit: {error}"
                if reason is None and r["bucket"] != "a_harness":
                    try:
                        assert r["row_audit"] is True
                        assert g["greedy_token_id"] == g["token_id"]
                        assert r["greedy_token_id"] == g["greedy_token_id"]
                        for field in ("greedy_top_two_ids", "greedy_top_two_logits", "greedy_logit_margin",
                                      "near_tie", "greedy_canonical"):
                            assert r[field] == g[field], "greedy row audit"
                        top, logits = r["greedy_top_two_ids"], r["greedy_top_two_logits"]
                        assert len(top) == len(logits) == 2 and top[0] != top[1]
                        assert top[0] == r["greedy_token_id"] and np.isfinite(logits).all()
                        margin = logits[0] - logits[1]
                        assert margin >= 0 and r["greedy_logit_margin"] == margin
                        assert r["near_tie"] == (margin <= NEAR_TIE_MARGIN)
                        assert margin != 0 or top[0] < top[1]
                        if task == "halueval_qa":
                            assert r["historical_match"] == (r["historical_token_id"] == g["greedy_token_id"])
                        disp = r["disposition"]
                        assert disp in ("YES", "NO", "non-canonical")
                        correct = disp == ("YES" if row["label"] == 0 else "NO")
                        bucket = "d_valid_correct" if correct else "b_noncompliant" if disp == "non-canonical" else "c_valid_wrong"
                        assert r["bucket"] == bucket and r["sample_error"] == int(not correct)
                        assert r["strict_non_yes"] == int(disp != "YES") and r["token_count"] == 1
                        if temp == 0:
                            assert r["uniform"] is None
                            assert r["greedy_canonical"] == (disp != "non-canonical")
                        else:
                            if sampling_failure is None:
                                assert r["uniform"] == float(u[(mi, task)][TEMPS.index(temp), idx, draw])
                            assert np.isfinite(r["surprise"]) and r["surprise"] >= 0
                            assert all(np.isfinite(r[k]) and 0 <= r[k] <= 1 + 1e-12
                                       for k in ("p_valid", "p_yes", "p_no"))
                            assert abs(r["p_valid"] - r["p_yes"] - r["p_no"]) <= 1e-12
                    except (KeyError, TypeError, ValueError, AssertionError) as error:
                        reason = f"record/greedy row audit: {error}"
                result = fail_record(base, reason or sampling_failure, r) if reason or sampling_failure else dict(r)
                if sampling_failure:
                    result["sampling_verification_failed"] = True
                    result["infrastructure_failure"] = (bool(reason) or r.get("infrastructure_failure") is True
                                                        or (r.get("bucket") == "a_harness" and not historical_failure(r)))
                if historical:
                    if result["bucket"] != "a_harness":
                        result["observed_bucket"] = result["bucket"]
                    result.update(bucket="a_harness", historical_identity_failed=True)
                    result.setdefault("error", "historical_greedy_token_mismatch")
                accounted.append(result)
    assert len(accounted) == len(expected)
    return accounted, audits, issues


def diagnostic_means(rs):
    result = {}
    for scope, records in (("observed_only", [r for r in rs if r["bucket"] != "a_harness"]),
                           ("all_available_distribution", rs)):
        values = {"mean_exact_p_invalid": {}, "mean_surprise": [], "strict_non_yes": []}
        for r in records:
            p = r.get("p_valid")
            if isinstance(p, (int, float)) and np.isfinite(p) and 0 <= p <= 1 + 1e-12:
                values["mean_exact_p_invalid"][r["sample_idx"]] = 1 - p
            surprise = r.get("surprise")
            if isinstance(surprise, (int, float)) and np.isfinite(surprise) and surprise >= 0:
                values["mean_surprise"].append(surprise)
            if r.get("strict_non_yes") in (0, 1):
                values["strict_non_yes"].append(r["strict_non_yes"])
        result[scope] = {}
        for name, vs in values.items():
            # Probability mass is one prefix/T diagnostic; surprise and answer bias are draw diagnostics.
            prompt_level = name == "mean_exact_p_invalid"
            vs = list(vs.values()) if prompt_level else vs
            planned = len({r["sample_idx"] for r in rs}) if prompt_level else len(rs)
            result[scope][name] = {"value": float(np.mean(vs)) if vs else None,
                                   "included": len(vs), "planned": planned,
                                   "unit": "prompts" if prompt_level else "records"}
    return result


def summarize_cell(rs, greedy=False, anli=False):
    counts = Counter(r["bucket"] for r in rs)
    a, b, c, d = (counts[k] for k in BUCKETS)
    n, observed = len(rs), b + c + d
    failures = [r for r in rs if r["bucket"] == "a_harness"]
    historical = sum(historical_failure(r) for r in failures)
    sampling = sum(r.get("sampling_verification_failed") is True for r in rs)
    infrastructure = sum(r.get("infrastructure_failure") is True or (
        not historical_failure(r) and r.get("sampling_verification_failed") is not True) for r in failures)
    status, reasons = cell_status(a, b, c, d, greedy, anli, historical, infrastructure, sampling)
    valid = c + d
    return {"status": status, "reasons": reasons, "planned": n, "observed": observed,
            "format_flags": format_flags(b, observed, greedy),
            "historical_identity_failures": historical, "infrastructure_failures": infrastructure,
            "sampling_verification_failures": sampling,
            "valid": valid, "counts": {k: counts[k] for k in BUCKETS},
            "rates_planned": {k: counts[k] / n for k in BUCKETS},
            "invalid_rate_observed": ratio(b, observed), "primary_error_rate": ratio(b + c, observed),
            "conditional_on_valid_error_rate": ratio(c, valid),
            "sample_error_rate": (b + c) / n if not a else None,
            "sample_error_bounds": [(b + c) / n, (a + b + c) / n],
            "diagnostics": diagnostic_means(rs),
            "failure_reasons": dict(Counter(r.get("error", "unspecified") for r in rs if r["bucket"] == "a_harness"))}


def reversal(rs, rows, task, pair_index):
    unit_ids = list(dict.fromkeys(r["sample_idx"] if task == "anli_r1" else r["stem_id"] for r in rows))
    unit_pos = {key: i for i, key in enumerate(unit_ids)}
    totals = np.zeros((len(unit_ids), len(TEMPS)))
    counts = np.zeros_like(totals)
    complete = True
    for r in rs:
        if r["T"] == 0:
            continue
        unit = r["sample_idx"] if task == "anli_r1" else r["stem_id"]
        i, j = unit_pos[unit], TEMPS.index(r["T"])
        if r["bucket"] == "a_harness":
            complete = False
        else:
            totals[i, j] += r["sample_error"]
            counts[i, j] += 1
    rng = np.random.RandomState(SEED + 40001 + pair_index)
    draws, rates = [], []
    means = None
    if complete:
        assert (counts == (K if task == "anli_r1" else 2 * K)).all()
        means = totals / counts
    # Keep identical joint-T cluster indices even when missing outcomes make every replicate unusable.
    for _ in range(2000):
        indices = rng.randint(0, len(unit_ids), size=len(unit_ids))
        if complete:
            m = means[indices].mean(axis=0)
            rates.append(m)
            draws.append((m[1] - m[0]) * (m[2] - m[1]))
    result = {"seed": SEED + 40001 + pair_index, "clusters": len(unit_ids), "attempted": 2000,
              "usable": len(draws), "point_R": None, "median_R": None, "R_interval": None,
              "sample_error_intervals": None}
    if complete:
        point = means.mean(axis=0)
        result.update(point_R=float((point[1] - point[0]) * (point[2] - point[1])),
                      median_R=float(np.median(draws)), R_interval=np.percentile(draws, [2.5, 97.5]).tolist(),
                      sample_error_intervals=np.percentile(rates, [2.5, 97.5], axis=0).T.tolist())
    else:
        result["reason"] = "harness failures: no complete full-denominator behavioural contrast"
    return result


def show(value):
    return "NA" if value is None else f"{value:.4f}"


def preserve_reports(run):
    existing = [run / name for name in DERIVED_REPORTS
                if (run / name).exists() or (run / name).is_symlink()]
    if not existing:
        return None
    superseded = run / "superseded"
    superseded.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
    archive = Path(tempfile.mkdtemp(prefix=stamp, dir=superseded))
    for path in existing:
        path.rename(archive / path.name)
    return archive


def main(value):
    run = output_dir(value)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["protocol"] == PROTOCOL
    assert manifest["gates"] == {"infrastructure_fraction": HARNESS_BUDGET, "historical_mismatches": 0,
                                 "sampling_verification_failures": 0,
                                 "observed_min": 1, "anli_greedy_canonical_min": 190}
    assert manifest["format_flags"] == {"sampled_invalid": INVALID_FLAG, "greedy_invalid": GREEDY_INVALID_FLAG}
    assert manifest["near_tie_margin"] == NEAR_TIE_MARGIN
    assert manifest["launch_files_sha256"] == provenance(), "launch implementation changed since launch"
    assert manifest["uniforms_sha256"] == sha256(run / "uniforms.npy") == UNIFORMS_SHA256
    analysis_files = file_hashes([HERE / "summarize_endpoint_b.py"])
    analysis_changed = analysis_files != manifest["analysis_files_sha256_at_launch"]
    rows, (_, u) = load_rows(), uniforms()
    archive = preserve_reports(run)
    report = {"protocol": PROTOCOL, "cells": [], "reversals": [], "raw_file_issues": {},
              "superseded_reports": str(archive.relative_to(run)) if archive else None,
              "amendment_commit": manifest["amendment_commit"], "amendment_tree": manifest["amendment_tree"],
              "analysis_files_sha256": analysis_files,
              "analysis_files_sha256_at_launch": manifest["analysis_files_sha256_at_launch"],
              "analysis_changed_since_launch": analysis_changed,
              "execution_status": manifest.get("execution_status", "interrupted/in-progress"),
              "stop_reason": manifest.get("stop_reason")}
    lines = [f"# A1 standalone behavioural Endpoint B: {run.name}", "",
             "Pilot-informed descriptive protocol; 24 sampled cells plus 8 greedy diagnostics. Phi retained.",
             "ANLI has no historical per-row token record; no full-panel transitive identity protection is available.",
             "Endpoint A, the 31-cell panel, detector AUROCs and fusion/surprise detector reversals are NOT COMPUTED.",
             f"Launch commit: {manifest['amendment_commit']}; tree: {manifest['amendment_tree']}.",
             f"Analysis SHA-256 (repo-relative paths): {json.dumps(analysis_files, sort_keys=True)}.",
             f"Analysis SHA-256 at launch: {json.dumps(manifest['analysis_files_sha256_at_launch'], sort_keys=True)}.",
             f"Disclosure: analysis code {'changed' if analysis_changed else 'unchanged'} since launch; launch implementation hashes match. Reporting fixes may rebuild these derived reports.",
             f"Prior derived reports: {report['superseded_reports'] or 'none to preserve'}.",
             "", "Counts and a/b/c/d rates use planned N. Primary error=(b+c)/(N-a); conditional error=c/(c+d).",
             "Sample_error is the full prompt/stem-unit mean, NA with any harness failure; its bounds retain all slots.",
             "Integrity gates: zero historical ID mismatches; zero sampling-verification failures; missing/infrastructure failures <=1% of N; observed outcomes >0.",
             "Only behavioural gate: ANLI greedy >=190/200 canonical. FAILED cells remain fully reported; no cascades.",
             "Descriptive format-usability flags only: invalid/observed >25% sampled or >5% greedy; these do not fail cells.", "",
             "| Model | Task | T | Status | N | a harness | b invalid | c valid wrong | d valid correct | a/N | b/N | c/N | d/N | primary error | conditional error | sample_error | bounds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    details, near_ties, hist, parser = [], [], [], []
    with (run / "ACCOUNTED_OUTPUTS.jsonl").open("w") as f, (run / "PROMPT_AUDIT.jsonl").open("w") as af:
        for mi, (model, _) in enumerate(MODELS):
            recs, audits, issues = account_records(run / f"{model.split('/')[1]}.jsonl", mi, rows, u)
            report["raw_file_issues"][model] = issues
            for r in recs:
                f.write(json.dumps(r, allow_nan=False) + "\n")
                if ("independent_parser_canonical" in r and "disposition" in r
                        and r["independent_parser_canonical"] != (r["disposition"] != "non-canonical")):
                    parser.append({k: r[k] for k in ("model", "task", "sample_idx", "T", "sample_index", "raw_text")})
            for a in audits:
                af.write(json.dumps(a, allow_nan=False) + "\n")
            for ti, (task, _, _, _) in enumerate(TASKS):
                task_recs = [r for r in recs if r["task"] == task]
                gs = [a for a in audits if a["task"] == task]
                available = [a for a in gs if a["greedy_logit_margin"] is not None]
                near = [a for a in available if a["near_tie"]]
                near_ties.append({"model": model, "task": task, "planned_prompts": len(gs),
                                  "available": len(available), "near_ties": len(near),
                                  "near_tie_historical_mismatches": sum(a["historical_match"] is False for a in near)})
                if task == "halueval_qa":
                    hist.append({"model": model, "planned": 1000,
                                 "matches": sum(a["historical_match"] is True for a in gs),
                                 "mismatches": sum(a["historical_match"] is False for a in gs),
                                 "near_tie_mismatches": sum(a["historical_match"] is False and a["near_tie"] is True for a in gs),
                                 "unavailable": sum(a["historical_match"] is None for a in gs)})
                for temp in (0.0,) + TEMPS:
                    cell = summarize_cell([r for r in task_recs if r["T"] == temp], temp == 0, task == "anli_r1")
                    cell.update(model=model, task=task, T=temp)
                    report["cells"].append(cell)
                    cs, rates = cell["counts"], cell["rates_planned"]
                    label = "greedy" if temp == 0 else str(temp)
                    lines.append(f"| {model.split('/')[1]} | {task} | {label} | {cell['status']} | {cell['planned']} | "
                                 + " | ".join(str(cs[k]) for k in BUCKETS) + " | "
                                 + " | ".join(show(rates[k]) for k in BUCKETS) + " | "
                                 + " | ".join(show(cell[k]) for k in ("primary_error_rate", "conditional_on_valid_error_rate", "sample_error_rate"))
                                 + f" | {cell['sample_error_bounds']} |")
                    details.append(f"- {model.split('/')[1]} / {task} / {label}: reasons={cell['reasons']}; "
                                   f"format flags={cell['format_flags']}; coverage={cell['observed']}/{cell['planned']}; "
                                   f"historical failure slots={cell['historical_identity_failures']}; "
                                   f"infrastructure failure slots={cell['infrastructure_failures']}; "
                                   f"inverse-CDF failure slots={cell['sampling_verification_failures']}; "
                                   f"invalid/observed={show(cell['invalid_rate_observed'])}; "
                                   + "; ".join(f"{scope} {name}={show(v['value'])} "
                                               f"({v['included']}/{v['planned']} {v['unit']})"
                                               for scope, metrics in cell["diagnostics"].items()
                                               for name, v in metrics.items()) + ".")
                report["reversals"].append({"model": model, "task": task,
                                           **reversal(task_recs, rows[task], task, mi * 2 + ti)})
    assert len(report["cells"]) == 32 and len(report["reversals"]) == 8
    report.update(near_ties=near_ties, historical_gate=hist, parser_disagreements=parser)
    lines += ["", "## Cell diagnostics", "",
              "Observed-only excludes harness records. All-available-distribution retains finite diagnostics from quarantined records; these are not trusted outcome rates.",
              "Exact P(invalid) counts distinct prefixes per T; surprise and strict-non-YES count records. Strict-non-YES uses actual available token outcomes, not expected 1-P(YES).",
              "Greedy has no temperature distribution or surprise: NA, with zero included. Failure categories can overlap within the same harness slot.", ""] + details
    lines += ["", "## Greedy margins and historical gate", "",
              f"Near tie: unscaled top-two logit gap <= {NEAR_TIE_MARGIN}; exact ties choose the lower token ID.",
              "Every available margin/top-two ID pair, including near ties, is in PROMPT_AUDIT.jsonl. Unavailable values are null."]
    lines += [f"- {json.dumps(v)}" for v in near_ties + hist]
    lines += ["", "## Behavioural joint-T reversals (descriptive)", "",
              "2,000 shared-T cluster resamples per pair; ANLI prompt / HaluEval paired stem. Format-usability flags do not suppress this analysis."]
    lines += [f"- {json.dumps(v)}" for v in report["reversals"]]
    lines += ["", f"Parser disagreements (diagnostic only): {len(parser)}. Details and unassignable raw lines are in CELL_SUMMARY.json.",
              "Raw per-model files are preserved. ACCOUNTED_OUTPUTS.jsonl includes every planned slot and explicit missing/failed records.",
              "One-token length termination is expected, not truncation. Harness failures are never model-invalid outcomes."]
    (run / "CELL_SUMMARY.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (run / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
