#!/usr/bin/env python3
"""Additive pass-one capture overlay; authoring does not authorize execution.

The original pass-two harness runs unchanged behind temporary, single-threaded
in-memory observation hooks. No sealed file is edited. Temperature replay uses
static projection rows, not banked W_s; see rpv_pass1_temperature.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import rpv_pass1_temperature as offline


def bank_source(logits, probabilities, committed_id, projection, ref, *, readout=False):
    """Bank float32 logits/int32 indices and float64 LSE for EVERY frozen T."""
    z = np.asarray(logits, dtype=np.float32)
    p = np.asarray(probabilities, dtype=np.float64)
    if z.ndim != 1 or p.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite or malformed full-vocabulary source")
    if np.sum(p) <= 0:
        raise ValueError("Empty source probability mass")
    # Steward correction 2026-09-16: an earlier comment here claimed this
    # renormalization matched the reference. It does NOT -- trace_pair_features
    # passes p_t straight into _support_spectrum with no renormalization. The step
    # is removed so the inline path treats p exactly as the reference does.
    # (fc_full_spectrum renormalizes over the support regardless, so this only
    # ever changed float rounding, never the mathematics.)
    idx = ref._topk_indices(p, 512)
    # Preserve the reference's support order; reject rounding/underflow that
    # actually replaces a larger logit by a smaller one at the cutoff.
    cutoff = np.partition(z, len(z) - len(idx))[len(z) - len(idx)]
    if float(np.min(z[idx])) < float(cutoff):
        raise ValueError("Probability top-k differs from logit top-k")
    lse = {}
    for temperature in offline.GRID:
        scaled = z.astype(np.float64) / temperature
        maximum = float(np.max(scaled))
        lse[str(temperature)] = maximum + float(np.log(np.sum(np.exp(scaled - maximum))))
    bank = dict(support_idx=idx.astype(np.int32).tolist(),
                support_logits=z[idx].astype(np.float32).tolist(),
                full_vocab_lse=lse, max_logit=float(np.max(z)),
                committed_token_logit=float(z[committed_id]),
                d_model=int(projection.hidden_size), vocab_size=int(z.size))
    W_s = projection.get_rows(idx)
    if W_s is None or not np.isfinite(W_s).all():
        raise ValueError("Missing/nonfinite support geometry")
    W_s = W_s.astype(np.float64)
    spectrum = ref.fc_full_spectrum(W_s, p[idx], projection.hidden_size)
    if not np.isfinite(spectrum).all():
        raise ValueError("Nonfinite inline spectrum")
    inline = ref._spectrum_stats(spectrum)
    # Readout trace probabilities use float32 safe_softmax, whereas block
    # logit lenses use float64 softmax_np. Record BOTH full-vocab routes so
    # the extra native-readout rounding is measurable rather than hidden.
    full = ref.softmax_np(z)
    full_spectrum = ref.fc_full_spectrum(W_s, full[idx], projection.hidden_size)
    if not np.isfinite(full_spectrum).all():
        raise ValueError("Nonfinite full-softmax spectrum")
    full_stats = ref._spectrum_stats(full_spectrum)
    replay = offline.reconstruct_source(bank, projection, ref, 1.0)
    for values in (inline, full_stats, replay):
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite source statistics")
    return bank, inline, full_stats, replay


def pass_one(trace, prefix_logits, projection, model, ref):
    """Use the prefix position and the identical pinned layer aggregation."""
    token = int(trace["gen_token_ids"][0])
    if token != int(np.argmax(prefix_logits)):
        raise ValueError("Committed token is not the greedy prefix argmax")
    n_layers = int(trace["n_layers"])
    window = ref._pinned_late_layers(n_layers)
    canonical = ref.pipeline.get_all_layer_indices(n_layers)
    by_index = {int(index): name for name, index in canonical.items()}
    hiddens = trace["last_prefix_hidden"]
    # A2 fails explicitly per row/model; never silently average fewer blocks.
    missing = [i for i in window["layers"] if by_index.get(i) not in hiddens]
    if missing:
        raise ValueError(f"A2 missing prefix blocks: {missing}")
    gamma = ref._extract_final_norm_gamma(model)
    if gamma is None:
        raise ValueError("Final RMSNorm gamma unavailable")
    computer = ref.pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    sources = [("readout", prefix_logits, trace["prefix_probs"][-1])]
    for index in window["layers"]:
        hidden = np.asarray(hiddens[by_index[index]], dtype=np.float32)
        if not np.isfinite(hidden).all():
            raise ValueError("Nonfinite prefix block hidden state")
        logits = projection.project(computer.rmsnorm(hidden, gamma))
        sources.append((f"block_{index}", logits, ref.softmax_np(logits)))
    banks, inline, full, replay = {}, {}, {}, {}
    for name, logits, probabilities in sources:
        banks[name], inline[name], full[name], replay[name] = bank_source(
            logits, probabilities, token, projection, ref, readout=name == "readout")
    for values in (inline, full, replay):
        values["aggregate"] = offline.aggregate(values)
    return dict(support_bank=banks, inline_t1=inline, full_softmax_t1=full,
                offline_t1=replay, pinned_layer_window=window,
                a2_all_prefix_blocks_present=True)


def exact_control(rows, path):
    """A1′ compares all inherited row fields, including rotation diagnostics."""
    if path is None:
        return dict(status="pending", reason="No committed-script smoke reference supplied")
    expected = json.loads(path.read_text())["rows"]
    if len(rows) != len(expected) or not rows:
        raise ValueError("A1-prime reference row count mismatch or empty smoke")
    differences = []

    def compare(a, b, location):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                differences.append(location + ": keys")
            for key in a.keys() & b.keys():
                compare(a[key], b[key], location + "/" + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append(location + ": length")
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, location + f"/{i}")
        elif isinstance(a, float) and isinstance(b, float):
            if a.hex() != b.hex():
                differences.append(location)
        elif type(a) is not type(b) or a != b:
            differences.append(location)

    compare(rows, expected, "rows")
    return dict(status="passed" if not differences else "failed",
                reference_sha256=offline.sha256(path), differences=differences,
                compared_rows=len(rows))


def capture(args):
    ref = offline.reference_module()
    pipeline = ref.pipeline
    manifest = offline.weight_manifest(args.model_path)
    original_load, original_trace = pipeline.load_model, pipeline.trace_sample
    original_softmax = pipeline.safe_softmax
    overlays, failures, context = {}, {}, {}
    counter = 0

    def load_local(model_id, config=None):
        result = offline.load_local_weights(ref, args.model_path, args.model_id,
                                            config, original_load)
        context["model"], context["projection"] = result[0], result[2]
        return result

    def observe_trace(**kwargs):
        nonlocal counter
        sample_idx = counter
        counter += 1
        prefix_count = len(pipeline.encode_text(kwargs["tokenizer"], kwargs["prompt"]))
        calls = 0
        prefix_logits = None

        def observe_softmax(logits):
            nonlocal calls, prefix_logits
            calls += 1
            if calls == prefix_count:
                prefix_logits = np.asarray(logits, dtype=np.float32).copy()
            return original_softmax(logits)

        pipeline.safe_softmax = observe_softmax
        try:
            trace = original_trace(**kwargs)
        finally:
            pipeline.safe_softmax = original_softmax
        try:
            if prefix_logits is None or calls != prefix_count + len(trace["gen_probs"]):
                raise ValueError("Trace softmax call schedule changed")
            if not trace["gen_token_ids"]:
                raise ValueError("No committed token (possibly EOS)")
            overlays[sample_idx] = pass_one(trace, prefix_logits, context["projection"],
                                          context["model"], ref)
            overlays[sample_idx]["wrapped_prompt_sha256"] = hashlib.sha256(
                kwargs["prompt"].encode("utf-8")).hexdigest()
        except Exception as exc:
            # Preserve pass-two reference eligibility independently of overlay
            # failure. Full-run acceptance requires zero unexplained failures.
            failures[sample_idx] = f"{type(exc).__name__}: {exc}"
        return trace

    pipeline.load_model, pipeline.trace_sample = load_local, observe_trace
    try:
        reference = ref.trace_pair_features(
            args.model_id, args.benchmark, args.data, limit=args.limit,
            max_new_tokens=1, k_support=512, seed=args.seed)
    finally:
        pipeline.load_model, pipeline.trace_sample = original_load, original_trace
        pipeline.safe_softmax = original_softmax
    prompts, labels, data_hash = ref._load_calibration_jsonl(str(args.data))
    rows = []
    for inherited in reference.rows:
        i = inherited["sample_idx"]
        if i not in overlays:
            continue
        # null_ratio_post_rank1 remains exactly the inherited pass-two value:
        # it uses h_prev from pass one AND p_t/h_t from pass two, comparator only.
        rows.append(dict(sample_idx=i, label=inherited["label"],
                         prompt_sha256=hashlib.sha256(prompts[i].encode("utf-8")).hexdigest(),
                         pass_two=inherited, **overlays[i]))
    if not rows:
        raise ValueError("No paired usable rows; refuse empty success artifact")
    maximum = offline.discrepancies(rows, "inline_t1", "offline_t1")
    full_maximum = offline.discrepancies(rows, "full_softmax_t1", "offline_t1")
    return dict(schema="rpv-pass1/v1", model_id=args.model_id,
                model_path=str(args.model_path.resolve()), weight_manifest=manifest,
                reference_hashes=offline.reference_hashes(),
                overlay_hashes={p.name: offline.sha256(p) for p in
                                [Path(__file__), offline.HERE / "rpv_pass1_temperature.py",
                                 offline.HERE / "PRE_REGISTRATION_RPV_PASS1.md"]},
                benchmark=args.benchmark, data_sha256=data_hash, seed=args.seed,
                limit=args.limit, max_new_tokens=1, k_support=512,
                temperature_grid=offline.GRID, rows=rows,
                pass_two_rows=reference.rows, pass_two_drops=reference.drops,
                pass_two_diagnostics=reference.diagnostics,
                pass_one_failures=failures,
                pass_one_only_indices=sorted(set(overlays) - {r["sample_idx"] for r in rows}),
                a1_prime=exact_control(reference.rows, args.a1_reference),
                a4_inline_discrepancy=offline.resolution_report(rows, maximum),
                a4_full_softmax_discrepancy=offline.resolution_report(rows, full_maximum),
                confidence_note="p_max and exp(-surprise) equal to within an additive 1e-10 floor; p_max is excluded as a separate comparator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Original model identity for prompt strategy")
    parser.add_argument("--model-path", required=True, type=Path, help="Immutable local model snapshot")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--a1-reference", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative; zero means all rows")
    offline.write_json(args.out, capture(args))


if __name__ == "__main__":
    main()
