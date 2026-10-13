#!/usr/bin/env python3
"""Local CPU adapter for the real torch capture and sealed commit-moment panel."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
import types


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SEAL_DIR = REPO / "modal" / "seal"
TORCH_SOURCE = REPO / "modal" / "modal_app.py"
DATA = REPO / "stage_b" / "data" / "anli_R1_seed20260612_n200.jsonl"
REFERENCE = REPO / "modal" / "refs_upload" / "anli_r1.matrix.npz"
SEED = 20260612
NBOOT = 2000
TRANSFORMERS_VERSION = "5.19.0"

# bf16 reconstruction accumulates in fp32 then rounds before o_proj; tolerate that
# rounding, but require amplitude as well as direction (cosine alone misses gains).
THRESHOLDS = {
    "bf16": {"cos_min": 0.999, "relative_l2_max": 0.02,
             "row_sum_error_max": 0.01, "causal_leak_max": 1e-7,
             "norm_relative_l2_max": 0.01, "logit_relative_l2_max": 0.02},
    "fp32": {"cos_min": 0.99999, "relative_l2_max": 1e-4,
             "row_sum_error_max": 1e-5, "causal_leak_max": 1e-7,
             "norm_relative_l2_max": 1e-5, "logit_relative_l2_max": 1e-4},
}
STUBS = ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "pri_v2_mlx_pipeline")
SEALED_MODULES = (
    "pri_calibrator", "pri_runtime", "comprehensive_run", "confluence_calibrator",
    "pri_v2_io_plugins", "diagnose_inter_head_disagreement", "analyze_adaptive_step",
    "test_shadow_ambiguity",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_torch_definitions(report):
    # Importing modal_app would execute App/Volume/Secret/Image declarations.
    # Compile only these original, undecorated definitions; never execute its module body.
    names = {"_import_seal", "_Capture", "_forward", "NumpyProjection",
             "_output_weight_numpy", "_readout_panel", "_parse_panel_cells", "_metric_of"}
    tree = ast.parse(TORCH_SOURCE.read_text(), filename=str(TORCH_SOURCE))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names or any(node.decorator_list for node in nodes):
        raise RuntimeError("torch source definitions changed; review the local allowlist")
    ro = next(node for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "RO_KEYS" for t in node.targets))
    module = types.ModuleType("local_detector_torch_source")
    module.__file__ = str(TORCH_SOURCE)
    module.SEAL_REMOTE = str(SEAL_DIR)
    module.RO_KEYS = ast.literal_eval(ro.value)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TORCH_SOURCE), "exec"),
         module.__dict__)
    report["provenance"]["torch_definitions"] = sorted(names)
    report["provenance"]["modal_isolation"] = "AST allowlist; no modal import or decorators"
    return module


def load_seal(source, report):
    # Both path controls must name the vendor directory: CC otherwise prepends the sibling repo.
    os.environ["CONFLUENCE_T0_REPO"] = str(SEAL_DIR)
    sys.path.insert(0, str(SEAL_DIR))
    if any(name in sys.modules for name in (*SEALED_MODULES, *STUBS)):
        raise RuntimeError("sealed/MLX modules already imported; run this CLI in a fresh process")
    result = source._import_seal()
    for name in SEALED_MODULES:
        actual = Path(sys.modules[name].__file__).resolve()
        if actual != (SEAL_DIR / f"{name}.py").resolve():
            raise RuntimeError(f"seal provenance mismatch: {name} came from {actual}")
    report["provenance"]["seal_paths"] = {
        "SEAL_REMOTE": source.SEAL_REMOTE,
        "CONFLUENCE_T0_REPO": os.environ["CONFLUENCE_T0_REPO"],
    }
    report["provenance"]["source_sha256"] = {
        str(p): sha256(p) for p in [Path(__file__), TORCH_SOURCE, REFERENCE,
                                  SEAL_DIR / "stage_b" / "fusion_signs.json",
                                  *(SEAL_DIR / f"{name}.py" for name in SEALED_MODULES)]
    }
    return result


def record_imports(report):
    imports = report["provenance"].setdefault("imports", {})
    for name, module in sorted(list(sys.modules.items())):
        path = getattr(module, "__file__", None)
        entry = {"__file__": str(path) if path is not None else None,
                 "stub": name in STUBS}
        if imports.get(name) != entry:
            print(f"IMPORT {name}: __file__={entry['__file__']!r} stub={entry['stub']}", flush=True)
        imports[name] = entry


def load_model(args, report):
    import torch
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoModelForCausalLM, AutoTokenizer

    report["provenance"].update({"transformers_version": transformers.__version__,
                                 "torch_version": torch.__version__})
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"this audited adapter requires Transformers=={TRANSFORMERS_VERSION}; "
                           f"found {transformers.__version__}")
    cached = try_to_load_from_cache(args.model_id, "config.json", revision=args.revision)
    if not isinstance(cached, str):
        raise RuntimeError(f"config.json for {args.model_id}@{args.revision} is not cached; "
                           "finish the download before running")
    # Do not resolve the config symlink: its parent is the snapshot SHA, not the blob directory.
    snapshot = Path(cached).absolute().parent
    if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
        raise RuntimeError(f"cannot establish a resolved snapshot SHA from {cached}")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    report["provenance"].update({
        "model_id": args.model_id, "requested_revision": args.revision,
        "model_snapshot_sha": snapshot.name, "tokenizer_snapshot_sha": snapshot.name,
        "snapshot_path": str(snapshot), "config_sha256": sha256(cached),
        "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "dtype": str(dtype), "device": "cpu", "attention_implementation": "eager",
        "python_version": sys.version, "local_files_only": True,
    })
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    # Explicit placement also avoids automatic offload/meta parameters on a 16-GB machine.
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False,
        torch_dtype=dtype, device_map="cpu", attn_implementation="eager")
    model.eval()
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.device.type != "cpu":
            raise RuntimeError(f"single-device CPU load failed: {name} is on {tensor.device}")
    actual_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    report["provenance"]["parameter_dtypes"] = actual_dtypes
    if actual_dtypes != [str(dtype)]:
        raise RuntimeError(f"requested {dtype}, loaded parameter dtypes {actual_dtypes}")
    return tok, model


def assert_structure(model, target_layer_map):
    try:
        layers = model.model.layers
        tags = target_layer_map(len(layers))
        if len(set(tags.values())) != 3:
            raise ValueError("three distinct target layers required")
        for idx in tags.values():
            attn = layers[idx].self_attn
            for name in ("v_proj", "o_proj"):
                proj = getattr(attn, name)
                if not callable(proj) or proj.weight.ndim != 2:
                    raise ValueError(f"layer {idx} self_attn.{name} must be a dense linear module")
        norm = model.model.norm
        if "rmsnorm" not in type(norm).__name__.lower() or norm.weight.ndim != 1:
            raise ValueError("model.model.norm must expose multiplicative RMSNorm weights")
        eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
        if eps is None or not math.isclose(float(eps), 1e-6, rel_tol=1e-9):
            raise ValueError(f"native norm epsilon={eps}; sealed PRI requires 1e-6")
        if getattr(norm, "bias", None) is not None:
            raise ValueError("sealed PRI does not implement final-norm bias")
        if getattr(model.get_output_embeddings(), "bias", None) is not None:
            raise ValueError("sealed projection does not implement output bias")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "structural-incompatibility: requires model.model.layers[i].self_attn.v_proj, "
            "self_attn.o_proj on all three targets, final self_attn, and model.model.norm; "
            f"{exc}") from exc
    return tags


def prompt_ids(tok, prompt, model_type):
    if not tok.chat_template:
        raise RuntimeError("tokenizer.chat_template is required by the torch detector prompt policy")
    kwargs = {"enable_thinking": False} if model_type == "qwen3" else {}
    # tokenize=True avoids adding special tokens a second time to an already-rendered template.
    # Transformers 5 returns a BatchEncoding unless return_dict=False.
    return [int(t) for t in tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, return_dict=False,
        add_generation_prompt=True, **kwargs)]


def gap(actual, expected):
    import numpy as np

    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("non-finite or shape-incompatible reconstruction/parity vectors")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"cosine": float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0,
            "relative_l2": float(np.linalg.norm(a - b) / max(nb, 1e-12)),
            "max_absolute": float(np.max(np.abs(a - b))),
            "rms": float(np.sqrt(np.mean((a - b) ** 2)))}


class ValidationProbe:
    """Observe native outputs without replacing the real _Capture or _forward."""

    def __init__(self, model, tags):
        self.handles = []
        self.native = {}
        self.probabilities = {}
        self.norm = None
        self.tags = tags
        for tag, idx in tags.items():
            def capture(module, inputs, output, tag=tag):
                value = output[0] if isinstance(output, tuple) else output
                self.native[tag] = value[0, -1].detach().float().cpu().numpy().copy()
            self.handles.append(model.model.layers[idx].self_attn.register_forward_hook(capture))

        def capture_norm(module, inputs, output):
            self.norm = output[0, -1].detach().float().cpu().numpy().copy()

        def capture_probabilities(module, inputs, output):
            import torch

            if output.attentions is None:
                raise RuntimeError("native attention probabilities unavailable; eager attention required")
            for tag, idx in tags.items():
                attn = output.attentions[idx]
                if attn is None or attn.ndim != 4 or attn.shape[0] != 1:
                    raise RuntimeError(f"invalid native attention shape at {tag}")
                # Inspect all query rows: the final query alone cannot reveal future-token leakage.
                a = attn[0].detach().float()
                self.probabilities[tag] = {
                    "finite": bool(torch.isfinite(a).all()),
                    "minimum": float(a.min()),
                    "row_sum_error_max": float((a.sum(-1) - 1).abs().max()),
                    "causal_leak_max": float(torch.triu(a, diagonal=1).abs().max()),
                    "shape": list(attn.shape),
                }
        self.handles.append(model.model.norm.register_forward_hook(capture_norm))
        self.handles.append(model.register_forward_hook(capture_probabilities))

    def reset(self):
        self.native.clear()
        self.probabilities.clear()
        self.norm = None

    def remove(self):
        for handle in self.handles:
            handle.remove()


def checked_forward(source, model, ids, cap, probe, tags, pri, scale, thresholds, record):
    import numpy as np
    import torch

    probe.reset()
    result = source._forward(model, ids, cap, tags)
    logits, caps, _, _, h = result
    record["layers"] = {}
    errors = []
    with torch.no_grad():
        for tag, idx in tags.items():
            stats = dict(probe.probabilities[tag])
            record["layers"][tag] = {"index": idx, **stats}
            w = torch.as_tensor(caps[tag][0], dtype=torch.float32, device="cpu")
            v = torch.as_tensor(cap.values[tag], dtype=torch.float32, device="cpu")
            if w.shape[0] % v.shape[0] or w.shape[1] != v.shape[1]:
                raise RuntimeError(f"GQA reconstruction shape mismatch at {tag}")
            # ctx[h,d] = sum_t A[h,t] V[KV(h),t,d]; o_proj tests the native contribution.
            repeated = v.repeat_interleave(w.shape[0] // v.shape[0], dim=0)
            o_proj = model.model.layers[idx].self_attn.o_proj
            ctx = torch.einsum("ht,htd->hd", w, repeated).reshape(1, -1).to(o_proj.weight.dtype)
            reconstructed = o_proj(ctx)[0].detach().float().cpu().numpy()
            g = gap(reconstructed, probe.native[tag])
            record["layers"][tag]["reconstruction"] = g
            ok = (stats["finite"] and stats["minimum"] >= 0
                  and stats["row_sum_error_max"] <= thresholds["row_sum_error_max"]
                  and stats["causal_leak_max"] <= thresholds["causal_leak_max"]
                  and g["cosine"] >= thresholds["cos_min"]
                  and g["relative_l2"] <= thresholds["relative_l2_max"])
            record["layers"][tag]["pass"] = bool(ok)
            if not ok:
                errors.append(f"attention/reconstruction {tag}")
    detector_norm = pri.rmsnorm(h, pri.final_norm_gamma)
    detector_logits = pri.output_projection.project(detector_norm)
    norm_gap = gap(detector_norm, probe.norm)
    logit_gap = gap(detector_logits, logits)
    record["parity"] = {
        "formula": "(W_u / logits_scaling) @ sealed_RMSNorm(h, gamma, eps=1e-6)",
        "norm": norm_gap, "logits": logit_gap,
        "argmax_matches": bool(np.argmax(detector_logits) == np.argmax(logits)),
        "logits_scaling": scale,
    }
    if scale != 1:
        record["parity"]["uncorrected_bare_W_u_logits"] = gap(detector_logits * scale, logits)
    parity_ok = (norm_gap["relative_l2"] <= thresholds["norm_relative_l2_max"]
                 and logit_gap["relative_l2"] <= thresholds["logit_relative_l2_max"]
                 and logit_gap["cosine"] >= thresholds["cos_min"])
    record["parity"]["pass"] = bool(parity_ok)
    if not parity_ok:
        errors.append("instrument deviation: native final norm/logit parity")
    record["pass"] = not errors
    if errors:
        raise RuntimeError("validation failed: " + "; ".join(errors))
    return result


def probabilities(logits):
    import numpy as np

    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite native logits")
    p = np.exp(logits - logits.max())
    p /= p.sum()
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1, atol=1e-12):
        raise RuntimeError("invalid native logit probabilities")
    return p


def require_scores(values, cells, kind, row, report):
    import numpy as np

    bad = []
    for cell, value in zip(cells, values):
        reason = "unavailable" if value is None else "non_finite" if not np.isfinite(value) else None
        if reason:
            bad.append({"sample_idx": row, "kind": kind, "cell": str(cell), "reason": reason,
                        "value": repr(value),
                        "sealed_may_have_masked_nonfinite": kind == "ACE" and value is None})
    if bad:
        report["score_failures"].extend(bad)
        for reason in ("unavailable", "non_finite"):
            count = sum(item["reason"] == reason for item in bad)
            report["counts"][f"{reason}_cells"] += count
            report["counts"][f"{reason}_rows"] += int(count > 0)
        raise RuntimeError(f"sample {row}: invalid {kind} scores: {bad}; calibration forbidden")
    return [float(value) for value in values]


def run(args, report, out):
    import numpy as np

    source = load_torch_definitions(report)
    tok, model = load_model(args, report)
    SEAL, PIPE, CR, CC, _, target_layer_map = load_seal(source, report)
    record_imports(report)
    tags = assert_structure(model, target_layer_map)
    report["target_layers"] = tags
    ace_panel = list(SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS)
    readout_panel = source._parse_panel_cells(source._readout_panel(str(REFERENCE), len(ace_panel)))
    if len(readout_panel) != len(CC.READOUT_PANEL) or set(readout_panel) != set(CC.READOUT_PANEL):
        raise RuntimeError("reference readout panel does not match the sealed six readout cells")
    if any(source._metric_of(cell) is None for cell in readout_panel):
        raise RuntimeError("unmapped readout cell in reference")
    panel = ace_panel + readout_panel
    report["panel"] = [str(cell) for cell in panel]
    prompts, labels, data_hash = CR._load_calibration_jsonl(str(DATA))
    if len(prompts) != 200:
        raise RuntimeError(f"expected the fixed 200-row ANLI dataset, found {len(prompts)}")
    prompts, labels = prompts[:args.n], labels[:args.n]
    report["data"] = {"path": str(DATA), "file_sha256": sha256(DATA),
                      "sealed_full_data_hash": data_hash, "selected_indices": list(range(args.n)),
                      "label_meanings": {"0": "YES expected (entailed)", "1": "NO expected (contradiction)"}}
    selected_hash = hashlib.sha256("".join(f"{int(y)}\t{p}\n" for p, y in zip(prompts, labels)).encode()).hexdigest()
    report["data"]["selected_data_hash"] = selected_hash
    if any(int((labels == y).sum()) < 2 for y in (0, 1)):
        raise RuntimeError("calibration requires at least two selected rows of each label; increase --n")
    report["prompt_policy"] = {"chat_template": tok.chat_template, "tokenize": True,
                               "add_generation_prompt": True, "truncation": False,
                               "enable_thinking": False if model.config.model_type == "qwen3" else None,
                               "batch_size": 1, "padding": False, "use_cache": False}
    encoded = []
    context_limit = int(model.config.max_position_embeddings)
    for i, prompt in enumerate(prompts):
        ids = prompt_ids(tok, prompt, model.config.model_type)
        encoded.append(ids)
        if len(ids) < 2 or len(ids) > args.max_prompt_tokens or len(ids) + 1 > context_limit:
            report["rejections"].append({"sample_idx": i, "reason": "prompt_length",
                                         "tokens": len(ids), "context_limit": context_limit})
    report["prompt_token_counts"] = [len(ids) for ids in encoded]
    if report["rejections"]:
        report["counts"]["rejected"] = len(report["rejections"])
        raise RuntimeError("prompt-length rejection(s); no truncation or subset calibration; see report")

    gamma = model.model.norm.weight.detach().float().cpu().numpy()
    W_u = source._output_weight_numpy(model)
    scale = float(getattr(model.config, "logits_scaling", 1.0))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"invalid logits_scaling={scale}")
    if model.config.model_type != "granitemoe" and scale != 1:
        raise RuntimeError("instrument deviation: unreviewed architecture-specific logits scaling")
    # Native Granite uses logits/6: W_eff=W_u/6 also scales Fisher eigenvalues by 1/36.
    # Correct the adapter projection for ALL readout kernels, not merely the parity display.
    if scale != 1:
        # In fp32 the exported NumPy array may share storage with model weights.
        W_u = W_u / scale
    proj = source.NumpyProjection(W_u)
    pri = PIPE.PRIComputer(proj, final_norm_gamma=gamma)
    report["readout_instrument"] = {"norm_epsilon": 1e-6, "logits_scaling": scale,
                                    "effective_projection": "W_u / logits_scaling",
                                    "changed_from_modal_bare_W_u": scale != 1,
                                    "byte_comparable_to_mlx_seal": False}
    cap = source._Capture(model, tags, len(model.model.layers) - 1)
    probe = ValidationProbe(model, tags)
    rows, keep = [], []
    try:
        for i in (0, 1):
            record = {"sample_idx": i, "phase": "D0"}
            report["validation"].append(record)
            result = checked_forward(source, model, encoded[i], cap, probe, tags, pri, scale,
                                     report["thresholds"], record)
            gid = int(np.argmax(result[0]))
            commit = tok.decode([gid], skip_special_tokens=True).strip().upper()
            record["commit_token"] = commit
            record["commit_is_yes_no"] = commit in ("YES", "NO")
            record_b = {"sample_idx": i, "phase": "D1"}
            report["validation"].append(record_b)
            checked_forward(source, model, encoded[i] + [gid], cap, probe, tags, pri, scale,
                            report["thresholds"], record_b)
        if not all(r.get("commit_is_yes_no", True) for r in report["validation"]):
            raise RuntimeError("validation failed: first two D0 argmax tokens must be YES/NO")
        report["validation_pass"] = True
        print("VALIDATION PASS: three layers, D0/D1, two examples; starting extraction", flush=True)

        for i, ids in enumerate(encoded):
            report["counts"]["attempted"] += 1
            try:
                d0 = {"sample_idx": i, "phase": "D0"}
                report["extraction_checks"].append(d0)
                logits_a, caps, vcaps, nkv, h_prev = checked_forward(
                    source, model, ids, cap, probe, tags, pri, scale, report["thresholds"], d0)
                p_a = probabilities(logits_a)
                gid = int(np.argmax(p_a))
                surprise = float(-np.log(p_a[gid] + 1e-300))
                commit = tok.decode([gid], skip_special_tokens=True).strip().upper()
                d0.update({"commit_token_id": gid, "commit_token": commit,
                           "expected": "YES" if int(labels[i]) == 0 else "NO"})
                ace = require_scores(
                    [SEAL._compute_attention_score(c, caps, nkv, v_norm_captures=vcaps) for c in ace_panel],
                    ace_panel, "ACE", i, report)
                d1 = {"sample_idx": i, "phase": "D1"}
                report["extraction_checks"].append(d1)
                logits_b, _, _, _, h_t = checked_forward(
                    source, model, ids + [gid], cap, probe, tags, pri, scale, report["thresholds"], d1)
                p_b = probabilities(logits_b)
                comp = pri.compute_step(
                    h_t=h_t, h_prev=h_prev, p_t=p_b, S_t=surprise, alpha=1.0,
                    topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                    v3_capture_raw=False, v3_capture_centered=False)
                spec, _, _ = CR._support_spectrum(proj, p_b, int(proj.hidden_size), CR.K_SUPPORT_DEFAULT)
                st = CR._spectrum_stats(spec)
                values = {"null_ratio_post_rank1": comp.get("null_ratio_post_rank1"),
                          "fisher_eff_rank": st.get("fisher_eff_rank"),
                          "spectral_entropy": st.get("spectral_entropy"),
                          "neg_shadow_logvol_r1": st.get("neg_shadow_logvol_r1"),
                          "surprise": surprise, "p_max": float(p_b.max())}
                ro = require_scores([values[source._metric_of(c)] for c in readout_panel],
                                    readout_panel, "readout", i, report)
                rows.append(ace + ro)
                keep.append(i)
                report["counts"]["extracted"] = len(keep)
                report["counts"]["yes_no"] += int(commit in ("YES", "NO"))
            except Exception as exc:
                report["counts"]["rejected"] += 1
                report["rejections"].append({"sample_idx": i, "reason": str(exc)})
                raise
            print(f"EXTRACT {i + 1}/{args.n}", flush=True)
    finally:
        probe.remove()
        cap.remove()
        # On failure only completed finite rows are saved, explicitly marked partial; never calibrated.
        matrix = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(panel))
        np.savez(out / "matrix.npz", score_matrix=matrix, labels=labels[keep],
                 sample_idx=np.asarray(keep, dtype=np.int64),
                 panel=json.dumps([str(c) for c in panel]),
                 meta=json.dumps({"complete": len(keep) == args.n, "planned_n": args.n,
                                  "model_snapshot_sha": report["provenance"]["model_snapshot_sha"],
                                  "backend": "local-torch-cpu", "precision": args.precision,
                                  "comparable": False, "data_hash": selected_hash}))
        report["matrix_path"] = str(out / "matrix.npz")
    if len(keep) != args.n or not np.isfinite(matrix).all():
        raise RuntimeError("incomplete/non-finite matrix; calibration forbidden")
    report["yes_no_commit_rate"] = report["counts"]["yes_no"] / args.n
    if report["yes_no_commit_rate"] < 0.5:
        raise RuntimeError("YES/NO commit rate below sealed 50% task-attempt gate; calibration forbidden")
    common = {"sample_idx": np.asarray(keep), "labels": labels, "data_hash": selected_hash}
    mm = CC.merge_matrices(
        {**common, "score_matrix": matrix[:, :len(ace_panel)], "panel": ace_panel,
         "slug": args.model_id.split("/")[-1]},
        {**common, "score_matrix": matrix[:, len(ace_panel):], "panel": readout_panel}, max_dropped=0)
    print("CALIBRATION: sealed selector, 2000 bootstraps, fusion and shuffled-label controls", flush=True)
    profile = CC.calibrate_merged(mm, n_bootstrap=NBOOT, seed=SEED,
                                  model_id=args.model_id, benchmark=args.task)
    # The sealed helper looks up refs/main, which need not be the requested revision.
    profile["provenance"]["sealed_helper_snapshot_sha"] = profile["provenance"].get("model_snapshot_sha")
    profile["provenance"]["model_snapshot_sha"] = report["provenance"]["model_snapshot_sha"]
    report["calibration"] = profile
    for endpoint in ("primary_full_panel", "secondary_geometric_only"):
        ep = profile[endpoint]
        required = [ep.get(k) for k in ("oob_auroc_median", "oob_auroc_ci_lo", "oob_auroc_ci_hi")]
        if any(v is None or not np.isfinite(v) for v in required):
            raise RuntimeError(f"sealed selector did not return a finite AUROC interval: {endpoint}")
    report["status"] = "complete"


def json_safe(value):
    # Sealed diagnostics may use NaN for undefined optional statistics; keep strict JSON.
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="main", help="cached HF revision; resolved SHA is recorded")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--task", choices=["anli_r1"], default="anli_r1")
    parser.add_argument("--n", type=int, default=200, help="first N rows of the fixed 200-row dataset")
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--out", type=Path, help="fresh output directory inside this folder's runs/")
    args = parser.parse_args()
    if not 4 <= args.n <= 200 or args.max_prompt_tokens < 2:
        parser.error("require 4 <= --n <= 200 and --max-prompt-tokens >= 2")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_id.split("/")[-1])
    runs = (HERE / "runs").resolve()
    out = (args.out or runs / slug).expanduser().resolve()
    if not runs.is_relative_to(HERE):
        parser.error("runs/ must remain inside the adapter directory (no external symlink)")
    if not out.is_relative_to(runs) or out == runs:
        parser.error("--out must be a model/run directory below " + str(runs))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"refusing to overwrite existing artifacts in {out}; use a fresh --out")
    out.mkdir(parents=True, exist_ok=True)
    # This process only reads cached snapshots and leaves source directories untouched.
    sys.dont_write_bytecode = True
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0", "USE_FLAX": "0"})
    start = time.perf_counter()
    report = {"status": "running", "verification": "not run by Codex",
              "provenance": {}, "thresholds": THRESHOLDS[args.precision],
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "counts": {"planned": args.n, "attempted": 0, "extracted": 0, "rejected": 0,
                         "non_finite_rows": 0, "unavailable_rows": 0, "non_finite_cells": 0,
                         "unavailable_cells": 0, "yes_no": 0},
              "validation": [], "validation_pass": False, "extraction_checks": [],
              "score_failures": [], "rejections": [], "calibration": None}
    exit_code = 0
    try:
        run(args, report, out)
    except (Exception, KeyboardInterrupt) as exc:
        exit_code = 1
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        report["wall_seconds"] = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        report["counts"]["not_extracted"] = args.n - report["counts"]["extracted"]
        report["counts"]["not_attempted"] = args.n - report["counts"]["attempted"]
        record_imports(report)
        (out / "report.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
        print(f"{report['status'].upper()}: {out / 'report.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
