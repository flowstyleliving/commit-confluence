"""Independent, label-blind Modal pilot. Verification: not run by Codex.

Only smoke/extract allocate GPU via a budget-reserving CPU coordinator. No secrets.
"""
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
import uuid

import modal

from make_freeze import (HERE, ROOT, TASKS, DATA_SHA, GateFailure, atomic, check_freeze,
                         gate, is_gate_failure, json_bytes, require, sha, utc)

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"
REVISION = "7f1c97f440f06ce36705e4f2b843edb5925f4498"
MOUNT = Path("/cache")
OUT = MOUNT / "router-pilot"
APP_NAME = "router-pilot"
VOLUME = "router-pilot-cache"
CAP, RESERVE, SHUTDOWN = 7200, 600, 30
# final_v_norm_bos is deprecated: V at position 0 depends only on the shared first token.
ATT = ("final_js", "final_js_no_bos", "final_js_kv_groups", "final_bos_mass",
       "final_v_norm_max", "final_v_norm_lastq_weighted")
# Descriptive replacement: how much the final position draws from the position-0 sink.
SINK = "sink_draw"
SINK_DEFINITION = "mean over heads h of A_h[T-1, 0] * ||V_h[0]|| (MHA: query head h uses KV head h)"
ROUTER = ("router_entropy_norm", "router_margin", "router_top8_mass_precast")
OUTPUT = ("out_p_max", "out_entropy", "out_yesno_logodds")
PINS = ("transformers==4.57.6", "huggingface_hub==0.36.0", "accelerate==1.11.0",
        "safetensors==0.6.2", "numpy==2.2.6", "tokenizers==0.22.1",
        "pandas==2.3.3", "scikit-learn==1.7.2", "scipy==1.15.3")
TORCH = "torch==2.8.0+cu126"
RECIPE = {"base": "debian_slim", "python": "3.12", "torch": TORCH,
          "torch_index": "https://download.pytorch.org/whl/cu126", "pins": PINS}
RECIPE_SHA = hashlib.sha256(json_bytes(RECIPE)).hexdigest()
app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
claims = modal.Dict.from_name("router-pilot-budget-claim", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(TORCH, index_url=RECIPE["torch_index"])
         .pip_install(*PINS)
         .env({"HF_HOME": "/cache/hub", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": "/payload/exploratory/router-pilot",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
         .add_local_dir(str(ROOT / "modal/seal"), "/payload/modal/seal", copy=True,
                        ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
         .add_local_dir(str(ROOT / "exploratory/depth-curve/data"),
                        "/payload/exploratory/depth-curve/data", copy=True))
for filename in ("PRE_REGISTRATION.md", "modal_router_pilot.py", "score_router_pilot.py", "make_freeze.py"):
    image = image.add_local_file(str(HERE / filename),
                                f"/payload/exploratory/router-pilot/{filename}", copy=True)
for filename in ("stem_grouped_sensitivity.py", "rescore_grid_a.py", "depth_score.py", "score_grid_b.py"):
    image = image.add_local_file(str(ROOT / "exploratory/depth-curve" / filename),
                                f"/payload/exploratory/depth-curve/{filename}", copy=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def publish_json(path, value):
    # Coordinator claim serializes state writes; ledger paths are UUID-unique.
    # No per-file remote lock can be stranded by a killed coordinator.
    atomic(path, json_bytes(value), serialized=True)


def label_blind_rows(task):
    """Hash the full frozen file; decode ONLY prompt and source_idx fields, never labels."""
    path = ROOT / "exploratory/depth-curve/data" / f"{task}_n200.jsonl"
    gate(sha(path) == DATA_SHA[task], "data hash mismatch")
    decoder = json.JSONDecoder()
    prompts, keys = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.match(r'\s*\{\s*"prompt"\s*:\s*', line)
        require(match is not None, "frozen prompt field layout changed")
        offset = match.end()
        prompt, end = decoder.raw_decode(line[offset:])
        require(isinstance(prompt, str), "prompt type")
        prompts.append(prompt)
        if task == "anli_r1":
            match = re.search(r"Premise: (.*?)\nHypothesis:", prompt, re.S)
            require(match is not None, "missing premise")
            keys.append(match.group(1))
        else:
            # Restrict search to text after the prompt string: source metadata only.
            tail = line[offset + end:]
            match = re.search(r'"source_idx"\s*:\s*(\d+)', tail)
            require(match is not None, "missing source_idx")
            keys.append(match.group(1))
    from collections import Counter
    sizes = sorted(Counter(keys).values())
    gate(len(prompts) == 200, "row count")
    gate(sizes == ([1] * 190 + [2] * 5 if task == "anli_r1" else [2] * 100),
         "stem multiplicities")
    return prompts


def manifests_and_tokens():
    """Runs in CPU coordinator before GPU dispatch; validates all 400 prompts."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=False,
                                       local_files_only=True)
    gate(tok.bos_token_id == tok.eos_token_id == 50279, "BOS/EOS mismatch")
    sets = {word: [i for i in range(len(tok))
                   if tok.decode([i], skip_special_tokens=True).strip().upper() == word]
            for word in ("YES", "NO")}
    gate(all(sets.values()), "empty YES/NO token set")
    tokens, manifests = {}, {}
    for task in TASKS:
        rows, tokens[task] = [], []
        for i, prompt in enumerate(label_blind_rows(task)):
            messages = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            native = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            gate(ids == native, "chat serialization mismatch")
            gate(2 <= len(ids) <= 900 and ids[0] == 50279 and ids.count(50279) == 1,
                 "length/BOS gate")
            gate(text.count(prompt) == 1, "ambiguous rendered prompt suffix")
            suffix = text.split(prompt, 1)[1]
            rows.append({"row": i, "length": len(ids),
                         "token_sha256": hashlib.sha256(json_bytes(ids)).hexdigest(),
                         "rendered_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "suffix_sha256": hashlib.sha256(suffix.encode()).hexdigest()})
            tokens[task].append(ids)
        manifests[task] = {"schema": "router-pilot-manifest/1", "task": task,
                           "model": MODEL, "revision": REVISION, "data_sha256": DATA_SHA[task],
                           "token_hash_encoding": "json_bytes(list[int])",
                           "yes_no_token_ids": sets, "rows": rows}
    return manifests, tokens


def import_seal():
    """Same import-only MLX shim as the depth lane; sealed Python sources unchanged."""
    import importlib.machinery
    import types
    sys.path.insert(0, str(ROOT / "modal/seal"))
    for name in ("mlx", "mlx.core", "mlx.nn", "mlx_lm", "pri_v2_mlx_pipeline"):
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        sys.modules[name] = module
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.modules["mlx"].nn = sys.modules["mlx.nn"]
    def unavailable(*args, **kwargs):
        raise RuntimeError("MLX/runtime/scoring entry point forbidden in extractor")
    sys.modules["mlx_lm"].load = unavailable
    sys.modules["mlx_lm"].generate = unavailable
    sys.modules["pri_v2_mlx_pipeline"].safe_auroc = unavailable
    import pri_calibrator as SEAL
    return SEAL


def check_time(deadline):
    gate(time.time() < deadline, "budget deadline reached")


def tensor_gate(t, shape, dtype):
    import torch
    require(tuple(t.shape) == tuple(shape), "capture tensor shape")
    gate(t.dtype == dtype and t.device.type == "cuda" and t.device.index == 0,
         "capture dtype/device")
    gate(bool(torch.isfinite(t).all()), "non-finite tensor")


def compare(a, b):
    import torch
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    denom = torch.linalg.vector_norm(b)
    gate(float(denom) > 0, "zero reconstruction reference")
    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    rel = float(torch.linalg.vector_norm(a - b) / denom)
    gate(math.isfinite(cos) and math.isfinite(rel) and cos >= .999 and rel <= .01,
         f"reconstruction cosine/relative-L2 gate (cos {cos:.6f}, rel {rel:.6f})")
    return {"cosine": cos, "relative_l2": rel}


def capture(model, SEAL, ids, sets, reconstruct=False, router_on=True):
    """One forward; hook errors propagate. Remove hooks before manual projections."""
    import numpy as np
    import torch
    T = len(ids)
    captures, counts, handles = {}, {}, []
    def hook(li, kind):
        def record(module, inputs, output):
            key = (li, kind)
            counts[key] = counts.get(key, 0) + 1
            gate(counts[key] == 1, "hook fired more than once")
            if kind == "v":
                tensor_gate(output, (1, T, 2048), torch.bfloat16)
                norms = torch.linalg.vector_norm(output.detach().float().reshape(T, 16, 128), dim=-1)
                captures[key] = norms.T.contiguous().cpu().numpy()
                if reconstruct:
                    captures[(li, "values")] = output.detach().clone()
            elif kind == "o":
                tensor_gate(output, (1, T, 2048), torch.bfloat16)
                if reconstruct:
                    captures[key] = output[0, -1].detach().clone()
            elif kind == "attn":
                tensor_gate(output[0], (1, T, 2048), torch.bfloat16)
                tensor_gate(output[1], (1, 16, T, T), torch.bfloat16)
            else:
                tensor_gate(inputs[0], (1, T, 2048), torch.bfloat16)
                tensor_gate(output[0], (1, T, 2048), torch.bfloat16)
                tensor_gate(output[1], (T, 64), torch.bfloat16)
                captures[(li, "gate_last")] = output[1][-1].detach().clone()
                if reconstruct:
                    captures[(li, "mlp_input")] = inputs[0][0, -1:].detach().clone()
                    captures[(li, "mlp_output")] = output[0][0, -1].detach().clone()
        return record
    start = time.perf_counter()
    try:
        for li, block in enumerate(model.model.layers):
            for module, kind in ((block.self_attn.v_proj, "v"), (block.self_attn.o_proj, "o"),
                                 (block.self_attn, "attn"), (block.mlp, "mlp")):
                handles.append(module.register_forward_hook(hook(li, kind)))
        gate(len(handles) == 64, "module coverage")
        with torch.no_grad():
            out = model(torch.tensor([ids], device="cuda:0"), use_cache=False,
                        output_attentions=True, output_router_logits=router_on, return_dict=True)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    gate(len(counts) == 64 and all(c == 1 for c in counts.values()), "hook coverage")
    gate(len(out.attentions) == 16, "attention coverage")
    if router_on:
        gate(len(out.router_logits) == 16, "router coverage")
    logits = out.logits[0, -1].float()
    gate(bool(torch.isfinite(logits).all()), "non-finite output logits")
    if not router_on:
        return {"logits": logits.cpu().numpy()}
    att, sink, router, expert_ids, weights, gates = [], [], [], [], [], []
    with torch.no_grad():
        for li, block in enumerate(model.model.layers):
            w = out.attentions[li][0, :, -1, :]
            w_last = w.float().cpu().numpy()
            vn = captures[(li, "v")]
            require(vn.shape == (16, T), "v-norm capture shape")
            gate(np.isfinite(vn).all(), "non-finite v-norm capture")
            values = []
            for metric in ATT:
                score = SEAL._compute_attention_score((0, "Attention", metric),
                    {"final": [w_last]}, {"final": 16},
                    v_norm_captures={"final": [vn]})
                gate(score is not None and np.isfinite(score), f"sealed score undefined: {metric} block {li}")
                values.append(float(score))
            att.append(values)
            require(w_last.shape == vn.shape == (16, T), "sink-draw head alignment")
            sink.append(float(np.mean(w_last[:, 0].astype(np.float64) * vn[:, 0].astype(np.float64))))
            tensor_gate(out.router_logits[li], (T, 64), torch.bfloat16)
            raw = out.router_logits[li][T - 1]
            gate(torch.equal(raw, captures[(li, "gate_last")]), "router return/capture mismatch")
            p = torch.softmax(raw, dim=-1, dtype=torch.float32)
            topw, topids = torch.topk(p, 8, dim=-1)
            sorted_p = p.sort(descending=True).values
            router.append([float(-torch.special.xlogy(p, p).sum() / math.log(64)),
                           float(sorted_p[7] - sorted_p[8]), float(topw.sum())])
            expert_ids.append(topids.cpu().numpy())
            weights.append(topw.cpu().numpy())
            if reconstruct:
                v = captures[(li, "values")].view(1, T, 16, 128).transpose(1, 2)
                context = torch.matmul(w[None, :, None, :], v).transpose(1, 2).reshape(1, 2048)
                o_gate = compare(block.self_attn.o_proj(context), captures[(li, "o")])
                x = captures[(li, "mlp_input")]
                applied = topw.to(x.dtype)
                mine = torch.zeros_like(x)
                # Exact 4.57.6 order: expert index ascending, bf16 multiply and index_add_.
                for ei in range(64):
                    positions = torch.where(topids == ei)[0]
                    if positions.numel():
                        contribution = block.mlp.experts[ei](x) * applied[positions[0]]
                        mine.index_add_(0, torch.zeros(1, dtype=torch.long, device=x.device),
                                        contribution.to(x.dtype))
                gates.append({"block": li, "o_proj": o_gate,
                              "moe": compare(mine, captures[(li, "mlp_output")])})
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    output = [float(p.max()), float(-(p * logp).sum()),
              float(torch.logsumexp(logp[sets["YES"]], 0) - torch.logsumexp(logp[sets["NO"]], 0))]
    argmax = int(logits.argmax())
    result = {"attention": np.asarray(att, dtype=np.float64),
              "sink_draw": np.asarray(sink, dtype=np.float64),
              "router": np.asarray(router, dtype=np.float64), "output": np.asarray(output),
              "expert_ids": np.asarray(expert_ids), "expert_weights": np.asarray(weights),
              "prompt_len": T, "argmax_id": argmax, "yes_no": argmax in sets["YES"] + sets["NO"],
              "logits": logits.cpu().numpy(), "gates": gates,
              "seconds": time.perf_counter() - start}
    for key in ("attention", "sink_draw", "router", "output", "expert_weights"):
        gate(np.isfinite(result[key]).all(), "non-finite feature")
    return result


def load_model():
    import torch
    from collections import Counter
    from transformers import OlmoeForCausalLM
    gate(torch.cuda.device_count() == 1, "one GPU required")
    device = torch.cuda.get_device_properties(0)
    gate("A10G" in device.name and 22 * 2**30 <= device.total_memory <= 25 * 2**30,
         "A10G 24GB required")
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # Weights were downloaded by the CPU coordinator; no GPU-billed download.
    model = OlmoeForCausalLM.from_pretrained(MODEL, revision=REVISION, token=False,
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map={"": "cuda:0"})
    model.eval()
    c = model.config
    gate((c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.hidden_size,
          c.num_experts, c.num_experts_per_tok) == (16, 16, 16, 2048, 64, 8), "architecture")
    gate(c.clip_qkv is None and c.norm_topk_prob is False, "clip/topk normalization")
    gate(c._attn_implementation == "eager" and not getattr(c, "quantization_config", None),
         "attention/quantization config")
    gate(len(model.model.layers) == 16 and not model.training, "layer count/eval")
    hist = Counter(str(p.dtype) for p in model.parameters())
    for p in model.parameters():
        gate(p.device.type == "cuda" and p.device.index == 0 and p.dtype == torch.bfloat16,
             "native bf16 residency")
    for b in model.buffers():
        gate(b.device.type == "cuda" and b.device.index == 0, "buffer residency")
    for block in model.model.layers:
        a, m = block.self_attn, block.mlp
        gate(a.head_dim == c.hidden_size // 16 == a.v_proj.out_features // 16 == 128,
             "head dimension")
        gate(a.num_heads == a.num_key_value_heads == 16 and a.attention_dropout == 0,
             "attention heads/dropout")
        gate(m.num_experts == len(m.experts) == 64 and m.top_k == 8 and not m.norm_topk_prob,
             "MoE dimensions")
    versions = {name.split("==")[0]: importlib.metadata.version(name.split("==")[0])
                for name in (TORCH,) + PINS}
    gate(all(versions[s.split("==")[0]] == s.split("==")[1] for s in (TORCH,) + PINS),
         "installed package drift")
    gate(sys.version_info[:2] == (3, 12), "Python version")
    return model, {"gpu": device.name, "gpu_bytes": device.total_memory, "versions": versions,
                   "python": sys.version, "parameter_dtype_histogram": dict(hist),
                   "image_recipe_sha256": RECIPE_SHA, "image_recipe": RECIPE,
                   "cuda": torch.version.cuda}


@app.cls(image=image, gpu="A10G", volumes={"/cache": vol}, timeout=6600, startup_timeout=120,
         retries=0, max_containers=1, scaledown_window=2)
class GPUWorker:
    @modal.method()
    def run(self, kind, task, tokens, manifests, freeze, deadline, invocation, image_id):
        try:
            return self._run(kind, task, tokens, manifests, freeze, deadline, invocation, image_id)
        except BaseException as exc:
            if not is_gate_failure(exc):
                raise
            # Ordinary return data survives Modal's exception-deserialization fallback.
            return {"gate_failure": {"type": type(exc).__name__, "message": str(exc),
                                     "traceback": traceback.format_exc()[-8000:]}}

    def _run(self, kind, task, tokens, manifests, freeze, deadline, invocation, image_id):
        import numpy as np
        import torch
        # No hidden model setup in @enter: all loading falls inside this timed invocation.
        check_time(deadline)
        vol.reload()
        active = claims.get("active")
        require(active and invocation == active["id"] + "-gpu", "unreserved GPU invocation")
        require(claims.put("started-" + invocation, True, skip_if_exists=True),
                "duplicate GPU invocation/retry forbidden")
        check_freeze(freeze, 1 if kind == "smoke" else 2)
        for t in TASKS:
            require(len(tokens[t]) == len(manifests[t]["rows"]) == 200, "shipped row count")
            for ids, row in zip(tokens[t], manifests[t]["rows"]):
                require(hashlib.sha256(json_bytes(ids)).hexdigest() == row["token_sha256"],
                        "shipped token identity")
            if kind == "extract":
                require(hashlib.sha256(json_bytes(manifests[t])).hexdigest() ==
                        freeze["entries"][-1]["manifests"][f"{t}.manifest.json"],
                        "shipped frozen manifest identity")
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model, provenance = load_model()
        provenance["modal_image_id"] = image_id
        load_seconds = time.perf_counter() - start
        SEAL = import_seal()
        if kind == "smoke":
            timings, checks = [], {}
            for t in TASKS:
                sets = manifests[t]["yes_no_token_ids"]
                longest = max(range(200), key=lambda i: len(tokens[t][i]))
                checks[t] = {"longest_row": longest, "rows": []}
                for i in sorted(set(range(10)) | {longest}):
                    check_time(deadline)
                    row = capture(model, SEAL, tokens[t][i], sets, reconstruct=i < 2)
                    timings.append(row["seconds"])
                    off = capture(model, SEAL, tokens[t][i], sets, router_on=False)
                    delta = float(np.max(np.abs(row["logits"] - off["logits"])))
                    gate(int(row["logits"].argmax()) == int(off["logits"].argmax()) and delta <= .001,
                         f"router-output neutrality (max abs {delta:.6g})")
                    checks[t]["rows"].append({"row": i, "neutrality_max_abs": delta,
                                                "reconstruction": row["gates"], "finite": True,
                                                "hooks_once": True})
                    del row, off
                previous, reserved = None, []
                for repeat in range(3):
                    check_time(deadline)
                    row = capture(model, SEAL, tokens[t][longest], sets)
                    timings.append(row["seconds"])
                    reserved.append(torch.cuda.memory_reserved())
                    if previous is not None:
                        gate(np.array_equal(previous["expert_ids"], row["expert_ids"]),
                             "expert determinism")
                        for key in ("attention", "sink_draw", "router", "output"):
                            gate(np.max(np.abs(previous[key] - row[key])) <= 1e-5,
                                 f"feature determinism: {key}")
                    previous = {key: row[key] for key in ("expert_ids", "attention", "sink_draw", "router", "output")}
                    del row
                gate(max(reserved) - min(reserved) < .5 * 2**30, "memory growth")
                checks[t]["repeats_reserved_bytes"] = reserved
                checks[t]["determinism"] = True
            peak = torch.cuda.max_memory_reserved()
            gate(peak <= 20 * 2**30, f"peak memory {peak / 2**30:.2f} GiB")
            p95 = float(np.percentile(timings, 95))
            remaining = deadline - time.time()
            projection = load_seconds + 400 * p95 * 1.2
            # Two separately launched extraction cells both reload; stricter than §5 alone.
            gate(projection <= remaining and projection + load_seconds + 2 * SHUTDOWN <= remaining,
                 f"projected extraction {projection:.0f}s exceeds budget {remaining:.0f}s")
            return {"passed": True, "checks": checks, "load_seconds": load_seconds,
                    "p95_forward_seconds": p95, "projected_seconds": projection,
                    "projected_with_two_loads_seconds": projection + load_seconds,
                    "remaining_seconds_at_gate": remaining, "peak_reserved_bytes": peak,
                    "provenance": provenance}
        rows = []
        sets = manifests[task]["yes_no_token_ids"]
        for ids in tokens[task]:
            check_time(deadline)
            row = capture(model, SEAL, ids, sets)
            rows.append({k: row[k] for k in ("attention", "sink_draw", "router", "output", "expert_ids",
                         "expert_weights", "prompt_len", "argmax_id", "yes_no")})
            del row
            gate(torch.cuda.max_memory_reserved() <= 20 * 2**30, "extraction peak memory")
        require(len(rows) == 200, "incomplete cell")
        meta = {"schema": "router-pilot-npz/1", "task": task, "model": MODEL, "revision": REVISION,
                "data_sha256": DATA_SHA[task], "manifest_sha256": freeze["entries"][-1]["manifests"][f"{task}.manifest.json"],
                "freeze_entry": freeze["entries"][-1], "invocation": invocation,
                "attention_metrics": ATT, "router_metrics": ROUTER, "output_metrics": OUTPUT,
                "sink_metric": SINK, "sink_definition": SINK_DEFINITION,
                "provenance": provenance}
        arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
        arrays["metadata"] = np.asarray(json.dumps(meta, allow_nan=False))
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        check_time(deadline)
        # CPU coordinator owns publication; a timed-out worker cannot write a late NPZ/status.
        return {"npz": buffer.getvalue(), "provenance": provenance}


def event(invocation, phase, **extra):
    publish_json(OUT / "ledger" / f"{time.time_ns()}-{uuid.uuid4().hex}.json",
                 {"invocation": invocation, "phase": phase, "utc": utc(),
                  "unix": time.time(), **extra})
    vol.commit()


def ledger_used():
    entries = [read_json(p) for p in (OUT / "ledger").glob("*.json")]
    starts = {r["invocation"]: r for r in entries if r["phase"] == "start" and r["gpu"] != "CPU"}
    ends = {r["invocation"]: r for r in entries if r["phase"] == "end" and r["gpu"] != "CPU"}
    return sum(ends[k]["charged_seconds"] if k in ends else r["reserved_seconds"]
               for k, r in starts.items()), starts.keys() - ends.keys()


def publish_failure(kind, exc, freeze):
    """One classifier/publisher for preflight, call, publication and reconciliation failures."""
    failed = is_gate_failure(exc)
    if kind == "smoke":
        path = OUT / f"smoke.failed.{freeze['entries'][-1]['entry']}.json"
        if not path.exists():
            publish_json(path, {"passed": False, "gate_failure": failed,
                                "type": type(exc).__name__, "message": str(exc)[:2000],
                                "traceback": "".join(traceback.format_exception(exc))[-8000:],
                                "freeze_entry": freeze["entries"][-1], "utc": utc()})
    if failed or kind != "smoke":
        stop_all(f"{kind} {'gate failure' if failed else 'abort'}: {str(exc)[:300]}", freeze)
    vol.commit()
    if failed and not isinstance(exc, GateFailure):
        return GateFailure(f"GATE: {type(exc).__name__}: {exc}")
    return exc


def stop_all(reason, freeze):
    for task in TASKS:
        path = OUT / f"{task}.status.json"
        if not path.exists():
            publish_json(path, {"status": "NOT-RUN", "task": task, "reason": reason,
                               "utc": utc(), "freeze_entry": freeze["entries"][-1]})
    vol.commit()


@app.function(image=image, volumes={"/cache": vol}, timeout=7500,
              retries=0, max_containers=1, scaledown_window=2)
def coordinate(kind, task, freeze, image_id):
    invocation = uuid.uuid4().hex
    gpu_id = invocation + "-gpu"
    durable, storage_failed, pending_failure = False, False, None

    def storage(operation, *args, **kwargs):
        nonlocal storage_failed
        try:
            return operation(*args, **kwargs)
        except BaseException:
            # Even if a later write works, keep ownership for reconciliation after an
            # uncertain storage result. Never unlock in a storage-error finally block.
            storage_failed = True
            raise

    # Dict put(if_not_exists) is atomic across separate `modal run` launches.
    claim = {"id": invocation, "expires": time.time() + 7500,
             "kind": kind, "task": task, "freeze": freeze}
    acquired = claims.put("active", claim, skip_if_exists=True)
    if not acquired:
        prior = claims.get("active")
        if (prior and time.time() > prior["expires"]
                and claims.put("reconciliation", invocation, skip_if_exists=True)):
            # Keep the claim as a permanent stop. A killed controller is never a retry.
            # Failure here retains both claims for executor inspection/reconciliation.
            vol.reload()
            publish_failure(prior.get("kind", kind),
                            GateFailure("GATE: expired controller reservation; pilot stopped"),
                            prior.get("freeze", freeze))
            _, open_ids = ledger_used()
            for key in open_ids:
                starts = [read_json(p) for p in (OUT / "ledger").glob("*.json")]
                rec = next(r for r in starts if r["phase"] == "start" and r["invocation"] == key)
                event(key, "end", gpu=rec["gpu"], recovered=True,
                      charged_seconds=rec["reserved_seconds"])
            event(invocation, "end", gpu="CPU", reconciled=prior["id"])
            claims.pop("reconciliation")
        raise RuntimeError("active or expired pilot claim; no GPU launched")
    try:
        storage(vol.reload)
        storage(event, invocation, "start", gpu="CPU", kind=kind, task=task)
        if kind == "smoke":
            entry = freeze["entries"][-1]["entry"]
            attempt = OUT / f"smoke.attempt.{entry}.json"
            require(not storage(attempt.exists),
                    "smoke already attempted under this freeze entry; amend first")
            require(not storage((OUT / f"smoke.failed.{entry}.json").exists),
                    "smoke already failed under this freeze entry; amend first")
            storage(publish_json, attempt, {"invocation": invocation, "utc": utc(),
                                           "freeze_entry": freeze["entries"][-1]})
            storage(vol.commit)
        # All preflight follows the durable per-entry smoke attempt marker.
        check_freeze(freeze, 1 if kind == "smoke" else 2)
        used, unclosed = storage(ledger_used)
        gate(not unclosed, "unclosed GPU reservation; pilot stopped")
        gate(used + RESERVE + SHUTDOWN + 1 < CAP, "budget exhausted")
        # Any NOT-RUN stops the whole pilot. Existing successful cells stay immutable.
        for t in TASKS:
            status = OUT / f"{t}.status.json"
            if storage(status.exists):
                require(storage(read_json, status)["status"] == "ok" and kind == "extract" and t != task,
                        "terminal cell; refusing launch")
            npz = OUT / f"{t}.router.npz"
            require(not storage(npz.exists) or (storage(status.exists) and kind == "extract" and t != task),
                    "existing NPZ; refusing launch")
        if kind == "smoke":
            require(not storage((OUT / "smoke.report.json").exists), "smoke already passed")
            gate(not any(storage(read_json, p).get("gate_failure")
                         for p in storage(lambda: list(OUT.glob("smoke.failed.*.json")))),
                 "a measured-gate failure stopped the pilot")
        else:
            report = storage(read_json, OUT / "smoke.report.json")
            require(report["passed"] and report["freeze_entry"] == freeze["entries"][-2],
                    "missing/mismatched successful smoke")
            check_freeze(freeze, 2, OUT)
        # Protected pinned download FIRST; tokenizer/model only read this local cache.
        # The outer publisher classifies Hub failures through both exception-chain links.
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, token=False)
        manifests, tokens = manifests_and_tokens()
        for t in TASKS:
            path = OUT / f"{t}.manifest.json"
            if storage(path.exists):
                gate(storage(path.read_bytes) == json_bytes(manifests[t]), "prompt manifest mismatch")
            elif kind == "smoke":
                storage(publish_json, path, manifests[t])
            else:
                raise RuntimeError("missing frozen manifest")
        storage(vol.commit)
        # Persist the exact stage history before GPU allocation.
        freeze_path = OUT / f"freeze-{freeze['entries'][-1]['entry']}.json"
        if storage(freeze_path.exists):
            require(storage(read_json, freeze_path) == freeze, "freeze history drift")
        else:
            storage(publish_json, freeze_path, freeze)
        seconds = int(CAP - used - RESERVE - SHUTDOWN)
        gate(seconds > 0, "no invocation budget")
        deadline = time.time() + seconds
        storage(claims.put, "active", {**claim, "expires": deadline + RESERVE})
        storage(event, gpu_id, "start", gpu="NVIDIA A10G", kind=kind, task=task,
                reserved_seconds=seconds + SHUTDOWN, deadline=deadline,
                modal_timeout_seconds=seconds, image_id=image_id,
                image_recipe_sha256=RECIPE_SHA)
        remote = GPUWorker.with_options(timeout=seconds)().run.remote
        call_error = None
        launch = time.perf_counter()
        try:
            result = remote(kind, task, tokens, manifests, freeze, deadline, gpu_id, image_id)
        except BaseException as exc:
            call_error = exc
        finally:
            # Capture immediately on BOTH return and exception, before any storage work.
            charged = time.perf_counter() - launch + SHUTDOWN
            returned = time.time()
        pending_failure = call_error
        # Exhaustion takes terminal precedence even when the remote call also failed.
        try:
            gate(returned < deadline and used + charged + RESERVE + SHUTDOWN + 1 < CAP,
                 "budget exhausted after GPU invocation")
        except GateFailure as exc:
            pending_failure = exc
        try:
            if pending_failure is None and "gate_failure" in result:
                failure = result["gate_failure"]
                pending_failure = GateFailure(f"GATE: remote {failure['type']}: {failure['message']}\n"
                                              f"{failure['traceback']}")
        finally:
            # Even a malformed return envelope must close the observed call interval.
            storage(event, gpu_id, "end", gpu="NVIDIA A10G", charged_seconds=charged,
                    completed=pending_failure is None)
        if pending_failure is not None:
            raise pending_failure
        result["provenance"]["modal_image_id"] = image_id
        if kind == "smoke":
            result.update({"freeze_entry": freeze["entries"][-1], "utc": utc(),
                           "manifests": {f"{t}.manifest.json": storage(sha, OUT / f"{t}.manifest.json")
                                         for t in TASKS}})
            storage(publish_json, OUT / "smoke.report.json", result)
        else:
            path = OUT / f"{task}.router.npz"
            storage(atomic, path, result["npz"], serialized=True)
            storage(publish_json, OUT / f"{task}.status.json", {
                "status": "ok", "task": task, "utc": utc(), "npz_sha256": storage(sha, path),
                "freeze_entry": freeze["entries"][-1], "provenance": result["provenance"]})
        storage(vol.commit)
        storage(event, invocation, "end", gpu="CPU", completed=True)
        durable = True
        return {"completed": kind, "task": task}
    except BaseException as exc:
        failure = storage(publish_failure, kind, pending_failure or exc, freeze)
        storage(event, invocation, "end", gpu="CPU", completed=False,
                gate_failure=is_gate_failure(failure))
        # A start without a durable end keeps its full reservation and the claim.
        _, unclosed = storage(ledger_used)
        durable = not unclosed
        if failure is not exc:
            raise failure from exc
        raise
    finally:
        if durable and not storage_failed:
            try:
                active = storage(claims.get, "active")
                if active and active["id"] == invocation:
                    storage(claims.pop, "active")
            except BaseException as exc:
                failure = storage(publish_failure, kind, exc, freeze)
                storage(event, invocation, "release-failed", gpu="CPU")
                if failure is not exc:
                    raise failure from exc
                raise


def local_freeze(stage):
    freeze = read_json(HERE / "FREEZE.json")
    check_freeze(freeze, stage, HERE if stage == 2 else None)
    return freeze


@app.local_entrypoint()
def smoke():
    freeze = local_freeze(1)
    try:
        coordinate.remote("smoke", "", freeze, image.object_id)
    finally:
        download(smoke_only=True)


@app.local_entrypoint()
def extract(task: str):
    require(task in TASKS, "unknown task")
    coordinate.remote("extract", task, local_freeze(2), image.object_id)


@app.local_entrypoint()
def download(smoke_only: bool = False):
    """Download immutable artifacts; never overwrite a different local byte stream."""
    for entry in vol.iterdir("router-pilot", recursive=True):
        relative = Path(entry.path.lstrip("/")).relative_to("router-pilot")
        if relative.suffix not in (".json", ".npz") or ".pending-" in relative.name:
            continue
        if smoke_only and not (relative.name in {"smoke.report.json", *(f"{t}.manifest.json" for t in TASKS)}
                               or relative.name.startswith(("smoke.failed.", "smoke.attempt."))
                               or relative.name in {f"{t}.status.json" for t in TASKS}):
            continue
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe volume path")
        target = HERE / relative
        data = b"".join(vol.read_file(entry.path))
        if target.exists():
            require(target.read_bytes() == data, "local artifact already exists with different bytes")
        else:
            atomic(target, data)
    print("Download complete; no scores computed.")
