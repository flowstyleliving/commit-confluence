"""Grid-B per-layer depth-curve extractor (Modal) — PRE_REGISTRATION_EXPANSION.md.

Same instrument as the registered grid-A extractor (modal_depth.py): per prompt ONE
t=0 forward, eager attention with FULL retention of every block, last-query row per
block scored by the SEALED kernel under tag "final" (the four weight-only cells).
ONE capture mode grid-wide. Gates identical. This file adds ONLY what grid B needs:

  * a frozen MODEL REGISTRY (pinned HF revisions, expected layer/head/KV counts,
    per-model load mode + GPU shape) — revision and layer-count mismatches abort;
  * a shared DECODER DESCRIPTOR so wrapper checkpoints (Mistral3ForConditional-
    Generation) and text-tower loads (Gemma3) expose decoder layers / text config /
    o_proj paths to the SAME gate and capture code as plain Llama;
  * custom loaders: nf4 with a vision-aware skip list (Mistral Small 3.2), and the
    Medium-3.5 FP8-origin -> deterministic dequant -> BF16-compute path (recorded);
  * TERMINAL STATUS FILES: every attempted cell writes <slug>.status.json with
    status ok|aborted (+reason). The scorer refuses to run without 12 terminal
    statuses; aborted cells are confirmatory FAILURES per the prereg.
  * SMOKE (gates-only): per model, the 2-row o_proj/YES-NO gate + tokenize all 200
    rows x both tasks, verify single-BOS at position 0 and token budget, write a
    prompt-token manifest (sha256 per row); the confirmatory extraction re-derives
    prompt tokens and FAILS CLOSED on any manifest mismatch. For the Mistral pair,
    smoke also cross-checks AutoTokenizer chat-template ids against mistral-common
    (ids must match or the cell aborts). Smoke never computes a metric or AUROC.

DISCIPLINE (registered): prints gates/progress only; never computes or prints an
AUROC. Scoring happens locally in score_grid_b.py ONCE after all 12 cells reach a
terminal state. COMPARABILITY: torch lane, NON-byte-comparable; never pool.

Run:
  modal run exploratory/depth-curve/modal_depth_b.py::smoke  --model-key llama31_8b
  modal run exploratory/depth-curve/modal_depth_b.py::extract --model-key llama31_8b --task anli_r1
"""
import os
from pathlib import Path

import modal

APP_NAME = "cc-depth-grid-b"
VOL_NAME = "model-cache"
MNT = "/models"
SEAL_REMOTE = "/seal"
PKG_REMOTE = "/pkg"
OUT_DIR = "depth_grid_b"
DEPTH_MAX_TOKENS = 900
SCHEMA = "furnace-depth-curve/1.1-gridB"
PREREG = "exploratory/depth-curve/PRE_REGISTRATION_EXPANSION.md"

FROZEN_DATA_SHA256 = {
    "anli_r1": "57ad341f2c29c886a726b7c62b7371be8c064b04b9b96e98324c931157d4f55b",
    "halueval_qa": "a841d096a3f41162a685994655e5fdd0974176ee35797e73be99e29e5d1c15e0",
}

# Filled at FREEZE TIME (after gates-only smoke, before any outcome-bearing
# extraction): sha256 of each smoke-written prompt manifest file, keyed
# "<slug>.<task>". Extraction REFUSES to run while an entry is missing, and
# fails closed if the volume manifest's bytes do not hash to the frozen value —
# a post-freeze smoke can therefore never silently replace a manifest
# (round-5 MAJOR-3). Smoke likewise refuses to overwrite a manifest that has a
# frozen entry.
_MANIFEST_PENDING = "PENDING-SMOKE-NOT-A-HASH"
# FROZEN 2026-08-17 from the gates-only smoke (transformers 5.15.0 image, one
# toolchain for all six models; local copies committed under manifests_gridb/).
# Mistral-Small/anli_r1 keeps the sentinel DELIBERATELY: its smoke gate failed
# (row-1 commit 'To', the registered behavioral gate) so it has no manifest and
# its extraction attempt will abort into a registered confirmatory failure.
FROZEN_MANIFEST_SHA256 = {
    "Llama-3.1-8B-Instruct.anli_r1": "ec30755c7280fdc42f28755c74ff5d3b5b59d3efb0f0a4d520fac4092e351199",
    "Llama-3.1-8B-Instruct.halueval_qa": "075e01093a4848475291ab153aa33abdd0005a3ef90046b034d54de98ee100be",
    "Llama-3.1-70B-Instruct.anli_r1": "04a818afc06838de2e0c19d57a293630e847c92abb9d9a5e2721ae54c6059f2a",
    "Llama-3.1-70B-Instruct.halueval_qa": "40569a0f9fc2f05b49e1b2f217dad957f31e706da806919b57385621b6b7031d",
    "Mistral-Small-3.2-24B-Instruct-2506.anli_r1": _MANIFEST_PENDING,
    "Mistral-Small-3.2-24B-Instruct-2506.halueval_qa": "cdf8712562d397c6fd15858de1d25af278606ed9d54f6062e5a2fa58469bcd3f",
    "Mistral-Medium-3.5-128B.anli_r1": "3fac6bdf67f356fe20783c55e351ec5ec15e0734388585aaacfa3bad482a887b",
    "Mistral-Medium-3.5-128B.halueval_qa": "97b1ba6ebd34b76b88f099302ae4b8887c294edaafe99456bbd9331f31dd2b91",
    "gemma-3-12b-it.anli_r1": "37eec4eddae99f41a2de3f81b43dde60382886125ec7c6ab7f458536c9779e19",
    "gemma-3-12b-it.halueval_qa": "3750f6db6fdd3d71fcba81d611e89a31acbd519dda6c56f4633f398d4f619e47",
    "gemma-3-27b-it.anli_r1": "e993a97d4da585cbb097e21d198d499b81232e56749af4914b8057a621a4dce0",
    "gemma-3-27b-it.halueval_qa": "65edb68a2ed64c85f833093495bcb7b3725330c9974dc3151de029ce7adfd460",
    # 405B stretch (registered §7 enablement, MK go 2026-08-17): hashes frozen
    # from its gates-only smoke (cos 0.99999 ×4, 882/882 Linear4bit, 8×A100).
    "Llama-3.1-405B-Instruct.anli_r1": "ea796f73453509e920899139a3697ba1f6fcd02069bb2ee1fdb1bfc067b0d00a",
    "Llama-3.1-405B-Instruct.halueval_qa": "c014438f2355cf4dba0b1a28c88d11f871fcfb55de1afa14cd6b553079a0c8a1",
}

# Filled at FREEZE TIME for the Medium cell: the dequant method AND GPU shape
# smoke selected. Extraction refuses to run mistral_medium_35 while either is
# None, aborts on method mismatch, and aborts if launched on any other GPU shape
# (round-5 MAJOR-5 + round-6: hardware must be frozen with the method).
# FROZEN 2026-08-17 from smoke: transformers auto-dequantizes the FP8 checkpoint
# to bf16 on A100 (capability 8.0 < 8.9) inside from_pretrained; verified 1233/1233
# params bf16, 616 decoder Linears plain, quantizer enforcement PASS, cos 1.0.
FROZEN_MEDIUM_DEQUANT_METHOD = "from_pretrained(bf16)"
FROZEN_MEDIUM_GPU = "A100-80GB:4"
DEPTH_CELL_DETAILS = ("final_js", "final_js_no_bos", "final_js_kv_groups", "final_bos_mass")
TASKS = ("anli_r1", "halueval_qa")

# ── frozen registry (PRE_REGISTRATION_EXPANSION.md §1) ──────────────────────────
# load modes: "base_nf4" = June BASE._load (plain CausalLM, bnb nf4);
#             "mistral3_nf4" = wrapper + vision-aware nf4 skip list;
#             "mistral3_fp8_dequant_bf16" = FP8-origin checkpoint, deterministic
#                 dequant to BF16 storage+compute (method recorded);
#             "gemma3_nf4" = AutoModelForCausalLM text tower (wrapper fallback).
REGISTRY = {
    "llama31_8b": dict(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
        load="base_nf4", gpu="A100-80GB",
        exp_layers=32, exp_heads=32, exp_kv=8, mistral_xcheck=False),
    "llama31_70b": dict(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        revision="1605565b47bb9346c5515c34102e054115b4f98b",
        load="base_nf4", gpu="A100-80GB",
        exp_layers=80, exp_heads=64, exp_kv=8, mistral_xcheck=False),
    "mistral_small_32": dict(
        model_id="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        revision="95a6d26c4bfb886c58daf9d3f7332c857cb27b43",
        load="mistral3_nf4", gpu="A100-80GB",
        exp_layers=40, exp_heads=32, exp_kv=8, mistral_xcheck=True),
    "mistral_medium_35": dict(
        model_id="mistralai/Mistral-Medium-3.5-128B",
        revision="22b2b868a15677cfa6061277ed2f653d1349a9ab",
        load="mistral3_fp8_dequant_bf16", gpu="A100-80GB:4",
        exp_layers=88, exp_heads=96, exp_kv=8, mistral_xcheck=True),
    "gemma3_12b": dict(
        model_id="google/gemma-3-12b-it",
        revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
        load="gemma3_nf4", gpu="A100-80GB",
        exp_layers=48, exp_heads=16, exp_kv=8, mistral_xcheck=False),
    "gemma3_27b": dict(
        model_id="google/gemma-3-27b-it",
        revision="005ad3404e59d6023443cb575daa05336842228a",
        load="gemma3_nf4", gpu="A100-80GB",
        exp_layers=62, exp_heads=32, exp_kv=16, mistral_xcheck=False),
    # STRETCH CELL (prereg §1/§7): descriptive-only, OUTSIDE every confirmatory
    # denominator; runs only on MK's explicit go (given 2026-08-17). Default
    # 8×A100-80 per the registration; the 4× downgrade path (nine-condition smoke
    # gate) is deliberately NOT implemented — default hardware only.
    "llama31_405b": dict(
        model_id="meta-llama/Llama-3.1-405B-Instruct",
        revision="be673f326cab4cd22ccfef76109faf68e41aa5f1",
        load="base_nf4", gpu="A100-80GB:8",
        exp_layers=126, exp_heads=128, exp_kv=8, mistral_xcheck=False),
}

_HERE = Path(__file__).parent
_MODAL_DIR = (_HERE / ".." / ".." / "modal").resolve()

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name("huggingface")

# Pinned image (Codex round-1 MAJOR-6: prospective dependency pinning).
# Repinned 2026-08-17 after smoke exposed version skew: the 2026 Mistral
# checkpoints (Small 3.2 tokenizer mapping, Medium 3.5 "TokenizersBackend")
# require the current transformers 5.x line. All six smokes rerun on THIS image
# so every manifest shares one toolchain.
PINNED = [
    "torch==2.13.0", "transformers==5.15.0", "accelerate==1.14.0",
    "numpy==2.3.5", "scipy==1.18.0", "sentencepiece==0.2.2",  # numpy<2.4 per mistral-common
    "safetensors==0.8.0", "huggingface_hub==1.27.0", "hf_transfer==0.1.9",
    "datasets==5.0.1", "bitsandbytes==0.50.1", "mistral-common==1.11.7",
    "pandas==3.0.5", "scikit-learn==1.9.0",
]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*PINNED)
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{MNT}/hub",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(_MODAL_DIR / "seal"), SEAL_REMOTE, copy=True,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
    .add_local_dir(str(_MODAL_DIR / "seal"), f"{PKG_REMOTE}/seal", copy=True,
                   ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc")
    .add_local_file(str(_MODAL_DIR / "modal_app.py"), f"{PKG_REMOTE}/cc_modal_app.py", copy=True)
)


# ── in-container helpers (imported lazily inside functions) ─────────────────────
def _slug(conf):
    return conf["model_id"].split("/")[-1]


def _status_path(task, conf):
    return f"{MNT}/{OUT_DIR}/{task}/{_slug(conf)}.status.json"


def _write_status(task, conf, status, reason="", extra=None):
    import json
    d = f"{MNT}/{OUT_DIR}/{task}"
    os.makedirs(d, exist_ok=True)
    payload = {"status": status, "reason": reason, "model_key": conf["_key"],
               "model_id": conf["model_id"], "revision": conf["revision"],
               "task": task, "schema": SCHEMA}
    if extra:
        payload.update(extra)
    sp = _status_path(task, conf)
    with open(sp + ".tmp", "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(sp + ".tmp", sp)
    vol.commit()


def _patch_mlx_stub():
    """cc_modal_app installs a stub `mlx` module so the sealed (MLX-based) kernel
    imports on Linux. transformers 5.x probes `isinstance(x, mx.array)` inside
    accelerate's multi-GPU output dispatch; the stub lacked `array`, crashing
    ONLY multi-device models (Medium, 405B). Give the stub an inert class so the
    isinstance probe is False and transformers falls through to the torch check.
    The June file itself stays untouched (grid-A provenance)."""
    import sys
    m = sys.modules.get("mlx.core")
    if m is not None and not hasattr(m, "array"):
        class _NeverAnMlxArray:  # noqa: N801 — inert sentinel type
            pass
        m.array = _NeverAnMlxArray


def _resolve_descriptor(model, conf):
    """One shared decoder descriptor (Codex round-1 MAJOR-7 / residual-7).

    Returns dict with: layers (ModuleList), text_config, n_layers, n_heads, n_kv.
    Verified against the frozen registry expectations — mismatch raises.
    """
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None) or cfg
    root = getattr(model, "model", model)
    if hasattr(root, "language_model") and hasattr(root.language_model, "layers"):
        layers = root.language_model.layers          # Mistral3/Gemma3 wrapper path
    elif hasattr(root, "layers"):
        layers = root.layers                          # plain CausalLM path
    elif hasattr(root, "model") and hasattr(root.model, "layers"):
        layers = root.model.layers
    else:
        raise RuntimeError(f"cannot resolve decoder layers on {type(model).__name__}")
    n_layers = len(layers)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv = int(getattr(text_cfg, "num_key_value_heads", n_heads))
    if n_layers != conf["exp_layers"] or n_heads != conf["exp_heads"] or n_kv != conf["exp_kv"]:
        raise RuntimeError(
            f"DESCRIPTOR MISMATCH vs frozen registry: got layers/heads/kv "
            f"{n_layers}/{n_heads}/{n_kv}, expected "
            f"{conf['exp_layers']}/{conf['exp_heads']}/{conf['exp_kv']}")
    return {"layers": layers, "text_config": text_cfg,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv": n_kv}


def _load_model(conf):
    """Load per registry mode. Returns (tok, model, precision_label, load_notes)."""
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    mid, rev = conf["model_id"], conf["revision"]
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    notes = {}

    def bnb(skip):
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=skip)

    if conf["load"] == "base_nf4":
        model = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, device_map="auto", dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=bnb(["lm_head", "embed_tokens"]))
        label = "nf4"
    elif conf["load"] == "mistral3_nf4":
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            mid, revision=rev, device_map="auto", dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=bnb(["lm_head", "embed_tokens", "vision_tower",
                                     "multi_modal_projector"]))
        label = "nf4"
    elif conf["load"] == "gemma3_nf4":
        # Text tower via AutoModelForCausalLM (Gemma3ForCausalLM). If transformers
        # routes to the wrapper instead, the descriptor resolves language_model.
        model = AutoModelForCausalLM.from_pretrained(
            mid, revision=rev, device_map="auto", dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=bnb(["lm_head", "embed_tokens", "vision_tower",
                                     "multi_modal_projector"]))
        label = "nf4"
    elif conf["load"] == "mistral3_fp8_dequant_bf16":
        # FP8-origin checkpoint (prereg §1): deterministic dequant -> BF16.
        # Tried in order; the successful method is recorded in provenance.
        from transformers import Mistral3ForConditionalGeneration
        model, method = None, None
        errs = []
        try:  # (a) plain bf16 load — transformers dequantizes if it supports it
            model = Mistral3ForConditionalGeneration.from_pretrained(
                mid, revision=rev, device_map="auto", dtype=torch.bfloat16,
                attn_implementation="eager")
            method = "from_pretrained(bf16)"
        except Exception as e:  # noqa: BLE001
            errs.append(f"(a) {type(e).__name__}: {e}")
        if model is None:
            try:  # (b) load quantized then dequantize()
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    mid, revision=rev, device_map="auto",
                    attn_implementation="eager")
                model = model.dequantize()
                method = "load+dequantize()"
            except Exception as e:  # noqa: BLE001
                errs.append(f"(b) {type(e).__name__}: {e}")
        if model is None:
            raise RuntimeError("Medium-3.5 dequant failed on this hardware: "
                               + " | ".join(errs))
        # enforce: no fp8 dtypes remain in decoder linears
        bad = [n for n, p in model.named_parameters()
               if "layers" in n and p.dtype not in (torch.bfloat16, torch.float32)]
        if bad:
            raise RuntimeError(f"dequant left non-bf16 decoder params, e.g. {bad[:5]}")
        label = "fp8origin-dequant-bf16"
        notes["dequant_method"] = method
    else:
        raise RuntimeError(f"unknown load mode {conf['load']!r}")

    got_rev = getattr(model.config, "_commit_hash", None)
    if got_rev != rev:
        raise RuntimeError(f"revision mismatch: loaded {got_rev} != pinned {rev}")
    model.eval()

    # No-offload assertion (round-5 MAJOR-5): every parameter must live on CUDA.
    offloaded = sorted({str(p.device) for p in model.parameters()
                        if p.device.type != "cuda"})
    if offloaded:
        raise RuntimeError(f"CPU/disk offload detected (devices {offloaded}) — "
                           f"refusing: capture faithfulness requires full-GPU residency")
    # Hardware + quantizer provenance
    notes["gpu_names"] = [torch.cuda.get_device_name(i)
                          for i in range(torch.cuda.device_count())]
    dt_hist = {}
    for p_ in model.parameters():
        dt_hist[str(p_.dtype)] = dt_hist.get(str(p_.dtype), 0) + 1
    notes["param_dtype_hist"] = dt_hist
    return tok, model, label, notes


def _enforce_decoder_quantizers(desc, conf):
    """Walk the DESCRIPTOR-resolved decoder blocks only (round-6: a name-based
    '.layers.' filter would sweep the deliberately-unquantized vision tower).
    Enforce per load mode: *_nf4 -> every 2-D-weight leaf module in every decoder
    block is bitsandbytes Linear4bit; fp8-dequant -> plain nn.Linear in bf16.
    Returns the full type histogram for provenance."""
    import torch
    import torch.nn as nn
    lin_types, offenders = {}, []
    want_4bit = conf["load"].endswith("_nf4")
    n_audited = 0
    for bi, block in enumerate(desc["layers"]):
        for n_, m_ in block.named_modules():
            if list(m_.children()):
                continue
            wt = getattr(m_, "weight", None)
            if wt is None:
                continue
            tname = type(m_).__name__
            # Classify by TYPE first (round-7: packed Linear4bit storage must not
            # be silently skipped on an ndim test); ndim gates only the plain path.
            is_bnb_linear = tname == "Linear4bit"
            # Linear4bit subclasses nn.Linear — exclude it explicitly so the
            # Medium plain-BF16 predicate can never accept a quantized module
            # (round-8 B2).
            is_plain_linear = (isinstance(m_, nn.Linear) and not is_bnb_linear
                               and wt.ndim == 2)
            if not (is_bnb_linear or is_plain_linear):
                continue
            n_audited += 1
            lin_types[tname] = lin_types.get(tname, 0) + 1
            if want_4bit and not is_bnb_linear:
                offenders.append((bi, n_, tname))
            if conf["load"] == "mistral3_fp8_dequant_bf16" and not (
                    is_plain_linear and wt.dtype == torch.bfloat16):
                offenders.append((bi, n_, f"{tname}/{wt.dtype}"))
    # Coverage floor: every decoder block has at least q/k/v/o projections, so an
    # audit that saw fewer than 4 linears per block is itself broken (round-7:
    # an empty audit must fail, not pass).
    floor = 4 * len(desc["layers"])
    if n_audited < floor:
        raise RuntimeError(f"QUANTIZER AUDIT INCOMPLETE: saw {n_audited} linear "
                           f"modules < floor {floor} — classification is missing "
                           f"module types; refusing")
    if offenders:
        raise RuntimeError(f"QUANTIZER ENFORCEMENT FAILED for mode {conf['load']}: "
                           f"e.g. {offenders[:5]} (total {len(offenders)})")
    return lin_types


def _chat_ids(tok, prompt):
    """Native chat template, single user turn, generation prompt appended."""
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _verify_bos(tok, ids, model_id):
    """Exactly ONE BOS in the whole sequence, at position 0 (round-5 MAJOR-2)."""
    bos = tok.bos_token_id
    if bos is None:
        raise RuntimeError(f"{model_id}: tokenizer has no BOS token id")
    if len(ids) < 2 or ids[0] != bos:
        raise RuntimeError(f"{model_id}: position 0 is not BOS (got {ids[:3]})")
    n_bos = sum(1 for t in ids if t == bos)
    if n_bos != 1:
        raise RuntimeError(f"{model_id}: {n_bos} BOS tokens in sequence (want exactly 1)")


def _prompt_manifest(tok, prompts, model_id):
    import hashlib
    rows = []
    for i, p in enumerate(prompts):
        ids = _chat_ids(tok, p)
        _verify_bos(tok, ids, model_id)
        if len(ids) > DEPTH_MAX_TOKENS:
            raise RuntimeError(f"row {i}: {len(ids)} tokens > {DEPTH_MAX_TOKENS}")
        import numpy as np
        rows.append(hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest())
    return rows


def _mistral_common_xcheck(tok, prompts, model_id, revision):
    """Cross-check AutoTokenizer chat-template ids against mistral-common.
    Registered rule: any row mismatch aborts the cell (two implementations must
    agree before the template is trusted)."""
    from huggingface_hub import hf_hub_download
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    path = hf_hub_download(model_id, "tekken.json", revision=revision)
    mt = MistralTokenizer.from_file(path)
    n_mismatch = 0
    first = None
    for i, p in enumerate(prompts):
        hf_ids = _chat_ids(tok, p)
        mc = mt.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=p)]))
        if list(mc.tokens) != list(hf_ids):
            n_mismatch += 1
            if first is None:
                first = {"row": i, "hf_head": hf_ids[:12], "mc_head": list(mc.tokens)[:12],
                         "hf_len": len(hf_ids), "mc_len": len(mc.tokens)}
    return n_mismatch, first


def _oproj_cos_gate_b(model, tok, desc, prompt):
    """The June validate() o_proj reconstruction gate with descriptor-resolved paths.

    Semantics IDENTICAL to grid A: capture the final block's per-position V (v_proj
    output) and the real post-o_proj attention output via hooks; take the SCORED
    attention row w[H, T] from output_attentions; reconstruct
    ctx[h] = sum_t w[h,t] * V_rep[h,t]; apply the module's own o_proj; cosine
    against the real output at the last position. This ties the tensors we score
    to the computation the model actually performed (NOT circular: w comes from
    output_attentions, V from v_proj — o_proj never sees its own input twice).
    """
    import numpy as np
    import torch

    final_idx = desc["n_layers"] - 1
    attn = desc["layers"][final_idx].self_attn
    n_kv = desc["n_kv"]
    store = {}

    hv = attn.v_proj.register_forward_hook(
        lambda _m, _i, out: store.__setitem__("v", out[0].detach().float().cpu()))
    ho = attn.o_proj.register_forward_hook(
        lambda _m, _i, out: store.__setitem__(
            "attn_out", out[0, -1].detach().float().cpu().numpy()))
    try:
        ids = _chat_ids(tok, prompt)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=model.device),
                        output_attentions=True, use_cache=False)
    finally:
        # Remove BOTH hooks before any manual module call: the reconstruction below
        # invokes attn.o_proj directly, and a live hook would overwrite the captured
        # reference vector before it is read (round-5 MAJOR-1, hook reentrancy).
        hv.remove(); ho.remove()

    # snapshot captures immediately; nothing below may touch `store`
    real_out = store["attn_out"].copy()
    v_out = store["v"].clone()

    logits = out.logits[0, -1].float().cpu().numpy()
    p = np.exp(logits - logits.max()); p /= p.sum()
    gid = int(np.argmax(p))
    commit = tok.decode([gid], skip_special_tokens=True).strip().upper()

    att = out.attentions[final_idx]
    w = att[0, :, -1, :].float().cpu()                      # [H, T]
    H, T = int(w.shape[0]), int(w.shape[1])
    if v_out.shape[0] != T or v_out.shape[1] % n_kv != 0:
        raise RuntimeError(f"v_proj capture shape {tuple(v_out.shape)} "
                           f"incompatible with T={T}, n_kv={n_kv}")
    d_head = v_out.shape[1] // n_kv
    vv = v_out.view(T, n_kv, d_head).permute(1, 0, 2)       # [n_kv, T, d]
    if H % n_kv != 0:
        raise RuntimeError(f"H={H} not divisible by n_kv={n_kv}")
    with torch.no_grad():
        v_rep = vv.repeat_interleave(H // n_kv, dim=0)      # [H, T, d]
        ctx = torch.einsum("ht,htd->hd", w, v_rep).reshape(1, -1)
        oproj_dtype = getattr(attn.o_proj, "compute_dtype", None) or attn.o_proj.weight.dtype
        if not (isinstance(oproj_dtype, torch.dtype)
                and torch.empty((), dtype=oproj_dtype).is_floating_point()):
            oproj_dtype = getattr(model, "dtype", torch.float32)
        ctx = ctx.to(device=attn.o_proj.weight.device, dtype=oproj_dtype)
        my_out = attn.o_proj(ctx)[0].detach().float().cpu().numpy()
    cos = float(np.dot(my_out, real_out) /
                (np.linalg.norm(my_out) * np.linalg.norm(real_out) + 1e-9))
    return cos, commit in ("YES", "NO"), repr(tok.decode([gid])), (H, T)


def _run_extract(conf, task):
    import hashlib
    import json
    import sys
    import time

    import numpy as np
    import torch

    if PKG_REMOTE not in sys.path:
        sys.path.insert(0, PKG_REMOTE)
    import cc_modal_app as BASE
    SEAL, PIPE, CR, CC, io_plugins, target_layer_map = BASE._import_seal()
    _patch_mlx_stub()

    by_detail = {c[2]: tuple(c) for c in SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS}
    missing = [d for d in DEPTH_CELL_DETAILS if d not in by_detail]
    if missing:
        raise RuntimeError(f"sealed panel missing cells: {missing}")
    cells = [by_detail[d] for d in DEPTH_CELL_DETAILS]

    data = f"{MNT}/data/{task}_n200.jsonl"
    if not os.path.exists(data):
        raise FileNotFoundError(f"missing data file on volume: {data}")
    data_sha = hashlib.sha256(open(data, "rb").read()).hexdigest()
    if data_sha != FROZEN_DATA_SHA256[task]:
        raise RuntimeError(f"DATA HASH MISMATCH {task}: {data_sha}")
    prompts, labels, dh = CR._load_calibration_jsonl(data)
    if len(prompts) != 200:
        raise RuntimeError(f"expected 200 rows, got {len(prompts)}")

    # Medium method freeze (round-5 MAJOR-5): outcome extraction may not run the
    # dequant cell until the smoke-selected method is frozen into this file.
    if conf["load"] == "mistral3_fp8_dequant_bf16":
        if FROZEN_MEDIUM_DEQUANT_METHOD is None:
            raise RuntimeError("Medium dequant method not frozen — smoke + freeze "
                               "commit must precede extraction")

    tok, model, precision, load_notes = _load_model(conf)
    if conf["load"] == "mistral3_fp8_dequant_bf16":
        if load_notes.get("dequant_method") != FROZEN_MEDIUM_DEQUANT_METHOD:
            raise RuntimeError(f"dequant method {load_notes.get('dequant_method')!r} "
                               f"!= frozen {FROZEN_MEDIUM_DEQUANT_METHOD!r}")
    desc = _resolve_descriptor(model, conf)
    load_notes["decoder_linear_types"] = _enforce_decoder_quantizers(desc, conf)
    n_layers, n_heads, n_kv = desc["n_layers"], desc["n_heads"], desc["n_kv"]
    print(f"[gridB] {conf['model_id']} {task} precision={precision} "
          f"layers={n_layers} H={n_heads} kv={n_kv}", flush=True)

    # Prompt manifest: must exist, hash to the FREEZE-pinned value, carry the right
    # identity fields, and reproduce exactly (round-5 MAJOR-3 + MINOR-2).
    man_path = f"{MNT}/{OUT_DIR}/manifests/{_slug(conf)}.{task}.prompts.json"
    if not os.path.exists(man_path):
        raise RuntimeError(f"missing smoke prompt manifest: {man_path} — run smoke first")
    man_bytes = open(man_path, "rb").read()
    man_sha = hashlib.sha256(man_bytes).hexdigest()
    frozen_sha = FROZEN_MANIFEST_SHA256.get(f"{_slug(conf)}.{task}")
    if frozen_sha != man_sha:
        raise RuntimeError(f"MANIFEST NOT FROZEN OR TAMPERED: file sha {man_sha} != "
                           f"frozen entry {frozen_sha!r} — extraction refuses")
    man = json.loads(man_bytes)
    if not (man.get("model_id") == conf["model_id"]
            and man.get("revision") == conf["revision"]
            and man.get("task") == task and man.get("schema") == SCHEMA
            and isinstance(man.get("rows"), list) and len(man["rows"]) == 200):
        raise RuntimeError(f"manifest identity fields invalid: "
                           f"{ {k: man.get(k) for k in ('model_id','revision','task','schema')} }")
    frozen_rows = man["rows"]
    live_rows = _prompt_manifest(tok, prompts, conf["model_id"])
    if live_rows != frozen_rows:
        bad = next(i for i in range(len(live_rows)) if live_rows[i] != frozen_rows[i])
        raise RuntimeError(f"PROMPT MANIFEST MISMATCH at row {bad} — refusing to extract")

    # gates (rows 0-1) — enforced on the RAW cosine, never the rounded display
    # value (round-6 new-defect fix: 0.99897 must not round up past the bar)
    gate = {"rows": []}
    raw_cos = []
    for i in (0, 1):
        cos, is_yn, commit, wshape = _oproj_cos_gate_b(model, tok, desc, prompts[i])
        raw_cos.append(cos)
        gate["rows"].append({"row": i, "oproj_recon_cos": round(cos, 5),
                             "oproj_recon_cos_raw": float(cos),
                             "commit_is_yes_no": bool(is_yn), "commit_token": commit})
        print(f"[gridB] gate row {i}: cos={cos:.7f} yes_no={is_yn} commit={commit}",
              flush=True)
    gate["GATE_cos_ok"] = all(c >= 0.999 for c in raw_cos)
    gate["GATE_yes_no_ok"] = all(r["commit_is_yes_no"] for r in gate["rows"])
    if not (gate["GATE_cos_ok"] and gate["GATE_yes_no_ok"]):
        raise RuntimeError(f"gridB gate FAILED: {json.dumps(gate)}")

    n_rows = len(prompts)
    scores = np.full((n_rows, n_layers, len(cells)), np.nan, dtype=np.float64)
    gen_ids = np.zeros(n_rows, dtype=np.int64)
    commit_p = np.zeros(n_rows, dtype=np.float64)
    yes_no = np.zeros(n_rows, dtype=bool)
    attn_dtype_seen = set()
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        ids = _chat_ids(tok, prompt)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=model.device),
                        output_attentions=True, use_cache=False)
        if out.attentions is None or len(out.attentions) != n_layers:
            got = None if out.attentions is None else len(out.attentions)
            raise RuntimeError(f"row {i}: attentions missing/short ({got}/{n_layers})")
        logits = out.logits[0, -1].float().cpu().numpy().astype(np.float64)
        p = np.exp(logits - logits.max()); p /= p.sum()
        gid = int(np.argmax(p))
        gen_ids[i] = gid
        commit_p[i] = float(p[gid])
        yes_no[i] = tok.decode([gid], skip_special_tokens=True).strip().upper() in ("YES", "NO")
        T = len(ids)
        for li in range(n_layers):
            att = out.attentions[li]
            if att is None or len(att.shape) != 4:
                raise RuntimeError(f"row {i} block {li}: bad attention tensor")
            if int(att.shape[1]) != n_heads or int(att.shape[-1]) != T or int(att.shape[-2]) != T:
                raise RuntimeError(f"row {i} block {li}: {tuple(att.shape)} != [1,{n_heads},{T},{T}]")
            attn_dtype_seen.add(str(att.dtype))
            w_last = att[0, :, -1, :].float().cpu().numpy()
            caps = {"final": [w_last]}
            nkv_map = {"final": n_kv}
            for k, cell in enumerate(cells):
                sc = SEAL._compute_attention_score(cell, caps, nkv_map, v_norm_captures=None)
                if sc is None or not np.isfinite(sc):
                    raise RuntimeError(f"row {i} block {li} cell {cell}: bad score {sc!r}")
                scores[i, li, k] = float(sc)
        del att, out
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if i % 25 == 0:
            print(f"[gridB] {i}/{n_rows} yes_no_so_far={int(yes_no[:i + 1].sum())}", flush=True)

    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite scores after loop")
    frac_yn = float(yes_no.mean())
    print(f"[gridB] done in {time.time() - t0:.0f}s: yes_no_rate={frac_yn:.2%}", flush=True)
    if frac_yn < 0.5:
        raise RuntimeError(f"YES/NO commit rate {frac_yn:.2%} < 50% — refusing to save")

    import bitsandbytes
    import transformers
    devmap = {}
    for n_, p_ in model.named_parameters():
        devmap.setdefault(str(p_.device), 0)
        devmap[str(p_.device)] += 1
    peak = {f"cuda:{d}": int(torch.cuda.max_memory_allocated(d))
            for d in range(torch.cuda.device_count())}
    meta = {
        "schema": SCHEMA, "model": conf["model_id"], "task": task,
        "model_key": conf["_key"], "revision_pinned": conf["revision"],
        "precision": precision, "n_layers": n_layers, "n_heads": n_heads,
        "n_kv_heads": n_kv, "n_rows": n_rows, "metrics": list(DEPTH_CELL_DETAILS),
        "backend": "modal-torch", "comparable": False,
        "data_path": data, "data_sha256": data_sha, "data_hash_loader": dh,
        "yes_no_commit_rate": round(frac_yn, 4), "gate": gate, "prereg": PREREG,
        "capture_mode": "full_eager_retention",
        "attn_dtypes_seen": sorted(attn_dtype_seen),
        "load_notes": load_notes,
        "wrapper_class": type(model).__name__,
        "device_param_counts": devmap, "cuda_peak_alloc_bytes": peak,
        "prompt_manifest_file_sha256": man_sha,
        "prompt_row_hashes": frozen_rows,  # per-row sha256 (round-5 MINOR)
        "provenance": {
            "hf_model_revision": getattr(model.config, "_commit_hash", None),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "bitsandbytes": bitsandbytes.__version__, "numpy": np.__version__,
            "pinned_image": PINNED,
            "seal_pri_calibrator_sha256": hashlib.sha256(
                open(f"{SEAL_REMOTE}/pri_calibrator.py", "rb").read()).hexdigest(),
            "depth_max_tokens": DEPTH_MAX_TOKENS,
        },
    }
    outdir = f"{MNT}/{OUT_DIR}/{task}"
    os.makedirs(outdir, exist_ok=True)
    # atomic writes (round-5 MINOR-3): tmp + os.replace, no partial artifacts
    npz_final = f"{outdir}/{_slug(conf)}.depth.npz"
    np.savez(npz_final + ".tmp.npz",
             scores=scores, labels=np.asarray(labels, dtype=np.int64),
             sample_idx=np.arange(n_rows, dtype=np.int64),
             gen_token_ids=gen_ids, commit_p=commit_p, yes_no=yes_no.astype(np.int64),
             metrics=json.dumps(list(DEPTH_CELL_DETAILS)), meta=json.dumps(meta))
    os.replace(npz_final + ".tmp.npz", npz_final)
    gpath = f"{outdir}/{_slug(conf)}.gates.json"
    with open(gpath + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(gpath + ".tmp", gpath)
    return {"model": conf["model_id"], "task": task, "precision": precision,
            "n_layers": n_layers, "yes_no_commit_rate": round(frac_yn, 4),
            "gate_cos": [r["oproj_recon_cos"] for r in gate["rows"]],
            "out": f"{outdir}/{_slug(conf)}.depth.npz"}


def _extract_body(model_key: str, task: str, gpu_label: str):
    conf = dict(REGISTRY[model_key]); conf["_key"] = model_key
    # TERMINAL-STATE IMMUTABILITY (round-5 MAJOR-4): checked BEFORE the try block —
    # an existing terminal status (ok OR aborted) or an existing npz may never be
    # overwritten by a rerun; a rescue would need a preregistered amendment.
    sp = _status_path(task, conf)
    npz = f"{MNT}/{OUT_DIR}/{task}/{_slug(conf)}.depth.npz"
    if os.path.exists(sp):
        raise SystemExit(f"TERMINAL STATUS EXISTS ({sp}) — reruns are forbidden; "
                         f"an aborted cell stays aborted per the prereg")
    if os.path.exists(npz):
        # Round-6 M4 residual: an atomic npz with NO status means the process was
        # killed in the write window between npz os.replace() and status creation.
        # The npz is complete (atomicity) and the scorer fully re-validates it, so
        # RECOVER THE STATUS ONLY — no re-extraction, no rescue.
        # NOTE: the ORIGINAL run's hardware lives in the npz meta (gpu_names /
        # device map); the recovery invocation's shape is recorded separately and
        # never attributed to the artifact (round-7). Recovery needs no Medium
        # GPU check — the npz was produced by a run that passed all checks.
        _write_status(task, conf, "ok",
                      reason="recovered: complete npz present, status missing "
                             "(kill inside the npz->status write window)",
                      extra={"recovered": True, "recovery_gpu_label": gpu_label,
                             "original_hardware": "see npz meta gpu_names/device map"})
        print("GRIDB_EXTRACT_RESULT ok (status recovered, no re-extraction)", flush=True)
        return {"recovered": True, "out": npz}
    if model_key == "mistral_medium_35":
        if FROZEN_MEDIUM_GPU is None:
            raise SystemExit("Medium GPU shape not frozen — smoke + freeze commit "
                             "must precede extraction")
        if gpu_label != FROZEN_MEDIUM_GPU:
            raise SystemExit(f"Medium must run on frozen GPU shape "
                             f"{FROZEN_MEDIUM_GPU!r}, not {gpu_label!r}")
    if model_key == "llama31_405b" and gpu_label != "A100-80GB:8":
        raise SystemExit(f"405B stretch must run on its registered default "
                         f"A100-80GB:8, not {gpu_label!r} (round-10: enforced "
                         f"in-body, not just via _fn_for)")
    try:
        res = _run_extract(conf, task)
        res["gpu_label"] = gpu_label
        _write_status(task, conf, "ok", extra={"result": res, "gpu_label": gpu_label})
        print("GRIDB_EXTRACT_RESULT ok", flush=True)
        return res
    except Exception as e:  # noqa: BLE001
        _write_status(task, conf, "aborted",
                      reason=f"{type(e).__name__}: {e}", extra={"gpu_label": gpu_label})
        print(f"GRIDB_EXTRACT_RESULT aborted: {e}", flush=True)
        raise


def _smoke_body(model_key: str):
    """Gates-only smoke: load, descriptor, 2-row o_proj/YES-NO gate, tokenize all
    200 rows x both tasks (single-BOS + budget), write prompt manifests, Mistral
    cross-check where registered. NEVER computes a metric."""
    import hashlib
    import json
    import sys

    if PKG_REMOTE not in sys.path:
        sys.path.insert(0, PKG_REMOTE)
    import cc_modal_app as BASE
    _SEAL, _PIPE, CR, _CC, _io, _tlm = BASE._import_seal()
    _patch_mlx_stub()

    conf = dict(REGISTRY[model_key]); conf["_key"] = model_key
    tok, model, precision, load_notes = _load_model(conf)
    desc = _resolve_descriptor(model, conf)
    load_notes["decoder_linear_types"] = _enforce_decoder_quantizers(desc, conf)
    report = {"model_key": model_key, "model_id": conf["model_id"],
              "revision": conf["revision"], "precision": precision,
              "load_notes": load_notes, "wrapper_class": type(model).__name__,
              "n_layers": desc["n_layers"], "n_heads": desc["n_heads"],
              "n_kv": desc["n_kv"], "tasks": {}}

    os.makedirs(f"{MNT}/{OUT_DIR}/manifests", exist_ok=True)
    # CELL-GRANULAR fail-closed (the registered unit is the CELL): each task's
    # gates are evaluated independently; a failing task gets NO manifest (its
    # extraction will abort into a registered failure) while the other task may
    # still pass and receive its manifest. The gate RULES are unchanged from the
    # registered instrument — only smoke's granularity matches the prereg's unit.
    failed_tasks = {}
    for task in TASKS:
        try:
            data = f"{MNT}/data/{task}_n200.jsonl"
            data_sha = hashlib.sha256(open(data, "rb").read()).hexdigest()
            if data_sha != FROZEN_DATA_SHA256[task]:
                raise RuntimeError(f"DATA HASH MISMATCH {task}")
            prompts, _labels, _dh = CR._load_calibration_jsonl(data)
            rows = _prompt_manifest(tok, prompts, conf["model_id"])
            xcheck = None
            if conf["mistral_xcheck"]:
                n_bad, first = _mistral_common_xcheck(tok, prompts, conf["model_id"],
                                                      conf["revision"])
                xcheck = {"mismatches": n_bad, "first": first}
                if n_bad:
                    raise RuntimeError(f"mistral-common cross-check FAILED: {xcheck}")
            cos, is_yn, commit, _ = _oproj_cos_gate_b(model, tok, desc, prompts[0])
            cos2, is_yn2, commit2, _ = _oproj_cos_gate_b(model, tok, desc, prompts[1])
            task_rep = {"gate": [
                {"row": 0, "cos": round(cos, 5), "yes_no": bool(is_yn), "commit": commit},
                {"row": 1, "cos": round(cos2, 5), "yes_no": bool(is_yn2), "commit": commit2}],
                "n_rows": len(rows), "max_tokens_ok": True, "mistral_xcheck": xcheck}
            report["tasks"][task] = task_rep
            # FAIL-CLOSED gates (round-5 MAJOR-2), enforced on raw values
            if not (cos >= 0.999 and cos2 >= 0.999):
                raise RuntimeError(f"SMOKE GATE FAIL {model_key}/{task}: o_proj cos "
                                   f"{cos:.7f}/{cos2:.7f} < 0.999")
            if not (is_yn and is_yn2):
                raise RuntimeError(f"SMOKE GATE FAIL {model_key}/{task}: commit not "
                                   f"YES/NO ({commit}/{commit2})")
            man_path = f"{MNT}/{OUT_DIR}/manifests/{_slug(conf)}.{task}.prompts.json"
            frozen_entry = FROZEN_MANIFEST_SHA256.get(f"{_slug(conf)}.{task}")
            if (os.path.exists(man_path) and frozen_entry
                    and frozen_entry != _MANIFEST_PENDING):
                raise RuntimeError(f"manifest {man_path} is FROZEN — post-freeze "
                                   f"smoke may not replace it")
            with open(man_path + ".tmp", "w") as f:
                json.dump({"model_id": conf["model_id"], "revision": conf["revision"],
                           "task": task, "rows": rows, "schema": SCHEMA}, f)
            os.replace(man_path + ".tmp", man_path)
            task_rep["manifest_sha256"] = hashlib.sha256(
                open(man_path, "rb").read()).hexdigest()
            print(f"[smoke] {model_key} {task}: cos={cos:.5f}/{cos2:.5f} "
                  f"yes_no={is_yn}/{is_yn2} commit={commit}/{commit2} "
                  f"manifest_sha={task_rep['manifest_sha256'][:16]}…", flush=True)
        except Exception as e:  # noqa: BLE001 — recorded per task, no manifest
            failed_tasks[task] = f"{type(e).__name__}: {e}"
            report["tasks"].setdefault(task, {})["smoke_failure"] = failed_tasks[task]
            print(f"[smoke] {model_key} {task}: FAILED — {e}", flush=True)
    report["failed_tasks"] = failed_tasks
    spath = f"{MNT}/{OUT_DIR}/manifests/{_slug(conf)}.smoke.json"
    with open(spath + ".tmp", "w") as f:
        json.dump(report, f, indent=2)
    os.replace(spath + ".tmp", spath)
    vol.commit()
    print("SMOKE_RESULT\n" + json.dumps(report, indent=2), flush=True)
    if failed_tasks:
        raise RuntimeError(f"SMOKE completed with failed task(s) {list(failed_tasks)} "
                           f"— report + passing manifests written; failing cells "
                           f"will abort at extraction into registered failures")
    return report


# ── Modal function per GPU shape (Modal binds gpu at decoration time) ───────────
_COMMON = dict(image=image, volumes={MNT: vol}, secrets=[hf_secret])


@app.function(gpu="A100-80GB", timeout=60 * 60 * 6, **_COMMON)
def extract_a100(model_key: str, task: str):
    return _extract_body(model_key, task, "A100-80GB")


@app.function(gpu="A100-80GB:4", timeout=60 * 60 * 12, **_COMMON)
def extract_a100x4(model_key: str, task: str):
    return _extract_body(model_key, task, "A100-80GB:4")


@app.function(gpu="H200:2", timeout=60 * 60 * 12, **_COMMON)
def extract_h200x2(model_key: str, task: str):
    return _extract_body(model_key, task, "H200:2")


@app.function(gpu="A100-80GB:8", timeout=60 * 60 * 12, **_COMMON)
def extract_a100x8(model_key: str, task: str):
    return _extract_body(model_key, task, "A100-80GB:8")


@app.function(gpu="A100-80GB", timeout=60 * 60 * 4, **_COMMON)
def smoke_a100(model_key: str):
    return _smoke_body(model_key)


@app.function(gpu="A100-80GB:4", timeout=60 * 60 * 6, **_COMMON)
def smoke_a100x4(model_key: str):
    return _smoke_body(model_key)


@app.function(gpu="H200:2", timeout=60 * 60 * 6, **_COMMON)
def smoke_h200x2(model_key: str):
    return _smoke_body(model_key)


@app.function(gpu="A100-80GB:8", timeout=60 * 60 * 6, **_COMMON)
def smoke_a100x8(model_key: str):
    return _smoke_body(model_key)


def _fn_for(conf, kind, gpu_override=None):
    gpu = gpu_override or conf["gpu"]
    table = {("A100-80GB", "extract"): extract_a100,
             ("A100-80GB:4", "extract"): extract_a100x4,
             ("H200:2", "extract"): extract_h200x2,
             ("A100-80GB:8", "extract"): extract_a100x8,
             ("A100-80GB", "smoke"): smoke_a100,
             ("A100-80GB:4", "smoke"): smoke_a100x4,
             ("H200:2", "smoke"): smoke_h200x2,
             ("A100-80GB:8", "smoke"): smoke_a100x8}
    return table[(gpu, kind)]


@app.local_entrypoint()
def smoke(model_key: str, gpu: str = ""):
    conf = REGISTRY[model_key]
    print(_fn_for(conf, "smoke", gpu or None).remote(model_key))


@app.local_entrypoint()
def extract(model_key: str, task: str, gpu: str = ""):
    conf = REGISTRY[model_key]
    # GPU override for OUTCOME runs is restricted to the registered Medium
    # fallback only (round-5: unrestricted override would leave hardware
    # unrecorded/unregistered).
    if gpu and not (model_key == "mistral_medium_35" and gpu == "H200:2"):
        raise SystemExit(f"gpu override {gpu!r} not registered for {model_key} — "
                         f"only mistral_medium_35 may fall back to H200:2")
    print(_fn_for(conf, "extract", gpu or None).remote(model_key, task))
