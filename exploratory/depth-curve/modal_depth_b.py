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

--out-dir selects the artifact namespace (npz + gates.json + status.json) and
DEFAULTS TO THE UNREGISTERED TREE (`depth_grid_b_vnorm`). Every cell in the
registered `depth_grid_b` tree is already terminal, so an unregistered
candidate-#16 run must not target it. Prompt manifests always resolve against the
registered tree (MANIFEST_DIR) because FROZEN_MANIFEST_SHA256 pins them there.
The terminal-state immutability guard is UNCHANGED — it now guards whichever
namespace the run targets.

Additive v-norm channel (unregistered, candidate #16): every block's value-vector
norms are captured via v_proj hooks and scored into a SEPARATE `<slug>.vnorm.npz`
sidecar. The registered `<slug>.depth.npz` keeps exactly its original eight arrays.
"""
import os
from pathlib import Path

import modal

APP_NAME = "cc-depth-grid-b"
VOL_NAME = "model-cache"
MNT = "/models"
SEAL_REMOTE = "/seal"
PKG_REMOTE = "/pkg"
# ── output namespaces ───────────────────────────────────────────────────────────
# `depth_grid_b` is the REGISTERED grid-B tree. Every one of its cells is already
# TERMINAL, and _extract_body's immutability guard refuses any rerun of a cell with
# an existing status or npz. Candidate #16 is unregistered descriptive work that was
# never part of PRE_REGISTRATION_EXPANSION.md, so it writes to its OWN namespace.
#
# THIS IS NAMESPACE SEPARATION, NOT A GATE OVERRIDE. The immutability check itself
# is untouched — not weakened, not skipped, not special-cased. It now guards the
# chosen namespace, so a #16 run passes it cleanly by not colliding with a
# registered cell at all, and a second #16 run of the same cell is still refused.
OUT_DIR_REGISTERED = "depth_grid_b"
OUT_DIR_DEFAULT = "depth_grid_b_vnorm"
# Prompt manifests are READ-ONLY INPUTS pinned by FROZEN_MANIFEST_SHA256 and stay on
# the registered tree regardless of out_dir. Redirecting them would break the frozen
# manifest verification and abort every cell on "missing smoke prompt manifest".
MANIFEST_DIR = OUT_DIR_REGISTERED
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

# ── ADDITIVE v-norm channel (added 2026-09-01; candidate #16 gating build step) ──
# The registered channel above is UNCHANGED: `scores` keeps shape [n_rows, n_layers, 4],
# `metrics` keeps the frozen 4-list, and the four cells are still computed with
# v_norm_captures=None on the untouched code path. This adds a SEPARATE, PARALLEL
# output channel `v_norm_scores` of shape [n_rows, n_layers, 3], written to its OWN
# `<slug>.vnorm.npz` sidecar, so the third ACE attention instrument finally gets
# per-layer depth coverage.
#
# HARD INVARIANT (sharper in grid B than in grid A): nothing in the additive path may
# raise, and nothing additive is serialized by the registered `np.savez` call.
# `_extract_body` turns ANY exception from `_run_extract` into a PERMANENT `aborted`
# terminal status that the prereg forbids re-running — so an additive bug must never
# be able to burn a cell. Every failure mode degrades to NaN in `v_norm_scores` plus
# a diagnostic counter.
#
# ORDER IS THE SEALED ORDER: pri_calibrator.ATTENTION_METRICS_V_NORMS is
# ("v_norm_bos", "v_norm_max", "v_norm_lastq_weighted"), and the trailing axis of
# `v_norm_scores` follows it exactly. `_run_extract` re-derives this tuple from the
# sealed constant at runtime and disables the channel on any mismatch, so a typo in
# a detail key cannot silently produce an all-NaN column.
# CONSUMERS MUST INDEX BY NAME via the `v_norm_metrics` array, never by position:
# index 0 is v_norm_bos, NOT the headline lastq_weighted metric.
#
# All three reduce the SAME captured (n_kv, T) norms, so adding the two extra
# metrics costs no extra capture and no extra forward — only two more calls to the
# sealed scorer per (row, block). Confirmed against
# diagnose_inter_head_disagreement._mean_v_norm_bos (:491) and _mean_v_norm_max
# (:503): both take `v_norms` alone.
ADDITIVE_DEPTH_CELL_DETAILS = (
    "final_v_norm_bos",
    "final_v_norm_max",
    "final_v_norm_lastq_weighted",
)

# `v_norm_capture_mode` in REGISTERED meta is drawn from this closed set of two
# module-level literals and nothing else. Free-form disable/error text (which is
# runtime-derived, unbounded, and can embed an arbitrary exception message) goes
# ONLY to the sidecar's `additive_capture_diagnostics.reason`. This keeps the
# registered meta's value schema fixed and its serialization risk nil.
_VNORM_MODE_ON = "per_block_v_proj_hook"
_VNORM_MODE_OFF = "disabled"

# ── INERT SENTINELS, allocated ONCE at import ───────────────────────────────────
# Every additive fallback is a REBIND to one of these, never a fresh literal.
# `MemoryError` is an `Exception`, so an unguarded `{}` or `[]` in a failure
# handler can raise from inside the handler and escape — in grid B that would
# permanently burn a claimed cell. Constructing the fallback is the thing that
# fails, so the fallback must already exist. NOTHING may mutate these.
_VNORM_EMPTY_MAP = {}          # read-only stand-in for a per-row capture dict
_VNORM_EMPTY_SEQ = ()          # read-only stand-in for add_cells / handle lists
_VNORM_DIAG_FALLBACK = {"enabled": False, "mode": _VNORM_MODE_OFF,
                        "diagnostics_error": "diagnostics unavailable"}
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


def _status_path(task, conf, out_dir=OUT_DIR_DEFAULT):
    return f"{MNT}/{out_dir}/{task}/{_slug(conf)}.status.json"


def _write_status(task, conf, status, reason="", extra=None, out_dir=OUT_DIR_DEFAULT):
    import json
    d = f"{MNT}/{out_dir}/{task}"
    os.makedirs(d, exist_ok=True)
    payload = {"status": status, "reason": reason, "model_key": conf["_key"],
               "model_id": conf["model_id"], "revision": conf["revision"],
               "task": task, "schema": SCHEMA, "out_dir": out_dir}
    if extra:
        payload.update(extra)
    # MUST pass out_dir through: the status file has to land in the SAME namespace
    # the immutability guard checked, or a rerun could slip past it.
    sp = _status_path(task, conf, out_dir)
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


# ── additive v-norm capture helpers (never raise into the registered path) ──────
# Byte-identical to the same block in modal_depth.py (grid A) — one instrument,
# two extractors. Keep them in sync.
def _vnorm_new_state():
    """Mutable state for the additive channel: GPU norm store, per-block fire
    counts, hook errors, and scoring diagnostics.

    Counter semantics: the per-(row, block) CAPTURE checks are counted once per
    block; only `ok` / `nonfinite` / `score_error` are per-(row, block, metric).
    `missing` (hook never fired), `multi_fire` (fired 2+ times) and `hook_failed`
    (fired but the body errored) are DISTINCT buckets — the previous version
    tested `vn is None` before the fire count, so a double-fire that also failed
    capture was misreported as `missing`.
    """
    return {
        "store": {},        # block index -> torch tensor [n_kv, T] fp32 (GPU, current row)
        "counts": {},       # block index -> hook fire count for the current row
        "errs": {},         # block index -> first hook error string (whole run)
        "global_errs": [],  # non-block-scoped failures (drain / removal / sidecar)
        "diag": {
            # ── CAPTURE outcome, exactly ONE per (row, block) ─────────────────
            # Mutually exclusive by construction: _vnorm_block_norms increments
            # exactly one of these and returns immediately, and the call site
            # never re-enters the capture phase for the same block. So
            # blocks_ok + missing + multi_fire + hook_failed + shape_mismatch
            # + capture_error == n_rows * n_layers whenever the channel is on.
            "blocks_ok": 0, "missing": 0, "multi_fire": 0, "hook_failed": 0,
            "shape_mismatch": 0, "capture_error": 0,
            # ── POST-CAPTURE failure, counted SEPARATELY ──────────────────────
            # A block can be captured successfully and still fail while scoring
            # or assigning. That is not a capture failure and must not be added
            # to n_blocks_failed, or captured + failed would exceed expected.
            "post_capture_error": 0,
            # ── per-(row, block, METRIC) ──────────────────────────────────────
            "ok": 0, "nonfinite": 0, "score_error": 0,
            # ── sidecar row-identity mirror ───────────────────────────────────
            "mirror_error": 0,
            "first_capture_error": None, "first_post_capture_error": None,
            "first_score_error": None, "first_mirror_error": None,
        },
    }


def _vnorm_exc_text(exc):
    """Format an exception with a CONSTANT fallback.

    `f"{exc}"` calls the exception's own `__str__`, which is arbitrary code and
    can itself raise (or raise again while being formatted). Every additive
    error string in this file is produced here, so a hostile `__str__` degrades
    to a constant instead of escaping into the registered path.
    """
    try:
        return f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001
        try:
            return str(type(exc).__name__)
        except Exception:  # noqa: BLE001
            return "<unformattable exception>"


def _vnorm_note(state, counter=None, first_key=None, exc=None, text=None,
                block=None):
    """The ONE diagnostic sink for the additive path. Structurally non-raising.

    Every recording site routes through here: hook-body failures, drain
    failures, capture failures, post-capture failures, scoring failures, mirror
    failures, sidecar failures and hook-removal failures. Each of the three
    steps below (format / count / store) is independently guarded with a
    constant fallback, so no combination of a broken exception, a mangled state
    dict or a full container can produce an exception at a call site.
    """
    try:
        msg = text if text is not None else _vnorm_exc_text(exc)
        msg = str(msg)[:400]
    except Exception:  # noqa: BLE001
        msg = "<unformattable exception>"
    try:
        d = state["diag"]
        if counter is not None and counter in d:
            d[counter] += 1
        if first_key is not None and d.get(first_key) is None:
            d[first_key] = msg
    except Exception:  # noqa: BLE001
        pass
    try:
        if block is not None:
            state["errs"].setdefault(block, msg)
        elif first_key is None:
            state["global_errs"].append(msg)
    except Exception:  # noqa: BLE001
        pass


def _claim_cell_exclusive(json, claim_path, payload):
    """Atomically reserve one (out_dir, task, model) cell.

    `os.open(..., O_CREAT | O_EXCL)` is a single atomic syscall: exactly one of
    two concurrent invocations can create the file, so this closes the
    check-then-write race that `os.path.exists(npz)` alone cannot. The claim is
    STICKY — it is never removed on failure, matching the lane's fail-closed
    discipline, and it is rewritten to state="complete" once the registered
    artifact is on disk.

    Returns (ok, reason). Callers must treat ok=False as fatal BEFORE loading a
    model; this is deliberately not fail-soft, because its whole purpose is to
    stop a second writer.
    """
    fd = None
    try:
        os.makedirs(os.path.dirname(claim_path), exist_ok=True)
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        existing = "<unreadable>"
        try:
            with open(claim_path) as f:
                existing = f.read()[:600]
        except Exception:  # noqa: BLE001
            pass
        return False, existing
    except Exception as exc:  # noqa: BLE001
        return False, f"claim create failed: {_vnorm_exc_text(exc)}"
    try:
        os.write(fd, json.dumps(payload, indent=2).encode("utf-8"))
    except Exception:  # noqa: BLE001
        pass  # the claim's EXISTENCE is the lock; its contents are advisory
    finally:
        try:
            os.close(fd)
        except Exception:  # noqa: BLE001
            pass
    return True, ""


def _claim_mark_complete(json, claim_path, payload):
    """Rewrite a held claim to state="complete". Best-effort: by the time this
    runs the registered artifact already exists, and the npz is the real
    completion evidence, so a failure here is recorded and ignored."""
    try:
        with open(claim_path + ".tmp", "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(claim_path + ".tmp", claim_path)
        return True
    except Exception:  # noqa: BLE001
        return False


def _vnorm_install_hooks(layers, n_kv, head_dim_expected, state):
    """Install observational forward hooks on EVERY block's `self_attn.v_proj`.

    Each hook reshapes the v_proj output `[1, T, n_kv*head_dim]` to
    `[T, n_kv, head_dim]`, casts to FP32 *before* squaring (sealed
    `_capture_value_norms` rationale: bf16/fp16 exponent range clips large value
    vectors), takes the L2 norm over `head_dim`, and stores ONLY the resulting
    `(n_kv, T)` tensor — the full V tensor is never retained. Norms stay on the
    GPU (~n_kv*T*4 bytes per block, ~3.6 MB total at 126 blocks / 8 KV / T=900)
    and are drained to numpy once per forward, so no hook forces a mid-forward
    GPU->CPU sync.

    The `[T, n_kv, head_dim]` (head-major) reshape is the SAME one the o_proj
    faithfulness gate already uses to rebuild the attention output from V — grid A
    via modal_app._Capture, grid B via _oproj_cos_gate_b's v_out.view(T, n_kv,
    d_head). That reconstruction is gated at cos >= 0.999 on rows 0-1 of every
    cell, so a wrong V layout would fail the gate before extraction ever starts.

    Returns `(handles, reason)`. `reason` is "" on success; a non-empty reason
    means NO hooks were installed and the additive channel must stay disabled.
    This function performs its structural pre-flight BEFORE installing anything
    and NEVER raises — a structural surprise disables the additive channel
    rather than aborting a registered extraction.
    """
    try:
        import torch  # bound for the closure below; inside the guard on purpose

        for li, blk in enumerate(layers):
            attn = getattr(blk, "self_attn", None)
            if attn is None:
                return [], f"block {li}: no .self_attn"
            vp = getattr(attn, "v_proj", None)
            if vp is None:
                return [], (f"block {li}: self_attn has no .v_proj "
                            f"(fused-QKV or non-standard attention layout)")
            # Mirror modal_app._Capture's refusal: raw v_proj must BE the value
            # tensor attention consumes (false for value-norm / K==V families).
            if hasattr(attn, "v_norm") or getattr(attn, "use_k_eq_v", False):
                return [], (f"block {li}: attention has v_norm/use_k_eq_v — raw "
                            f"v_proj capture would be unfaithful")
            of = getattr(vp, "out_features", None)
            if not isinstance(of, int) or of <= 0:
                return [], f"block {li}: v_proj.out_features unusable ({of!r})"
            if of % n_kv != 0:
                return [], (f"block {li}: v_proj out_features {of} not divisible "
                            f"by n_kv={n_kv}")
            if head_dim_expected is not None and (of // n_kv) != head_dim_expected:
                return [], (f"block {li}: derived head_dim {of // n_kv} != config "
                            f"head_dim {head_dim_expected}")
        store, counts = state["store"], state["counts"]
    except Exception as exc:  # noqa: BLE001
        return [], f"preflight {_vnorm_exc_text(exc)}"

    def _make(li):
        def _hook(_mod, _inp, out):
            # EVERY statement is inside the try. A hook that raises does so inside
            # the registered forward, where grid B's _extract_body would convert it
            # into a PERMANENT `aborted` terminal status. The fire count is the
            # first statement inside the try, so double-fire detection survives:
            # a block whose v_proj fires twice in one forward is ambiguous data and
            # must be dropped, not averaged.
            try:
                counts[li] = counts.get(li, 0) + 1
                v = out[0] if isinstance(out, tuple) else out
                if v.ndim != 3 or int(v.shape[0]) != 1:
                    raise RuntimeError(
                        f"v_proj output {tuple(v.shape)} != [1, T, n_kv*head_dim]")
                t_len, width = int(v.shape[1]), int(v.shape[2])
                if width % n_kv != 0:
                    raise RuntimeError(
                        f"v_proj width {width} not divisible by n_kv={n_kv}")
                hd = width // n_kv
                x = v.detach().float().reshape(t_len, n_kv, hd)
                nrm = torch.linalg.vector_norm(x, dim=-1)      # [T, n_kv] fp32
                store[li] = nrm.transpose(0, 1).contiguous()   # [n_kv, T]
                del x, nrm
            except Exception as exc:  # noqa: BLE001 — must never perturb the forward
                # Through the sink, never a bare setdefault: formatting `exc`
                # here would be the last unguarded statement inside the
                # registered forward.
                _vnorm_note(state, block=li, exc=exc)
        return _hook

    handles = []
    try:
        for li, blk in enumerate(layers):
            handles.append(blk.self_attn.v_proj.register_forward_hook(_make(li)))
    except Exception as exc:  # noqa: BLE001
        _vnorm_remove_hooks(handles, state)
        return [], f"register {_vnorm_exc_text(exc)}"
    return handles, ""


def _vnorm_remove_hooks(handles, state):
    """Best-effort hook removal. A RemovableHandle that refuses to detach must not
    take a finished extraction down with it."""
    try:
        for h in handles:
            try:
                h.remove()
            except Exception as exc:  # noqa: BLE001
                _vnorm_note(state, exc=exc)
    except Exception:  # noqa: BLE001
        pass


def _vnorm_row_reset(state):
    """Drop the previous row's captures so a hook that fails to fire is detected
    as missing rather than silently reusing a stale row (cross-row bleed)."""
    try:
        state["store"].clear()
        state["counts"].clear()
    except Exception:  # noqa: BLE001
        pass


def _vnorm_drain(state):
    """Move this forward's per-block norms to CPU numpy and release the GPU
    buffers. Returns {block index: np.ndarray (n_kv, T) float32}.

    Transfers are BATCHED PER DEVICE: every block on one device has the same
    (n_kv, T) shape within a row, so they stack into a single D2H copy. The naive
    per-block version issued n_layers copies per row (25,200 for a 126-block model
    at 200 rows) instead of one per device per row (200, or 1,600 on 8xA100).
    Falls back to per-block copies for a device whose stack fails. Never raises —
    including under MemoryError: `out = {}` is an ALLOCATION and therefore lives
    inside the try, with the import-time `_VNORM_EMPTY_MAP` as the return-path
    fallback so no failure handler ever has to allocate.
    """
    out = None
    store = None
    try:
        import torch

        out = {}
        store = state["store"]
        by_dev = {}
        for li, t in store.items():
            by_dev.setdefault(str(getattr(t, "device", "cpu")), []).append(li)
        for dev, lis in by_dev.items():
            try:
                stacked = torch.stack([store[li] for li in lis], dim=0).cpu().numpy()
                for j, li in enumerate(lis):
                    out[li] = stacked[j]
            except Exception as exc:  # noqa: BLE001 — ragged shapes / OOM on this device
                _vnorm_note(state, exc=exc,
                            text=f"drain[{dev}] {_vnorm_exc_text(exc)} "
                                 f"(fell back per-block)")
                for li in lis:
                    try:
                        out[li] = store[li].cpu().numpy()
                    except Exception as exc2:  # noqa: BLE001
                        _vnorm_note(state, block=li, exc=exc2)
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, exc=exc)
    finally:
        try:
            if store is not None:
                store.clear()
        except Exception:  # noqa: BLE001
            pass
    return out if out is not None else _VNORM_EMPTY_MAP


def _vnorm_block_norms(v_by_block, li, n_kv, T, state):
    """Validate ONE block's captured norms, once per (row, block). Returns the
    (n_kv, T) array or None; never raises.

    The fire count is tested BEFORE `vn is None` so the three failure modes stay
    distinct: never-fired (`missing`), fired-twice (`multi_fire`), fired-but-body-
    errored (`hook_failed`).
    """
    try:
        diag = state["diag"]
        c = int(state["counts"].get(li, 0))
        if c == 0:
            diag["missing"] += 1
            return None
        if c != 1:
            diag["multi_fire"] += 1
            return None
        vn = v_by_block.get(li)
        if vn is None:
            diag["hook_failed"] += 1
            return None
        if vn.ndim != 2 or int(vn.shape[0]) != n_kv or int(vn.shape[1]) != T:
            diag["shape_mismatch"] += 1
            return None
        diag["blocks_ok"] += 1
        return vn
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="capture_error",
                    first_key="first_capture_error", exc=exc)
        return None


def _vnorm_score_cell(SEAL, np, cell, caps, nkv_map, vn, state):
    """Score one additive v-norm cell. Returns a float or None; never raises.

    `caps` is the SAME weights dict the registered cells were scored from, so the
    additive cell reads exactly the same attention row. `v_norm_bos` and
    `v_norm_max` ignore `caps` entirely (they reduce v_norms alone) but the sealed
    kernel still fetches `captures[layer][step]` before dispatching on metric, so
    the weights dict must be present for all three.
    """
    try:
        diag = state["diag"]
        sc = SEAL._compute_attention_score(cell, caps, nkv_map,
                                           v_norm_captures={"final": [vn]})
        if sc is None or not np.isfinite(sc):
            diag["nonfinite"] += 1
            return None
        diag["ok"] += 1
        return float(sc)
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="score_error",
                    first_key="first_score_error", exc=exc)
        return None


def _vnorm_meta(state, mode, reason, enabled, n_rows, n_layers, n_metrics):
    """Build the additive diagnostics block. Returns only JSON primitives and
    never raises — on internal failure it returns a minimal explanatory dict.

    `mode` is the stable literal that also goes into registered meta; `reason`
    is the free-form disable/error text and lives ONLY here.
    """
    try:
        d = state["diag"]
        first_hook_err = None
        if state["errs"]:
            k = sorted(state["errs"])[0]
            first_hook_err = f"block {k}: {state['errs'][k]}"
        attempted = int(d["ok"] + d["nonfinite"] + d["score_error"])
        # CAPTURE buckets only — post_capture_error is NOT summed in here, or a
        # block could be counted as both captured and failed.
        blocks_failed = int(d["missing"] + d["multi_fire"] + d["hook_failed"]
                            + d["shape_mismatch"] + d["capture_error"])
        blocks_attempted = int(d["blocks_ok"]) + blocks_failed
        return {
            "enabled": bool(enabled),
            "hook_target": "self_attn.v_proj",
            "capture_shape": "(n_kv_heads, T) fp32 L2 norms over head_dim",
            "mode": str(mode),
            "reason": str(reason),
            # Full-coverage denominator, independent of whether the channel ran.
            # When the channel is disabled every counter below is 0 and this stays
            # at full size, so 0/N reads unambiguously as "captured nothing".
            "n_metric_cells_expected": int(n_rows) * int(n_layers) * int(n_metrics),
            "n_metric_cells_attempted": attempted,
            "n_scored": int(d["ok"]),
            "n_nonfinite": int(d["nonfinite"]),
            "n_score_errors": int(d["score_error"]),
            "n_blocks_expected": int(n_rows) * int(n_layers),
            # INVARIANT a consumer can check: when enabled,
            #   n_blocks_attempted == n_blocks_expected
            #   n_blocks_captured + n_blocks_failed == n_blocks_attempted
            "n_blocks_attempted": blocks_attempted,
            "n_blocks_captured": int(d["blocks_ok"]),
            "n_blocks_failed": blocks_failed,
            "n_missing": int(d["missing"]),
            "n_multi_fire": int(d["multi_fire"]),
            "n_hook_failed": int(d["hook_failed"]),
            "n_shape_mismatch": int(d["shape_mismatch"]),
            "n_capture_errors": int(d["capture_error"]),
            # Disjoint from every capture bucket above; these blocks WERE
            # captured and then failed downstream.
            "n_post_capture_errors": int(d["post_capture_error"]),
            "n_mirror_errors": int(d["mirror_error"]),
            "n_blocks_with_hook_errors": len(state["errs"]),
            "first_hook_error": first_hook_err,
            "first_capture_error": d["first_capture_error"],
            "first_post_capture_error": d["first_post_capture_error"],
            "first_score_error": d["first_score_error"],
            "first_mirror_error": d["first_mirror_error"],
            "global_errors": [str(e) for e in state["global_errs"][:5]],
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": bool(enabled), "mode": str(mode),
                "diagnostics_error": _vnorm_exc_text(exc)}


def _vnorm_build_mirror(np, state, labels, gen_token_ids, commit_p, yes_no):
    """Copy the registered row-identity columns for the sidecar (MK decision,
    2026-09-01). Never raises: a column that cannot be built is omitted and
    recorded, and the sidecar is written without it.

    These four columns are already in memory at sidecar-write time, so this adds
    no compute and no new failure mode. Dtypes deliberately match the registered
    write EXACTLY (`labels`/`gen_token_ids`/`yes_no` int64, `commit_p` float64)
    so a consumer can assert bitwise equality against the registered npz.

    Purpose: `sample_idx` is `arange(200)` in every banked file and therefore
    cannot detect a row permutation — a guard that fires everywhere and proves
    nothing. Within a cell these four columns ARE a content fingerprint.
    """
    out = None
    try:
        out = {}   # allocation: inside the guard, like every other one
        for name, src, dtype in (("labels", labels, "int64"),
                                 ("gen_token_ids", gen_token_ids, "int64"),
                                 ("commit_p", commit_p, "float64"),
                                 ("yes_no", yes_no, "int64")):
            try:
                out[name] = np.asarray(src).astype(dtype, copy=True)
            except Exception as exc:  # noqa: BLE001
                _vnorm_note(state, counter="mirror_error",
                            first_key="first_mirror_error", exc=exc,
                            text=f"mirror[{name}] {_vnorm_exc_text(exc)}")
    except Exception as exc:  # noqa: BLE001
        _vnorm_note(state, counter="mirror_error",
                    first_key="first_mirror_error", exc=exc)
    return out if out is not None else _VNORM_EMPTY_MAP


def _vnorm_write_sidecar(np, json, out_path, v_norm_scores, mirror, n_rows,
                         vmeta, state):
    """Serialize the additive channel to its OWN npz, in its OWN call, AFTER the
    registered artifact is already on disk.

    This is the isolation the registered write requires: the additive arrays no
    longer ride inside the same `np.savez` as `scores` / `labels` / `sample_idx` /
    `gen_token_ids` / `commit_p` / `yes_no` / `metrics` / `meta`. A malformed
    additive array can no longer destroy a complete registered artifact at the
    final write — on the 405B cell that would have been ~40 min of 8xA100 lost
    after all compute was already paid for.

    `mirror` carries the row-identity columns copied from the registered arrays
    (see _vnorm_build_mirror); it may be empty or partial, and the sidecar is
    written either way.

    Atomic (tmp + os.replace), same discipline as the registered write. Never
    raises. Returns (path_or_None, reason) where reason is ALWAYS a plain str
    produced by _vnorm_exc_text — no `f"...{exc}"` outside a guard.
    """
    tmp = None
    try:
        tmp = str(out_path) + ".tmp.npz"
        if v_norm_scores is None:
            return None, "additive array was never allocated"
        arrays = {
            "v_norm_scores": np.asarray(v_norm_scores, dtype=np.float64),
            "v_norm_metrics": json.dumps(list(ADDITIVE_DEPTH_CELL_DETAILS)),
            "sample_idx": np.arange(int(n_rows), dtype=np.int64),
            "meta": json.dumps(vmeta),
        }
        # Row-identity mirror. Merged LAST but cannot shadow the keys above.
        try:
            for k, v in (mirror or _VNORM_EMPTY_MAP).items():
                if k not in arrays:
                    arrays[k] = v
        except Exception as exc:  # noqa: BLE001
            _vnorm_note(state, counter="mirror_error",
                        first_key="first_mirror_error", exc=exc)
        np.savez(tmp, **arrays)
        os.replace(tmp, out_path)
        return out_path, ""
    except Exception as exc:  # noqa: BLE001
        reason = "<unformattable exception>"
        try:
            reason = _vnorm_exc_text(exc)
        except Exception:  # noqa: BLE001
            pass
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:  # noqa: BLE001
            pass
        _vnorm_note(state, text=f"sidecar {reason}")
        return None, reason


def _run_extract(conf, task, out_dir=OUT_DIR_DEFAULT):
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

    # ADDITIVE channel: resolve the v-norm cell(s) from the SAME sealed panel.
    # A miss here disables the additive channel; it never aborts the registered cell.
    # `v_norm_mode` is ALWAYS one of the two module literals (registered meta
    # reads it verbatim). `v_norm_reason` carries the free-form explanation and
    # is written to the SIDECAR only.
    add_cells, v_norm_mode, v_norm_reason = _VNORM_EMPTY_SEQ, _VNORM_MODE_ON, ""
    try:
        sealed_v = tuple(f"final_{m}" for m in SEAL.ATTENTION_METRICS_V_NORMS)
    except Exception as _vexc:  # noqa: BLE001
        sealed_v = None
    add_missing = [d for d in ADDITIVE_DEPTH_CELL_DETAILS if d not in by_detail]
    if add_missing:
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = f"sealed panel missing additive cells {add_missing}"
    elif sealed_v is None:
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = "could not read sealed ATTENTION_METRICS_V_NORMS"
    elif sealed_v != tuple(ADDITIVE_DEPTH_CELL_DETAILS):
        # The trailing axis of v_norm_scores is positional; if the sealed metric
        # order ever drifts from this file's literal, disable rather than emit a
        # correctly-shaped array whose columns mean something else.
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = (f"additive order {list(ADDITIVE_DEPTH_CELL_DETAILS)} "
                         f"!= sealed order {list(sealed_v)}")
    else:
        add_cells = [by_detail[d] for d in ADDITIVE_DEPTH_CELL_DETAILS]

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
    # MANIFEST_DIR, not out_dir: manifests are read-only inputs pinned by
    # FROZEN_MANIFEST_SHA256 and always live on the registered tree. Redirecting
    # them with out_dir would break the frozen verification and abort every cell.
    man_path = f"{MNT}/{MANIFEST_DIR}/manifests/{_slug(conf)}.{task}.prompts.json"
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

    # ── ADDITIVE channel: parallel array + per-block v_proj hooks ────────────────
    # Installed AFTER the faithfulness gate so the gate's own v_proj/o_proj hooks
    # run exactly as before. Observational only (hooks return None), so the
    # registered four metrics are numerically untouched.
    # PREBIND INERT SENTINELS FIRST. Every name below is bound to an object that
    # already exists (None, or an import-time constant), so these four statements
    # cannot allocate and cannot raise. The failure handlers that follow rebind to
    # these same objects — they never construct a fallback, because constructing
    # the fallback is precisely what fails under MemoryError.
    v_state = None
    v_handles = _VNORM_EMPTY_SEQ
    v_norm_scores = None
    v_by_block = _VNORM_EMPTY_MAP
    try:
        v_state = _vnorm_new_state()   # allocates several dicts — must be guarded
    except Exception:  # noqa: BLE001 — handler is REBIND-ONLY, no allocation
        add_cells = _VNORM_EMPTY_SEQ
        v_norm_mode = _VNORM_MODE_OFF
        v_norm_reason = "additive state allocation failed"
    try:
        v_norm_scores = np.full((n_rows, n_layers, len(ADDITIVE_DEPTH_CELL_DETAILS)),
                                np.nan, dtype=np.float64)
    except Exception as _vexc:  # noqa: BLE001
        add_cells, v_norm_mode = _VNORM_EMPTY_SEQ, _VNORM_MODE_OFF
        try:
            v_norm_reason = f"alloc {_vnorm_exc_text(_vexc)}"
        except Exception:  # noqa: BLE001
            v_norm_reason = "additive score array allocation failed"
    if add_cells:
        try:
            head_dim_expected = getattr(desc["text_config"], "head_dim", None)
            if not isinstance(head_dim_expected, int):
                head_dim_expected = None
            v_handles, reason = _vnorm_install_hooks(
                desc["layers"], n_kv, head_dim_expected, v_state)
            if reason:
                add_cells, v_norm_mode = _VNORM_EMPTY_SEQ, _VNORM_MODE_OFF
                v_norm_reason = str(reason)
        except Exception as _vexc:  # noqa: BLE001 — handler must not allocate
            _vnorm_remove_hooks(v_handles, v_state)
            v_handles = _VNORM_EMPTY_SEQ
            add_cells = _VNORM_EMPTY_SEQ
            v_norm_mode = _VNORM_MODE_OFF
            try:
                v_norm_reason = f"setup {_vnorm_exc_text(_vexc)}"
            except Exception:  # noqa: BLE001
                v_norm_reason = "additive hook setup failed"
    # Guarded like every other additive statement: a broken stdout must not be
    # able to abort a registered cell from inside the additive channel.
    try:
        if add_cells:
            print(f"[gridB][vnorm] additive channel ON: "
                  f"{list(ADDITIVE_DEPTH_CELL_DETAILS)} via {len(v_handles)} "
                  f"v_proj hooks", flush=True)
        else:
            print(f"[gridB][vnorm] additive channel OFF — {v_norm_reason}", flush=True)
    except Exception:  # noqa: BLE001
        pass

    for i, prompt in enumerate(prompts):
        try:
            _vnorm_row_reset(v_state)
        except Exception:  # noqa: BLE001 — caller-level guard, no allocation
            pass
        ids = _chat_ids(tok, prompt)
        with torch.no_grad():
            out = model(torch.tensor([ids], device=model.device),
                        output_attentions=True, use_cache=False)
        # Drain the additive GPU norm buffers immediately after the forward (one
        # batched transfer; no mid-forward sync). The disabled branch and the
        # failure branch both REBIND to the import-time empty map — neither
        # allocates a `{}`, which is itself a MemoryError site.
        try:
            v_by_block = _vnorm_drain(v_state) if add_cells else _VNORM_EMPTY_MAP
        except Exception:  # noqa: BLE001
            v_by_block = _VNORM_EMPTY_MAP
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
            # Additive channel. Validate the block's capture ONCE, then score every
            # additive metric off it. The outer try is a structural guarantee that
            # no additive statement can raise into the registered loop, on top of
            # the helpers already being individually non-raising.
            if add_cells:
                # PHASE 1 — capture validation. Increments exactly one capture
                # bucket. Its own outer guard uses `capture_error`, the same
                # bucket _vnorm_block_norms uses internally, so a block can never
                # land in two capture buckets.
                vn = None
                try:
                    vn = _vnorm_block_norms(v_by_block, li, n_kv, T, v_state)
                except Exception as _vexc:  # noqa: BLE001 — unreachable; see report
                    _vnorm_note(v_state, counter="capture_error",
                                first_key="first_capture_error", exc=_vexc)
                # PHASE 2 — scoring + assignment. Failures here are POST-capture
                # and are counted in a disjoint bucket, so
                # n_blocks_captured + n_blocks_failed can never exceed
                # n_blocks_expected.
                if vn is not None:
                    try:
                        for k, vcell in enumerate(add_cells):
                            vsc = _vnorm_score_cell(SEAL, np, vcell, caps,
                                                    nkv_map, vn, v_state)
                            if vsc is not None:
                                v_norm_scores[i, li, k] = vsc
                    except Exception as _vexc:  # noqa: BLE001
                        _vnorm_note(v_state, counter="post_capture_error",
                                    first_key="first_post_capture_error",
                                    exc=_vexc)
        v_by_block = _VNORM_EMPTY_MAP   # release the row's captures; no allocation
        del att, out
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if i % 25 == 0:
            print(f"[gridB] {i}/{n_rows} yes_no_so_far={int(yes_no[:i + 1].sum())}", flush=True)

    # Prebound to the import-time constant, not a fresh dict: the pre-bind itself
    # was an unguarded allocation.
    v_norm_diag = _VNORM_DIAG_FALLBACK
    try:
        _vnorm_remove_hooks(v_handles, v_state)
        v_handles = _VNORM_EMPTY_SEQ
        _vnorm_row_reset(v_state)
        v_norm_diag = _vnorm_meta(v_state, v_norm_mode, v_norm_reason,
                                  bool(add_cells), n_rows, n_layers,
                                  len(ADDITIVE_DEPTH_CELL_DETAILS))
        json.dumps(v_norm_diag)  # serialization probe — must not fail at write time
        print(f"[gridB][vnorm] {json.dumps(v_norm_diag)}", flush=True)
    except Exception as _vexc:  # noqa: BLE001
        # Try for an informative dict; fall back to the import-time constant if
        # even that allocation fails.
        try:
            v_norm_diag = {"enabled": bool(add_cells), "mode": v_norm_mode,
                           "diagnostics_error": _vnorm_exc_text(_vexc)}
        except Exception:  # noqa: BLE001
            v_norm_diag = _VNORM_DIAG_FALLBACK

    # The additive channel is deliberately EXCLUDED from this check: NaN there is a
    # recorded gap, never a reason to abort a registered cell.
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
        "out_dir": out_dir,
        # ── ADDITIVE pointers ONLY ──────────────────────────────────────────
        # A module-level list literal, a CLOSED-SET literal, and an f-string over
        # the slug. `v_norm_capture_mode` is only ever _VNORM_MODE_ON or
        # _VNORM_MODE_OFF — no runtime-derived text, no exception message, no
        # unbounded string. All diagnostics and every disable/error reason live
        # in the SIDECAR meta.
        "additive_metrics": list(ADDITIVE_DEPTH_CELL_DETAILS),
        "v_norm_capture_mode": v_norm_mode,
        "additive_sidecar": f"{_slug(conf)}.vnorm.npz",
    }
    outdir = f"{MNT}/{out_dir}/{task}"
    os.makedirs(outdir, exist_ok=True)
    # ── REGISTERED WRITE: exactly the eight arrays the frozen extractor wrote ─────
    # Nothing additive is serialized here. A malformed additive array can no longer
    # destroy a complete registered artifact at the final write.
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

    # ── ADDITIVE STANZA — ONE NON-RAISING BOUNDARY ───────────────────────────────
    # The registered npz and gates.json are already on disk (atomically) above.
    # EVERYTHING from here to the end of the try is additive: vmeta construction,
    # the mirror build, the helper call, the tuple unpack, and every print. None
    # of it may raise: _extract_body turns any exception out of _run_extract into
    # a PERMANENT `aborted` terminal status, which would leave a COMPLETE
    # registered artifact on disk marked aborted and unrunnable forever — the
    # single worst outcome available here.
    vnorm_path, vnorm_reason = None, "not attempted"
    try:
        vmeta = {
            "schema": "furnace-depth-vnorm/1.0", "model": conf["model_id"],
            "task": task, "model_key": conf["_key"],
            "revision_pinned": conf["revision"],
            "depth_npz": f"{_slug(conf)}.depth.npz", "precision": precision,
            "n_layers": n_layers, "n_heads": n_heads, "n_kv_heads": n_kv,
            "n_rows": n_rows,
            "additive_metrics": list(ADDITIVE_DEPTH_CELL_DETAILS),
            "v_norm_capture_mode": v_norm_mode,
            "additive_capture_diagnostics": v_norm_diag,
            "data_sha256": data_sha, "backend": "modal-torch", "comparable": False,
            "registered": False, "out_dir": out_dir,
            "row_identity_mirror": ["labels", "gen_token_ids", "commit_p", "yes_no"],
            "row_identity_check": "REQUIRED",
            "note": "UNREGISTERED descriptive channel (candidate #16); NOT part of "
                    "PRE_REGISTRATION_EXPANSION.md and outside every confirmatory "
                    "denominator. "
                    "(1) Index the trailing axis of v_norm_scores BY NAME via "
                    "v_norm_metrics, never by position. "
                    "(2) BEFORE joining to depth_npz you MUST assert exact "
                    "equality of labels/gen_token_ids/commit_p/yes_no against "
                    "the registered file and REFUSE the join on any mismatch. "
                    "sample_idx is arange(n) and cannot detect a permutation — "
                    "writing these columns without enforcing them would repeat "
                    "exactly the mistake that made sample_idx vacuous.",
        }
        v_mirror = _vnorm_build_mirror(np, v_state, labels, gen_ids, commit_p,
                                       yes_no)
        vnorm_path, vnorm_reason = _vnorm_write_sidecar(
            np, json, f"{outdir}/{_slug(conf)}.vnorm.npz", v_norm_scores,
            v_mirror, n_rows, vmeta, v_state)
        try:
            print(f"[gridB][vnorm] sidecar={vnorm_path!r} reason={vnorm_reason!r} "
                  f"mirror={sorted(v_mirror)}", flush=True)
        except Exception:  # noqa: BLE001
            pass
    except Exception as _vexc:  # noqa: BLE001 — additive boundary; never escapes
        vnorm_path = None
        try:
            vnorm_reason = _vnorm_exc_text(_vexc)
        except Exception:  # noqa: BLE001
            vnorm_reason = "<unformattable exception>"
        try:
            print(f"[gridB][vnorm] sidecar stanza FAILED: {vnorm_reason} "
                  f"(registered artifact already written)", flush=True)
        except Exception:  # noqa: BLE001
            pass

    return {"model": conf["model_id"], "task": task, "precision": precision,
            "n_layers": n_layers, "yes_no_commit_rate": round(frac_yn, 4),
            "gate_cos": [r["oproj_recon_cos"] for r in gate["rows"]],
            "out": npz_final, "out_dir": out_dir,
            "vnorm_out": vnorm_path, "v_norm_capture_mode": v_norm_mode}


def _extract_body(model_key: str, task: str, gpu_label: str,
                  out_dir: str = OUT_DIR_DEFAULT):
    conf = dict(REGISTRY[model_key]); conf["_key"] = model_key
    # TERMINAL-STATE IMMUTABILITY (round-5 MAJOR-4): checked BEFORE the try block —
    # an existing terminal status (ok OR aborted) or an existing npz may never be
    # overwritten by a rerun; a rescue would need a preregistered amendment.
    #
    # UNCHANGED by the out_dir work. The rule is not weakened, skipped or
    # special-cased; it simply guards whichever namespace the run targets. Both
    # paths below derive from the SAME out_dir, so an unregistered candidate-#16
    # run passes by not colliding with a registered cell, while a second #16 run of
    # the same cell is still refused — and pointing out_dir at OUT_DIR_REGISTERED
    # still hits the frozen terminal statuses exactly as before.
    print(f"[gridB] out_dir={out_dir!r} (registered tree is {OUT_DIR_REGISTERED!r})",
          flush=True)
    sp = _status_path(task, conf, out_dir)
    npz = f"{MNT}/{out_dir}/{task}/{_slug(conf)}.depth.npz"
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
                             "original_hardware": "see npz meta gpu_names/device map"},
                      out_dir=out_dir)
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

    # ── EXCLUSIVE CLAIM — taken AFTER every immutability check above ─────────────
    # The status/npz checks are check-then-write and therefore cannot stop two
    # simultaneous FIRST attempts on a cell that has neither. os.open with
    # O_CREAT|O_EXCL is a single atomic syscall, so exactly one invocation wins.
    #
    # Placed LAST on purpose: the immutability guard stays the first and
    # authoritative gate and is not weakened, skipped or special-cased. The claim
    # only closes the residual first-attempt race that guard cannot see.
    import json as _json
    import time as _time
    claim_path = f"{MNT}/{out_dir}/{task}/{_slug(conf)}.claim.json"
    claim_payload = {
        "state": "in_progress", "model_key": model_key,
        "model_id": conf["model_id"], "revision": conf["revision"], "task": task,
        "out_dir": out_dir, "gpu_label": gpu_label, "schema": SCHEMA,
        "claimed_at_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "claimed_at_unix": _time.time(), "pid": os.getpid(),
        "note": "STICKY LOCK. Never auto-removed. state='in_progress' means the "
                "run was killed before writing a terminal status (spend limit, "
                "preemption, OOM) — this is exactly the 2026-08-18 405B "
                "emergency-stop shape. Such a cell is NO LONGER relaunchable "
                "as-is: inspect this file, confirm the process is dead, delete "
                "it, then relaunch. state='complete' means the npz beside it is "
                "final and the terminal status governs.",
    }
    claim_ok, claim_why = _claim_cell_exclusive(_json, claim_path, claim_payload)
    if not claim_ok:
        raise SystemExit(
            f"CELL ALREADY CLAIMED ({claim_path}) — another invocation holds "
            f"this (out_dir, task, model). Existing claim:\n{claim_why}\n"
            f"If that run is dead, delete the claim file to retry.")
    try:
        res = _run_extract(conf, task, out_dir)
        res["gpu_label"] = gpu_label
        _claim_mark_complete(_json, claim_path,
                             dict(claim_payload, state="complete", npz=npz))
        _write_status(task, conf, "ok", extra={"result": res, "gpu_label": gpu_label},
                      out_dir=out_dir)
        print("GRIDB_EXTRACT_RESULT ok", flush=True)
        return res
    except Exception as e:  # noqa: BLE001
        _write_status(task, conf, "aborted",
                      reason=f"{type(e).__name__}: {e}", extra={"gpu_label": gpu_label},
                      out_dir=out_dir)
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

    os.makedirs(f"{MNT}/{MANIFEST_DIR}/manifests", exist_ok=True)
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
            man_path = f"{MNT}/{MANIFEST_DIR}/manifests/{_slug(conf)}.{task}.prompts.json"
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
    spath = f"{MNT}/{MANIFEST_DIR}/manifests/{_slug(conf)}.smoke.json"
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
def extract_a100(model_key: str, task: str, out_dir: str = OUT_DIR_DEFAULT):
    return _extract_body(model_key, task, "A100-80GB", out_dir)


@app.function(gpu="A100-80GB:4", timeout=60 * 60 * 12, **_COMMON)
def extract_a100x4(model_key: str, task: str, out_dir: str = OUT_DIR_DEFAULT):
    return _extract_body(model_key, task, "A100-80GB:4", out_dir)


@app.function(gpu="H200:2", timeout=60 * 60 * 12, **_COMMON)
def extract_h200x2(model_key: str, task: str, out_dir: str = OUT_DIR_DEFAULT):
    return _extract_body(model_key, task, "H200:2", out_dir)


@app.function(gpu="A100-80GB:8", timeout=60 * 60 * 12, **_COMMON)
def extract_a100x8(model_key: str, task: str, out_dir: str = OUT_DIR_DEFAULT):
    return _extract_body(model_key, task, "A100-80GB:8", out_dir)


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
def extract(model_key: str, task: str, gpu: str = "",
            out_dir: str = OUT_DIR_DEFAULT):
    conf = REGISTRY[model_key]
    # GPU override for OUTCOME runs is restricted to the registered Medium
    # fallback only (round-5: unrestricted override would leave hardware
    # unrecorded/unregistered).
    if gpu and not (model_key == "mistral_medium_35" and gpu == "H200:2"):
        raise SystemExit(f"gpu override {gpu!r} not registered for {model_key} — "
                         f"only mistral_medium_35 may fall back to H200:2")
    print(_fn_for(conf, "extract", gpu or None).remote(model_key, task, out_dir))
