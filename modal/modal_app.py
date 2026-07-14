"""commit-confluence extractor on Modal (NVIDIA GPUs) — faithful PyTorch port of the sealed pipeline.

Design (mirrors stage_b/gemma4_full_extract.py, the validated gemma-4 extractor):
  * The model FORWARD is the only framework-specific part. Everything downstream is the SEALED
    code, imported verbatim from the vendored `cloud/seal/` (pri_calibrator, pri_runtime,
    comprehensive_run, confluence_calibrator, pri_v2_io_plugins, + transitive deps).
  * MLX is Apple-only and cannot run here, so we inject stub `mlx`/`mlx.core`/`mlx.nn` +
    `pri_v2_mlx_pipeline` modules before importing the seal code. Verified locally: the kernels we
    call (PRIComputer.compute_step, _support_spectrum/_spectrum_stats, _compute_attention_score,
    merge_matrices/calibrate_merged) never touch `mx` or the MLX pipeline.
  * The MLX OutputProjection (quantized get_rows/project) is replaced by NumpyProjection over a
    DENSE W_u pulled from the HF model — the only substitution in the geometry path.

Faithfulness to the seal:
  * Commit moment = gen_step 1 with the D0/D1 split PRESERVED: forward A on the (chat-templated)
    prompt -> D0 (surprise, h_prev, ACE caps); commit gid = argmax(D0); forward B on [prompt+gid]
    -> D1 (p_t, p_max, h_t). This is the off-by-one fix, identical to gemma4_full_extract.
  * attn_implementation="eager" so `outputs.attentions` ARE the model's own softmax weights — ACE
    morphology reads the model's attention directly (no recompute; the cos~1.0 gate becomes a
    reconstruction *check* in validate()).
  * Prompts go through the tokenizer chat template (the gemma-4 lesson: instruction-tuned models
    won't attempt the YES/NO task on a raw prompt). validate() asserts the commit token is YES/NO.

COMPARABILITY: torch + bf16 (or 4-bit) != MLX-4bit seal. Every result here is NON-byte-comparable,
a standalone exploratory panel like the gemma-4 cell. NEVER pool with the sealed/byte-comparable cells.

Volume layout (model-cache @ /models):
  /models/hub/                      HF weight cache
  /models/data/<task>_n<N>.jsonl    benchmark prompts (uploaded)
  /models/refs/<task>.matrix.npz    seal-reference matrix (readout panel order; uploaded)
  /models/profiles_ext/<task>/<slug>.matrix.npz + .profile.json   outputs

Run:
  modal run cloud/modal_app.py::validate --model-id Qwen/Qwen2.5-32B-Instruct --task anli_r1
  modal run cloud/modal_app.py::extract  --model-id Qwen/Qwen2.5-32B-Instruct --task anli_r1
"""
import ast
import os
from pathlib import Path

import modal

APP_NAME = "commit-confluence"
VOL_NAME = "model-cache"
MNT = "/models"
SEAL_REMOTE = "/seal"
MAX_PROMPT_TOKENS = 4096
SEED = 20260612
NBOOT = 2000

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.46.0",
        "accelerate>=0.34.0",
        "numpy<2",
        "pandas",
        "scikit-learn>=1.3",
        "scipy",
        "sentencepiece",
        "safetensors",
        "huggingface_hub",
        "hf_transfer",
        "datasets>=2.20.0",
        "bitsandbytes",  # 4-bit path (load_in_4bit) for 70B on a single 80GB GPU
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{MNT}/hub",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(
        str(Path(__file__).parent / "seal"),
        SEAL_REMOTE,
        copy=True,
        ignore=lambda p: "__pycache__" in p.parts or p.suffix == ".pyc",
    )
)

# 32B fits bf16 on one 80GB. For 70B you MUST either pass --load-in-4bit (single 80GB) OR set
# GPU_CONFIG = "A100-80GB:2" for bf16 across 2 GPUs — the default below will OOM on 70B bf16.
GPU_CONFIG = "A100-80GB"


# --------------------------------------------------------------------------------------
# seal import shim (runs inside the container)
# --------------------------------------------------------------------------------------
def _import_seal():
    """Inject MLX/pipeline stubs, then import the sealed modules verbatim."""
    import importlib.machinery
    import sys
    import types

    if SEAL_REMOTE not in sys.path:
        sys.path.insert(0, SEAL_REMOTE)

    # A hand-made ModuleType has __spec__=None, which makes importlib.util.find_spec(name) RAISE
    # ValueError("name.__spec__ is None"). transformers calls find_spec("mlx") at import
    # (is_mlx_available), so every stub MUST carry a real spec. With a spec present, transformers
    # then checks installed metadata -> PackageNotFoundError -> correctly sees mlx as UNAVAILABLE,
    # while the seal's `import mlx`/`import mlx_lm` still resolve to these stubs.
    def _stub(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = m
        elif getattr(m, "__spec__", None) is None:
            m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        return m

    for name in ("mlx", "mlx.core", "mlx.nn"):
        _stub(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.modules["mlx"].nn = sys.modules["mlx.nn"]

    # pri_runtime hard-imports `from mlx_lm import load, generate` (raises otherwise). We never
    # call them on Modal (the model forward is torch), so a stub with no-op callables suffices.
    if not hasattr(sys.modules.get("mlx_lm"), "load"):
        mlx_lm = _stub("mlx_lm")
        def _unavailable(*a, **k):
            raise RuntimeError("mlx_lm is stubbed on the Modal (torch) backend and must not be called")
        mlx_lm.load = _unavailable
        mlx_lm.generate = _unavailable

    if not hasattr(sys.modules.get("pri_v2_mlx_pipeline"), "safe_auroc"):
        stub = _stub("pri_v2_mlx_pipeline")

        def safe_auroc(labels, scores):  # faithful copy of pri_runtime.safe_auroc
            import numpy as np
            from sklearn.metrics import roc_auc_score
            labels = np.asarray(labels).astype(int)
            scores = np.asarray(scores)
            if len(np.unique(labels)) < 2:
                return np.nan
            try:
                return float(roc_auc_score(labels, scores))
            except Exception:
                return np.nan

        stub.safe_auroc = safe_auroc
        sys.modules["pri_v2_mlx_pipeline"] = stub

    import comprehensive_run as CR
    import confluence_calibrator as CC
    import pri_calibrator as SEAL
    import pri_runtime as PIPE
    import pri_v2_io_plugins as io_plugins
    from diagnose_inter_head_disagreement import _target_layer_map

    return SEAL, PIPE, CR, CC, io_plugins, _target_layer_map


# --------------------------------------------------------------------------------------
# model load + attention/value/hidden capture
# --------------------------------------------------------------------------------------
PRECISIONS = ("nf4", "int8", "bf16", "fp32")


def _resolve_precision(precision, load_in_4bit):
    # Back-compat: legacy callers pass load_in_4bit; ladder callers pass precision=.
    # nf4 reproduces the historical load_in_4bit=True config byte-for-byte.
    if precision:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        return precision
    return "nf4" if load_in_4bit else "bf16"


def _out_slug(model_id, precision):
    # Per-rung artifact name. nf4 keeps the legacy bare name so existing 4-bit profiles
    # stay addressable; higher-bit rungs get a __<precision> suffix so a ladder never
    # clobbers itself.
    base = model_id.split("/")[-1]
    return base if precision == "nf4" else f"{base}__{precision}"


def _load(model_id: str, load_in_4bit: bool = False, precision: str = ""):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = _resolve_precision(precision, load_in_4bit)
    quantized = precision in ("nf4", "int8")  # weights stored below 16-bit

    # A comment isn't a guard: refuse 70B/72B at >=16-bit on a single GPU (it OOMs) — require a
    # sub-16-bit rung (nf4/int8) or 2+ GPUs.
    parts = str(GPU_CONFIG).split(":", 1)
    n_gpu = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1
    if (not quantized) and any(s in model_id.lower() for s in ("70b", "72b")) and n_gpu < 2:
        raise ValueError(
            f"{model_id} at precision={precision} will OOM on GPU_CONFIG={GPU_CONFIG!r}; use a "
            f"sub-16-bit rung (precision=nf4|int8) on one 80GB GPU, or GPU_CONFIG='A100-80GB:2'.")

    tok = AutoTokenizer.from_pretrained(model_id)
    compute_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    kw = dict(attn_implementation="eager", device_map="auto", torch_dtype=compute_dtype)
    if quantized:
        from transformers import BitsAndBytesConfig
        # Keep lm_head + embed_tokens OUT of quantization at EVERY rung: W_u must stay a floating
        # tensor for the geometry kernels (a quantized lm_head trips _output_weight_numpy's guard;
        # untied models like Llama-3.3-70B would otherwise quantize it).
        skip = ["lm_head", "embed_tokens"]
        if precision == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=skip,
            )
        else:  # int8
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
        kw.pop("torch_dtype")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model, precision


class _Capture:
    """Hooks the tagged attention layers (value vectors) + the final decoder layer (residual h)."""

    def __init__(self, model, tags, final_idx):
        self.handles = []
        self.values = {}          # tag -> value_states [n_kv, T, hd] for last forward
        self.h = {"final": None}  # residual stream at the final decoder layer output
        self.attn_out = {"final": None}  # final self_attn output [d] (post o_proj) for the cos gate
        layers = model.model.layers
        cfg = model.config
        self.n_kv = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))

        # Value-norm capture assumes raw v_proj == the value vectors used by attention (true for
        # Llama/Qwen/Mixtral). Models that apply a value norm or share K=V (gemma-4) need the
        # G4Wrap path; refuse rather than emit silently-wrong v-norms.
        for tag, idx in tags.items():
            attn = layers[idx].self_attn
            if hasattr(attn, "v_norm") or getattr(attn, "use_k_eq_v", False):
                raise NotImplementedError(
                    f"layer {idx} attention has v_norm/use_k_eq_v; raw v_proj capture is unfaithful "
                    f"for this architecture — port the G4Wrap value path before running it.")

        def make_v_hook(tag):
            def hook(mod, inp, out):
                v = out  # v_proj output: [B, T, n_kv*hd]
                B, T, _ = v.shape
                if v.shape[-1] % self.n_kv != 0:
                    raise RuntimeError(f"v_proj width {v.shape[-1]} is not divisible by n_kv={self.n_kv}")
                hd = v.shape[-1] // self.n_kv
                vv = v.reshape(B, T, self.n_kv, hd)[0].transpose(0, 1)  # [n_kv, T, hd]
                self.values[tag] = vv.detach().float().cpu().numpy()
            return hook

        for tag, idx in tags.items():
            self.handles.append(layers[idx].self_attn.v_proj.register_forward_hook(make_v_hook(tag)))

        def first_tensor(out, where):
            t = out[0] if isinstance(out, tuple) else out
            if not hasattr(t, "shape") or len(t.shape) < 3:
                raise RuntimeError(f"{where} hook expected [B,T,D] tensor output, got {type(out).__name__}")
            return t

        def h_hook(mod, inp, out):
            hs = first_tensor(out, "final decoder layer")
            # Decoder-layer output is before model.model.norm, so this is the pre-final-norm residual.
            self.h["final"] = hs[0, -1].detach().float().cpu().numpy()  # [d], pre-final-norm
        self.handles.append(layers[final_idx].register_forward_hook(h_hook))

        def attn_hook(mod, inp, out):
            ao = first_tensor(out, "final self_attn")
            self.attn_out["final"] = ao[0, -1].detach().float().cpu().numpy()  # [d], post o_proj
        self.handles.append(layers[final_idx].self_attn.register_forward_hook(attn_hook))

    def remove(self):
        for hd in self.handles:
            hd.remove()


def _forward(model, ids, cap, tags):
    """One forward. Returns (logits_last [V], caps{tag:[w (H,T)]}, vcaps{tag:[vn (n_kv,T)]}, nkv, h_final)."""
    import numpy as np
    import torch

    # NOTE: output_attentions=True materializes ALL layers' [B,H,T,T] weights though only the 3
    # tagged layers are read (~0.4GB at T~200/70B; grows O(T^2)). Acceptable for n=200 short prompts;
    # TODO if prompts get long: capture last-query weights via hooks on the 3 tagged attn modules
    # and set output_attentions=False. device_map="auto" gathers logits/attentions to one device.
    # Clear hook caches so a hook that fails to fire raises here instead of reusing stale values
    # from a previous forward (cross-forward bleed).
    cap.values.clear(); cap.h["final"] = None; cap.attn_out["final"] = None
    with torch.no_grad():
        out = model(torch.tensor([ids], device=model.device), output_attentions=True, use_cache=False)
    if out.attentions is None:
        raise RuntimeError("out.attentions is None — eager attention not active for this model/"
                           "transformers version; ACE capture is impossible.")
    logits = out.logits[0, -1].float().cpu().numpy().astype(np.float64)
    caps, vcaps, nkv = {}, {}, {}
    expected_h = int(getattr(model.config, "num_attention_heads"))
    expected_kv = int(getattr(model.config, "num_key_value_heads", expected_h))
    for tag, idx in tags.items():
        if tag not in cap.values:
            raise RuntimeError(f"value hook did not fire for tag {tag} (layer {idx})")
        att = out.attentions[idx]
        if att is None or len(att.shape) != 4:
            shape = None if att is None else tuple(att.shape)
            raise RuntimeError(f"attention for tag {tag} (layer {idx}) has shape {shape}; expected [B,H,T,T]")
        if int(att.shape[1]) != expected_h:
            raise RuntimeError(
                f"attention for tag {tag} (layer {idx}) has H={int(att.shape[1])}; "
                f"expected num_attention_heads={expected_h} post-GQA-repeat "
                f"(num_key_value_heads={expected_kv}).")
        if int(att.shape[-2]) != len(ids) or int(att.shape[-1]) != len(ids):
            raise RuntimeError(
                f"attention for tag {tag} (layer {idx}) has shape {tuple(att.shape)}; "
                f"expected sequence axes {len(ids)}x{len(ids)} with use_cache=False.")
        w_last = att[0, :, -1, :].float().cpu().numpy()                  # [H, T]
        vv = cap.values[tag]                                             # [n_kv, T, hd]
        vn = np.linalg.norm(vv, axis=-1)                                 # [n_kv, T]
        caps[tag] = [w_last]
        vcaps[tag] = [vn]
        nkv[tag] = int(vv.shape[0])
    if cap.h["final"] is None:
        raise RuntimeError("final-layer residual hook did not fire")
    return logits, caps, vcaps, nkv, cap.h["final"]


class NumpyProjection:
    """Dense-W_u stand-in for the MLX OutputProjection (only project + get_rows are used)."""

    def __init__(self, W_u):
        import numpy as np
        W = np.asarray(W_u, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"output projection weight must be [vocab,d], got shape {W.shape}")
        self.W = np.ascontiguousarray(W)
        self.vocab_size, self.hidden_size = self.W.shape

    def project(self, dh):
        import numpy as np
        h = np.asarray(dh, dtype=np.float32)
        if h.ndim == 2 and 1 in h.shape:
            h = h.reshape(-1)
        elif h.ndim != 1:
            raise ValueError(f"project expected hidden vector [d], got shape {h.shape}")
        if h.shape[0] != self.hidden_size:
            raise ValueError(f"project hidden size mismatch: got {h.shape[0]}, expected {self.hidden_size}")
        return np.asarray(self.W @ h, dtype=np.float32).reshape(-1)

    def get_rows(self, indices):
        import numpy as np
        idx = np.asarray(indices, dtype=np.int32)
        return self.W[idx].astype(np.float32) if idx.size else None


def _output_weight_numpy(model):
    import torch
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("model.get_output_embeddings() did not expose a weight tensor")
    w = out.weight
    # 4-bit footgun: a bitsandbytes Params4bit lm_head stores uint8-packed weights. `.to(float32)`
    # reinterprets the PACKED BYTES (not a dequantization) -> silently-garbage W_u -> garbage
    # null_ratio/fisher. Check the SOURCE dtype is floating BEFORE casting (the post-cast check
    # below always passes). If lm_head is quantized, refuse: keep it unquantized (default skip list).
    if not torch.is_floating_point(w):
        raise RuntimeError(
            f"output projection weight has non-floating dtype {w.dtype} (likely a quantized "
            f"lm_head, e.g. bitsandbytes Params4bit). Casting it to float32 would be garbage — "
            f"keep lm_head unquantized (llm_int8_skip_modules default) or run bf16.")
    W_t = w.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if W_t.device.type != "cpu" or W_t.dtype != torch.float32:
        raise RuntimeError(f"output projection weight is {W_t.dtype} on {W_t.device}; expected float32 on cpu")
    return W_t.numpy()


def _chat_ids(tok, prompt, model_id=None):
    if not hasattr(tok, "chat_template") or tok.chat_template is None:
        name = model_id or getattr(tok, "name_or_path", "<unknown model>")
        raise RuntimeError(f"{name} tokenizer is missing tokenizer.chat_template; chat template is required")
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    enc = tok(text)["input_ids"]
    import numpy as np
    return [int(t) for t in np.array(enc).reshape(-1)]


def _readout_panel(seal_ref_path, n_ace):
    import json
    import numpy as np
    ref = np.load(seal_ref_path, allow_pickle=True)
    raw = ref["panel"]
    panel = json.loads(str(raw)) if isinstance(raw.tolist(), str) else [str(x) for x in raw]
    return panel[n_ace:]


RO_KEYS = ["null_ratio_post_rank1", "fisher_eff_rank", "spectral_entropy",
           "neg_shadow_logvol_r1", "surprise", "p_max"]


def _metric_of(cell):
    s = str(cell)
    for m in RO_KEYS:
        if m in s:
            return m
    return None


def _parse_panel_labels(raw_panel_list, SEAL):
    """Parse panel cell strings into _cell_label hashes.

    Accepts a list of stringy cell reprs (e.g. ["(1, 'attention', ...)", ...])
    and returns the SEAL._cell_label(tuple) list used for hash validation.
    Uses ast.literal_eval instead of eval — these are trusted artifact strings
    from extract(), but literal_eval is the right tool for the job.
    """
    return [SEAL._cell_label(c) for c in _parse_panel_cells(raw_panel_list)]


def _parse_panel_cells(raw_panel_list):
    """Parse stringy cell reprs into tuples. Shared by hash validation and scoring."""
    import ast
    result = []
    for c in raw_panel_list:
        try:
            cell = ast.literal_eval(str(c))
        except (ValueError, SyntaxError):
            cell = c
        result.append(tuple(cell))
    return result


def _label_panel(SEAL, panel):
    return [SEAL._cell_label(tuple(c)) for c in panel]


def _load_profile(SEAL, model_id, task, precision):
    """Load only the calibration profile JSON (no matrix I/O)."""
    import json

    out_slug = _out_slug(model_id, precision)
    base = f"{MNT}/profiles_ext/{task}/{out_slug}"
    profile_path = f"{base}.profile.json"
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"missing calibration profile on Modal volume: {profile_path}. "
            f"Run extract() for this model/task/precision before using the guard.")
    with open(profile_path) as f:
        profile = json.load(f)
    return profile


def _load_guard_artifacts(SEAL, model_id, task, precision):
    import json
    import numpy as np

    out_slug = _out_slug(model_id, precision)
    base = f"{MNT}/profiles_ext/{task}/{out_slug}"
    profile_path = f"{base}.profile.json"
    matrix_path = f"{base}.matrix.npz"
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"missing calibration profile on Modal volume: {profile_path}. "
            f"Run extract() for this model/task/precision before using the guard.")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"missing calibration matrix on Modal volume: {matrix_path}. "
            f"The guard derives its threshold from the fitted calibration scores.")
    with open(profile_path) as f:
        profile = json.load(f)
    matrix = np.load(matrix_path, allow_pickle=True)
    return profile, matrix


# --------------------------------------------------------------------------------------
# stress-data builders: exploratory shared-label YES/NO tasks staged directly on Modal
# --------------------------------------------------------------------------------------
ANLI_PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)

TRUTHFULQA_PROMPT_TEMPLATE = (
    "Instruction: Read the question and proposed answer, then decide whether "
    "the proposed answer is true. Answer YES if the proposed answer is true, "
    "NO if it is false or misleading.\n\n"
    "Question: {question}\n"
    "Proposed answer: {answer}\n"
    "Answer:"
)

HALUEVAL_PROMPTS = {
    "halueval_qa": (
        "Instruction: Read the supporting knowledge, question, and proposed answer. "
        "Answer YES if the proposed answer is supported by the knowledge, NO if it "
        "contains hallucinated or unsupported information.\n\n"
        "Knowledge: {knowledge}\n"
        "Question: {question}\n"
        "Proposed answer: {answer}\n"
        "Answer:"
    ),
    "halueval_dialogue": (
        "Instruction: Read the supporting knowledge, dialogue history, and proposed response. "
        "Answer YES if the proposed response is supported by the knowledge and dialogue, "
        "NO if it contains hallucinated or unsupported information.\n\n"
        "Knowledge: {knowledge}\n"
        "Dialogue history: {dialogue_history}\n"
        "Proposed response: {answer}\n"
        "Answer:"
    ),
    "halueval_summarization": (
        "Instruction: Read the source document and proposed summary. Answer YES if the "
        "summary is faithful to the document, NO if it contains hallucinated or unsupported "
        "information.\n\n"
        "Document: {document}\n"
        "Proposed summary: {answer}\n"
        "Answer:"
    ),
}


def _norm_space(text, limit=None):
    text = " ".join(str(text).split())
    if limit is not None and len(text) > limit:
        return text[:limit].rsplit(" ", 1)[0]
    return text


def _write_jsonl(path, rows):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _copy_panel_ref(task, ref_task="anli_r1"):
    import shutil
    os.makedirs(f"{MNT}/refs", exist_ok=True)
    src = f"{MNT}/refs/{ref_task}.matrix.npz"
    dst = f"{MNT}/refs/{task}.matrix.npz"
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"missing reference matrix {src}; upload/copy a prior matrix first. "
            f"The stress builder uses it only for readout-panel order.")
    shutil.copyfile(src, dst)
    return dst


def _manifest_for(task, rows, source, seed, notes=None):
    import hashlib
    import json
    data_path = f"{MNT}/data/{task}_n{len(rows)}.jsonl"
    _write_jsonl(data_path, rows)
    ref_path = _copy_panel_ref(task)
    manifest = {
        "schema": "furnace_stress_data_v1",
        "task": task,
        "source": source,
        "seed": seed,
        "n_rows": len(rows),
        "n_pos_label1": sum(int(r["label"]) == 1 for r in rows),
        "n_neg_label0": sum(int(r["label"]) == 0 for r in rows),
        "data_path": data_path,
        "ref_path": ref_path,
        "data_sha256": hashlib.sha256(open(data_path, "rb").read()).hexdigest(),
        "notes": notes or [],
    }
    with open(f"{MNT}/data/{task}_n{len(rows)}.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _build_anli_round(task, n, seed):
    import numpy as np
    from datasets import load_dataset

    round_id = task.rsplit("_r", 1)[-1]
    split = f"dev_r{round_id}"
    ds = load_dataset("facebook/anli", split=split)
    target_each = n // 2
    rng = np.random.RandomState(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    yes_rows, no_rows, seen = [], [], set()
    for idx in order:
        ex = ds[int(idx)]
        if int(ex["label"]) == 1:
            continue
        prompt = ANLI_PROMPT_TEMPLATE.format(
            premise=_norm_space(ex["premise"]),
            hypothesis=_norm_space(ex["hypothesis"]),
        )
        if prompt in seen:
            continue
        seen.add(prompt)
        row = {"prompt": prompt, "label": 0 if int(ex["label"]) == 0 else 1,
               "meta": {"dataset": "facebook/anli", "split": split, "source_idx": int(idx)}}
        if row["label"] == 0 and len(yes_rows) < target_each:
            yes_rows.append(row)
        elif row["label"] == 1 and len(no_rows) < target_each:
            no_rows.append(row)
        if len(yes_rows) >= target_each and len(no_rows) >= target_each:
            break
    if len(yes_rows) < target_each or len(no_rows) < target_each:
        raise RuntimeError(f"{task} insufficient rows: label0={len(yes_rows)} label1={len(no_rows)}")
    rows = yes_rows + no_rows
    rng.shuffle(rows)
    return _manifest_for(task, rows, {"dataset": "facebook/anli", "split": split}, seed,
                         notes=["ANLI R2/R3 stress task; no sealed-20 denominator change."])


def _extract_truthfulqa_pair(ex):
    targets = ex.get("mc1_targets") or ex.get("mc2_targets") or {}
    choices = list(targets.get("choices") or [])
    labels = list(targets.get("labels") or [])
    true_answers = [c for c, y in zip(choices, labels) if int(y) == 1 and str(c).strip()]
    false_answers = [c for c, y in zip(choices, labels) if int(y) == 0 and str(c).strip()]
    if not true_answers or not false_answers:
        return None
    return true_answers[0], false_answers[0]


def _build_truthfulqa(task, n, seed):
    import random
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows, stems = [], 0
    for idx in indices:
        pair = _extract_truthfulqa_pair(ds[idx])
        if pair is None:
            continue
        question = _norm_space(ds[idx]["question"])
        true_answer, false_answer = pair
        rows.extend([
            {"prompt": TRUTHFULQA_PROMPT_TEMPLATE.format(question=question, answer=_norm_space(true_answer)),
             "label": 0,
             "meta": {"dataset": "truthfulqa/truthful_qa", "config": "multiple_choice", "source_idx": int(idx),
                      "kind": "true"}},
            {"prompt": TRUTHFULQA_PROMPT_TEMPLATE.format(question=question, answer=_norm_space(false_answer)),
             "label": 1,
             "meta": {"dataset": "truthfulqa/truthful_qa", "config": "multiple_choice", "source_idx": int(idx),
                      "kind": "false"}},
        ])
        stems += 1
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"{task} insufficient rows: {len(rows)}")
    rows = rows[:n]
    rng.shuffle(rows)
    return _manifest_for(task, rows, {"dataset": "truthfulqa/truthful_qa", "config": "multiple_choice"}, seed,
                         notes=[f"Stem-paired TruthfulQA MC stress task; {stems} stems for {len(rows)} rows."])


def _build_halueval(task, n, seed):
    import random
    from datasets import load_dataset

    config = task.replace("halueval_", "")
    ds = load_dataset("pminervini/HaluEval", config, split="data")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    rows, stems = [], 0
    for idx in indices:
        ex = ds[idx]
        if task == "halueval_qa":
            vals = {
                "knowledge": _norm_space(ex.get("knowledge", ""), limit=1800),
                "question": _norm_space(ex.get("question", "")),
                "right": _norm_space(ex.get("right_answer", "")),
                "hallucinated": _norm_space(ex.get("hallucinated_answer", "")),
            }
        elif task == "halueval_dialogue":
            vals = {
                "knowledge": _norm_space(ex.get("knowledge", ""), limit=1600),
                "dialogue_history": _norm_space(ex.get("dialogue_history", ""), limit=1600),
                "right": _norm_space(ex.get("right_response", "")),
                "hallucinated": _norm_space(ex.get("hallucinated_response", "")),
            }
        elif task == "halueval_summarization":
            vals = {
                "document": _norm_space(ex.get("document", ""), limit=2600),
                "right": _norm_space(ex.get("right_summary", "")),
                "hallucinated": _norm_space(ex.get("hallucinated_summary", "")),
            }
        else:
            raise ValueError(f"unsupported HaluEval task: {task}")
        if not vals["right"] or not vals["hallucinated"]:
            continue
        prompt_tpl = HALUEVAL_PROMPTS[task]
        base = {k: v for k, v in vals.items() if k not in ("right", "hallucinated")}
        rows.extend([
            {"prompt": prompt_tpl.format(**base, answer=vals["right"]),
             "label": 0,
             "meta": {"dataset": "pminervini/HaluEval", "config": config, "source_idx": int(idx),
                      "kind": "right"}},
            {"prompt": prompt_tpl.format(**base, answer=vals["hallucinated"]),
             "label": 1,
             "meta": {"dataset": "pminervini/HaluEval", "config": config, "source_idx": int(idx),
                      "kind": "hallucinated"}},
        ])
        stems += 1
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"{task} insufficient rows: {len(rows)}")
    rows = rows[:n]
    rng.shuffle(rows)
    return _manifest_for(task, rows, {"dataset": "pminervini/HaluEval", "config": config}, seed,
                         notes=[f"Stem-paired HaluEval {config} stress task; {stems} stems for {len(rows)} rows.",
                                "Long contexts are whitespace-normalized and char-limited for attention capture."])


STRESS_TASKS = (
    "anli_r2",
    "anli_r3",
    "truthfulqa_mc",
    "halueval_qa",
    "halueval_dialogue",
    "halueval_summarization",
)


@app.function(image=image, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60)
def build_stress_data(task: str = "all", n: int = 200, seed: int = SEED):
    import json
    import traceback
    if n % 2:
        raise ValueError("stress data n must be even for balanced paired/binary sampling")
    tasks = list(STRESS_TASKS) if task == "all" else [task]
    manifests, errors = [], []
    for t in tasks:
        try:
            if t not in STRESS_TASKS:
                raise ValueError(f"unknown stress task {t!r}; expected one of {STRESS_TASKS} or 'all'")
            if t.startswith("anli_r"):
                manifest = _build_anli_round(t, n, seed)
            elif t == "truthfulqa_mc":
                manifest = _build_truthfulqa(t, n, seed)
            elif t.startswith("halueval_"):
                manifest = _build_halueval(t, n, seed)
            else:
                raise AssertionError(t)
            manifests.append(manifest)
        except Exception as e:
            errors.append({"task": t, "error_type": type(e).__name__, "error": str(e),
                           "traceback_tail": traceback.format_exc().splitlines()[-12:]})
    vol.commit()
    out = {"built": manifests, "errors": errors}
    print("STRESS_DATA_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
    return out


def _winner_endpoint(profile, endpoint):
    if endpoint not in ("geometric", "primary"):
        raise ValueError("endpoint must be 'geometric' or 'primary'")
    key = "secondary_geometric_only" if endpoint == "geometric" else "primary_full_panel"
    ep = profile.get(key) or {}
    winner = ep.get("winner")
    if not winner:
        raise ValueError(f"profile endpoint {key!r} has no winner")
    return key, ep, winner


def _youden_threshold(signed_scores, labels):
    """Return a simple held-calibration threshold: signed score >= threshold => risk class."""
    import numpy as np

    s = np.asarray(signed_scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    finite = np.isfinite(s) & np.isfinite(y)
    s, y = s[finite], y[finite]
    if len(s) < 4 or len(np.unique(y)) < 2:
        raise ValueError("not enough finite two-class calibration scores to derive a threshold")
    vals = np.unique(s)
    if len(vals) == 1:
        return float(vals[0])
    mids = (vals[:-1] + vals[1:]) / 2.0
    candidates = np.concatenate(([vals[0] - 1e-9], mids, [vals[-1] + 1e-9]))
    pos = y == 1
    neg = ~pos
    best_t, best_j = float(candidates[0]), -1e9
    for t in candidates:
        pred = s >= t
        tpr = float((pred & pos).sum() / max(pos.sum(), 1))
        fpr = float((pred & neg).sum() / max(neg.sum(), 1))
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def _guard_decision(score, threshold, abstain_lower, abstain_upper, calibration_scores=None):
    """Return ALLOW/ABSTAIN/BLOCK decision using frozen per-endpoint guard policy bounds.

    Parameters are baked at extract() time and validated before this call — no
    live recomputation of Youden's J from the calibration matrix.

    If calibration_scores is provided, computes the real empirical percentile
    of the signed score within the calibration distribution (not a fake 0.5).
    """
    import numpy as np

    delta = float(score - threshold)
    abstain_lo = float(abstain_lower)
    abstain_hi = float(abstain_upper)
    band = float(abstain_hi - abstain_lo) / 2.0

    # Compute real percentile from calibration scores if available
    if calibration_scores is not None and len(calibration_scores) > 0:
        cal = np.asarray(calibration_scores, dtype=np.float64)
        cal = cal[np.isfinite(cal)]
        if len(cal) > 0:
            percentile = float((cal <= score).mean())
        else:
            percentile = 0.5
    else:
        percentile = 0.5

    if abstain_lo <= score <= abstain_hi:
        state = "ABSTAIN"
        reason = "score is inside the frozen calibration uncertainty band"
    elif delta > 0:
        state = "BLOCK"
        reason = "signed detector score exceeds the frozen calibrated risk threshold"
    else:
        state = "ALLOW"
        reason = "signed detector score is below the frozen calibrated risk threshold"
    return {
        "state": state,
        "reason": reason,
        "threshold": float(threshold),
        "delta": delta,
        "abstain_band": band,
        "abstain_lower": abstain_lo,
        "abstain_upper": abstain_hi,
        "calibration_percentile": percentile,
        "frozen_policy": True,
    }


def _score_prompt_guard(SEAL, PIPE, CR, model, tok, model_id, prompt, winner, panel):
    import numpy as np

    labels = _label_panel(SEAL, panel)
    if winner not in labels:
        raise KeyError(f"winner {winner!r} is not in the direct score panel; fusion winners need a fusion scorer")
    winner_idx = labels.index(winner)
    winner_cell = tuple(panel[winner_idx])

    n_layers = len(model.model.layers)
    _, _, _, _, _, target_layer_map = _import_seal()
    tags = target_layer_map(n_layers)
    final_idx = n_layers - 1
    cap = _Capture(model, tags, final_idx)
    try:
        ids = _chat_ids(tok, prompt, model_id)
        logits_a, caps, vcaps, nkv, h_prev = _forward(model, ids, cap, tags)
        p_a = np.exp(logits_a - logits_a.max()); p_a /= p_a.sum()
        gid = int(np.argmax(p_a))
        surprise = float(-np.log(p_a[gid] + 1e-300))
        commit_piece = tok.decode([gid], skip_special_tokens=True)

        if winner.startswith("attention["):
            raw = SEAL._compute_attention_score(winner_cell, caps, nkv, v_norm_captures=vcaps)
            if raw is None or not np.isfinite(raw):
                raise RuntimeError(f"attention winner {winner!r} did not produce a finite score")
            return {
                "raw_score": float(raw),
                "commit_token": repr(commit_piece),
                "commit_probability": float(p_a[gid]),
                "commit_surprise": surprise,
                "pre_detokenization": True,
                "locus": "t=0 attention morphology",
                "requires_commit_forward": False,
            }

        metric = _metric_of(winner_cell)
        if metric is None:
            raise KeyError(f"cannot map winner {winner!r} to a supported readout metric")
        logits_b, _, _, _, h_t = _forward(model, ids + [gid], cap, tags)
        p_b = np.exp(logits_b - logits_b.max()); p_b /= p_b.sum()
        gamma = model.model.norm.weight.detach().float().cpu().numpy()
        W_u = _output_weight_numpy(model)
        proj = NumpyProjection(W_u)
        pri = PIPE.PRIComputer(proj, final_norm_gamma=gamma)
        comp = pri.compute_step(h_t=h_t, h_prev=h_prev, p_t=p_b, S_t=surprise, alpha=1.0,
                                topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                                v3_capture_raw=False, v3_capture_centered=False)
        spec, _, _ = CR._support_spectrum(proj, p_b, int(proj.hidden_size), CR.K_SUPPORT_DEFAULT)
        st = CR._spectrum_stats(spec)
        values = {
            "null_ratio_post_rank1": float(comp.get("null_ratio_post_rank1", np.nan)),
            "fisher_eff_rank": float(st["fisher_eff_rank"]),
            "spectral_entropy": float(st["spectral_entropy"]),
            "neg_shadow_logvol_r1": float(st["neg_shadow_logvol_r1"]),
            "surprise": surprise,
            "p_max": float(p_b.max()),
        }
        raw = values[metric]
        if not np.isfinite(raw):
            raise RuntimeError(f"readout winner {winner!r} produced non-finite score {raw}")
        return {
            "raw_score": float(raw),
            "commit_token": repr(commit_piece),
            "commit_probability": float(p_a[gid]),
            "commit_surprise": surprise,
            "pre_detokenization": False,
            "pre_response_text": True,
            "locus": "gen_step=1 readout geometry",
            "requires_commit_forward": True,
        }
    finally:
        cap.remove()


# --------------------------------------------------------------------------------------
# validate: prove the capture is faithful before any full run (the cos~1.0 / YES-NO gate)
# --------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60)
def validate(model_id: str, task: str, n: int = 200, load_in_4bit: bool = False, precision: str = ""):
    import numpy as np
    import torch

    SEAL, PIPE, CR, CC, io_plugins, target_layer_map = _import_seal()
    data = f"{MNT}/data/{task}_n{n}.jsonl"
    prompts, labels, _ = CR._load_calibration_jsonl(data)
    tok, model, precision = _load(model_id, load_in_4bit, precision)
    n_layers = len(model.model.layers)
    tags = target_layer_map(n_layers)
    final_idx = n_layers - 1
    cap = _Capture(model, tags, final_idx)

    report = {"model": model_id, "task": task, "precision": precision, "n_layers": n_layers, "tags": tags, "examples": []}
    for i in (0, 1):
        ids = _chat_ids(tok, prompts[i], model_id)
        logits, caps, vcaps, nkv, h = _forward(model, ids, cap, tags)
        p = np.exp(logits - logits.max()); p /= p.sum()
        gid = int(np.argmax(p))
        top = np.argsort(p)[::-1][:6]
        commit = tok.decode([gid]).strip().upper()
        # o_proj reconstruction vs the model's OWN attention output (the cos~1.0 faithfulness gate):
        # ctx[h,d] = Σ_t w[h,t]·v_rep[h,t,d]  ->  o_proj  ==  the captured self_attn output.
        fa = model.model.layers[final_idx].self_attn
        oproj_device = fa.o_proj.weight.device
        oproj_dtype = getattr(fa.o_proj, "compute_dtype", None) or fa.o_proj.weight.dtype
        def is_floating_dtype(dtype):
            return isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point()
        if not is_floating_dtype(oproj_dtype):
            oproj_dtype = getattr(model, "dtype", torch.float32)
        if not is_floating_dtype(oproj_dtype):
            oproj_dtype = torch.float32
        w = torch.as_tensor(caps["final"][0], device=oproj_device, dtype=torch.float32)      # [H, T]
        vv = torch.as_tensor(cap.values["final"], device=oproj_device, dtype=torch.float32)  # [n_kv, T, hd]
        H = w.shape[0]; nrep = H // vv.shape[0]
        if H % vv.shape[0] != 0 or w.shape[1] != vv.shape[1]:
            raise RuntimeError(f"o_proj reconstruction shape mismatch: w={tuple(w.shape)} v={tuple(vv.shape)}")
        v_rep = vv.repeat_interleave(nrep, dim=0)                # [H, T, hd]
        ctx = torch.einsum("ht,htd->hd", w, v_rep).reshape(1, -1).to(dtype=oproj_dtype)
        my_out = fa.o_proj(ctx)[0].detach().float().cpu().numpy()
        real_out = cap.attn_out["final"]
        cos = float(np.dot(my_out, real_out) /
                    (np.linalg.norm(my_out) * np.linalg.norm(real_out) + 1e-9))
        report["examples"].append({
            "label": int(labels[i]),
            "commit_token": repr(tok.decode([gid])),
            "commit_is_yes_no": commit in ("YES", "NO"),
            "commit_p": round(float(p[gid]), 4),
            "top6": [(repr(tok.decode([int(t)])), round(float(p[t]), 4)) for t in top],
            "caps_rowsum_mean": round(float(caps["final"][0].sum(-1).mean()), 4),
            "vn_shape": list(vcaps["final"][0].shape),
            "nkv_final": nkv["final"],
            "h_finite": bool(np.isfinite(h).all()),
            "oproj_recon_cos": round(cos, 5),
            "maxabs": round(float(np.max(np.abs(my_out - real_out))), 4),
        })
    cap.remove()
    # Hard gates: the capture must be faithful AND the model must attempt the task.
    report["GATE_cos_ok"] = all(e["oproj_recon_cos"] >= 0.999 for e in report["examples"])
    report["GATE_yes_no_ok"] = all(e["commit_is_yes_no"] for e in report["examples"])
    report["GATE_PASS"] = report["GATE_cos_ok"] and report["GATE_yes_no_ok"]
    # Persist to the volume so the result survives Modal's "final app logs" flush race + tqdm
    # ANSI overwrites (which can clobber the locally-printed return value).
    import json
    os.makedirs(f"{MNT}/validate", exist_ok=True)
    with open(f"{MNT}/validate/{_out_slug(model_id, precision)}_{task}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    vol.commit()
    if not report["GATE_PASS"]:
        report["WARNING"] = ("VALIDATION FAILED — do NOT trust extract() until cos>=0.999 "
                             "(capture faithful) and commit is YES/NO (task attempted).")
        raise RuntimeError("cc-modal validation gate FAILED:\n" + json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------------------
# extract: the full faithful extractor -> matrix.npz -> calibrate_merged -> verdict
# --------------------------------------------------------------------------------------
@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60 * 6)
def extract(model_id: str, task: str, n: int = 200, load_in_4bit: bool = False, precision: str = "", max_dropped: int = 0, abstain_band: float = 0.10):
    import json
    import numpy as np

    SEAL, PIPE, CR, CC, io_plugins, target_layer_map = _import_seal()
    ACE_PANEL = SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS
    data = f"{MNT}/data/{task}_n{n}.jsonl"
    seal_ref = f"{MNT}/refs/{task}.matrix.npz"
    readout_panel = _readout_panel(seal_ref, len(ACE_PANEL))
    # Fail loudly if any readout cell doesn't map to a known metric (else myro[None] KeyError mid-run).
    unmapped = [str(c) for c in readout_panel if _metric_of(c) is None]
    if unmapped:
        raise ValueError(f"readout panel cells with no metric mapping: {unmapped}")

    prompts, labels, dh = CR._load_calibration_jsonl(data)
    tok, model, precision = _load(model_id, load_in_4bit, precision)
    n_layers = len(model.model.layers)
    tags = target_layer_map(n_layers)
    final_idx = n_layers - 1
    cap = _Capture(model, tags, final_idx)

    gamma = model.model.norm.weight.detach().float().cpu().numpy()
    W_u = _output_weight_numpy(model)
    proj = NumpyProjection(W_u)
    pri = PIPE.PRIComputer(proj, final_norm_gamma=gamma)
    dmodel = int(proj.hidden_size)
    print(f"[cc-modal] {model_id} {task} layers={n_layers} tags={tags} d={dmodel} "
          f"ace={len(ACE_PANEL)} ro={len(readout_panel)} V={proj.vocab_size}", flush=True)

    ace_rows, ro_rows, keep, n_yes_no = [], [], [], 0
    for i, (p_, y) in enumerate(zip(prompts, labels)):
        try:
            ids = _chat_ids(tok, p_, model_id)
            logitsA, caps, vcaps, nkv, h_prev = _forward(model, ids, cap, tags)  # D0 + ACE caps
            pA = np.exp(logitsA - logitsA.max()); pA /= pA.sum()
            gid = int(np.argmax(pA)); surprise = float(-np.log(pA[gid] + 1e-300))
            commit_is_yn = tok.decode([gid], skip_special_tokens=True).strip().upper() in ("YES", "NO")
            ace_row = []
            for cell in ACE_PANEL:
                sc = SEAL._compute_attention_score(cell, caps, nkv, v_norm_captures=vcaps)
                ace_row.append(float(sc) if (sc is not None and np.isfinite(sc)) else 0.0)
            logitsB, _, _, _, h_t = _forward(model, ids + [gid], cap, tags)        # D1
            pB = np.exp(logitsB - logitsB.max()); pB /= pB.sum()
            p_t, p_max = pB, float(pB.max())
            comp = pri.compute_step(h_t=h_t, h_prev=h_prev, p_t=p_t, S_t=surprise, alpha=1.0,
                                    topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
                                    v3_capture_raw=False, v3_capture_centered=False)
            spec, _, _ = CR._support_spectrum(proj, p_t, dmodel, CR.K_SUPPORT_DEFAULT)
            st = CR._spectrum_stats(spec)
            myro = {"null_ratio_post_rank1": float(comp.get("null_ratio_post_rank1", np.nan)),
                    "fisher_eff_rank": float(st["fisher_eff_rank"]),
                    "spectral_entropy": float(st["spectral_entropy"]),
                    "neg_shadow_logvol_r1": float(st["neg_shadow_logvol_r1"]),
                    "surprise": surprise, "p_max": p_max}
            ro_row = [myro[_metric_of(c)] for c in readout_panel]
            if all(np.isfinite(v) for v in ro_row):
                ace_rows.append(ace_row); ro_rows.append(ro_row); keep.append(i)
                n_yes_no += int(commit_is_yn)
            else:
                dropped_so_far = (i + 1) - len(keep)
                if dropped_so_far > max_dropped:
                    raise RuntimeError(
                        f"ex{i} produced non-finite readout values and dropped_so_far={dropped_so_far} "
                        f"exceeds max_dropped={max_dropped}; aborting extraction")
        except Exception as e:
            import traceback
            print(f"ex{i} FAIL {type(e).__name__}: {e}", flush=True)
            dropped_so_far = (i + 1) - len(keep)
            if i < 2 or dropped_so_far > max_dropped:
                traceback.print_exc()
            if dropped_so_far > max_dropped:
                raise RuntimeError(
                    f"ex{i} failed and dropped_so_far={dropped_so_far} exceeds "
                    f"max_dropped={max_dropped}; aborting extraction") from e
        if i % 25 == 0:
            print(f"  {i}/{len(prompts)} kept={len(keep)} yes_no={n_yes_no}", flush=True)
    cap.remove()

    n_dropped = len(prompts) - len(keep)
    frac_yes_no = n_yes_no / max(len(keep), 1)
    print(f"[cc-modal] dropped {n_dropped}/{len(prompts)} (non-finite readout); "
          f"YES/NO commit rate {frac_yes_no:.2%} over kept", flush=True)
    # Fail closed: don't calibrate a silently-shrunk or task-not-attempted sample.
    if n_dropped > max_dropped:
        raise RuntimeError(f"dropped {n_dropped}/{len(prompts)} samples > max_dropped={max_dropped}; "
                           f"refusing calibration (re-run with a higher --max-dropped to override).")
    if frac_yes_no < 0.5:
        # the gemma-4 failure mode: model isn't attempting the task -> signals are noise
        raise RuntimeError(f"YES/NO commit rate {frac_yes_no:.2%} < 50% — task not attempted "
                           f"(prompt-format/chat-template problem); refusing to emit a verdict. "
                           f"Run validate first.")

    yk = np.array([int(labels[i]) for i in keep])
    slug = model_id.split("/")[-1]
    out_slug = _out_slug(model_id, precision)
    ace_d = {"sample_idx": np.array(keep), "labels": yk, "score_matrix": np.array(ace_rows),
             "panel": list(ACE_PANEL), "slug": slug, "data_hash": dh}
    ro_d = {"sample_idx": np.array(keep), "labels": yk, "score_matrix": np.array(ro_rows),
            "panel": list(readout_panel), "data_hash": dh}
    mm = CC.merge_matrices(ace_d, ro_d, max_dropped=0)
    outdir = f"{MNT}/profiles_ext/{task}"
    os.makedirs(outdir, exist_ok=True)
    np.savez(f"{outdir}/{out_slug}.matrix.npz", score_matrix=mm["score_matrix"], labels=mm["labels"],
             sample_idx=mm["sample_idx"], panel=json.dumps([str(c) for c in mm["panel"]]),
             meta=json.dumps({"model": model_id, "task": task, "n": mm["n_aligned"], "precision": precision,
                              "backend": "modal-torch", "comparable": False}))
    prof = CC.calibrate_merged(mm, n_bootstrap=NBOOT, seed=SEED, model_id=model_id, benchmark=task)
    prof["comparability"] = {"byte_comparable_to_mlx_seal": False, "backend": "modal-torch", "precision": precision}
    prof["extraction_audit"] = {"n_prompts": len(prompts), "n_kept": len(keep),
                                "n_dropped": n_dropped, "yes_no_commit_rate": round(frac_yes_no, 4)}

    # ── frozen guard policy ──────────────────────────────────────────────────
    # Bake the threshold + abstain bounds at extraction time so the guard never
    # recomputes Youden's J from a potentially stale matrix.  Schema version
    # pins the policy format; data hashes catch stale-artifact pairs.
    import hashlib as _hl
    def _sha256_json(obj):
        return _hl.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

    prof["guard_policy"] = {}
    for ep_key, ep_name in [("secondary_geometric_only", "geometric"),
                             ("primary_full_panel", "primary")]:
        ep = prof.get(ep_key, {})
        w = ep.get("winner")
        if not w:
            continue
        marg = (ep.get("winner_marginal") or
                (ep.get("full_sample_marginals") or {}).get(w) or {})
        sign = int(marg.get("sign", 1))
        labels_y = mm["labels"]
        panel_labels = [SEAL._cell_label(tuple(c)) for c in mm["panel"]]
        j = None
        for ji, label in enumerate(panel_labels):
            if label == w:
                j = ji
                break
        if j is None:
            continue
        signed_cal = np.asarray(mm["score_matrix"][:, j], dtype=np.float64) * float(sign)
        threshold = float(_youden_threshold(signed_cal, labels_y))

        # latency classification
        is_ace = any(kw in str(w).lower() for kw in ("attention", "bos_mass",
                     "js", "v_norm", "lastq_weighted"))
        is_readout = any(kw in str(w).lower() for kw in ("null_ratio", "fisher_eff",
                         "spectral_entropy", "neg_shadow"))
        is_fusion = "fusion" in str(w).lower()
        if is_fusion:
            gs, lc = False, "unsupported"
        elif is_ace:
            gs, lc = True, "fast"
        elif is_readout:
            gs, lc = False, "requires_commit_forward"
        else:
            gs, lc = True, "fast"  # default: assume single-forward

        cached_ace = [str(c) for c in ACE_PANEL]
        cached_full = [str(c) for c in mm["panel"]]
        prof["guard_policy"][ep_name] = {
            "schema_version": "furnace-guard/0.1",
            "winner": w,
            "winner_index": j,
            "sign": sign,
            "threshold": threshold,
            "abstain_lower": round(threshold - abstain_band, 6),
            "abstain_upper": round(threshold + abstain_band, 6),
            "abstain_band_baked": abstain_band,
            "guard_supported": gs,
            "latency_class": lc,
            "data_hash": mm["ace_data_hash"],
            "panel_label_hash": _sha256_json(panel_labels),
            "endpoint_key": ep_key,
            "endpoint_name": ep_name,
            "deployable": bool(ep.get("deployable")),
            "winner_oob_ci_lo": ep.get("oob_auroc_ci_lo"),
            "winner_oob_median": ep.get("oob_auroc_median"),
            "ace_panel": cached_ace,
            "guard_panel": cached_full,
            "calibration_scores": signed_cal.tolist(),
        }
    with open(f"{outdir}/{out_slug}.profile.json", "w") as f:
        json.dump(prof, f, indent=2, default=str)
    vol.commit()

    ge = prof.get("secondary_geometric_only", {}); pr = prof.get("primary_full_panel", {})
    verdict = {"task": task, "model": model_id, "precision": precision, "n_aligned": mm["n_aligned"],
               "n_dropped": n_dropped, "yes_no_commit_rate": round(frac_yes_no, 4),
               "geom_winner": ge.get("winner"), "geom_ci_lo": ge.get("oob_auroc_ci_lo"),
               "geom_deployable": ge.get("deployable"),
               "primary_winner": pr.get("winner"), "primary_ci_lo": pr.get("oob_auroc_ci_lo"),
               "primary_deployable": pr.get("deployable"),
               "controls_pass": prof.get("controls", {}).get("pass"),
               "NOTE": "NON-byte-comparable (torch backend); standalone exploratory cell."}
    print("CC_MODAL_RESULT\n" + json.dumps(verdict, indent=2, default=str), flush=True)
    return verdict


@app.function(image=image, gpu=GPU_CONFIG, volumes={MNT: vol}, secrets=[hf_secret], timeout=60 * 60 * 2)
def guard_prompt(model_id: str, task: str, prompt: str, n: int = 200,
                 load_in_4bit: bool = False, precision: str = "",
                 endpoint: str = "geometric", abstain_band: float = 0.10,
                 max_prompt_tokens: int = MAX_PROMPT_TOKENS):
    """Score one prompt against a fitted confluence profile and return a fail-closed guard state.

    This is the product-facing path for the TUI/wrapper. It computes the selected Furnace
    metric before any response text is generated. For Qwen2.5-32B the winning cells are ACE
    attention cells, so the score is available from the prompt-only forward pass.
    """
    import json
    import numpy as np

    SEAL, PIPE, CR, CC, io_plugins, target_layer_map = _import_seal()
    resolved_precision = _resolve_precision(precision, load_in_4bit)
    base = {
        "model": model_id,
        "task": task,
        "precision": resolved_precision,
        "endpoint": endpoint,
        "backend": "modal-torch",
        "byte_comparable_to_mlx_seal": False,
        "winner": "<none>",
        "signed_score": 0.0,
        "response_text_emitted": False,
    }
    try:
        # Load profile only (no matrix I/O). If the frozen policy has a cached
        # guard_panel, we can skip the matrix entirely on the hot path.
        profile = _load_profile(SEAL, model_id, task, resolved_precision)
        endpoint_key, ep, winner = _winner_endpoint(profile, endpoint)
        controls_pass = bool((profile.get("controls") or {}).get("pass"))
        deployable = bool(ep.get("deployable"))
        if not controls_pass or not deployable:
            out = {
                **base,
                "state": "DEFER",
                "reason": "profile is not deployable or shuffled-label controls did not pass",
                "winner": winner,
                "endpoint_key": endpoint_key,
                "profile_deployable": deployable,
                "controls_pass": controls_pass,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out

        # ── frozen guard policy validation ──────────────────────────────────
        gp = (profile.get("guard_policy") or {}).get(endpoint)
        if not gp:
            out = {
                **base,
                "state": "DEFER",
                "reason": f"no frozen guard policy for endpoint {endpoint!r} — "
                          f"re-run extract() on this model/task/precision to bake one",
                "winner": winner,
                "endpoint_key": endpoint_key,
                "profile_deployable": deployable,
                "controls_pass": controls_pass,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        if gp.get("schema_version") != "furnace-guard/0.1":
            out = {
                **base,
                "state": "DEFER",
                "reason": f"guard policy schema {gp.get('schema_version')!r} is not supported",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out

        # Validate frozen policy against live artifacts
        import hashlib as _hl
        import numpy as _np

        required = {
            "schema_version", "winner", "winner_index", "sign", "threshold",
            "abstain_lower", "abstain_upper", "guard_supported",
            "latency_class", "data_hash", "panel_label_hash",
        }
        missing_policy = required - set(gp)
        if missing_policy:
            out = {
                **base,
                "state": "DEFER",
                "reason": f"guard policy missing required fields: {sorted(missing_policy)}",
                "winner": winner,
                "guard_supported": False,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        sign = int(gp["sign"])
        if sign not in (-1, 1) or gp["sign"] not in (-1, 1):
            out = {
                **base,
                "state": "DEFER",
                "reason": f"guard policy sign must be exactly -1 or 1, got {gp['sign']!r}",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        if gp.get("winner") != winner:
            out = {
                **base,
                "state": "DEFER",
                "reason": f"guard policy winner {gp.get('winner')!r} != live endpoint winner {winner!r}",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        threshold = float(gp["threshold"])
        abstain_lower = float(gp["abstain_lower"])
        abstain_upper = float(gp["abstain_upper"])
        if not all(_np.isfinite(v) for v in (threshold, abstain_lower, abstain_upper)):
            out = {
                **base,
                "state": "DEFER",
                "reason": "guard policy threshold/bounds must be finite",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        if not (abstain_lower <= threshold <= abstain_upper):
            out = {
                **base,
                "state": "DEFER",
                "reason": "guard policy abstain bounds must bracket threshold",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        # data_hash: only validate when both sides are non-empty (legacy profiles
        # may not have ace_data_hash in the profile dict).
        baked_hash = gp.get("data_hash", "")
        profile_hash = profile.get("ace_data_hash", "")
        if not baked_hash or not profile_hash:
            out = {
                **base,
                "state": "DEFER",
                "reason": "guard policy data_hash and profile ace_data_hash must both be non-empty",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        if baked_hash != profile_hash:
            out = {
                **base,
                "state": "DEFER",
                "reason": "guard policy data_hash mismatch — stale artifact pair",
                "winner": winner,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out

        # Validate panel_label_hash. When guard_panel is cached (Fix 2), validate
        # against the cached labels — no matrix needed, no disk I/O.
        cached_panel_labels = gp.get("guard_panel")
        if cached_panel_labels and len(cached_panel_labels) > 0:
            runtime_panel_labels = _parse_panel_labels(cached_panel_labels, SEAL)
        else:
            # Legacy path: load matrix for panel parsing (disk I/O required)
            matrix = _load_guard_artifacts(SEAL, model_id, task, resolved_precision)[1]
            panel_raw = matrix["panel"]
            # Handle 0-D numpy scalar (json.dumps stored as scalar array in NPZ)
            if hasattr(panel_raw, 'ndim') and panel_raw.ndim == 0:
                panel_labels = json.loads(str(panel_raw.item()))
            else:
                panel_labels = [str(c) for c in panel_raw]
            runtime_panel_labels = _parse_panel_labels(panel_labels, SEAL)
        live_panel_hash = _hl.sha256(
            json.dumps(runtime_panel_labels, sort_keys=True, default=str).encode()
        ).hexdigest()
        if gp.get("panel_label_hash") != live_panel_hash:
            out = {
                **base,
                "state": "DEFER",
                "reason": "guard policy panel_label_hash mismatch — "
                          "profile was extracted with a different panel than this matrix",
                "winner": winner,
                "guard_supported": False,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        if not gp.get("guard_supported"):
            out = {
                **base,
                "state": "DEFER",
                "reason": f"winner {winner!r} is not supported for guard mode "
                          f"(latency_class={gp.get('latency_class', 'unsupported')!r})",
                "winner": winner,
                "guard_supported": False,
                "latency_class": gp.get("latency_class"),
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out

        sign = int(gp.get("sign", 1))
        threshold = float(gp["threshold"])
        abstain_lower = float(gp.get("abstain_lower", threshold - 0.10))
        abstain_upper = float(gp.get("abstain_upper", threshold + 0.10))
        cal_scores = gp.get("calibration_scores")  # may be None for legacy profiles

        tok, model, resolved_precision = _load(model_id, load_in_4bit, resolved_precision)

        # ── exact tokenizer-based prompt-length gate ──────────────────────
        prompt_token_count = len(_chat_ids(tok, prompt, model_id))
        if prompt_token_count > max_prompt_tokens:
            out = {
                **base,
                "state": "DEFER",
                "reason": f"prompt exceeds {max_prompt_tokens} token limit "
                          f"after model tokenizer/chat template ({prompt_token_count} tokens)",
                "winner": winner,
                "signed_score": 0.0,
                "prompt_tokens": prompt_token_count,
                "max_prompt_tokens": max_prompt_tokens,
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out

        # Use cached guard panel from frozen policy (no disk I/O on the hot path).
        # Falls back to loading the reference matrix for legacy profiles.
        cached_full = gp.get("guard_panel")
        if cached_full and len(cached_full) > 0:
            guard_panel = _parse_panel_cells(cached_full)
        else:
            ACE_PANEL = SEAL.ATTENTION_PANEL_T0_WITH_V_NORMS
            seal_ref = f"{MNT}/refs/{task}.matrix.npz"
            readout_panel = _readout_panel(seal_ref, len(ACE_PANEL))
            guard_panel = list(ACE_PANEL) + _parse_panel_cells(readout_panel)
        score_info = _score_prompt_guard(SEAL, PIPE, CR, model, tok, model_id, prompt, winner, guard_panel)
        # ── pre-detokenization contract enforcement ─────────────────────
        if not bool(score_info.get("pre_detokenization")):
            out = {
                **base,
                "state": "DEFER",
                "reason": "selected guard metric is not available before detokenization "
                          "(requires commit forward pass); gate refuses to score",
                "winner": winner,
                "signed_score": 0.0,
                "metric_locus": score_info.get("locus"),
                "pre_detokenization": False,
                "pre_response_text": bool(score_info.get("pre_response_text", True)),
            }
            print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
            return out
        signed_score = float(score_info["raw_score"]) * float(sign)
        decision = _guard_decision(signed_score, threshold, abstain_lower, abstain_upper, cal_scores)
        out = {
            **base,
            **decision,
            "winner": winner,
            "endpoint_key": endpoint_key,
            "profile_deployable": deployable,
            "controls_pass": controls_pass,
            "guard_supported": True,
            "latency_class": gp.get("latency_class"),
            "winner_oob_ci_lo": gp.get("winner_oob_ci_lo"),
            "winner_oob_median": gp.get("winner_oob_median"),
            "winner_sign": sign,
            "raw_score": float(score_info["raw_score"]),
            "signed_score": signed_score,
            "threshold": threshold,
            "abstain_lower": abstain_lower,
            "abstain_upper": abstain_upper,
            "metric_locus": score_info.get("locus"),
            "pre_detokenization": bool(score_info.get("pre_detokenization")),
            "pre_response_text": bool(score_info.get("pre_response_text", True)),
            "commit_token": score_info.get("commit_token"),
            "commit_probability": score_info.get("commit_probability"),
            "commit_surprise": score_info.get("commit_surprise"),
            "requires_commit_forward": bool(score_info.get("requires_commit_forward")),
        }
    except Exception as e:
        out = {
            **base,
            "state": "DEFER",
            "reason": f"guard failed closed: {type(e).__name__}: {e}",
            "error_type": type(e).__name__,
            "winner": locals().get("winner", "<none>"),
            "signed_score": 0.0,
        }
    print("FURNACE_GUARD_RESULT\n" + json.dumps(out, indent=2, default=str), flush=True)
    return out


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-32B-Instruct", task: str = "anli_r1",
         n: int = 200, mode: str = "validate", load_in_4bit: bool = False,
         precision: str = "", max_dropped: int = 0, prompt: str = "",
         endpoint: str = "geometric", abstain_band: float = 0.10,
         max_prompt_tokens: int = MAX_PROMPT_TOKENS):
    if mode == "validate":
        print(validate.remote(model_id, task, n, load_in_4bit, precision))
    elif mode == "build-stress-data":
        print(build_stress_data.remote(task, n, SEED))
    elif mode == "guard":
        result = guard_prompt.remote(model_id, task, prompt, n, load_in_4bit, precision,
                                     endpoint, abstain_band, max_prompt_tokens)
        print("FURNACE_GUARD_RESULT\n" + json.dumps(result, indent=2, default=str), flush=True)
    else:
        print(extract.remote(model_id, task, n, load_in_4bit, precision, max_dropped, abstain_band))
