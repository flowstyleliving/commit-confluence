"""Modal infrastructure scaffold for commit-confluence extraction on larger (NVIDIA) models.

This is the INFRA skeleton, not the extractor yet. It proves the Modal path end to end:
build a torch+transformers image, mount a persistent HF weight cache + HF token secret,
grab a GPU, load a model, and report its config. The real extractor (a PyTorch port of the
MLX ACE/readout capture that emits the same matrix.npz schema) drops into `extract()` below.

Why a new backend at all: the sealed pipeline is MLX (Apple-Silicon only). Modal runs on
NVIDIA GPUs, so model forwards must use PyTorch/HF. Calibration stays the identical numpy/sklearn
code. ===> Any Modal result is NON-byte-comparable to the seal (different framework + dtype),
exactly like the gemma-4 mlx-vlm cell. It is a standalone exploratory panel; never pool it with
the sealed or byte-comparable cells.

Usage (after `pip install modal` and `modal setup`):
    modal run modal/modal_app.py                         # smoke test default model
    modal run modal/modal_app.py --model-id meta-llama/Llama-3.3-70B-Instruct
"""
import modal

app = modal.App("commit-confluence-extract")

# --- image: torch + HF stack. Pin transformers high enough for the target archs. ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.46",
        "accelerate>=0.34",
        "numpy",
        "scikit-learn",
        "sentencepiece",
        "safetensors",
        "huggingface_hub",
        # "bitsandbytes",  # uncomment for 4-bit loading of 70B+ on a single 80GB GPU
    )
)

# --- persistent HF weight cache so cold starts don't re-download tens of GB ---
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE = "/cache"

# --- HF token for gated models (Llama/Gemma). Create once with your local token:
#       modal secret create huggingface HF_TOKEN=$(cat ~/.cache/huggingface/token)
hf_secret = modal.Secret.from_name("huggingface")

GPU_CONFIG = "A100-80GB"  # see README for sizing; "H100", "A100-40GB", "A100-80GB:2" also valid


def _load(model_id: str):
    """Load tokenizer + model with eager attention (so ACE hooks can see real weights)."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = HF_CACHE
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # flash/sdpa fuse softmax and hide attention weights
    )
    model.eval()
    return tok, model


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={HF_CACHE: hf_cache},
    secrets=[hf_secret],
    timeout=60 * 60 * 2,
)
def smoke(model_id: str):
    """Load a model and report its shape — confirms image, GPU, cache, and secret all work."""
    import torch

    tok, model = _load(model_id)
    cfg = model.config
    hf_cache.commit()  # persist any newly downloaded weights
    return {
        "model": model_id,
        "n_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "param_dtype": str(next(model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0),
        "mem_gb": round(torch.cuda.memory_allocated() / 1e9, 1),
    }


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={HF_CACHE: hf_cache},
    secrets=[hf_secret],
    timeout=60 * 60 * 6,
)
def extract(model_id: str, task: str, data_jsonl: bytes, seal_ref_npz: bytes) -> bytes:
    """PLACEHOLDER for the PyTorch ACE+readout extractor.

    Port target (mirrors stage_b/gemma4_full_extract.py), per tagged layer {final, mid, last_minus_1}:
      caps[tag]  : last-query softmax attention weights, shape [H, T]
      vcaps[tag] : per-KV-head value norms,              shape [n_kv, T]
      nkv[tag]   : int number of KV heads
    via a forward hook on each tagged attention module that recomputes
      softmax(q @ k^T * scale + causal_mask)[last_query]   (post q/k-norm + RoPE, eager attn)
    plus logits (surprise/p_max/gid) and hidden states h_prev (D0) / h_t (D1=[prompt+gid]),
    and W_u rows for the support spectrum (model.get_output_embeddings().weight).
    VALIDATE the recompute against the model's own o_proj output (cos must be ~1.0) before trusting.
    Feed the same ATTENTION_PANEL_T0_WITH_V_NORMS + _compute_attention_score + calibrate_merged.
    Returns the matrix.npz bytes; calibration runs here or downloaded and run locally.
    """
    raise NotImplementedError("extractor port pending — see README and decisions")


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen2.5-14B-Instruct"):
    print(smoke.remote(model_id))
