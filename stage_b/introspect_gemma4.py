"""Introspect gemma-4-12B-it (gemma4_unified) to scope the adapter. Read-only; no seal touched."""
import inspect
import mlx_lm
from mlx_lm import load

MID = "mlx-community/gemma-4-12B-it-qat-4bit"
print("mlx_lm version:", mlx_lm.__version__)
model, tok = load(MID)
print("model class:", type(model).__module__, type(model).__name__)

cfg = getattr(model, "args", None) or getattr(model, "config", None)
if cfg is not None:
    for f in ["model_type", "num_hidden_layers", "hidden_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "sliding_window", "sliding_window_pattern",
              "query_pre_attn_scalar", "rope_theta", "rope_local_base_freq", "tie_word_embeddings",
              "vocab_size", "attn_logit_softcapping", "final_logit_softcapping", "rms_norm_eps"]:
        if hasattr(cfg, f):
            print(f"  cfg.{f} = {getattr(cfg, f)}")

print("dir(model):", [a for a in dir(model) if not a.startswith("_")][:50])

# locate the transformer layer stack
layers = None
for path in ["model.layers", "language_model.layers", "language_model.model.layers", "layers"]:
    obj, ok = model, True
    for p in path.split("."):
        if hasattr(obj, p):
            obj = getattr(obj, p)
        else:
            ok = False
            break
    if ok and hasattr(obj, "__len__"):
        layers = obj
        print(f"LAYERS at '{path}' count={len(obj)}")
        break

if layers is not None:
    l0 = layers[0]
    print("layer0 attrs:", [a for a in dir(l0) if not a.startswith("_")][:30])
    attn = getattr(l0, "self_attn", None) or getattr(l0, "attention", None)
    if attn is not None:
        print("ATTN class:", type(attn).__name__)
        print("ATTN attrs:", [a for a in dir(attn) if not a.startswith("_")][:40])
        for a in ["n_heads", "n_kv_heads", "num_heads", "num_kv_heads", "head_dim", "scale",
                  "is_sliding", "use_sliding_window", "q_norm", "k_norm"]:
            if hasattr(attn, a):
                print(f"   attn.{a} = {getattr(attn, a)}")
        try:
            print("\n===== ATTN __call__ SOURCE =====\n" + inspect.getsource(type(attn).__call__))
        except Exception as e:
            print("no attn source:", e)
    # embed / norm / head
    for nm in ["embed_tokens", "norm", "lm_head"]:
        for base in [model, getattr(model, "model", None), getattr(model, "language_model", None)]:
            if base is not None and hasattr(base, nm):
                print(f"FOUND {nm} on {type(base).__name__}")
                break
print("INTROSPECT_DONE")
