"""Blueprint dump for a manual gemma-4 text forward (ACE attn capture + readout). Read-only."""
import inspect
from mlx_vlm import load

MID = "mlx-community/gemma-4-12B-it-qat-4bit"
m, proc = load(MID)
print("top model:", type(m).__module__, type(m).__name__)
lm = getattr(m, "language_model", m)
print("language_model:", type(lm).__name__)
inner = getattr(lm, "model", lm)
print("inner:", type(inner).__name__, "attrs:", [a for a in dir(inner) if not a.startswith("_")][:35])
layers = inner.layers
print("n_layers:", len(layers))
for nm, obj in [("inner.embed_tokens", getattr(inner, "embed_tokens", None)),
                ("inner.norm", getattr(inner, "norm", None)),
                ("lm.lm_head", getattr(lm, "lm_head", None)),
                ("m.lm_head", getattr(m, "lm_head", None))]:
    print(f"  {nm}: {'present '+type(obj).__name__ if obj is not None else 'absent'}")
try:
    print("embed_tokens.weight shape:", inner.embed_tokens.weight.shape)
except Exception as e:
    print("embed shape err:", e)
for lab, obj in [("Model.__call__", type(m)), ("LM.__call__", type(lm)),
                 ("inner.__call__", type(inner)), ("layer.__call__", type(layers[0]))]:
    try:
        print(f"\n===== {lab} =====\n" + inspect.getsource(obj.__call__))
    except Exception as e:
        print(f"{lab}: no src ({e})")
for i in [0, 5]:
    a = layers[i].self_attn
    print(f"\nlayer{i} attn: layer_type={getattr(a,'layer_type','?')} n_heads={a.n_heads} "
          f"n_kv={a.n_kv_heads} head_dim={a.head_dim} scale={getattr(a,'scale','?')} "
          f"sliding={getattr(a,'is_sliding','?')} k_eq_v={getattr(a,'use_k_eq_v',getattr(a,'kv_shared_only','?'))}")
tok = getattr(proc, "tokenizer", proc)
print("\ntokenizer:", type(tok).__name__, "chat_template:", bool(getattr(tok, "chat_template", None)))
print("STRUCT_DONE")
