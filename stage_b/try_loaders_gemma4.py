"""Probe whether mlx-lm(main) or mlx-vlm can load gemma4_unified, and dump its attention source."""
import importlib.util as u, inspect

def has(m):
    return u.find_spec(m) is not None

MID = "mlx-community/gemma-4-12B-it-qat-4bit"
import mlx_lm
print("mlx_lm", mlx_lm.__version__, "gemma4_unified_module:", has("mlx_lm.models.gemma4_unified"))

# --- attempt mlx-lm load ---
try:
    from mlx_lm import load
    m, t = load(MID)
    print("MLX_LM_LOAD_OK", type(m).__name__)
except Exception as e:
    print("MLX_LM_LOAD_FAIL", type(e).__name__, str(e)[:160])

# --- attempt mlx-vlm load ---
print("mlx_vlm installed:", has("mlx_vlm"))
if has("mlx_vlm"):
    try:
        from mlx_vlm import load as vload
        m, proc = vload(MID)
        print("MLX_VLM_LOAD_OK", type(m).__name__)
        layers = None
        for path in ["language_model.model.layers", "language_model.layers",
                     "model.language_model.layers", "model.layers"]:
            o, ok = m, True
            for pp in path.split("."):
                if hasattr(o, pp):
                    o = getattr(o, pp)
                else:
                    ok = False
                    break
            if ok and hasattr(o, "__len__"):
                layers = o
                print("LAYERS at", path, "n=", len(o))
                break
        if layers is not None:
            a = getattr(layers[0], "self_attn", None) or getattr(layers[0], "attention", None)
            if a is not None:
                print("ATTN", type(a).__name__,
                      "attrs:", [x for x in dir(a) if not x.startswith("_")][:30])
                try:
                    print("ATTN_SRC_BEGIN\n" + inspect.getsource(type(a).__call__) + "\nATTN_SRC_END")
                except Exception as e:
                    print("no attn src:", e)
    except Exception as e:
        import traceback
        print("MLX_VLM_LOAD_FAIL", type(e).__name__, str(e)[:200])
        print(traceback.format_exc()[:1800])
print("LOADERS_DONE")
