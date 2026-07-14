"""Can the t0 extraction pipeline be imported+reused under the gemma4 venv (newer mlx-lm)?
If yes, we monkeypatch only the loader (mlx-vlm) and reuse ALL extraction code. Read-only probe."""
import sys, os, inspect
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
T0 = os.environ.get("CONFLUENCE_T0_REPO", os.path.join(REPO_ROOT, "vendor", "t0_core"))
sys.path.insert(0, T0)
sys.path.insert(0, os.path.join(T0, "exploratory/shadow-ambiguity"))

for m in ["pipeline", "pri_runtime", "model_adapters", "pri_v2_io_plugins", "io_plugins",
          "comprehensive_run", "test_shadow_ambiguity"]:
    try:
        __import__(m)
        print("IMPORT_OK  ", m)
    except Exception as e:
        print("IMPORT_FAIL", m, "->", type(e).__name__, str(e)[:170])

try:
    import pipeline
    print("\nload_model sig:", inspect.signature(pipeline.load_model))
    try:
        print("trace_sample sig:", inspect.signature(pipeline.trace_sample))
    except Exception as e:
        print("trace_sample sig err:", e)
    try:
        c = pipeline.Config()
        print("Config fields:", [a for a in dir(c) if not a.startswith("_")][:45])
    except Exception as e:
        print("Config err:", e)
    # what does load_model return / what does projection look like?
    for nm in ["PRIComputer", "load_model", "trace_sample", "OutputProjection", "Projection"]:
        print(f"  has pipeline.{nm}:", hasattr(pipeline, nm))
except Exception as e:
    print("pipeline inspect failed:", type(e).__name__, str(e)[:160])
print("REUSE_PROBE_DONE")
