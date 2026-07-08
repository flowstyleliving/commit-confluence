"""Validate gemma-4 text forward + hidden/logit capture + locate W_u (readout half)."""
import inspect
import mlx.core as mx
from mlx_vlm import load

MID = "mlx-community/gemma-4-12B-it-qat-4bit"
m, proc = load(MID)
lm = m.language_model

try:
    print("LOGITS_FROM_HIDDEN_SRC:\n" + inspect.getsource(lm.logits_from_hidden))
except Exception as e:
    print("no logits_from_hidden src:", e)
print("lm attrs:", [a for a in dir(lm) if not a.startswith("_")][:45])

tok = getattr(proc, "tokenizer", proc)
msgs = [{"role": "user", "content": "Is the sky blue? Answer yes or no."}]
import numpy as np
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
if hasattr(enc, "input_ids"):          # BatchEncoding
    enc = enc["input_ids"]
ids = [int(t) for t in np.array(enc).reshape(-1)]
print("prompt token len:", len(ids))
x = mx.array([ids])

out = m(x, capture_layer_ids=list(range(48)), return_hidden=True)
print("out type:", type(out).__name__)
print("logits shape:", out.logits.shape)
hs = out.hidden_states
print("hidden_states: n=", len(hs) if hs else None, "each:", hs[0].shape if hs else None)

logits = out.logits[0, -1]
p = mx.softmax(logits.astype(mx.float32))
amax = int(mx.argmax(p))
pmax = float(p[amax])
surprise = float(-mx.log(p[amax] + 1e-10))
print(f"commit: p_max={pmax:.4f} surprise={surprise:.4f} argmax={amax} tok={tok.decode([amax])!r}")
print("READOUT_PROBE_DONE")
