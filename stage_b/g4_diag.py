"""Diagnostic: is the gemma-4 commit a YES/NO answer, or noise from a prompt-format bug?

For triviaqa examples 0,1: print RAW prompt, the gemma-4 strategy-wrapped prompt, the
gemma-3-12b strategy-wrapped prompt, whether they match, whether the wrapped text carries
gemma chat-template markers, and the model's commit (argmax) top-6 tokens for each variant.
Run in the gemma4 venv.
"""
import sys, os, json
import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.expanduser("~/Documents/t0-morphology-furnace"))
sys.path.insert(0, os.path.expanduser("~/Documents/t0-morphology-furnace/exploratory/shadow-ambiguity"))
sys.path.insert(0, os.path.expanduser("~/Documents/commit-confluence"))
sys.path.insert(0, os.path.expanduser("~/Documents/commit-confluence/stage_b"))

import comprehensive_run as CR
import pri_v2_io_plugins as io_plugins
from mlx_vlm import load as vload

G4_ID = "mlx-community/gemma-4-12B-it-qat-4bit"
G3_ID = "mlx-community/gemma-3-12b-it-4bit"
DATA = os.path.expanduser("~/Documents/commit-confluence/stage_b/data/triviaqa_paired_seed20260612_n200.jsonl")

prompts, labels, _ = CR._load_calibration_jsonl(DATA)
m, proc = vload(G4_ID)
tok = getattr(proc, "tokenizer", proc)
lm = m.language_model


def commit_top6(text):
    enc = tok(text)["input_ids"]
    ids = [int(t) for t in np.array(enc).reshape(-1)]
    out = lm(mx.array([ids]))
    logits = np.asarray(out.logits[0, -1].astype(mx.float32), dtype=np.float64)
    order = np.argsort(logits)[::-1][:6]
    p = np.exp(logits - logits.max()); p /= p.sum()
    return [(int(t), repr(tok.decode([int(t)])), round(float(p[t]), 4)) for t in order], len(ids)


s4 = io_plugins.get_prompt_strategy(G4_ID)
s3 = io_plugins.get_prompt_strategy(G3_ID)
MARK = "<start_of_turn>"

for i in (0, 1):
    p, y = prompts[i], labels[i]
    raw = p
    w4 = s4(p, tok)
    w3 = s3(p, tok)
    print("=" * 80)
    print(f"EX {i}  label={y}")
    print(f"--- RAW prompt (last 200 chars) ---\n...{raw[-200:]}")
    print(f"\n--- gemma-4 strategy wrap (last 200) ---\n...{w4[-200:]}")
    print(f"--- gemma-3-12b strategy wrap (last 200) ---\n...{w3[-200:]}")
    print(f"\nw4==w3 ? {w4 == w3}   w4 has {MARK!r}? {MARK in w4}   raw has it? {MARK in raw}")
    # try the official chat template too
    try:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        chat_has = MARK in chat
    except Exception as e:
        chat = None; chat_has = f"ERR {e}"
    print(f"apply_chat_template available? {chat is not None}  has marker? {chat_has}")

    top_raw, n_raw = commit_top6(w4)
    print(f"\nCOMMIT top-6 via gemma-4 strategy wrap (ntok={n_raw}):")
    for t in top_raw:
        print("   ", t)
    if chat is not None:
        top_chat, n_chat = commit_top6(chat)
        print(f"COMMIT top-6 via apply_chat_template (ntok={n_chat}):")
        for t in top_chat:
            print("   ", t)
