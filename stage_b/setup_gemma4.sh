#!/bin/bash
# Parallel gemma4 venv + model + introspection. Does NOT touch the seal venv or the sealed t0 core.
LOG="$HOME/Documents/commit-confluence/stage_b/_ext_logs/gemma4_setup.log"
VENV="$HOME/Documents/commit-confluence/.venv_gemma4"
PYBASE="$(command -v python3.12 || command -v python3.11 || command -v python3)"
{
  echo "=== setup start ($(date)) base python: $PYBASE ==="
  "$PYBASE" --version
  if [ ! -x "$VENV/bin/python" ]; then echo "creating venv"; "$PYBASE" -m venv "$VENV"; fi
  "$VENV/bin/pip" install -q --upgrade pip 2>&1 | tail -1
  echo "installing mlx + mlx-lm (latest) + hf_hub ..."
  "$VENV/bin/pip" install -q -U mlx mlx-lm huggingface_hub 2>&1 | tail -6
  echo "=== version / gemma4 support ==="
  "$VENV/bin/python" -c "import mlx_lm,importlib.util as u; print('mlx_lm',mlx_lm.__version__); print('gemma4_module', u.find_spec('mlx_lm.models.gemma4') is not None); print('gemma4_unified_module', u.find_spec('mlx_lm.models.gemma4_unified') is not None)" 2>&1
  echo "=== download gemma-4-12B-it-qat-4bit ==="
  "$VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/gemma-4-12B-it-qat-4bit'))" 2>&1 | tail -2
  echo "=== introspect ==="
  "$VENV/bin/python" "$HOME/Documents/commit-confluence/stage_b/introspect_gemma4.py" 2>&1
  echo "=== setup done ($(date)) ==="
} > "$LOG" 2>&1
