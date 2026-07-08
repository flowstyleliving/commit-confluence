#!/bin/bash
LOG="$HOME/Documents/commit-confluence/stage_b/_ext_logs/gemma4_loaders.log"
VENV="$HOME/Documents/commit-confluence/.venv_gemma4"
{
  echo "=== install mlx-lm git main ($(date)) ==="
  "$VENV/bin/pip" install -q -U "git+https://github.com/ml-explore/mlx-lm" 2>&1 | tail -3
  echo "=== install mlx-vlm ==="
  "$VENV/bin/pip" install -q -U mlx-vlm 2>&1 | tail -3
  echo "=== loader probe ==="
  "$VENV/bin/python" "$HOME/Documents/commit-confluence/stage_b/try_loaders_gemma4.py" 2>&1
  echo "=== done ($(date)) ==="
} > "$LOG" 2>&1
