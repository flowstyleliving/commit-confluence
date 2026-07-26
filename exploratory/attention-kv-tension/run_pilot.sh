#!/usr/bin/env bash
# Exploratory KV-tension pilot.
#
# This is NOT a sealed ACE rerun. It reuses the sealed ANLI data only as a
# first pilot substrate for the opt-in --attention-kv-tension panel.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="${DATA:-$REPO_ROOT/experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl}"
OUT="${OUT:-$REPO_ROOT/exploratory/attention-kv-tension/pilot_outputs/2026-06-09/anli_r1_t0_kv_tension}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
TASK_LABEL="${TASK_LABEL:-kv_tension_pilot_anli_r1_t0_20260609}"

MODELS=(
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
)

mkdir -p "$OUT"
LOG="$OUT/pilot.log"

{
  echo "[kv-tension-pilot] start: $(date)"
  echo "[kv-tension-pilot] pid=$$"
  echo "[kv-tension-pilot] repo=$REPO_ROOT"
  echo "[kv-tension-pilot] data=$DATA"
  echo "[kv-tension-pilot] out=$OUT"
  echo "[kv-tension-pilot] n_bootstrap=$N_BOOTSTRAP"
  echo "[kv-tension-pilot] models=${#MODELS[@]}"
} | tee -a "$LOG"

cd "$REPO_ROOT" || exit 1

for M in "${MODELS[@]}"; do
  NAME="${M##*/}"
  PROFILE_OUT="$OUT/${NAME}.profile.json"

  if [[ -f "$PROFILE_OUT" ]]; then
    echo "[kv-tension-pilot] skip exists: $NAME" | tee -a "$LOG"
    continue
  fi

  echo "" | tee -a "$LOG"
  echo "[kv-tension-pilot] === model: $M  $(date) ===" | tee -a "$LOG"

  if PYTHONUNBUFFERED=1 .venv/bin/python -u pri_calibrator.py \
      --model "$M" \
      --data "$DATA" \
      --out "$PROFILE_OUT" \
      --task-label "$TASK_LABEL" \
      --t0-commit \
      --attention-kv-tension \
      --attention-only \
      --n-bootstrap "$N_BOOTSTRAP" \
      > "$OUT/${NAME}.log" 2>&1; then
    echo "[kv-tension-pilot] done: $NAME  $(date)" | tee -a "$LOG"
  else
    echo "[kv-tension-pilot] FAILED: $NAME  $(date)" | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
echo "[kv-tension-pilot] complete: $(date)" | tee -a "$LOG"
