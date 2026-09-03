#!/bin/bash
# Grid-B staged extraction driver — PRE_REGISTRATION_EXPANSION.md §7.
# Detached: run under nohup; fires each cell as its own `modal run --detach`,
# waits for TERMINAL STATUS FILES on the volume between stages. Never scores.
set -u
MODAL=/Users/msrk/Library/Python/3.9/bin/modal
CD=/Users/msrk/Documents/commit-confluence
LOGD=$CD/exploratory/depth-curve/_gridb_logs
# Volume namespace for this run's npz + status files. Defaults to the REGISTERED
# grid-B tree, so an unmodified invocation still targets the frozen cells (and is
# correctly refused by the terminal-state immutability guard). Override for an
# unregistered lane, e.g. OUT_DIR=depth_grid_b_vnorm ./run_grid_b.sh
OUT_DIR=${OUT_DIR:-depth_grid_b}
cd "$CD" || exit 1

slug_of() {
  case "$1" in
    llama31_8b) echo "Llama-3.1-8B-Instruct";;
    llama31_70b) echo "Llama-3.1-70B-Instruct";;
    mistral_small_32) echo "Mistral-Small-3.2-24B-Instruct-2506";;
    mistral_medium_35) echo "Mistral-Medium-3.5-128B";;
    gemma3_12b) echo "gemma-3-12b-it";;
    gemma3_27b) echo "gemma-3-27b-it";;
  esac
}

launch_cell() { # key task
  local key=$1 task=$2
  nohup "$MODAL" run --detach exploratory/depth-curve/modal_depth_b.py::extract \
    --model-key "$key" --task "$task" --out-dir "$OUT_DIR" \
    > "$LOGD/ex_${key}_${task}.log" 2>&1 &
  echo "$(date +%H:%M:%S) launched $key/$task"
}

wait_stage() { # "key:task key:task ..."  max_minutes
  local cells="$1" maxmin=$2 waited=0
  while true; do
    local missing=0
    for ct in $cells; do
      local key="${ct%%:*}" task="${ct##*:}"
      local slug; slug=$(slug_of "$key")
      if ! "$MODAL" volume ls model-cache "$OUT_DIR/$task" 2>/dev/null \
          | grep -q "$slug.status.json"; then
        missing=$((missing+1))
      fi
    done
    if [ "$missing" -eq 0 ]; then echo "$(date +%H:%M:%S) stage terminal"; return 0; fi
    if [ "$waited" -ge $((maxmin*60)) ]; then
      echo "$(date +%H:%M:%S) STAGE TIMEOUT ($missing cells non-terminal)"; return 1
    fi
    sleep 120; waited=$((waited+120))
  done
}

echo "=== STAGE 1: small models (8 cells) ==="
S1=""
for key in llama31_8b mistral_small_32 gemma3_12b gemma3_27b; do
  for task in anli_r1 halueval_qa; do launch_cell "$key" "$task"; S1="$S1 $key:$task"; done
done
wait_stage "$S1" 240 || echo "stage1 timeout — continuing (statuses may still land)"

echo "=== STAGE 2: Llama-3.1-70B (2 cells) ==="
S2=""
for task in anli_r1 halueval_qa; do launch_cell llama31_70b "$task"; S2="$S2 llama31_70b:$task"; done
wait_stage "$S2" 300 || echo "stage2 timeout — continuing"

echo "=== STAGE 3: Mistral-Medium-3.5 (2 cells) ==="
S3=""
for task in anli_r1 halueval_qa; do launch_cell mistral_medium_35 "$task"; S3="$S3 mistral_medium_35:$task"; done
wait_stage "$S3" 600 || echo "stage3 timeout"

echo "=== FINAL CHECK: all 12 terminal? ==="
ALL=""
for key in llama31_8b llama31_70b mistral_small_32 mistral_medium_35 gemma3_12b gemma3_27b; do
  for task in anli_r1 halueval_qa; do ALL="$ALL $key:$task"; done
done
if wait_stage "$ALL" 5; then
  touch "$LOGD/GRID_B_ALL_TERMINAL"
  echo "$(date +%H:%M:%S) ALL 12 TERMINAL — marker written"
else
  echo "$(date +%H:%M:%S) NOT all terminal at driver end"
fi
