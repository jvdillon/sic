#!/bin/bash
# Usage: ./run_exps.sh [-g GPUS] EXP1 EXP2 ...
# GPUS is a comma-separated list of GPU IDs (default: 0).
# Runs experiments across specified GPUs, one per GPU, advancing when one finishes.
set -u

GPU_LIST="0"
if [ "${1:-}" = "-g" ]; then
  GPU_LIST=$2
  shift 2
fi

IFS=',' read -ra GPUS <<< "$GPU_LIST"

EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then
  echo "Usage: $0 [-g GPU_IDS] EXP1 EXP2 ..." >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
IDX=0
TOTAL=${#EXPS[@]}
declare -A PIDS   # pid per gpu
declare -A NAMES  # exp name per gpu

run_next() {
  local gpu=$1
  if [ $IDX -ge $TOTAL ]; then
    PIDS[$gpu]=0
    return
  fi
  local exp=${EXPS[$IDX]}
  IDX=$((IDX + 1))
  echo "[GPU $gpu] Starting $exp"
  CUDA_VISIBLE_DEVICES=$gpu uvr "$DIR/$exp.py" &
  PIDS[$gpu]=$!
  NAMES[$gpu]=$exp
}

for gpu in "${GPUS[@]}"; do
  run_next "$gpu"
done

while true; do
  any_running=false
  for gpu in "${GPUS[@]}"; do
    pid=${PIDS[$gpu]:-0}
    if [ "$pid" -ne 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "[GPU $gpu] ${NAMES[$gpu]} finished OK"
      else
        echo "[GPU $gpu] ${NAMES[$gpu]} failed (exit $rc)"
      fi
      run_next "$gpu"
    fi
    if [ "${PIDS[$gpu]:-0}" -ne 0 ]; then
      any_running=true
    fi
  done
  $any_running || break
  sleep 5
done

echo "All done."
