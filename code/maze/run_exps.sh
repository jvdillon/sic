#!/bin/bash
# Usage: ./run_exps.sh x07c x07d x07e x07g x07i x07k x07l
# Runs experiments across 2 GPUs, one per GPU, moving on when one finishes/fails.
set -u

EXPS=("$@")
if [ ${#EXPS[@]} -eq 0 ]; then
  echo "Usage: $0 EXP1 EXP2 ..." >&2
  exit 1
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
IDX=0
TOTAL=${#EXPS[@]}
PIDS=()   # pid per gpu slot
NAMES=()  # exp name per gpu slot

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

# Start first two.
run_next 0
run_next 1

# Wait for any to finish, then start next.
while true; do
  any_running=false
  for gpu in 0 1; do
    pid=${PIDS[$gpu]}
    if [ "$pid" -ne 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null
      rc=$?
      if [ $rc -eq 0 ]; then
        echo "[GPU $gpu] ${NAMES[$gpu]} finished OK"
      else
        echo "[GPU $gpu] ${NAMES[$gpu]} failed (exit $rc)"
      fi
      run_next $gpu
    fi
    if [ "${PIDS[$gpu]}" -ne 0 ]; then
      any_running=true
    fi
  done
  $any_running || break
  sleep 5
done

echo "All done."
