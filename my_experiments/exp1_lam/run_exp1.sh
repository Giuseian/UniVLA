#!/bin/bash
# =============================================================================
# Experiment 1 - LAM code stability (our own metric, not in the paper)
# Do the task-centric latent action codes stay the same when we change things
# that do not matter for the task (light, colors, blur, noise, small camera shift)?
# Compares task-centric codes (indices) vs task-irrelevant codes (indices_uncontrol)
# of the released stage-2 LAM. Uses only ~3 GB of GPU.
#
# Usage (INSIDE the container):
#   bash my_experiments/exp1_lam/run_exp1.sh test   # quick check: Goal only, 20 pairs (~5 min)
#   bash my_experiments/exp1_lam/run_exp1.sh full   # 4 suites x 200 pairs (~30-60 min)
#
# Output: /data/eval_logs/exp1_lam/<test|full>/
#   results.md    table of results (open it to read the numbers)
#   results.json  same numbers, machine-readable
#   examples.png  what the perturbations look like
#   codes.npz     all the codes, for further analysis
#   console.log   full console output
# =============================================================================
set -eo pipefail

MODE=${1:?"Usage: $0 <test|full>"}
OUT=/data/eval_logs/exp1_lam/$MODE
SCRIPT=/workspace/univla/my_experiments/exp1_lam/lam_stability.py
MIN_FREE_MB=4000

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free" -lt "$MIN_FREE_MB" ]; then
    echo "[exp1] ERROR: only ${free} MB free on the GPU (need >= ${MIN_FREE_MB} MB)."; exit 1
fi
echo "[exp1] Free GPU memory: ${free} MB - OK"

case "$MODE" in
    test) ARGS="--suites goal --pairs_per_suite 20" ;;
    full) ARGS="--suites spatial object goal long --pairs_per_suite 200" ;;
    *) echo "[exp1] Unknown mode '$MODE' (use test|full)"; exit 1 ;;
esac

mkdir -p "$OUT"
echo "[exp1] === mode: $MODE | started: $(date) ==="
python "$SCRIPT" $ARGS --out_dir "$OUT" 2>&1 | tee "$OUT/console.log"
echo "[exp1] === finished: $(date) ==="
