#!/bin/bash
# =============================================================================
# Submits the LIBERO evaluation of a model fine-tuned on Franklin (run it on fe01).
#
# Usage:
#   bash my_experiments/franklin/submit_eval.sh <test|full> <run_name> [suite]
#     test: partition debug (15 min), 1 trial per task = 10 episodes -> checks that it works
#     full: partition gpua, 50 trials per task = 500 episodes (as Exp 0 and the paper)
#     suite: default libero_goal
#
# Example:
#   bash my_experiments/franklin/submit_eval.sh test exp3_human_goal
#
# Results: bash my_experiments/franklin/summarize_evals.sh
# =============================================================================
set -eo pipefail

MODE=${1:?"Usage: $0 <test|full> <run_name> [suite]"}
RUN_NAME=${2:?"missing run_name"}
SUITE=${3:-libero_goal}
DATA=/work/giannotti/univla

ls -d "$DATA/runs/$RUN_NAME"/univla-* >/dev/null 2>&1 || { echo "[submit] no model in $DATA/runs/$RUN_NAME"; exit 1; }

case "$MODE" in
    test) PART=debug; TIME=00:15:00; TRIALS=1;  OUT_NAME="${RUN_NAME}_test" ;;
    full) PART=gpua;  TIME=12:00:00; TRIALS=50; OUT_NAME="$RUN_NAME" ;;
    *) echo "[submit] unknown mode '$MODE' (use test|full)"; exit 1 ;;
esac

if [ "$MODE" == "full" ] && ls "$DATA/eval_logs/$OUT_NAME/$SUITE"/EVAL-*.txt >/dev/null 2>&1; then
    echo "[submit] $DATA/eval_logs/$OUT_NAME/$SUITE already has results: delete them first."; exit 1
fi
mkdir -p "$DATA/eval_logs"

sbatch --partition "$PART" --time "$TIME" --job-name "eval_$OUT_NAME" \
    --export=ALL,RUN_NAME="$RUN_NAME",SUITE="$SUITE",TRIALS="$TRIALS",OUT_NAME="$OUT_NAME" \
    "$(dirname "$0")/eval_libero.sbatch"
echo "[submit] log: $DATA/eval_logs/slurm-eval_$OUT_NAME-<jobid>.log"
