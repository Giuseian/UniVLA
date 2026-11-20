#!/bin/bash
# =============================================================================
# Submits a UniVLA fine-tuning job on Franklin (run it on the login node fe01).
#
# Usage:
#   bash my_experiments/franklin/submit_finetune.sh <test|full> <model> <dataset> <run_name>
#     test: partition debug (max 15 min), 10 steps, saves at the end -> checks that everything works
#     full: partition gpua-longrun (max 3 days), 30000 steps (paper), checkpoint every 10000
#
# Example:
#   bash my_experiments/franklin/submit_finetune.sh test univla-7b-human-pt libero_goal_no_noops exp3_human_goal
#
# Follow the job:  squeue -u $USER
# Read the log:    tail -f /work/giannotti/univla/runs/slurm-<run_name>-<jobid>.log
# Cancel the job:  scancel <jobid>
# =============================================================================
set -eo pipefail

MODE=${1:?"Usage: $0 <test|full> <model> <dataset> <run_name>"}
MODEL=${2:?"missing model"}
DATASET=${3:?"missing dataset"}
RUN_NAME=${4:?"missing run_name"}
DATA=/work/giannotti/univla

[ -d "$DATA/models/$MODEL" ] || { echo "[submit] model not found: $DATA/models/$MODEL"; exit 1; }
[ -d "$DATA/datasets/libero_rlds/$DATASET" ] || { echo "[submit] dataset not found: $DATA/datasets/libero_rlds/$DATASET"; exit 1; }

case "$MODE" in
    # QOS limits on Franklin: debug max 8 CPUs per job, gpua-long max 96
    test) PART=debug;        TIME=00:15:00;   STEPS=10;    SAVE=10;    CPUS=8;  RUN_NAME="${RUN_NAME}_test" ;;
    full) PART=gpua-longrun; TIME=3-00:00:00; STEPS=30000; SAVE=10000; CPUS=32 ;;
    *) echo "[submit] unknown mode '$MODE' (use test|full)"; exit 1 ;;
esac

if [ -d "$DATA/runs/$RUN_NAME" ] && [ "$MODE" == "full" ]; then
    echo "[submit] $DATA/runs/$RUN_NAME already exists: choose another run_name or delete it first."; exit 1
fi
mkdir -p "$DATA/runs"

sbatch --partition "$PART" --time "$TIME" --job-name "$RUN_NAME" --cpus-per-task "$CPUS" \
    --export=ALL,MODEL="$MODEL",DATASET="$DATASET",RUN_NAME="$RUN_NAME",MAX_STEPS="$STEPS",SAVE_STEPS="$SAVE" \
    "$(dirname "$0")/finetune_libero.sbatch"
echo "[submit] log: $DATA/runs/slurm-$RUN_NAME-<jobid>.log"
