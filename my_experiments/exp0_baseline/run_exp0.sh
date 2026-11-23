#!/bin/bash
# =============================================================================
# Experiment 0 - Baseline
# Evaluates the released UniVLA checkpoints (fine-tuned by the authors) on LIBERO,
# with the same settings as the paper (bf16, 50 trials per task, 10 tasks = 500 episodes).
# Compared against paper Table I, "Full" row:
#   Spatial 96.5 | Object 96.8 | Goal 95.6 | Long 92.0 | Avg 95.2
#
# Usage (INSIDE the container, preferably within tmux):
#   bash my_experiments/exp0_baseline/run_exp0.sh <suite> [seed]
#     suite: spatial | object | goal | long | all
#     seed:  default 7 (the authors' default)
#
# Output: /data/eval_logs/exp0_baseline/<suite>/
#   EVAL-*.txt          official script log (per-episode results)
#   console_seed<N>.log full console output
#   rollouts/           episode videos
# =============================================================================
set -eo pipefail

SUITE=${1:?"Usage: $0 <spatial|object|goal|long|all> [seed]"}
SEED=${2:-7}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)   # absolute: run_suite does cd into the output folder

MODELS=/data/models/univla-7b-224-sft-libero
OUT=/data/eval_logs/exp0_baseline
EVAL_SCRIPT=/workspace/univla/experiments/robot/libero/run_libero_eval.py
MIN_FREE_MB=17000   # the bf16 model needs ~16 GB

declare -A TASK=( [spatial]=libero_spatial [object]=libero_object [goal]=libero_goal [long]=libero_10 )
declare -A CKPT=( [spatial]=univla-libero-spatial [object]=univla-libero-object [goal]=univla-libero-goal [long]=univla-libero-10 )

check_gpu() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "[exp0] ERROR: only ${free} MB free on the GPU (need >= ${MIN_FREE_MB} MB)."
        echo "[exp0] Is someone else using the GPU? Check with nvidia-smi."
        exit 1
    fi
    echo "[exp0] Free GPU memory: ${free} MB - OK"
}

run_suite() {
    local s=$1
    if [ -z "${TASK[$s]}" ]; then
        echo "[exp0] Unknown suite: '$s' (use spatial|object|goal|long|all)"; exit 1
    fi
    check_gpu
    local dir="$OUT/$s"
    mkdir -p "$dir"
    cd "$dir"   # videos are saved under ./rollouts, so they land here
    echo "[exp0] === Suite: $s (${TASK[$s]}) | seed: $SEED | started: $(date) ==="
    python "$EVAL_SCRIPT" \
        --task_suite_name "${TASK[$s]}" \
        --pretrained_checkpoint "$MODELS/${CKPT[$s]}" \
        --action_decoder_path "$MODELS/${CKPT[$s]}/action_decoder.pt" \
        --num_trials_per_task 50 \
        --save_video True \
        --local_log_dir "$dir" \
        --run_id_note "seed$SEED" \
        --seed "$SEED" \
        2>&1 | tee "$dir/console_seed$SEED.log"
    echo "[exp0] === Suite $s finished: $(date) ==="
}

if [ "$SUITE" == "all" ]; then
    for s in goal spatial object long; do run_suite "$s"; done
else
    run_suite "$SUITE"
fi

bash "$SCRIPT_DIR/summarize_exp0.sh"
