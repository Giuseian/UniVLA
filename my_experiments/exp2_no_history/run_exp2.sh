#!/bin/bash
# =============================================================================
# Experiment 2 - No history
# Same as Experiment 0 (same released checkpoints, bf16, 500 episodes per suite),
# but WITHOUT the history latent actions in the prompt (--use_history False).
# Only Goal and Long, compared against paper Table V:
#   without history: Goal 95.0 | Long 88.1
#   with history:    Goal 95.6 | Long 92.0   (our Exp 0: 92.6 | 90.4)
# NOTE: the released checkpoints were TRAINED with history; here we only remove
# it at inference time.
#
# Usage (INSIDE the container, preferably within tmux):
#   bash my_experiments/exp2_no_history/run_exp2.sh <goal|long|all> [seed]
#
# Output: /data/eval_logs/exp2_no_history/<suite>/  (same layout as Exp 0)
# =============================================================================
set -eo pipefail

SUITE=${1:?"Usage: $0 <goal|long|all> [seed]"}
SEED=${2:-7}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)   # absolute: run_suite does cd into the output folder

MODELS=/data/models/univla-7b-224-sft-libero
OUT=/data/eval_logs/exp2_no_history
EVAL_SCRIPT=/workspace/univla/experiments/robot/libero/run_libero_eval.py
MIN_FREE_MB=17000   # the bf16 model needs ~16 GB

declare -A TASK=( [goal]=libero_goal [long]=libero_10 )
declare -A CKPT=( [goal]=univla-libero-goal [long]=univla-libero-10 )

check_gpu() {
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt "$MIN_FREE_MB" ]; then
        echo "[exp2] ERROR: only ${free} MB free on the GPU (need >= ${MIN_FREE_MB} MB)."
        echo "[exp2] Is someone else using the GPU? Check with nvidia-smi."
        exit 1
    fi
    echo "[exp2] Free GPU memory: ${free} MB - OK"
}

run_suite() {
    local s=$1
    if [ -z "${TASK[$s]}" ]; then
        echo "[exp2] Unknown suite: '$s' (use goal|long|all)"; exit 1
    fi
    check_gpu
    local dir="$OUT/$s"
    mkdir -p "$dir"
    cd "$dir"   # videos are saved under ./rollouts, so they land here
    echo "[exp2] === Suite: $s (${TASK[$s]}) | NO history | seed: $SEED | started: $(date) ==="
    python "$EVAL_SCRIPT" \
        --task_suite_name "${TASK[$s]}" \
        --pretrained_checkpoint "$MODELS/${CKPT[$s]}" \
        --action_decoder_path "$MODELS/${CKPT[$s]}/action_decoder.pt" \
        --num_trials_per_task 50 \
        --use_history False \
        --save_video True \
        --local_log_dir "$dir" \
        --run_id_note "nohist--seed$SEED" \
        --seed "$SEED" \
        2>&1 | tee "$dir/console_seed$SEED.log"
    echo "[exp2] === Suite $s finished: $(date) ==="
}

if [ "$SUITE" == "all" ]; then
    for s in goal long; do run_suite "$s"; done
else
    run_suite "$SUITE"
fi

bash "$SCRIPT_DIR/summarize_exp2.sh"
