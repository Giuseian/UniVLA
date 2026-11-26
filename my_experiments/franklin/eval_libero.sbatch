#!/bin/bash
# =============================================================================
# SLURM job: evaluates a model fine-tuned on Franklin (output of finetune_libero.sbatch)
# on a LIBERO suite, with the same settings as Exp 0 (bf16, seed 7, 50 trials x 10 tasks).
#
# Do NOT launch it directly: use my_experiments/franklin/submit_eval.sh.
#   RUN_NAME   folder in /work/giannotti/univla/runs (e.g. exp3_human_goal)
#   SUITE      libero_goal | libero_spatial | libero_object | libero_10
#   TRIALS     trials per task (50 = official, 1 = quick test)
#   OUT_NAME   output folder name in /work/giannotti/univla/eval_logs
# =============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nv-a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/work/giannotti/univla/eval_logs/slurm-%x-%j.log
set -eo pipefail
: "${RUN_NAME:?}" "${SUITE:?}" "${TRIALS:?}" "${OUT_NAME:?}"
SEED=${SEED:-7}

DATA=/work/giannotti/univla
SIF=$DATA/containers/univla_cu121.sif
CODE=$HOME/projects/univla

# The fine-tuned model is the run sub-folder named after the experiment id (not adapter-tmp / wandb)
CKPT_HOST=$(ls -d "$DATA/runs/$RUN_NAME"/univla-* | head -1)
[ -f "$CKPT_HOST/model.safetensors.index.json" ] || { echo "[eval] no merged model in $CKPT_HOST"; exit 1; }
# Last saved action decoder (the one of the final step)
DECODER_HOST=$(ls "$CKPT_HOST"/action_decoder-*.pt | sort -V | tail -1)
CKPT=/data/${CKPT_HOST#$DATA/}
DECODER=/data/${DECODER_HOST#$DATA/}

OUT=/data/eval_logs/$OUT_NAME/$SUITE
mkdir -p "$DATA/eval_logs/$OUT_NAME/$SUITE"

# LIBERO keeps its config in ~/.libero by default; in Apptainer HOME is the Franklin home,
# so we keep it in the project cache and create it (default paths, /opt/LIBERO) if missing.
export LIBERO_CONFIG_PATH=/data/cache/libero

echo "[eval] job $SLURM_JOB_ID on $(hostname) | started $(date)"
echo "[eval] run=$RUN_NAME suite=$SUITE trials=$TRIALS seed=$SEED"
echo "[eval] model:   $CKPT"
echo "[eval] decoder: $DECODER"
echo "[eval] output:  $OUT"
echo "[eval] code version: $(git -C "$CODE" log --oneline -1)"
nvidia-smi --query-gpu=name,memory.total --format=csv

type module >/dev/null 2>&1 || source /etc/profile   # make `module` available in non-login shells
module load apptainer-1.4.1
apptainer exec --nv \
    --bind "$CODE":/workspace/univla,"$DATA":/data \
    --pwd "$OUT" \
    "$SIF" \
    bash -c "
        set -eo pipefail
        [ -f $LIBERO_CONFIG_PATH/config.yaml ] || echo n | python -c 'import libero.libero'
        python /workspace/univla/experiments/robot/libero/run_libero_eval.py \
            --task_suite_name $SUITE \
            --pretrained_checkpoint $CKPT \
            --action_decoder_path $DECODER \
            --num_trials_per_task $TRIALS \
            --save_video True \
            --local_log_dir $OUT \
            --run_id_note $RUN_NAME--seed$SEED \
            --seed $SEED
    "

echo "[eval] finished $(date)"
