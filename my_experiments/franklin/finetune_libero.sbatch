#!/bin/bash
# =============================================================================
# SLURM job: fine-tunes a pretrained UniVLA model on a LIBERO dataset on Franklin
# (Apptainer image + A100 80GB). Same recipe as the paper (vla-scripts/finetune_libero.py).
#
# Do NOT launch it directly: use my_experiments/franklin/submit_finetune.sh,
# which sets the variables below and the partition/time.
#   MODEL       folder in /work/giannotti/univla/models (e.g. univla-7b-human-pt)
#   DATASET     LIBERO RLDS dataset name (e.g. libero_goal_no_noops)
#   RUN_NAME    output folder name in /work/giannotti/univla/runs
#   MAX_STEPS   training steps (paper: 30000 for Goal)
#   SAVE_STEPS  checkpoint interval
#   NGPU        number of GPUs (default 4)
# =============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nv-a100-sxm4-80gb:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --output=/work/giannotti/univla/runs/slurm-%x-%j.log
set -eo pipefail
: "${MODEL:?}" "${DATASET:?}" "${RUN_NAME:?}" "${MAX_STEPS:?}" "${SAVE_STEPS:?}"
NGPU=${NGPU:-4}

# Keep the paper's global batch: 8 per GPU x 8 GPUs x 2 accumulation = 128 = 8 x NGPU x GRAD_ACC
GRAD_ACC=$(( 16 / NGPU ))

DATA=/work/giannotti/univla
SIF=$DATA/containers/univla_cu121.sif
CODE=$HOME/projects/univla
RUN_DIR=$DATA/runs/$RUN_NAME
mkdir -p "$RUN_DIR"

# W&B: no account needed, logs are kept locally in the run folder
export WANDB_MODE=offline
export WANDB_DIR=/data/runs/$RUN_NAME

echo "[ft] job $SLURM_JOB_ID on $(hostname) | started $(date)"
echo "[ft] model=$MODEL dataset=$DATASET run=$RUN_NAME steps=$MAX_STEPS save=$SAVE_STEPS gpus=$NGPU grad_acc=$GRAD_ACC"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "[ft] code version: $(git -C "$CODE" log --oneline -1)"

type module >/dev/null 2>&1 || source /etc/profile   # make `module` available in non-login shells
module load apptainer-1.4.1

# Download DINOv2 (used by the LAM) ONCE before starting the GPU processes:
# otherwise the 4 torchrun processes download it at the same time into the same
# torch.hub cache folder and crash (FileExistsError / Directory not empty).
apptainer exec --bind "$DATA":/data "$SIF" \
    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg'); print('[ft] DINOv2 cache OK:', torch.hub.get_dir())"

apptainer exec --nv \
    --bind "$CODE":/workspace/univla,"$DATA":/data \
    --pwd /workspace/univla \
    "$SIF" \
    torchrun --standalone --nnodes 1 --nproc-per-node "$NGPU" vla-scripts/finetune_libero.py \
        --vla_path /data/models/"$MODEL" \
        --lam_path /data/models/univla-latent-action-model/lam-stage-2.ckpt \
        --data_root_dir /data/datasets/libero_rlds \
        --dataset_name "$DATASET" \
        --run_root_dir /data/runs/"$RUN_NAME" \
        --adapter_tmp_dir /data/runs/"$RUN_NAME"/adapter-tmp \
        --batch_size 8 \
        --grad_accumulation_steps "$GRAD_ACC" \
        --max_steps "$MAX_STEPS" \
        --save_steps "$SAVE_STEPS"

echo "[ft] finished $(date)"
