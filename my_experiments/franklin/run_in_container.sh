#!/bin/bash
# Runs a command inside the UniVLA Apptainer image on Franklin, with the same paths as the
# Docker container on the 9950x: code -> /workspace/univla, data -> /data.
# Usage: bash my_experiments/franklin/run_in_container.sh <command...>
#   e.g. bash my_experiments/franklin/run_in_container.sh python my_script.py
# Add --nv automatically when a GPU is visible.
set -eo pipefail
DATA=/work/giannotti/univla
type module >/dev/null 2>&1 || source /etc/profile   # make `module` available in non-login shells
module load apptainer-1.4.1
NV=""; command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && NV="--nv"
apptainer exec $NV \
    --bind "$HOME/projects/univla":/workspace/univla,"$DATA":/data \
    --pwd /workspace/univla \
    "$DATA/containers/univla_cu121.sif" "$@"
