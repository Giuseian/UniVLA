#!/bin/bash
# =============================================================================
# Experiment 4 - Few robot data (paper Fig. 10, LIBERO-Goal)
# Fine-tunes the Full pretrained model (univla-7b) on 10% of LIBERO-Goal. Paper: 86.3.
# Same number of steps as the full-data run (30000): the paper does not say otherwise.
#
# Before the first run, create the 10% dataset (once):
#   srun -p cpu -c 4 --mem=32G -t 00:30:00 \
#     bash my_experiments/franklin/run_in_container.sh \
#     python my_experiments/exp4_data_efficiency/make_libero_subset.py
#
# Usage (on Franklin, login node fe01, from ~/projects/univla):
#   bash my_experiments/exp4_data_efficiency/submit_exp4.sh <test|full>
# =============================================================================
set -eo pipefail
MODE=${1:?"Usage: $0 <test|full>"}
bash "$(dirname "$0")/../franklin/submit_finetune.sh" "$MODE" univla-7b libero_goal_no_noops_mini exp4_10pct_goal
