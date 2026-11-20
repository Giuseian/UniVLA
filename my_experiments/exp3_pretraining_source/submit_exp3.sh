#!/bin/bash
# =============================================================================
# Experiment 3 - Which pretraining data? (paper Table I, LIBERO-Goal)
# Fine-tunes the models pretrained only on human videos (Ego4D) and only on Bridge
# on LIBERO-Goal, with the paper's recipe. Paper: Human 90.2 | Bridge 91.9 (Full 95.6).
#
# Usage (on Franklin, login node fe01, from ~/projects/univla):
#   bash my_experiments/exp3_pretraining_source/submit_exp3.sh <test|full>
# =============================================================================
set -eo pipefail
MODE=${1:?"Usage: $0 <test|full>"}
SUBMIT="$(dirname "$0")/../franklin/submit_finetune.sh"
bash "$SUBMIT" "$MODE" univla-7b-human-pt  libero_goal_no_noops exp3_human_goal
bash "$SUBMIT" "$MODE" univla-7b-bridge-pt libero_goal_no_noops exp3_bridge_goal
