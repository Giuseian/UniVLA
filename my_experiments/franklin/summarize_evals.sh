#!/bin/bash
# Prints the LIBERO results of the models fine-tuned on Franklin next to the paper.
# Usage (on fe01): bash my_experiments/franklin/summarize_evals.sh

DATA=/work/giannotti/univla
declare -A PAPER=( [exp3_human_goal]="90.2 (Tab. I, Human)" [exp3_bridge_goal]="91.9 (Tab. I, Bridge)" [exp4_10pct_goal]="86.3 (Fig. 10, 10%)" )

printf "\n%-24s | %-12s | %-22s | %-10s | %s\n" "Run" "Suite" "Paper" "Ours" "Episodes"
printf -- "-------------------------+--------------+------------------------+------------+---------\n"
for dir in "$DATA"/eval_logs/*/*/; do
    run=$(basename "$(dirname "$dir")"); suite=$(basename "$dir")
    for log in $(ls "$dir"EVAL-*.txt 2>/dev/null); do
        succ=$(grep "# successes:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
        eps=$(grep "# episodes completed so far:" "$log" | tail -1 | awk '{print $NF}')
        printf "%-24s | %-12s | %-22s | %-10s | %s\n" "$run" "$suite" "${PAPER[$run]:--}" "${succ:-n/a}%" "${eps:-0}"
    done
done
echo
