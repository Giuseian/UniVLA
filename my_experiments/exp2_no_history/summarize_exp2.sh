#!/bin/bash
# Prints the Experiment 2 results (no history) next to the paper and to our Exp 0 (with history).
# Usage (from the host): docker exec univla bash my_experiments/exp2_no_history/summarize_exp2.sh

OUT=/data/eval_logs/exp2_no_history
declare -A PAPER_NOHIST=( [goal]=95.0 [long]=88.1 )
declare -A EXP0_HIST=( [goal]=92.6 [long]=90.4 )

printf "\n%-6s | %-14s | %-20s | %-13s | %s\n" "Suite" "Paper no-hist" "Ours no-hist" "Ours Exp0 hist" "Episodes"
printf -- "-------+----------------+----------------------+----------------+---------\n"
for s in goal long; do
    for log in $(ls "$OUT/$s"/EVAL-*.txt 2>/dev/null); do
        succ=$(grep "# successes:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
        eps=$(grep "# episodes completed so far:" "$log" | tail -1 | awk '{print $NF}')
        status=""
        [ "${eps:-0}" -lt 500 ] && status=" (incomplete)"
        printf "%-6s | %-14s | %-20s | %-14s | %s\n" "$s" "${PAPER_NOHIST[$s]}" "${succ:-n/a}%$status" "${EXP0_HIST[$s]}" "${eps:-0}"
    done
done
echo
