#!/bin/bash
# Prints the Experiment 0 results next to the paper's numbers.
# Usage (inside the container): bash my_experiments/exp0_baseline/summarize_exp0.sh

OUT=/data/eval_logs/exp0_baseline
declare -A PAPER=( [spatial]=96.5 [object]=96.8 [goal]=95.6 [long]=92.0 )

printf "\n%-8s | %-6s | %-12s | %-8s | %s\n" "Suite" "Paper" "Ours" "Episodes" "Log"
printf -- "---------+--------+--------------+----------+-----\n"
for s in spatial object goal long; do
    for log in $(ls "$OUT/$s"/EVAL-*.txt 2>/dev/null); do
        # last line "# successes: X (Y%)" and number of completed episodes
        succ=$(grep "# successes:" "$log" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
        eps=$(grep "# episodes completed so far:" "$log" | tail -1 | awk '{print $NF}')
        status=""
        [ "${eps:-0}" -lt 500 ] && status=" (incomplete)"
        printf "%-8s | %-6s | %-12s | %-8s | %s\n" "$s" "${PAPER[$s]}" "${succ:-n/a}%$status" "${eps:-0}" "$(basename "$log")"
    done
done
echo
