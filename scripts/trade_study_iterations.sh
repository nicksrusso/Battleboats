#!/usr/bin/env bash
# scripts/trade_study_iterations.sh
#
# Sweep MCTS iteration counts to find the minimum for reliable termination.
# Each setting plays N games across N workers; settings run sequentially
# because each individual benchmark already saturates the CPU.
#
# After each run, the produced mcts_vs_random_<ts>.json is renamed to embed
# the iteration count, so the per-setting results are easy to grep / jq.
# Files land in runs/benchmarks/ alongside ad-hoc benchmark runs.

set -euo pipefail

# cd to repo root (parent of scripts/)
cd "$(dirname "$0")/.."

ITERATIONS=(25 50 75 100 125 150)
WORKERS=8
NUM_GAMES=8

echo "Trade study: ${#ITERATIONS[@]} settings × ${NUM_GAMES} games × ${WORKERS} workers"
echo "Iterations to sweep: ${ITERATIONS[*]}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo

study_t0=$(date +%s)

for iters in "${ITERATIONS[@]}"; do
  echo "================================================================"
  echo "  iterations=${iters}    ($(date '+%H:%M:%S'))"
  echo "================================================================"

  setting_t0=$(date +%s)
  poetry run python scripts/benchmark_godmode_mcts.py \
      --iterations "${iters}" \
      --workers "${WORKERS}" \
      --num-games "${NUM_GAMES}" \
      --no-debug-plot
  setting_elapsed=$(($(date +%s) - setting_t0))

  # Tag the just-produced JSON with the iteration count for easy lookup.
  latest=$(ls -t runs/benchmarks/mcts_vs_random_*.json 2>/dev/null | head -n 1 || true)
  if [[ -n "${latest}" ]]; then
    tagged="${latest%.json}_iter${iters}.json"
    mv "${latest}" "${tagged}"
    echo
    echo "  -> setting done in ${setting_elapsed}s"
    echo "  -> tagged: ${tagged}"
  fi
  echo
done

study_elapsed=$(($(date +%s) - study_t0))
echo "================================================================"
echo "  Trade study complete in ${study_elapsed}s"
echo "================================================================"
echo

# Best-effort tabular summary. Requires jq; skipped silently if absent.
if command -v jq >/dev/null 2>&1; then
  printf "%-10s %-8s %-8s %-8s %-8s\n" "iters" "mcts" "random" "trunc" "wall_s"
  printf "%-10s %-8s %-8s %-8s %-8s\n" "-----" "----" "------" "-----" "------"
  for iters in "${ITERATIONS[@]}"; do
    f=$(ls -t runs/benchmarks/mcts_vs_random_*_iter${iters}.json 2>/dev/null | head -n 1 || true)
    if [[ -n "${f}" ]]; then
      jq -r --arg i "${iters}" '
        .results
        | "\($i)\t\(.mcts_wins)\t\(.random_wins)\t\(.truncated)\t\(.total_wall_time_s | floor)"
      ' "${f}" | awk -F'\t' '{printf "%-10s %-8s %-8s %-8s %-8s\n", $1, $2, $3, $4, $5}'
    fi
  done
else
  echo "jq not installed; skipping summary table."
  echo "Per-setting JSON files: runs/benchmarks/mcts_vs_random_*_iter*.json"
fi
