#!/usr/bin/env bash
# Phase 3: composing options over time on unseen obstacle tasks (docs/DESIGN.md, experiment log E5).
# Usage: ./experiments/phase3.sh [run-name-regex]     (default: the E1 seeds, ^jem-s[0-9]$)
# For each matching run: four planners x {circle, cup} x horizon {20, 40}, 64 episodes each.
# Results are appended to the run's metrics.jsonl as down/<task>_h<H>/<planner>_{success,steps}.
set -uo pipefail
cd "$(dirname "$0")/.."
pattern="${1:-^jem-s[0-9]$}"
for run in runs/*/; do
    name=$(python3 -c "import json,sys; print(json.load(open('$run/config.json'))['meta']['name'])")
    [[ $name =~ $pattern ]] || continue
    for task in circle cup; do
        for horizon in 20 40; do
            uv run jem downstream "$run" --task $task --horizon $horizon > /dev/null 2>&1 \
                || echo "downstream failed: $run $task $horizon"
        done
    done
done
echo "phase 3 [$pattern] done"
