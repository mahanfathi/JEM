#!/usr/bin/env bash
# Phase 2: options that switch mid-trajectory (docs/DESIGN.md, experiment log E4).
# Usage: ./experiments/phase2.sh [e4]      (default: all sections). Sequential on purpose.
set -uo pipefail
cd "$(dirname "$0")/.."
sections=" ${*:-e4} "
run() { uv run jem train "$@" > /dev/null 2>&1 || echo "run failed: $*"; }
short=(train.steps=5000 train.eval_every=1000)

if [[ $sections == *" e4 "* ]]; then
    # Training demos switch expert with hazard h (mean segment 1/h steps). Plain objective vs
    # the switching-aware one that is told h (recency-weighted inference + outlier mixture).
    for h in 0.03125 0.125; do
        run --preset jem --name "jem-switch-h$h" --set "${short[@]}" env.switch_prob=$h
        run --preset jem --name "jem-switch-h$h-aware" --set "${short[@]}" env.switch_prob=$h objective.hazard=$h
    done
fi
echo "phase 2 [${sections}] done"
