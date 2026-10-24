#!/usr/bin/env bash
# Phase 2: options that switch mid-trajectory (docs/DESIGN.md, experiment log E4+).
# Usage: ./experiments/phase2.sh [e4]      (default: all sections). Sequential on purpose.
set -uo pipefail
cd "$(dirname "$0")/.."
sections=" ${*:-e4} "
run() { uv run jem train "$@" > /dev/null 2>&1 || echo "run failed: $*"; }
short=(train.steps=5000 train.eval_every=1000)

if [[ $sections == *" e4 "* ]]; then  # training data with hidden switches (mean segment 32 / 8 steps)
    run --preset jem --name jem-switch32 --set "${short[@]}" env.switch_prob=0.03125
    run --preset jem --name jem-switch8 --set "${short[@]}" env.switch_prob=0.125
fi
echo "phase 2 [${sections}] done"
