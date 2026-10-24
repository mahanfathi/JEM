#!/usr/bin/env bash
# Phase 1: the method, its ablations, and the 2022 baseline (experiments E1-E3).
# Usage: ./experiments/phase1.sh [e1] [e2] [e2b] [e3]      (default: e1 e2 e3)
# Runs are sequential on purpose: concurrent trainings thrash the GPU (22 vs 29 steps/s
# aggregate for three factored runs). Runs land in runs/; watch them with `uv run jem dash`.
set -uo pipefail
cd "$(dirname "$0")/.."
sections=" ${*:-e1 e2 e3} "
run() { uv run jem train "$@" > /dev/null 2>&1 || echo "run failed: $*"; }

if [[ $sections == *" e1 "* ]]; then  # method, three seeds
    for seed in 0 1 2; do
        run --preset jem --name "jem-s$seed" --set train.steps=10000 train.eval_every=1000 train.seed=$seed
    done
fi

if [[ $sections == *" e2 "* ]]; then  # ablations at 5k steps (compare with E1 at step 5000)
    short=(train.steps=5000 train.eval_every=1000)
    run --preset jem-em --name jem-em-s0 --set "${short[@]}"           # EM: no implicit gradient
    run --preset jem-d2 --name jem-d2 --set "${short[@]}"              # 2-D options
    run --preset jem-unnormed --name jem-unnormed --set "${short[@]}"  # unbounded features
    run --preset jem --name jem-joint --set "${short[@]}" model.arch=joint
fi

if [[ $sections == *" e2b "* ]]; then  # implicit gradient vs EM, equal 5k schedules, 3 seeds each
    for seed in 0 1 2; do
        run --preset jem --name "jem-ift5k-s$seed" --set train.steps=5000 train.eval_every=1000 train.seed=$seed
        [[ $seed == 0 ]] || run --preset jem-em --name "jem-em-s$seed" --set train.steps=5000 train.eval_every=1000 train.seed=$seed
    done
fi

if [[ $sections == *" e3 "* ]]; then  # the 2022 algorithm: faithful, then given every chance
    run --preset legacy --name legacy --set train.steps=20000 train.eval_every=2000
    run --preset legacy-tuned --name legacy-tuned --set train.steps=3000 train.eval_every=500 train.batch_size=64
fi
echo "phase 1 [${sections}] done"
