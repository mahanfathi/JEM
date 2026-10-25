#!/usr/bin/env bash
# Phase 4: the Franka Panda arm in MuJoCo Warp (experiment E6). Needs `uv sync --extra sim`.
# Usage: ./experiments/phase4.sh [e6] [e6d] [e6x]      (default: e6 e6d)
#   e6   the method on the arm (preset `panda`), three seeds
#   e6d  downstream planning for every E6 seed: the pillar (horizons 20 and 40) and the bin
#        (horizon 40, two-option plans)
#   e6x  ablations on seed 0 (paper §5.8): proposals (uniform | behaviour prior) x targets
#        (adjacent | up to 2 s later), each with and without consolidation into prototypes
# Runs are sequential on purpose (one GPU job at a time); watch them with `uv run jem dash`.
set -uo pipefail
cd "$(dirname "$0")/.."
sections=" ${*:-e6 e6d} "
run() { uv run --extra sim jem train "$@" > /dev/null 2>&1 || echo "run failed: $*"; }

if [[ $sections == *" e6 "* ]]; then
    for seed in 0 1 2; do
        run --preset panda --name "panda-s$seed" --set train.steps=10000 train.seed=$seed
    done
fi

if [[ $sections == *" e6d "* ]]; then
    for dir in runs/*/; do
        name=$(python3 -c "import json; print(json.load(open('$dir/config.json'))['meta']['name'])")
        [[ $name =~ ^panda-s[0-9]$ ]] || continue
        for spec in "pillar 20" "pillar 40" "bin 40"; do
            read -r task horizon <<< "$spec"
            uv run --extra sim jem downstream "$dir" --task "$task" --horizon "$horizon" \
                > /dev/null 2>&1 || echo "downstream failed: $dir $task $horizon"
        done
    done
fi

if [[ $sections == *" e6x "* ]]; then
    proto=(model.prototypes=6 objective.continuous_weight=0)  # consolidation, as in `panda`
    run --preset panda-uniform --name panda-planar                                # E6a
    run --preset panda-uniform --name panda-prior-only --set objective.proposal=prior  # E6b
    run --preset panda --name panda-no-consolidation --set model.prototypes=0 \
        objective.continuous_weight=1                                             # E6c
    run --preset panda-uniform --name panda-minimal --set "${proto[@]}"
    run --preset panda --name panda-no-prior --set objective.proposal=uniform
    run --preset panda --name panda-no-gap --set data.target_gap=0
fi
echo "phase 4 [${sections}] done"
