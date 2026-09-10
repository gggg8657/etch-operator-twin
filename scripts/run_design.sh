#!/usr/bin/env bash
# Clause 3, both protocols. T-pinned hands over the target's own dt; T-free
# searches total etch time and is the honest one. T-pinned was expected to be the
# easier of the two and measures worse (0.0098 vs 0.0061) -- it is a constraint,
# not a hint.
set -x
RUN=${1:-runs/seed1}
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
$PY scripts/design.py --run "$RUN" --data data --n-targets 20 --iters 200 \
  --restarts 3 --random-budget 256 --device cuda:0 --out "$RUN/design.json"
$PY scripts/design.py --run "$RUN" --data data --n-targets 20 --iters 200 \
  --restarts 3 --random-budget 256 --device cuda:0 --optimise-dt \
  --out "$RUN/design_Tfree.json"
