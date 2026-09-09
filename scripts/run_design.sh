#!/usr/bin/env bash
# Clause 3, both protocols. T-pinned is the optimistic bound (the target's own dt
# is handed over); T-free searches total etch time and is the honest one.
set -x
RUN=${1:-runs/seed1}
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1
$PY scripts/design.py --run "$RUN" --data data --n-targets 20 --iters 200 \
  --restarts 3 --random-budget 256 --device cuda:0 --out "$RUN/design.json"
$PY scripts/design.py --run "$RUN" --data data --n-targets 20 --iters 200 \
  --restarts 3 --random-budget 256 --device cuda:0 --optimise-dt \
  --out "$RUN/design_Tfree.json"
