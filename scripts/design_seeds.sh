#!/usr/bin/env bash
# Clause 3 on more than one seed. The T-free (honest) protocol only, with the
# same budget seed1 used, so the numbers are comparable across seeds.
set -x
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=1
for S in "$@"; do
  $PY scripts/design.py --run "runs/seed$S" --data data --n-targets 20 \
    --iters 200 --restarts 3 --random-budget 256 --device cuda:0 --optimise-dt \
    --out "runs/design_Tfree_seed$S.json" || echo "seed $S FAILED rc=$?"
done
