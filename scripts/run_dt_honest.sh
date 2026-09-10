#!/usr/bin/env bash
# H4: does clause 3 survive when the duration search is NOT warm-started at the
# target's true dt? Two honest initialisations, same model, same 20 targets, same
# budget as the published run.
set -x
PY=~/miniforge3/envs/pdeno/bin/python
RUN=${RUN:-runs/seed1}
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=1
for MODE in random mid; do
  $PY scripts/design.py --run "$RUN" --data data --n-targets 20 --iters 200 \
    --restarts 3 --random-budget 256 --device cuda:0 --optimise-dt \
    --dt-init "$MODE" --out "runs/design_Tfree_dtinit_$MODE.json" \
    || echo "dt-init $MODE FAILED rc=$?"
done
