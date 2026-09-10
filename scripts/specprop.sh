#!/usr/bin/env bash
# H18: does the architecture that CLEARS clause 2 also meet clause 1?
#
# COST IS ALREADY MEASURED AND THE CLAUSE-2 HALF IS CONFIRMED
# (runs/arch_cost_h18.json, paired workload-matched denominator, load 405 so
# these are lower bounds):
#
#   specprop m=8  ma=32   225.7 us/wafer   2267.9x   UNDER the 1000x budget
#   specprop m=8  ma=32   283,904 params
#   specprop m=4  ma=4    277.1 us/wafer   1847.7x   UNDER
#   specprop m=4  ma=16   293.8 us/wafer   1742.3x   UNDER
#   fno_w8m4L2 (the accuracy-admissible reference) 2580.4 us   198.4x   over
#   pw_wf8 (cheapest pointwise)                     944.0 us   542.3x   over
#
# These are the first architectures in this repo to clear 1000x. So this sweep
# is not asking whether they are fast -- that is measured -- it is asking the
# only remaining question: whether a model that is linear in phi can reach
# rel-L2 <= 0.05.
#
# PREDICTION, WRITTEN BEFORE THE SWEEP: 0.05-0.12 in-distribution terminal, i.e.
# it clears clause 2 and misses clause 1. If that is what happens, the frontier
# is pinned from BOTH sides for the first time -- a measured point under 1000x
# that is too inaccurate, and a measured point at 0.04717 that is too slow.
#
# FALSIFIER: if it reaches <= 0.05 on the point estimate, every seed and the
# bootstrap upper bound, then clause 1 and clause 2 hold on ONE model and the
# KPI passes on two of three clauses simultaneously for the first time. That
# outcome must survive the same checks as everything else here -- 3 seeds is a
# screen, and an 8-seed extension follows before any verdict.
#
# The expected failure mode is stated in advance so it cannot be rationalised
# afterwards: a propagator linear in phi cannot represent an advance that
# depends nonlinearly on phi, which is what an undercut is -- the advance under
# an overhang depends on the mask above it. runs/surface_representable.json
# finds an undercut in 249 of 250 trajectories, growing to 99.6% of frames by
# t=10. If this misses, that is the first place to look.
#
# Stride 10 (one application per wafer) and 800 epochs = 80*K, matching every
# other arm in this repo so the gradient-step count equals the K=1 anchor's.
#
#   scripts/specprop.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do
  # name          modes modes_a
  for cfg in "m4_ma4    4  4" \
             "m4_ma16   4  16" \
             "m8_ma32   8  32" \
             ; do
    echo "$cfg $s" >> "$JOBS"
  done
done

run_cell() {
  NAME=$1; M=$2; MA=$3; S=$4
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/${NAME}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== $NAME seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes "$M" --modes-a "$MA" \
      > logs/specprop_${NAME}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_${NAME}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1 $2 $3'
rm -f "$JOBS"; echo "specprop queue drained"
