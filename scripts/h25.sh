#!/usr/bin/env bash
# H25 accuracy: can a rank-constrained additive head learn?
#
# WHAT IS ALREADY MEASURED (runs/rank_cost.json, one invocation, within-run
# deltas so they are usable):
#
#   a_rank=0  1,070,144 params  609.8 us  1.00x
#   a_rank=16   404,544 params  471.2 us  1.29x
#   a_rank=8    204,864 params  412.1 us  1.48x
#   a_rank=4    105,024 params  384.4 us  1.59x
#
# and a 72.4 us floor in the stage that does not shrink with rank, so the route
# is capped at 1.68x even at rank 1. Clause 2's worst draw needs 1.9x, so this
# is a NECESSARY PART of the margin and not the margin. Said before the cost
# was measured and repeated here so a 1.59x is not retold as a pass.
#
# THE PRIOR BEING TESTED, AND IT IS THE WHOLE POINT. A rank-r head asserts the
# additive spectral response is SEPARABLE in the two frequency axes: the
# (2ma, ma) coefficient field is a sum of r outer products. Nothing in this
# repo has measured whether it is. The dense head spends 98.0% of the model's
# parameters on that field, which is either necessary or enormously wasteful,
# and this is the experiment that says which.
#
# PREDICTION, WRITTEN NOW. The anchor reads 0.08606/0.08699/0.08943 validation
# (seeds 1-3). I expect:
#   * a_rank=16 within 0.005 of the anchor  -- 404k params is still a lot of
#     coefficient capacity and 16 outer products is a rich basis;
#   * a_rank=4 worse by 0.01-0.04 -- 4 outer products over a 128x64 field is a
#     strong constraint and the etch response has a sharp trench-edge feature
#     that is not obviously separable;
#   * monotone in rank, because this is a nested family: rank r+1 can represent
#     everything rank r can.
#
# FALSIFIER, AND IT CUTS THE INTERESTING WAY: if a_rank=4 matches the anchor
# within the anchor's own seed spread (0.00337 over seeds 1-3), then 98% of
# this model's parameters are buying nothing, the dense head is the wrong
# design and not merely an expensive one, and the finding is about the
# architecture rather than about the cost. If instead accuracy degrades
# monotonically and steeply, the coefficient field is NOT separable, the cheap
# route is closed, and the 1.59x is unavailable at any accuracy worth having.
#
# NON-MONOTONICITY would falsify the nested-family reasoning above and would
# mean the comparison is dominated by optimisation rather than capacity --
# which is exactly what H17 found for the crossed split, so it is not an
# idle worry. 3 seeds is a SCREEN; 8 follow if it moves.
#
#   scripts/h25.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do for r in 4 8 16; do echo "$r $s" >> "$JOBS"; done; done

run_cell() {
  R=$1; S=$2
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/m4_ma64_r${R}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== m4_ma64_r${R} seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes 4 --modes-a 64 --a-rank "$R" \
      > logs/specprop_m4_ma64_r${R}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_m4_ma64_r${R}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1'
rm -f "$JOBS"; echo "h25 queue drained"
