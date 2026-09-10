#!/usr/bin/env bash
# H16: the accuracy of the architectures that fit the clause-2 budget.
#
# THE HYPOTHESIS, WRITTEN BEFORE THIS RAN, with a number so it can be wrong:
# the purely pointwise arms (scale=0, no spatial mixing at ANY resolution) will
# miss clause 1, landing at 0.06-0.15 in-distribution terminal band rel-L2
# against the 0.05 threshold, while the coarse-body arms land near the
# full-resolution FNO's measured 0.04717. The reason to expect the miss:
# runs/surface_representable.json finds a mask undercut in 249 of 250
# trajectories, growing to 99.6% of frames by t=10, and how far a front advances
# under an overhang depends on the mask geometry ABOVE it -- which a per-pixel
# function of (phi, x, y, recipe) cannot see in the field.
#
# THE FALSIFIER AND WHY IT MATTERS MORE THAN THE HYPOTHESIS: if a pointwise arm
# reaches <= 0.05, then no spatial mixing is needed on this dataset, and clause
# 1 and clause 2 hold on one model for the first time in this repo. That result
# would come with a limitation I am stating in advance rather than after seeing
# it: the mask geometry here is a TWO-PARAMETER family (`trench_width` and
# `mask_height` are conditioning inputs), so (x, y, trench_width, mask_height)
# determines the layout and a pointwise model can infer from the recipe what a
# spatial model would have to read from the field. On an arbitrary mask layout
# it could not. A pointwise pass would therefore be a real pass of the clause as
# written and NOT evidence that pointwise operators solve etch simulation.
#
# Every arm: stride 10 (one application per wafer, where the cost wins are),
# epochs 800 = 80*K so gradient steps match the K=1 anchor's ~22.5k -- the
# step-matching correction that moved K10 from 0.04781 to 0.02639 at 8 seeds
# and would otherwise put every arm here under a budget known to be 1.83x
# insufficient.
#
#   scripts/ladder.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/ladder

JOBS=$(mktemp)
for s in $SEEDS; do
  # name              arch        w  m  L  wf nl act  scale
  for cfg in \
    "pw_wf8_n1_relu   multiscale  8  4  2  8  1  relu 0" \
    "pw_wf32_n3_relu  multiscale  8  4  2  32 3  relu 0" \
    "pw_wf32_n3_gelu  multiscale  8  4  2  32 3  gelu 0" \
    "ms_s8_w8L2_wf16  multiscale  8  4  2  16 2  relu 8" \
    "ms_s4_w8L2_wf16  multiscale  8  4  2  16 2  relu 4" \
    "ms_s8_w16L4_wf16 multiscale  16 4  4  16 2  relu 8" \
    ; do
    echo "$cfg $s" >> "$JOBS"
  done
done

run_cell() {
  NAME=$1; ARCH=$2; W=$3; M=$4; L=$5; WF=$6; NL=$7; ACT=$8; SC=$9; S=${10}
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/ladder/${NAME}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== $NAME seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch "$ARCH" --width "$W" --modes "$M" --layers "$L" \
      --width-full "$WF" --n-local "$NL" --act "$ACT" --scale "$SC" \
      > logs/ladder_${NAME}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/ladder_${NAME}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1 $2 $3 $4 $5 $6 $7 $8 $9'
rm -f "$JOBS"; echo "ladder queue drained"
