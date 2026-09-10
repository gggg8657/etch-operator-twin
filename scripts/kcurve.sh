#!/usr/bin/env bash
# H5, the K-step (horizon collapse) curve.
#
# One change from the K=1 arm that produced runs/seed1..8: the operator is
# trained on (phi_t, recipe, K*dt) -> phi_{t+K} instead of (..., dt) -> phi_{t+1}.
# Architecture, width, modes, layers, epochs, optimiser and the normalisation
# constants are all held fixed, so an arm differs from the anchor only in how
# many dataset timesteps one application advances.
#
# Two variants per K:
#   nv  non-overlapping starts   -- T/K pairs per trajectory, the partition the
#                                   rollout is actually taken over
#   ov  overlapping starts       -- T-K+1 pairs, the DATA-MATCHED CONTROL. It
#                                   restores the pair count (and hence the
#                                   gradient-step count) to ~T at every K, which
#                                   is what separates a horizon effect from a
#                                   training-set-size effect.
# At K=10 on a T=10 trajectory the two coincide (T-K+1 = 1 = T/K), so only nv is
# run there and the ov cell is that same run by construction, not a second fit.
#
# K=1 is NOT re-run: tests/test_stride.py::test_stride_1_reproduces_the_original
# _one_step_dataset pins stride=1 to the historical dataset element-for-element,
# so runs/seed1..8 *are* the K=1 arm.
#
# Ordered seed-major so seeds 1-3 of every cell land first (a screen, per
# .overnight/RULES.md's seed-count lesson) and seeds 4-8 extend it to the
# 8-seed verdict without re-running anything.
#
#   scripts/kcurve.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3 4 5 6 7 8"}
mkdir -p logs runs/kcurve

JOBS=$(mktemp)
for s in $SEEDS; do
  for cell in "2 nv" "5 nv" "10 nv" "2 ov" "5 ov"; do
    set -- $cell
    echo "$1 $2 $s" >> "$JOBS"
  done
done

run_cell() {
  K=$1; VAR=$2; S=$3
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/kcurve/K${K}_${VAR}_s${S}
  OV=""; [ "$VAR" = ov ] && OV="--overlap-pairs"
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== K=$K $VAR seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 80 \
      --device cuda:0 --seed "$S" --stride "$K" $OV \
      > logs/kcurve_K${K}_${VAR}_s${S}.log 2>&1 || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/kcurve_K${K}_${VAR}_s${S}.log 2>&1 || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell
export GPU

xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1 $2'
rm -f "$JOBS"
echo "kcurve queue drained"
