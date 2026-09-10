#!/usr/bin/env bash
# H7, the accuracy cost of shrinking. See critique_log.md for the hypothesis and
# the decision rule, both written before this ran.
#
# runs/cost_floor.json prices the architectures; this measures what they lose.
# Arms are run at BOTH stride 10 (one application per wafer, where the cost wins
# are) and stride 1 (the control that separates a horizon penalty from a capacity
# penalty), because a stride-10 arm inherits K=10's 0.04847 before it is shrunk.
#
#   scripts/shrink.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/shrink

JOBS=$(mktemp)
for s in $SEEDS; do
  # width modes layers stride
  for cfg in "8 4 2 10" "8 8 2 10" "16 8 2 10" "16 8 4 10" "32 12 4 10" \
             "8 4 2 1" "16 8 4 1"; do
    echo "$cfg $s" >> "$JOBS"
  done
done

run_cell() {
  W=$1; M=$2; L=$3; K=$4; S=$5
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/shrink/w${W}m${M}L${L}_K${K}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== w=$W m=$M L=$L K=$K seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 80 \
      --device cuda:0 --seed "$S" --stride "$K" \
      --width "$W" --modes "$M" --layers "$L" \
      > logs/shrink_w${W}m${M}L${L}_K${K}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/shrink_w${W}m${M}L${L}_K${K}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1 $2 $3 $4'
rm -f "$JOBS"; echo "shrink queue drained"
