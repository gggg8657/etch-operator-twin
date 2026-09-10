#!/usr/bin/env bash
# H7, the accuracy cost of shrinking. See critique_log.md for the hypothesis and
# the decision rule, both written before this ran.
#
# runs/cost_floor.json prices the architectures; this measures what they lose.
# Arms are run at BOTH stride 10 (one application per wafer, where the cost wins
# are) and stride 1 (the control that separates a horizon penalty from a capacity
# penalty), because a stride-10 arm inherits K=10's 0.04847 before it is shrunk.
#
# EPOCHS ARE STEP-MATCHED, and that correction is the whole reason this file was
# revised before it ran. An arm at stride K trains on T/K pairs per trajectory,
# so at a fixed epoch count it also takes K times FEWER gradient steps than the
# stride-1 anchor. runs/kcurve.json, once eight partial checkpoints were excluded
# from it (commit ec16fcd), measures what that is worth at the deployed
# architecture: K10_nv at 80 epochs scores 0.04781 in-distribution terminal
# rel-L2, while K10_sm -- identical pairs, identical input distribution, epochs =
# 80*K so the gradient-step count matches the anchor's -- scores 0.02610. A
# factor of 1.83, and the same direction at K=5 (0.03336 vs 0.02330).
#
# So a stride-10 arm trained for 80 epochs is undertrained by a margin larger
# than the accuracy differences this sweep exists to resolve. Running it that way
# would have measured shrunken networks under a budget known to be insufficient,
# and H7's pre-registered decision rule -- "if w8/m4 already fails rel-L2 <= 0.05
# badly, the frontier is closed below 1000x" -- would then have fired on a
# training artefact rather than on a capacity limit, closing the only surviving
# route to clause 2 for the wrong reason.
#
# Epochs are therefore 80*stride, matching the anchor's ~22.5k gradient steps.
# The stride-1 control arms are unaffected (80*1 = 80), so they remain
# comparable to runs/seed1..8 element-for-element.
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
  EP=$((80 * K))
  echo "=== w=$W m=$M L=$L K=$K seed=$S epochs=$EP -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs "$EP" \
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
