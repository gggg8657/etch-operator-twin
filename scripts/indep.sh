#!/usr/bin/env bash
# H13: does training on displacement-covering data beat the crossed plateau?
#
# See critique_log.md for the hypothesis, its distinguishing predictions, the
# comparability rule and the confounds -- all written before this ran.
#
# The primary arm is SIZE-MATCHED (data_mixed_matched: 450 adaptive + 450
# independent = the anchor's 900 trajectories, 9000 pairs, 80 epochs, ~22.5k
# gradient steps), so it differs from runs/seed1..8 only in the dt coupling of
# half its data. Turn 5's lesson is that the step count on its own can
# manufacture a trend, so it is held fixed here.
#
# Both test splits in data_mixed_matched are byte copies of data/, so every arm
# old and new is scored on identical test data. The norm IS refit on the mixed
# training set, because standardising a wider displacement range against
# constants that never saw it would be wrong.
#
#   scripts/indep.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-2}; shift 2 || true
SEEDS=${*:-"1 2 3"}
DATA=${DATA:-data_mixed_matched}
TAG=${TAG:-mixmatched}
mkdir -p logs runs/indep

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s" >> "$JOBS"; done

run_indep() {
  S=$1
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/indep/${TAG}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== $TAG seed=$S data=$DATA -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 80 \
      --device cuda:0 --seed "$S" --data "$DATA" \
      > logs/indep_${TAG}_s${S}.log 2>&1 || { echo "TRAIN FAILED $RUN"; return 1; }
  for SPLIT in test test_crossed; do
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
        --data "$DATA" --split $SPLIT \
        >> logs/indep_${TAG}_s${S}.log 2>&1 || echo "EVAL $SPLIT FAILED $RUN"
  done
}
export -f run_indep; export GPU DATA TAG
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_indep $0'
rm -f "$JOBS"; echo "indep queue drained"
