#!/usr/bin/env bash
# The step-matched control for H5.
#
# An `nv` arm at stride K trains on T/K pairs per trajectory for the same 80
# epochs as the anchor, so it also takes K times fewer gradient steps. If the
# K-curve rises because the arms are undertrained rather than because a K-step
# map is harder, that is the explanation to kill first -- and the `ov` control
# does not kill it, because ov changes the input distribution as well (it trains
# on start times the rollout never visits).
#
# `sm` changes exactly one thing from `nv`: epochs = 80*K, so the gradient-step
# count matches the anchor's ~22.5k while the pairs, the input distribution and
# the architecture are the nv arm's.
#
#   scripts/kcurve_sm.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-1}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/kcurve
JOBS=$(mktemp)
for s in $SEEDS; do for K in 2 5 10; do echo "$K $s" >> "$JOBS"; done; done

run_sm() {
  K=$1; S=$2
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/kcurve/K${K}_sm_s${S}
  EP=$((80 * K))
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then echo "skip $RUN"; return 0; fi
  echo "=== K=$K sm seed=$S epochs=$EP -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs "$EP" \
      --device cuda:0 --seed "$S" --stride "$K" \
      > logs/kcurve_K${K}_sm_s${S}.log 2>&1 || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/kcurve_K${K}_sm_s${S}.log 2>&1 || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_sm; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_sm $0 $1'
rm -f "$JOBS"; echo "sm queue drained"
