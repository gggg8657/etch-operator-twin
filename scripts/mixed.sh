#!/usr/bin/env bash
# H12 (renumbered from H10; see critique_log.md): one operator trained
# jointly on several horizons.
#
# See critique_log.md for the hypothesis and its distinguishing prediction, both
# written before this ran. The point is input-state diversity, not horizon
# length: a stride-10 arm on a T=10 trajectory has exactly one start offset, so
# --overlap-pairs cannot help it the way it helps K=2 and K=5, while mixing
# strides makes every intermediate state an input at some horizon.
#
# Budget is matched to K10_sm's ~23.2k gradient steps: the mixed set holds 16200
# pairs, so at batch 32 an epoch is ~506 steps and 45 epochs is ~22.8k.
#
# Arms live in runs/mixed/, NOT runs/kcurve/, so kcurve_report's glob cannot
# pick them up and misgroup them by cfg["stride"].
#
#   scripts/mixed.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-2}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/mixed

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s" >> "$JOBS"; done

run_mixed() {
  S=$1
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/mixed/S1_2_5_10_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== mixed strides 1,2,5,10 eval@10 seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 45 \
      --device cuda:0 --seed "$S" --strides 1,2,5,10 --eval-stride 10 \
      > logs/mixed_s${S}.log 2>&1 || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/mixed_s${S}.log 2>&1 || { echo "EVAL FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      --split test_crossed >> logs/mixed_s${S}.log 2>&1 || echo "CROSSED EVAL FAILED $RUN"
}
export -f run_mixed; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_mixed $0'
rm -f "$JOBS"; echo "mixed queue drained"
