#!/usr/bin/env bash
# The specprop anchor at 8 seeds, so the H22 comparison has something to be
# compared AGAINST at this repo's verdict standard.
#
# WHY NOW. H22 (state_modes 4/8/16) is screening at 3 seeds on GPU 0. If it
# moves, the verdict needs 8 seeds per arm and an exact test -- and the anchor
# currently has 3. Training the anchor's missing five seeds in parallel on the
# other half of the lease costs nothing extra in wall-clock and removes the
# step that would otherwise sit between "the screen moved" and "the verdict".
# If H22 does NOT move, these five are still the anchor's own seed spread at
# full resolution, which every future specprop comparison needs.
#
# The anchor's 3 known seeds read 0.08606 / 0.08699 / 0.08943 on validation --
# a range of 0.00337, which is already comparable to effects this repo has
# reported. That is the reason for 8 and not 3.
#
#   scripts/anchor_seeds.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"4 5 6 7 8"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s" >> "$JOBS"; done

run_cell() {
  S=$1
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/m4_ma64_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== m4_ma64 seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes 4 --modes-a 64 --state-modes 0 \
      > logs/specprop_m4_ma64_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_m4_ma64_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0'
rm -f "$JOBS"; echo "anchor seed queue drained"
