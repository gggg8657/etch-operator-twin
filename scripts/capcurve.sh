#!/usr/bin/env bash
# H19: where is the capacity optimum for OUT-OF-DISTRIBUTION generalisation?
#
# WHAT DEMANDS THIS. runs/capacity_ood.json settles H17 at 8 seeds against 8:
# the 266,729-parameter `w16m8L4_K1` beats the deployed 26,248,025-parameter
# anchor on the crossed-dt split by -0.01145 (exact permutation p = 0.0255, 95%
# interval [-0.02127, -0.00163], excludes zero), while being worse in
# distribution by +0.00677 (p = 0.0002). First crossed-split win in this repo,
# and produced by making the model 98x smaller.
#
# But it is NOT "smaller is better". `w8m4L2_K1` at 10,897 parameters is worse
# than the anchor on BOTH splits (+0.03875 in-distribution, +0.02779 crossed).
# Three points, one at each end and one in the middle, and the middle one wins:
# that is a U-shape with no interior resolution.
#
# THE HYPOTHESIS, WRITTEN BEFORE THIS RUNS, with predictions so it can be wrong:
# crossed-split error is U-shaped in capacity with its minimum near 2-5 x 10^5
# parameters. Concretely I predict all three new arms land ABOVE w16m8L4_K1's
# 0.04289 -- w8m8L2 (35,473) near 0.06, w16m8L2 (135,049) near 0.045, and
# w32m12L4 (2,369,977) near 0.048 -- so the optimum sits at 266k within a factor
# of ~3 and the curve is not monotone.
#
# THE FALSIFIER, AND THE OUTCOME THAT WOULD CHANGE A CLAUSE:
#  * If the curve is monotone DECREASING across 35k -> 2.37M then "reduce
#    capacity" is the right rule and 266k is not an optimum, only the smallest
#    arm that still fits the data.
#  * If any arm beats 0.04289 far enough that its TRAJECTORY BOOTSTRAP UPPER
#    BOUND falls under 0.05, then clause 1 is met on the crossed split for the
#    first time in this project. w16m8L4_K1 currently reads point 0.04289, all
#    8 seeds under (max 0.04977), and bootstrap upper 0.05295 -- so it passes
#    two of the repo's three readings and fails the third. Roughly 0.040 on the
#    point estimate would carry the third.
#
# EVERY ARM IS STRIDE 1, matching the anchor, so capacity is the only thing that
# differs. Epochs are 80, the anchor's own budget, for the same reason -- a
# stride-1 arm at 80 epochs takes the anchor's ~22.5k gradient steps, so no
# step-matching correction applies here (that correction exists for stride>1).
#
# 8 SEEDS PER ARM, NOT 3. Crossed readings in this repo have reversed their sign
# at 3 seeds three separate times, and runs/seed_level_test.json measures every
# crossed interval 0.022-0.042 wide at every seed count including 8. A 3-seed
# capacity curve would be a picture of noise.
#
#   scripts/capcurve.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3 4 5 6 7 8"}
mkdir -p logs runs/shrink

JOBS=$(mktemp)
for s in $SEEDS; do
  # width modes layers stride -- the three interior capacities missing from the
  # stride-1 curve. 10,897 / 266,729 / 26,248,025 are already measured at 8
  # seeds, so these fill 35k, 135k and 2.37M between them.
  for cfg in "8 8 2 1" "16 8 2 1" "32 12 4 1"; do
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
rm -f "$JOBS"; echo "capcurve queue drained"
