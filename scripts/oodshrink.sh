#!/usr/bin/env bash
# H17: is the crossed-split failure a capacity problem rather than a physics one?
#
# THE OBSERVATION THAT DEMANDS THIS. Clause 1 has failed on the crossed-dt split
# for this entire project, and every attempt to move it treated the failure as a
# property of the data (displacement coverage) or of the horizon (the K-curve).
# runs/shrink.json shows something neither of those predicts: `w16m8L4_K1`, a
# 266,729-parameter FNO -- 98x smaller than the deployed 26,248,025-parameter
# one -- scores **0.04328** crossed-in-coverage terminal against the deployed
# anchor's **0.05433** at 8 seeds. All 3 of its seeds are under 0.05 (0.04218,
# 0.04291, 0.04476). It is the only sub-0.05 crossed number this repo has ever
# produced, and it was produced by making the model smaller.
#
# THE HYPOTHESIS: the deployed operator's out-of-distribution failure is partly
# overfitting, and reducing capacity improves crossed-split generalisation while
# costing little in distribution (w16m8L4_K1 is 0.02603 in-distribution against
# the anchor's 0.01890 -- worse, but far inside the clause).
#
# WHY IT IS ONLY A SCREEN NOW, AND WHAT WOULD SETTLE IT. Three seeds. The
# crossed split's per-seed spread is an order of magnitude larger than the
# in-distribution split's (anchor crossed range 0.03511 vs in-distribution
# 0.00172, measured), and runs/seed_level_test.json found every crossed interval
# 0.022-0.042 wide AT EVERY SEED COUNT INCLUDING 8 -- so seeds buy less
# resolution here than anywhere else in this repo, and 3 of them buy very
# little. The trajectory bootstrap already says so: the upper 95% bound on
# w16m8L4_K1's crossed mean is 0.05332, above the threshold. So the point
# estimate passes, every seed passes, and the interval does not.
#
# This run takes it from 3 seeds to 8, which is the repo's verdict standard, and
# the falsifier is stated in advance: if the added seeds pull the mean above
# 0.05, the 3-seed reading was noise -- which is what happened to a 3-seed
# crossed reading in this repo TWICE, most recently this turn, when K2_sm's
# crossed advantage over K=1 (-0.00263, sign-flip p=0.00858) reversed to a
# disadvantage (+0.00334, p=0.0098) on the same test at 8 seeds.
#
#   scripts/oodshrink.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"4 5 6 7 8"}
mkdir -p logs runs/shrink

JOBS=$(mktemp)
for s in $SEEDS; do
  # The two stride-1 configs from the shrink sweep: the one that produced the
  # sub-0.05 crossed reading, and the smaller one that did not (0.07953), so
  # the seed extension covers the comparison and not just the winner.
  for cfg in "16 8 4 1" "8 4 2 1"; do
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
rm -f "$JOBS"; echo "oodshrink queue drained"
