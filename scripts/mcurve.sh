#!/usr/bin/env bash
# H21: the accuracy-versus-multiplicative-bandwidth curve at fixed modes_a=64.
#
# WHAT THIS IS ATTACKING. `sp_m4_ma64` reads 0.08886 terminal band rel-L2 at
# 1283.8x. It is not representation-limited (oracle floor 0.00003, i.e. it sits
# 2900x above its own bound) and it is not optimisation-limited (the last 400 of
# 800 epochs, LR annealed to 0, moved validation by 5.1%). What is still
# throttled is the MULTIPLICATIVE band: the model's whole dependence on phi runs
# through the lowest `modes` x `modes` corners of the input spectrum, and
# `modes` is 4. The phi-blind null `m0_ma64` reads 0.32064, so that pathway is
# worth 3.6x while seeing 4 modes of 64.
#
# THE COST SUB-PREDICTION WAS FALSIFIED BEFORE THIS LAUNCHED, which is the whole
# point of pricing first. I predicted m=64 would be CHEAPER than m=16 for the
# same reason ma=64 measured cheaper than ma=63 (426 vs 510 us): keeping the
# whole spectrum needs no sub-slice or mask. Measured (runs/arch_cost_h21.json,
# workload-matched denominator, one CPU thread, fresh subprocess):
#
#   specprop_m4_ma64    401.8 us/wafer   1283.8x  [1169-1401]  UNDER budget
#   specprop_m32_ma64   482.2 us/wafer   1069.7x  [ 762-1213]  UNDER budget
#   specprop_m64_ma64   693.4 us/wafer    743.9x  [ 655- 864]  OVER  budget
#
# Cost is MONOTONE INCREASING in `modes`, and m=64 misses clause 2 by 1.34x. The
# masking argument does not transfer from the additive side to the
# multiplicative one, and the reason is plain in hindsight: `irfft2` costs the
# same whatever fraction of the spectrum is non-zero, but the multiplicative
# term is an einsum whose WORK scales with the area of the retained block. Free
# on one side, quadratic on the other. This is the second time this weekend that
# reasoning by analogy from the additive term to the multiplicative one was
# wrong, and the first (my Nyquist argument, falsified by a concurrent instance)
# was the same conflation of what a term can OUTPUT with what it can SEE.
#
# SO THE ARMS SPLIT INTO TWO ROLES, AND ONLY ONE OF THEM IS ABOUT THE CLAUSE:
#
#   m16, m32 -- clause-relevant. m32 is the widest multiplicative band that
#               prices under 1000x, and its range [762-1213] straddles the
#               threshold, so if it wins on accuracy the cost row needs
#               repeating at higher `rounds` before anything is claimed.
#   m64      -- NOT clause-relevant; it is over budget by 1.34x and cannot pass
#               clause 2 whatever it scores. It is trained anyway because it is
#               the END of the axis: if the full multiplicative band does not
#               close the accuracy gap, bandwidth is not the constraint at all
#               and the remaining suspect is LINEARITY in phi. That is worth 15
#               GPU-minutes and it is the measurement that decides the next
#               hypothesis rather than leaving it to argument.
#
# PREDICTION, WRITTEN NOW: m64_ma64 lands 0.04-0.08, i.e. improvement without
# clause 1, and the curve m4 -> m16 -> m32 -> m64 is monotone but saturating.
# Reasoning: the ma 32->64 step bought 3.35x because ma was PROVABLY binding
# (the model sat 3.8% above a computed floor); `modes` has no such bound, only
# an argument and a null. And the map stays LINEAR in phi, which the matched
# locality control says costs 1.88x on its own -- `pw_wf16n2_relu` (nonlinear,
# no spatial mixing) reads 0.08722 and `sp_m4_ma64` (linear, global mixing)
# reads 0.08886, while the FNO that has both reads 0.04717.
#
# FALSIFIER: if m64_ma64 lands within the seed spread of m4_ma64 (~0.002), the
# multiplicative bandwidth is not the constraint, linearity is, and the next
# change is a pointwise nonlinearity before the transform -- one elementwise
# ReLU, measured at 13.2 us against the ~109 us of headroom m4_ma64 leaves under
# the 511 us budget.
#
#   scripts/mcurve.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do
  for m in 16 32 64; do
    echo "$m $s" >> "$JOBS"
  done
done

run_cell() {
  M=$1; S=$2
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/m${M}_ma64_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== m${M}_ma64 seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes "$M" --modes-a 64 \
      > logs/specprop_m${M}_ma64_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_m${M}_ma64_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1'
rm -f "$JOBS"; echo "mcurve queue drained"
