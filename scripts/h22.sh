#!/usr/bin/env bash
# H22: does a NONLINEAR dependence on phi close the gap that bandwidth did not?
#
# THE STATE OF THE FRONTIER THIS ATTACKS. `sp_m4_ma64` reads 0.08886 terminal
# band rel-L2 (3 seeds) and prices at 401.8 us/wafer on an idle box. It misses
# clause 1 by 1.78x, and three explanations for that are already dead by
# measurement:
#   * representation  -- oracle projection floor at modes_a=64 is 0.00003, so it
#                        sits 2900x above its own bound;
#   * optimisation    -- the last 400 of 800 epochs, LR annealed to 0, moved
#                        validation by 5.1% (0.09146 -> 0.08679);
#   * recipe lookup   -- the pre-registered phi-blind null m0_ma64 reads 0.32064,
#                        3.6x worse, so the phi pathway does real work.
# H21 is testing the fourth: that the MULTIPLICATIVE bandwidth is the constraint.
#
# THIS IS THE FIFTH, AND IT IS THE ONE THE MATCHED CONTROLS POINT AT. The
# locality control pw_wf16n2_relu (nonlinear in phi, NO spatial mixing) reads
# 0.08722; sp_m4_ma64 (linear in phi, GLOBAL spatial mixing) reads 0.08886 --
# within 1.9% of each other. The FNO that has BOTH reads 0.04717, a factor of
# 1.88 better than either. So neither ingredient alone gets past ~0.088, and the
# question is whether they can be combined inside the four-operation cost
# structure rather than the FNO's twenty-five.
#
# THE MECHANISM, AND WHOSE IDEA IT WAS. Feeding a state_modes x state_modes
# spectral summary of the current field into the ADDITIVE head makes that head a
# nonlinear function of phi (it has a GELU) and couples a low-frequency summary
# of the input to EVERY output mode up to modes_a -- which a mode-diagonal
# multiplicative term cannot do at any width. It reuses the rfft2 already
# computed, so it adds no transform and no full-resolution activation. This is
# `codex`'s proposal in answer to the rung-4 question "how would you make this
# clause pass?"; my own registered fallback was a pointwise nonlinearity before
# the transform, which is strictly worse because it still routes through the
# mode-diagonal bottleneck afterwards.
#
# COST, MEASURED, AND WHAT IS STILL UNMEASURED. runs/arch_cost_h22.json, all
# rows in one invocation so they share machine load (loadavg 430):
#
#   specprop_m4_ma64        530.6 us   958.7x   <- anchor, SAME invocation
#   specprop_m4_ma64_sm4    617.3 us   824.1x   +86.7 us  (+16.3%)
#   specprop_m4_ma64_sm8    665.5 us   764.4x  +134.9 us  (+25.4%)
#   specprop_m4_ma64_sm16   619.8 us   820.8x   +89.2 us  (+16.8%)
#
# Every absolute figure there is under load and NONE of them is the clause
# reading: the same anchor prices at 401.8 us at loadavg 74 and 530.6 us at
# loadavg 430, a 1.32x spread that straddles 1000x. What the invocation DOES
# establish, because the load is shared, is the RELATIVE cost of the state head:
# +16.3% to +25.4%. codex estimated 25-55 us; the matched-load delta is 87-135
# us, so its estimate is optimistic by at least 1.6x. Whether sm4 clears 1000x
# is [not measured] and must be re-priced on an idle box -- which is worth doing
# only if the accuracy moves, so accuracy goes first.
#
# PREDICTION, WRITTEN NOW: sm8 lands 0.05-0.075. Reasoning, and it is an
# interpolation between measured points rather than a hope: the FNO buys 1.88x
# over either single-ingredient model by having both ingredients with 8 channels
# and 25 operations; this buys the missing ingredient through a 64-unit MLP on a
# 263-vector and 4 operations. Getting a fraction of 1.88x is likely, getting
# all of it is not. 0.05 exactly would be the boundary and I am not predicting a
# pass.
#
# FALSIFIER: if sm8 lands within the seed spread of m4_ma64 (~0.002), then
# nonlinearity delivered through the additive head is worth nothing here, the
# 1.88x the FNO holds is about its EIGHT CHANNELS rather than its nonlinearity,
# and the four-operation family is closed for clause 1. That would be a real
# result about this architecture class, not a null.
#
# state_modes 4 / 8 / 16 spans the summary width at matched cost within noise,
# so the curve says whether the effect saturates in the summary or in the head.
#
#   scripts/h22.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do
  for sm in 4 8 16; do
    echo "$sm $s" >> "$JOBS"
  done
done

run_cell() {
  SM=$1; S=$2
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/m4_ma64_sm${SM}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== m4_ma64_sm${SM} seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes 4 --modes-a 64 --state-modes "$SM" \
      > logs/specprop_m4_ma64_sm${SM}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_m4_ma64_sm${SM}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1'
rm -f "$JOBS"; echo "h22 queue drained"
