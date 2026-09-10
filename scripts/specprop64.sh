#!/usr/bin/env bash
# H20: the full-spectrum additive propagator, and the null that decides whether
# it is an operator or a lookup table.
#
# WHY THE H18 CONFIGS WERE DOOMED, MEASURED WITHOUT TRAINING
# (runs/spectral_floor_fine.json). SpectralPropagator emits a residual
# band-limited to modes_a. The best any such model can do is the spectral
# projection of the true residual, achievable by a perfect H and A, so the
# projection error is a LOWER BOUND on its rel-L2 at that modes_a. Measured on
# the test split, terminal step at stride 10:
#
#   ma=4   0.87906     ma=32  0.28516     ma=60  0.08383
#   ma=16  0.44070     ma=56  0.12467     ma=62  0.05322
#                                         ma=63  0.03298   <- first to admit 0.05
#                                         ma=64  0.00003   <- control: full spectrum
#
# So the floor crosses the clause only between ma=62 and ma=63, i.e. at the full
# spectrum minus one mode, because the residual of an SDF over a 10-step etch is
# BROADBAND -- the surface moves several micron and the change is localised at
# the interface. All three H18 arms (ma = 4, 16, 32) are provably unable to meet
# clause 1 and are being left to finish only because they TEST THIS BOUND: the
# floor predicts they land at or above 0.879, 0.441 and 0.285.
#
# THE FIX IS THE PROPERTY THE CLASS WAS DESIGNED AROUND: modes_a is free in
# OPERATIONS, because irfft2 costs the same whatever fraction of the spectrum is
# non-zero. Priced (runs/arch_cost_h20.json, paired workload-matched
# denominator): specprop m=4 ma=64 is 426.4 us/wafer = 1187.7x with a measured
# range of [1062-1218x] entirely above 1000x, still 4 full-field operations. Its
# oracle floor is 0.00003. So for the first time an architecture in this repo
# clears clause 2 with NO representational obstruction to clause 1.
#
# THE RISK, AND THE NULL THAT MEASURES IT. `A` depends only on the recipe, and
# in this dataset the initial geometry is determined by `trench_width` and
# `mask_height`, which ARE conditioning inputs. So a full-spectrum `A` can encode
# the entire terminal residual for a recipe and the model becomes a lookup table
# that ignores phi -- which would score well here and would not be an operator.
#
# `m0_ma64` is that null, and it is pre-registered rather than run afterwards:
# modes=0 removes the multiplicative term, leaving phi + irfft2(A(recipe)) whose
# residual is provably phi-independent (tests/test_arch.py pins max|diff| = 0).
#
# DECISION RULE, WRITTEN NOW:
#   * if m4_ma64 meets 0.05 and m0_ma64 does NOT, the phi-dependence is doing
#     real work and the model is an operator;
#   * if BOTH meet 0.05, the dataset cannot distinguish an operator from a
#     recipe lookup, the clause-1 pass is real but uninformative about operator
#     learning, and that must be said in the same sentence as the number;
#   * if neither meets it, the floor was not the binding constraint and the
#     optimisation is.
#
# PREDICTION: m4_ma64 lands 0.02-0.06; m0_ma64 lands within 0.01 of it, i.e. I
# expect the lookup degeneracy to be REAL on this dataset. Stating that in
# advance because it is the outcome that would most tempt a favourable reading.
#
# Stride 10, 800 epochs = 80*K, matching every other arm in this repo.
#
#   scripts/specprop64.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-2}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/specprop

JOBS=$(mktemp)
for s in $SEEDS; do
  for cfg in "m4_ma64  4  64" \
             "m0_ma64  0  64" \
             ; do
    echo "$cfg $s" >> "$JOBS"
  done
done

run_cell() {
  NAME=$1; M=$2; MA=$3; S=$4
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/specprop/${NAME}_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== $NAME seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch specprop --modes "$M" --modes-a "$MA" \
      > logs/specprop_${NAME}_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/specprop_${NAME}_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1 $2 $3'
rm -f "$JOBS"; echo "specprop64 queue drained"
