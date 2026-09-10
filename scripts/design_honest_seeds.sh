#!/usr/bin/env bash
# Clause 3, replicated across independently trained operators, under an
# initialisation the target does not supply.
#
# runs/design_Tfree_seed{2,4,5}.json all used --dt-init target, the leaky
# default: the duration search began at the target's own dt, which was derived
# from a simulator probe of the true recipe's rate. So the cross-seed
# replication of clause 3 that this repo reported was leaky on every seed, not
# just the headline one. These runs replace it.
#
# Both pre-registered honest modes are run on every seed, in one loop, exactly
# as scripts/run_dt_honest.sh did for seed1 -- reporting whichever mode happens
# to look better would be choosing a protocol after seeing the result.
#
#   scripts/design_honest_seeds.sh <gpu> <seed> [seed...]
set -u
GPU=$1; shift
PY=~/miniforge3/envs/pdeno/bin/python
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1
for S in "$@"; do
  for MODE in mid random; do
    OUT="runs/design_Tfree_dtinit_${MODE}_seed${S}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    echo "=== seed $S dt-init $MODE ==="
    $PY scripts/design.py --run "runs/seed$S" --data data --n-targets 20 \
      --iters 200 --restarts 3 --random-budget 256 --device cuda:0 \
      --optimise-dt --dt-init "$MODE" --out "$OUT" \
      || echo "seed $S mode $MODE FAILED rc=$?"
  done
done
echo "honest design replication drained"
