#!/usr/bin/env bash
# H19 at 8 seeds: the depth-vs-dt conditioning comparison, promoted from screen
# to verdict.
#
# WHY THIS NEEDS 8 SEEDS AND NOT 3. At 3 matched seeds depth conditioning wins
# on every seed -- 0.03688/0.03949/0.03530 against dt's 0.04720/0.04909/0.04523
# -- but a paired sign test on 3/3 cannot go below p = 0.25, and this repo has
# twice had a 3-seed crossed reading reverse when extended. The effect it claims
# (0.00995 mean, 21% relative) is large against the ~0.00386 yardstick, so the
# extension is expected to confirm rather than reverse; it is run because the
# repo's own rule says a headline needs 8, and this IS the headline: depth
# conditioning is the only thing measured so far that removes the probe which
# caps clause 2 at 12.8x, AND it meets clause 1 on every seed.
#
# THE ARMS ARE MATCHED IN EVERYTHING BUT THE CONDITIONING CHANNEL. Both are
# arch fno, width 8, modes 4, layers 2, stride 10, 800 epochs; the same seed
# gives both arms the same initialisation. Seeds 1-3 already exist for both and
# are skipped by the done.json guard, so this only adds 4-8.
#
#   scripts/h19_seeds.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"4 5 6 7 8"}
mkdir -p logs runs/depthcond runs/shrink

PY=~/miniforge3/envs/pdeno/bin/python
if [ ! -f data/norm_depth.json ]; then
  echo "data/norm_depth.json missing; running derive_depth.py first"
  $PY scripts/derive_depth.py || exit 1
fi

JOBS=$(mktemp)
for s in $SEEDS; do
  echo "depth $s" >> "$JOBS"
  echo "dt    $s" >> "$JOBS"
done

run_cell() {
  COND=$1; S=$2
  PY=~/miniforge3/envs/pdeno/bin/python
  if [ "$COND" = "depth" ]; then
    RUN=runs/depthcond/w8m4L2_K10_depth_s${S}; EXTRA="--cond depth"
    LOG=logs/depthcond_s${S}.log
  else
    RUN=runs/shrink/w8m4L2_K10_s${S}; EXTRA=""
    LOG=logs/shrink_w8m4L2_K10_s${S}.log
  fi
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== cond=$COND seed=$S -> $RUN ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --width 8 --modes 4 --layers 2 $EXTRA \
      > "$LOG" 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> "$LOG" 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0 $1'
rm -f "$JOBS"; echo "h19_seeds queue drained"
