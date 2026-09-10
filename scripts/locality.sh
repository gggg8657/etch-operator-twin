#!/usr/bin/env bash
# The architecture-matched locality test, which the ladder did NOT provide.
#
# WHY IT IS NEEDED. runs/ladder.json invites the reading that spatial mixing is
# what separates a working operator from a failing one: ms_s4_w8m4L2_wf16n2
# scores 0.04231 and pw_wf32n3 scores 0.05052. But those two arms differ in
# pointwise width (16 vs 32), pointwise depth (2 vs 3), parameter count (10,370
# vs 3,881) AND the presence of a spectral body. Nothing about locality follows
# from comparing them, and the critique log says so rather than letting the
# reading stand.
#
# THIS RUN HOLDS EVERYTHING FIXED AND TOGGLES ONE THING. Both arms are
# width_full=16, n_local=2, act=relu, stride 10, 800 epochs, seeds 1-3; the only
# difference is `--scale`: 0 (no body, no spatial mixing at any resolution)
# against 4 (a 32x32 spectral body). The scale=4 arm already exists as
# runs/ladder/ms_s4_w8L2_wf16_s*, so only the scale=0 arm is trained here and
# the comparison is against those runs.
#
# THE HYPOTHESIS: the body is worth its cost in accuracy. Falsified if the
# scale=0 arm at wf16/n2 matches or beats 0.04231 -- which would mean the
# multiscale arm's advantage over pw_wf32n3 came from its pointwise
# configuration and not from spatial mixing at all, and would make the coarse
# body pure overhead, since it is already measured to be cost-neutral
# (2926 us against the FNO's 2758 us, ranges overlapping).
#
# Note the scale=0 arm has FEWER parameters than the scale=4 one by
# construction: removing the body removes its weights. That asymmetry cannot be
# fixed without changing the pointwise path, which is the thing being held
# fixed, so it is stated rather than papered over -- if scale=0 wins it wins
# with less capacity, which strengthens the falsification; if it loses, capacity
# is a confound and the follow-up is a wider pointwise path at matched count.
#
#   scripts/locality.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-0}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/ladder

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s" >> "$JOBS"; done

run_cell() {
  S=$1
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/ladder/pw_wf16n2_relu_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== locality control (scale=0, wf16, n2) seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 \
      --arch multiscale --width 8 --modes 4 --layers 2 \
      --width-full 16 --n-local 2 --act relu --scale 0 \
      > logs/locality_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/locality_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0'
rm -f "$JOBS"; echo "locality queue drained"
