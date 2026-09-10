#!/usr/bin/env bash
# H19: does conditioning on DEPTH instead of dt cost accuracy?
#
# WHY THIS IS THE EXPERIMENT. runs/bench_workload.json measures a ceiling on the
# speedup clause that no architecture beats. The operator needs its horizon
# channel supplied before it can be queried; in dt mode that channel is
# log(K*dt), and a caller whose query is a target DEPTH must run
# eot.solver.probe_rate to obtain dt -- a real solver call, 39.9 ms -- so the
# speedup is bounded by solver/probe = 12.8x. That bound applies to the specprop
# rows a concurrent instance measured at 1848-2268x on cost just as much as to
# the 187x FNO: the numerator stops mattering once the caller has to run the
# solver to build the query. Conditioning on depth removes the probe by
# construction, because the depth IS the query.
#
# THE HYPOTHESIS, WRITTEN BEFORE THIS RAN: depth conditioning will be as good as
# or BETTER than dt conditioning -- at or below the dt arm's 0.04717 -- because
# depth is more directly related to the output than dt is. A dt-conditioned
# model must infer the etch rate from the recipe and multiply; a depth-
# conditioned one is told the answer's magnitude and needs only its shape.
#
# THE FALSIFIER: if depth conditioning is materially worse, then the dt channel
# was carrying something depth cannot replace. The candidate mechanism, so the
# falsification is informative rather than just negative: depth is a single
# scalar summarising the trench bottom, while dt plus the recipe determines the
# whole rate FIELD -- including lateral etch under the mask, which does not
# scale with bottom depth. If that is what dt was for, sidewall error should
# degrade more than bottom error.
#
# THE COMPARISON IS MATCHED IN EVERYTHING BUT THE CHANNEL. The dt arms are
# runs/shrink/w8m4L2_K10_s{1,2,3}: arch fno, width 8, modes 4, layers 2, stride
# 10, 800 epochs, seeds 1-3. These arms are identical in all of that and differ
# only in --cond. Same seeds, so a seed's initialisation is shared.
#
# STRIDE 10 IS NOT A CHOICE, IT IS THE ONLY WELL-POSED CASE. The etch rate falls
# as the trench deepens, so the depth advanced by application 1 exceeds that of
# application 2 and no single depth describes a multi-application rollout.
# eot.data._cond_for raises for stride != 10 rather than approximating.
#
#   scripts/depthcond.sh <gpu> [n_parallel] [seeds...]
set -u
GPU=${1:-1}; NPAR=${2:-3}; shift 2 || true
SEEDS=${*:-"1 2 3"}
mkdir -p logs runs/depthcond

PY=~/miniforge3/envs/pdeno/bin/python
if [ ! -f data/norm_depth.json ]; then
  echo "data/norm_depth.json missing; running derive_depth.py first"
  $PY scripts/derive_depth.py || exit 1
fi

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s" >> "$JOBS"; done

run_cell() {
  S=$1
  PY=~/miniforge3/envs/pdeno/bin/python
  RUN=runs/depthcond/w8m4L2_K10_depth_s${S}
  if [ -f "$RUN/done.json" ] && [ -f "$RUN/test_eval.json" ]; then
    echo "skip $RUN (done)"; return 0
  fi
  echo "=== depth-conditioned seed=$S -> $RUN ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run "$RUN" --epochs 800 \
      --device cuda:0 --seed "$S" --stride 10 --cond depth \
      --arch fno --width 8 --modes 4 --layers 2 \
      > logs/depthcond_s${S}.log 2>&1 \
      || { echo "TRAIN FAILED $RUN"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run "$RUN" --device cuda:0 \
      >> logs/depthcond_s${S}.log 2>&1 \
      || { echo "EVAL FAILED $RUN"; return 1; }
}
export -f run_cell; export GPU
xargs -a "$JOBS" -P "$NPAR" -L1 bash -c 'run_cell $0'
rm -f "$JOBS"; echo "depthcond queue drained"
