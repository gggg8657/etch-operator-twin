#!/usr/bin/env bash
# Seed spread for the conditioned arm. The clause-1 verdict (0.0400 against a
# 0.05 threshold) has only a 20% margin, and this workspace has already learned
# that two identical invocations can differ by more than an effect it reported
# as a finding. A verdict at this margin is not reportable on one seed.
set -u
PY=~/miniforge3/envs/pdeno/bin/python
GPU=$1; shift
for s in "$@"; do
  echo "=== seed $s on gpu $GPU ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py --run runs/seed$s --epochs 80 \
      --device cuda:0 --seed "$s" > logs/train_seed$s.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval.py --run runs/seed$s --device cuda:0 \
      > logs/eval_seed$s.log 2>&1
done
