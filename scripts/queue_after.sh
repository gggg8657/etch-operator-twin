#!/usr/bin/env bash
# Queue a job behind a PID on the same GPU, rather than oversubscribing the lease.
#   scripts/queue_after.sh <pid> <gpu> <command...>
set -u
WAITPID=$1; shift
GPU=$1; shift
while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
echo "=== pid $WAITPID gone at $(date -Is); starting on gpu $GPU ==="
export CUDA_VISIBLE_DEVICES=$GPU
"$@"
