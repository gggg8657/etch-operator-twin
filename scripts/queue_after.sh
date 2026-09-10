#!/usr/bin/env bash
# Queue a job behind a PID on the same GPU, rather than oversubscribing the lease.
#   scripts/queue_after.sh <pid> <gpu> <command...>
set -u
WAITPID=$1; shift
GPU=$1; shift
# Refuse a pid that is not a live process we could plausibly be waiting on.
# An empty pid (from a command substitution on a tmux session that had already
# exited) becomes `kill -0 ""`, which fails, so the loop falls through and the
# job starts immediately -- oversubscribing the lease. A pid of 1 never exits,
# so the job never starts and the turn's experiment silently does not run. Both
# happened; this refuses instead.
case "$WAITPID" in
  ''|*[!0-9]*) echo "queue_after: bad pid '$WAITPID'" >&2; exit 2;;
esac
if [ "$WAITPID" -le 2 ]; then
  echo "queue_after: refusing to wait on pid $WAITPID (never exits)" >&2; exit 2
fi
if ! kill -0 "$WAITPID" 2>/dev/null; then
  echo "queue_after: pid $WAITPID is already gone; starting now" >&2
else
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
fi
echo "=== pid $WAITPID gone at $(date -Is); starting on gpu $GPU ==="
export CUDA_VISIBLE_DEVICES=$GPU
"$@"
