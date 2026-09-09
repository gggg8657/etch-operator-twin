"""One writer per output. Enforced, because advice did not hold.

This repo has already withdrawn one headline number to a concurrent-writer bug:
two `train.py` processes were launched against `runs/base`, both opened
`log.jsonl` in append mode and both wrote `best.pt`, and the resulting 0.0400
rel-L2 belonged to neither model. Turn 7 then found *two `claude -p` processes
started 1 s apart with the same brief*, so the collision is not a slip that can
be avoided by being careful -- it is structural, and it needs a lock.

`fcntl.flock` on a sidecar `.lock` file, non-blocking. The lock dies with the
process, so a killed job does not wedge the directory.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import time
from pathlib import Path

_HELD = []


class RunLocked(RuntimeError):
    pass


def acquire(target, what="run", exit_on_conflict=True):
    """Take an exclusive lock covering `target` (a run dir or an output file)."""
    target = Path(target)
    lock_path = (target / ".lock") if target.is_dir() or not target.suffix \
        else target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        holder = fh.read().strip() or "(unknown)"
        msg = (f"REFUSING to write {target}: another process holds its lock.\n"
               f"  holder: {holder}\n"
               f"  This guard exists because two writers to one run directory "
               f"already cost this repo a withdrawn result. Choose a different "
               f"--run/--out, or wait for the holder to finish.")
        if exit_on_conflict:
            print(msg, file=sys.stderr, flush=True)
            raise SystemExit(3)
        raise RunLocked(msg)
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                         "argv": sys.argv, "acquired": time.time(), "what": what}))
    fh.flush()
    _HELD.append(fh)  # keep the handle alive for the process lifetime
    return lock_path


def mark_done(run, **extra):
    """Completion marker. A run without one is not scoreable, however many
    epochs its log appears to contain -- an interleaved log from two appenders
    can reach the epoch count with neither model finished."""
    run = Path(run)
    (run / "done.json").write_text(json.dumps(
        {"finished": time.time(), "pid": os.getpid(), "argv": sys.argv, **extra},
        indent=2))
