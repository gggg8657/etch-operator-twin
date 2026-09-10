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


def reclaim_orphan(run, what="train"):
    """Make a run directory killed mid-training restartable, exactly once.

    The `log.jsonl` guard in train.py exists because two trainers once
    interleaved their epochs into one log while overwriting each other's
    `best.pt` (commit 8cca0d1, a withdrawn 0.0400 clause-1 number). But that
    guard also refuses the *other* case: a directory whose only writer was
    SIGKILLed, which is what happened to four `runs/kcurve` arms when a parent
    process exited and took its process group with it.

    `acquire()` has already run by the time this is called, so the exclusive
    flock is held here and **no live process is writing this directory** -- the
    lock dies with its holder, so a free lock is proof of death, not of
    politeness. That makes the two cases distinguishable, and only this one
    safe to restart.

    Archive rather than append: appending is precisely what produced the
    interleaved log. Returns the archive path, or None if there was nothing to
    reclaim.
    """
    run = Path(run)
    if not (run / "log.jsonl").exists():
        return None
    if (run / "done.json").exists():
        raise SystemExit(
            f"{run} carries done.json: it is a finished run, not an orphan. "
            f"Score it or choose a fresh --run; --force to overwrite.")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    dest = run.parent / "_orphaned" / f"{run.name}_{stamp}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()
    for p in sorted(run.iterdir()):
        if p.name == ".lock":
            continue  # the live lock this process holds
        p.rename(dest / p.name)
    epochs = 0
    log = dest / "log.jsonl"
    if log.exists():
        epochs = sum(1 for ln in log.read_text().splitlines() if ln.strip())
    (dest / "orphaned.json").write_text(json.dumps(
        {"reclaimed_by_pid": os.getpid(), "reclaimed_at": time.time(),
         "original": str(run), "epoch_lines": epochs, "argv": sys.argv,
         "reason": "log.jsonl present, done.json absent, flock free -> the "
                   "only writer is dead. Archived, not appended to."},
        indent=2))
    print(f"reclaimed orphan {run} ({epochs} epoch lines) -> {dest}",
          file=sys.stderr, flush=True)
    return dest


# The GPU lease this track is bound to, from the brief: devices 0 and 1, never
# 2 or 3. Overridable so the module is not a hard-coded fact about one box.
GPU_LEASE_DEFAULT = "0,1"


def assert_gpu_lease(device: str = "cuda", env=None):
    """Refuse to run on a GPU outside this track's lease.

    A `scripts/train.py` from this repository was observed running with
    `CUDA_VISIBLE_DEVICES=2,3` on 2026-09-10, which is another track's lease. It
    exited before any allocation appeared on those devices -- every compute app
    on the box was on GPU 0 or 1 throughout -- so nothing is known to have been
    disturbed, and this exists so the next one cannot be.

    It was launched by the other loop sharing this working tree (D4), which is
    why the check belongs in the code rather than in a launcher: whichever loop
    starts a trainer, this fires. `CUDA_VISIBLE_DEVICES` is what the process can
    actually reach, so it is the thing to check, not the `--device` flag, which
    is an index *into* that set and is `cuda:0` in every one of this repo's
    scripts regardless of which physical GPU that is.
    """
    env = os.environ if env is None else env
    if not str(device).startswith("cuda"):
        return None
    vis = env.get("CUDA_VISIBLE_DEVICES")
    if vis is None or vis.strip() == "":
        # Unset means every GPU is visible, which includes devices outside the
        # lease. Refuse rather than gamble on which one torch picks.
        raise SystemExit(
            "REFUSING to start on cuda with CUDA_VISIBLE_DEVICES unset: every "
            f"GPU on the box would be reachable and this track's lease is "
            f"{env.get('EOT_GPU_LEASE', GPU_LEASE_DEFAULT)}. Set it explicitly.")
    lease = {d.strip() for d in env.get("EOT_GPU_LEASE", GPU_LEASE_DEFAULT).split(",")
             if d.strip()}
    asked = {d.strip() for d in vis.split(",") if d.strip()}
    outside = sorted(asked - lease)
    if outside:
        raise SystemExit(
            f"REFUSING to start: CUDA_VISIBLE_DEVICES={vis} includes "
            f"{', '.join(outside)}, outside this track's GPU lease "
            f"({sorted(lease)}). Those devices belong to another track and a job "
            f"of this repo's was seen with them visible on 2026-09-10. Relaunch "
            f"with a leased device, or set EOT_GPU_LEASE if the lease changed.")
    return sorted(asked)
