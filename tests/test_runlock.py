"""The concurrency guards. These exist because the repo lost a number to a race."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eot import runlock  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from seed_spread import completed  # noqa: E402


def _child_tries(target):
    code = (f'import sys; sys.path.insert(0, {str(ROOT)!r})\n'
            f'from eot import runlock\n'
            f'try:\n'
            f'    runlock.acquire({str(target)!r}, exit_on_conflict=False)\n'
            f'    print("ACQUIRED")\n'
            f'except runlock.RunLocked:\n'
            f'    print("REFUSED")\n')
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True).stdout.strip()


def test_second_writer_is_refused():
    d = Path(tempfile.mkdtemp()) / "run"
    d.mkdir(parents=True)
    runlock.acquire(d, exit_on_conflict=False)
    assert _child_tries(d) == "REFUSED"


def test_lock_is_released_when_holder_dies():
    d = Path(tempfile.mkdtemp()) / "run"
    d.mkdir(parents=True)
    code = (f'import sys, time; sys.path.insert(0, {str(ROOT)!r})\n'
            f'from eot import runlock\n'
            f'runlock.acquire({str(d)!r}, exit_on_conflict=False)\n')
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    # holder exited, so the lock must be free for the next writer
    assert _child_tries(d) == "ACQUIRED"


def _run(tmp, lines, epochs=4, done=False):
    d = Path(tmp)
    d.mkdir(parents=True, exist_ok=True)
    (d / "log.jsonl").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in lines))
    if done:
        runlock.mark_done(d, epochs=epochs)
    return d, {"epochs": epochs}


def test_complete_run_is_scoreable():
    d, cfg = _run(tempfile.mkdtemp() + "/a", [0, 1, 2, 3])
    assert completed(d, cfg)[0]


def test_unfinished_run_is_refused():
    d, cfg = _run(tempfile.mkdtemp() + "/b", [0, 1])
    ok, why = completed(d, cfg)
    assert not ok and "2/4" in why


def test_interleaved_log_is_refused_even_at_full_length():
    """Two appenders can reach the epoch count with neither model converged.
    A line count alone accepts this; the guard must not."""
    d, cfg = _run(tempfile.mkdtemp() + "/c", [0, 1, 0, 1])
    assert len(list((d / "log.jsonl").open())) == cfg["epochs"]  # line count says done
    ok, why = completed(d, cfg)
    assert not ok and "duplicate epoch" in why


def test_done_marker_is_accepted():
    d, cfg = _run(tempfile.mkdtemp() + "/d", [0, 1, 2, 3], done=True)
    ok, why = completed(d, cfg)
    assert ok and why == "done.json"




def _with_ckpt(d, eval_lag_s):
    """Give a run a best.pt and a test_eval.json `eval_lag_s` seconds apart."""
    best, ev = Path(d) / "best.pt", Path(d) / "test_eval.json"
    best.write_bytes(b"not-a-real-checkpoint")
    ev.write_text(json.dumps({"rollout": {"op": {"band": {"mean": 0.01}}}}))
    t = best.stat().st_mtime
    os.utime(ev, (t + eval_lag_s, t + eval_lag_s))
    return Path(d)


def test_eval_older_than_checkpoint_is_refused():
    """The `runs/base` failure of 2026-09-10, reduced to a test.

    A raced directory was excluded for logging 49/80 epochs. The surviving
    trainer then completed the log to a clean 0..79 and wrote a newer best.pt,
    so every structural check passed while test_eval.json still described the
    checkpoint from 14 minutes earlier. Admitting that run moved this repo's
    measured seed range from 0.00114 to 0.03721.
    """
    d, cfg = _run(tempfile.mkdtemp() + "/e", [0, 1, 2, 3])
    _with_ckpt(d, eval_lag_s=-859)
    ok, why = completed(d, cfg)
    assert not ok and "older than best.pt" in why


def test_staleness_beats_the_done_marker():
    """done.json is not a licence: a run can be marked done and then have its
    checkpoint overwritten, which leaves the eval describing nothing on disk."""
    d, cfg = _run(tempfile.mkdtemp() + "/f", [0, 1, 2, 3], done=True)
    _with_ckpt(d, eval_lag_s=-10)
    ok, why = completed(d, cfg)
    assert not ok and "older than best.pt" in why


def test_eval_newer_than_checkpoint_is_accepted():
    d, cfg = _run(tempfile.mkdtemp() + "/g", [0, 1, 2, 3])
    _with_ckpt(d, eval_lag_s=+7)   # the lag every clean run in this repo shows
    assert completed(d, cfg)[0]



def test_reclaim_orphan_archives_rather_than_appends():
    """A killed trainer's directory must be restartable, and its log must not be
    appended to -- appending is what produced the interleaved log in the first
    place."""
    run = Path(tempfile.mkdtemp()) / "K2_nv_s4"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}\n')
    (run / "best.pt").write_text("weights")

    dest = runlock.reclaim_orphan(run)

    assert dest is not None
    # the run directory is clear, so a fresh trainer can start
    assert not (run / "log.jsonl").exists()
    assert not (run / "best.pt").exists()
    # nothing was destroyed
    assert (dest / "log.jsonl").read_text() == '{"epoch": 1}\n{"epoch": 2}\n'
    assert (dest / "best.pt").read_text() == "weights"
    note = json.loads((dest / "orphaned.json").read_text())
    assert note["epoch_lines"] == 2
    assert note["original"] == str(run)


def test_reclaim_orphan_refuses_a_finished_run():
    """done.json means the run completed. Reclaiming it would silently discard a
    scoreable result, which is worse than the restart failing."""
    run = Path(tempfile.mkdtemp()) / "seed1"
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text('{"epoch": 1}\n')
    (run / "done.json").write_text("{}")
    try:
        runlock.reclaim_orphan(run)
    except SystemExit:
        pass
    else:
        raise AssertionError("a finished run must not be reclaimed")
    assert (run / "log.jsonl").exists()


def test_reclaim_orphan_is_a_noop_on_a_fresh_directory():
    run = Path(tempfile.mkdtemp()) / "fresh"
    run.mkdir(parents=True)
    assert runlock.reclaim_orphan(run) is None

def test_no_test_file_defines_a_test_the_runner_cannot_reach():
    """Every test file runs itself by walking globals() under __main__, so a
    test function appended BELOW that block is defined, collected by nothing and
    silently never executed -- while CI still reports the file green.

    I introduced exactly that twice on 2026-09-10 by appending new tests to the
    end of two files. This compares each file's top-level `def test_*` against
    the position of its runner, so the next append cannot go unnoticed.
    """
    import ast

    tests_dir = Path(__file__).resolve().parent
    offenders = []
    for f in sorted(tests_dir.glob("test_*.py")):
        src = f.read_text()
        tree = ast.parse(src)
        runner = [n for n in tree.body
                  if isinstance(n, ast.If) and "__main__" in ast.dump(n.test)]
        if not runner:
            continue  # a file with no self-runner is not covered by this rule
        cut = runner[0].lineno
        late = [n.name for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name.startswith("test_") and n.lineno > cut]
        if late:
            offenders.append(f"{f.name}: {', '.join(late)}")
    assert not offenders, (
        "test functions defined below the __main__ runner, so they never run:\n  "
        + "\n  ".join(offenders))


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
