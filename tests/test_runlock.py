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


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
