"""The paper draft's status block must match the run JSONs.

**The failure this exists to prevent, which already happened.**
`paper_draft.md` was the one document in this repo not generated from run
JSONs, and on 2026-09-11 its abstract still read "the speedup clause is
unreachable at 1000x under every reading we can defend" while
`runs/clock_matched_speedup.json` measured a median of 1225-1265x across four
readings. Its 4.2 headline reported 9.58x for the FNO family against a
denominator retired in its own 4.2.1.

Nothing there was fabricated -- every figure came from a run. The document
simply had no way to notice when its numbers stopped being the repo's best
measurement, and several turns of rereading did not catch it. This turns that
into a test failure.

The check is CONTENT equality: regenerating the block must produce no diff. It
deliberately does not compare file mtimes against the source JSONs, because a
sweep rewriting a JSON without changing a derived number would fail the suite
while nothing was stale, and a test that fires during every run gets ignored.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paper_status_block_is_current():
    r = subprocess.run([sys.executable, "scripts/paper_status.py", "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"{r.stdout}{r.stderr}\nRun `python scripts/paper_status.py` to "
        f"regenerate the block from the run JSONs.")


def test_generated_block_is_delimited_and_not_hand_edited():
    """The markers must survive, or the next regeneration duplicates the block."""
    t = (ROOT / "paper_draft.md").read_text()
    b = "<!-- BEGIN GENERATED paper_status -- do not edit by hand -->"
    e = "<!-- END GENERATED paper_status -->"
    assert t.count(b) == 1, f"expected exactly one BEGIN marker, found {t.count(b)}"
    assert t.count(e) == 1, f"expected exactly one END marker, found {t.count(e)}"
    assert t.index(b) < t.index(e)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                fails += 1
                print(f"  FAIL {name}: {exc}")
    print("FAILED" if fails else "all paper-status tests pass")
    sys.exit(1 if fails else 0)
