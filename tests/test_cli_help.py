"""Every script's `--help` must render.

**The defect this exists to prevent, which was live on `main`.** An `--act`
help string in `scripts/train.py` contained a bare `%` -- "i.e. 55% of the
entire clause-2 budget". argparse runs `help % params` on every help string, so
`% o` was parsed as an `%o` octal conversion and `python scripts/train.py
--help` died with `TypeError: %o format: an integer is required, not dict`.

It had been that way since the string was written and nothing caught it,
because nothing in this repo ever runs `--help`: every job is launched from a
driver script with the flags already spelled out. The cost is not the crash --
it is that the one command a human would use to find out what a script takes
was broken, in a repo whose whole method is that a reader can re-derive every
number from the scripts.

This walks every script exposing a CLI and renders its help.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scripts_with_a_cli():
    out = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        src = p.read_text()
        if "ArgumentParser" in src and "__main__" in src:
            out.append(p)
    return out


def test_every_script_help_renders():
    broken = {}
    checked = 0
    for p in _scripts_with_a_cli():
        r = subprocess.run([sys.executable, str(p), "--help"],
                           capture_output=True, text=True, cwd=ROOT,
                           timeout=180)
        checked += 1
        if r.returncode != 0:
            broken[p.name] = r.stderr.strip().splitlines()[-1] if r.stderr \
                else f"exit {r.returncode}"
    assert checked > 0, "found no scripts with a CLI; the discovery is wrong"
    assert not broken, f"--help fails for: {broken}"


def test_no_bare_percent_in_an_argparse_help_string():
    """A `%` inside an argparse help string must be doubled.

    `test_every_script_help_renders` already catches this by executing, but it
    costs a subprocess per script; this reads the source and names the line,
    which is what a person fixing it wants.
    """
    import re
    offenders = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        in_help = False
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            if re.search(r"\bhelp\s*=", ln):
                in_help = True
            if in_help:
                for lit in re.findall(r'"([^"]*)"', ln):
                    if re.search(r"(?<!%)%(?!%)", lit):
                        offenders.append(f"{p.name}:{i}: {lit[:60]}")
            if in_help and ln.rstrip().endswith(")"):
                in_help = False
    assert not offenders, "bare % in argparse help:\n  " + "\n  ".join(offenders)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print("FAILED" if fails else "all CLI help tests pass")
    sys.exit(1 if fails else 0)
