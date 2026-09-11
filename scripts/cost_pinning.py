"""H23: the between-invocation spread in clause-2 cost is core and frequency
state on this box, not machine load.

    python scripts/cost_pinning.py --config specprop_m4_ma64_K10 --invocations 14

**What forced this.** `runs/cost_repro_specprop_m4_ma64_K10.json` measured 14
invocations at loadavg ~420 and got 449.8-553.4 us (median 526.0, 3/14 meeting
1000x). The matching idle-box run,
`runs/cost_repro_specprop_m4_ma64_K10_IDLEBOX.json`, was queued specifically to
test the load hypothesis I had registered as "the most plausible mechanism". It
came back at loadavg ~20 with 350.2-637.1 us, **median 575.3 and 1/14 meeting
1000x** -- SLOWER on the median and 1.82x wider, against a prediction that an
idle box would be faster and tighter.

**So the load hypothesis is falsified in the direction opposite to prediction**,
and it was mine. What replaces it is a property of the box that is visible
without running anything:

* the cpufreq governor is `powersave` and idle cores sit at **800 MHz** against
  a 2.1 GHz base (`/proc/cpuinfo`, INTEL XEON PLATINUM 8558). A workload of
  ~500 us of tiny dispatches never runs long enough to ramp the clock, so on an
  IDLE box it is measured on a cold, slow core; on a LOADED box every core is
  already boosted by its neighbours. `time.process_time()` counts CPU-seconds,
  so a 2.6x slower clock inflates the reading 2.6x directly.
* the box has **4 NUMA nodes** over 2 sockets, 192 CPUs, and nothing in this
  repo's timing path pins anything. An unpinned thread on an idle box is free
  to migrate across NUMA domains between rounds; on a loaded box it tends to
  stay where it was placed because every other core is busy.

Both explain "idle is slower and wider". They are different fixes, so this
separates them with a 2x2 rather than assuming one.

**The design.** Two factors, crossed, 14 independent invocations per cell, all
four cells interleaved within one wall-clock window so they share whatever the
box is doing:

* `pin`     -- taskset to one fixed physical core, vs the current unpinned protocol
* `rampup`  -- a long untimed warmup (default 300 rollouts) to drive the clock
               up before timing, vs the current 3

**Predictions, written before the run** (`--invocations 14`, config
`specprop_m4_ma64_K10`, whose four published readings span 401.8-590.5 us):

1. `rampup` is the dominant factor: it moves the median by more than `pin` does.
   Stated because the frequency ratio (2.1/0.8 = 2.6x) is large enough to cover
   the whole observed spread and the migration story is not obviously that big.
2. `pin` reduces the between-invocation SPREAD more than it moves the median.
3. The `pin+rampup` cell has the tightest spread of the four.
4. Recorded frequency at timing rises with `rampup` and is what carries the
   median difference. If the medians move and the frequencies do not, prediction
   1 is false and I will say so.

**The adoption rule, fixed in advance so this cannot become protocol-shopping.**
A cell becomes the repo's clause-2 protocol ONLY if it reduces the
between-invocation spread. Whatever median that cell reports is then the
clause-2 number **even if it is worse than the number this repo has published**,
and the other three cells stay in the JSON. Picking the fastest cell is exactly
the move the brief forbids and it is not available here: the criterion is
variance, and it is written down before the numbers exist.

Writes `runs/cost_pinning_<config>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.arch_cost import SPECS  # noqa: E402

# The child differs from scripts/cost_floor.py's only in what it REPORTS: the
# core it actually ran on and that core's clock, sampled around the timed
# region. The timing statements are identical so the readings stay comparable.
_CHILD = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, sys
sys.path.insert(0, {root!r})

def _cpu_now():
    # field 39 of /proc/self/stat is the last CPU this task ran on
    with open("/proc/self/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[36])

def _khz(c):
    try:
        with open(f"/sys/devices/system/cpu/cpu{{c}}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip())
    except OSError:
        return None

import torch
torch.set_num_threads(1)
{build}
if hasattr(M, "eval"):
    M.eval()
torch.manual_seed(0)
phi = torch.randn(1, 1, {H}, {H}); cond = torch.randn(1, 7)
wall, cpu, cores, khz = [], [], [], []
with torch.no_grad():
    for _ in range({n_warm}):
        {call}
    khz_pre = _khz(_cpu_now())
    for _ in range({n_rep}):
        w0 = time.perf_counter(); c0 = time.process_time()
        {call}
        wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
        c = _cpu_now(); cores.append(c); khz.append(_khz(c))
n_par = sum(p.numel() for p in M.parameters()) if hasattr(M, "parameters") else 0
print(json.dumps({{"wall": wall, "cpu": cpu, "params": n_par,
                   "cores": cores, "khz": khz, "khz_pre": khz_pre,
                   "affinity_n": len(os.sched_getaffinity(0))}}))
'''


def time_model(spec, n_rep, n_warm, pin=None, H=128):
    body = _CHILD.format(root=str(ROOT), build=spec["build"], call=spec["call"],
                         H=H, n_rep=n_rep, n_warm=n_warm)
    cmd = [sys.executable, "-c", body]
    if pin is not None:
        cmd = ["taskset", "-c", str(pin)] + cmd
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="specprop_m4_ma64_K10")
    ap.add_argument("--invocations", type=int, default=14)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--pin-core", type=int, default=8,
                    help="a core on NUMA node0; the GPU lease does not use it")
    ap.add_argument("--rampup", type=int, default=300,
                    help="untimed rollouts in the long-warmup arms")
    ap.add_argument("--workload", default="runs/arch_cost_h21.json")
    ap.add_argument("--denominator", default="terminal_one_apply")
    ap.add_argument("--target", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    spec = SPECS[a.config]
    out = a.out or f"runs/cost_pinning_{a.config}.json"
    runlock.acquire(out, what="cost_pinning")

    dens = (json.loads(Path(a.workload).read_text())
            ["protocol"]["budget_provenance"]["denominators_cpu_s"])
    solver = dens[a.denominator]
    # The same two guards scripts/cost_reproducibility.py carries, for the same
    # reason: I reinstated the retired fixed-duration denominator once already
    # by copying a flag from a script that legitimately takes it.
    assert abs(solver - 0.2767124970000001) > 1e-6, "retired fixed-duration denominator"
    assert solver > 0.4, f"denominator {solver} implausibly small"

    cells = {
        "unpinned_warm3":    dict(pin=None,       n_warm=3),
        "unpinned_ramp":     dict(pin=None,       n_warm=a.rampup),
        "pinned_warm3":      dict(pin=a.pin_core, n_warm=3),
        "pinned_ramp":       dict(pin=a.pin_core, n_warm=a.rampup),
    }
    rows = {k: [] for k in cells}

    # Interleaved: invocation i of every cell before invocation i+1 of any, so a
    # drift in what the box is doing hits all four cells equally instead of
    # landing on whichever cell ran last.
    for i in range(a.invocations):
        for name, cfg in cells.items():
            before = os.getloadavg()[0]
            r = time_model(spec, n_rep=a.rounds, n_warm=cfg["n_warm"],
                           pin=cfg["pin"])
            cpu = np.asarray(r["cpu"], dtype=float)
            khz = [k for k in r["khz"] if k]
            rows[name].append({
                "i": i, "loadavg": before,
                "median_per_wafer_s": float(np.median(cpu)),
                "within_spread_factor": float(cpu.max() / cpu.min()),
                "speedup": float(solver / np.median(cpu)),
                "n_distinct_cores": len(set(r["cores"])),
                "median_khz": float(np.median(khz)) if khz else None,
                "affinity_n": r["affinity_n"],
            })
            print(f"  [{i+1}/{a.invocations}] {name:16s} "
                  f"{np.median(cpu)*1e6:7.1f} us  "
                  f"{solver/np.median(cpu):8.1f}x  "
                  f"cores={len(set(r['cores'])):2d}  "
                  f"{(np.median(khz)/1e6 if khz else float('nan')):.2f} GHz",
                  flush=True)

    summary = {}
    for name, rs in rows.items():
        us = np.array([r["median_per_wafer_s"] for r in rs]) * 1e6
        sp = np.array([r["speedup"] for r in rs])
        kh = [r["median_khz"] for r in rs if r["median_khz"]]
        summary[name] = {
            "median_us": float(np.median(us)),
            "min_us": float(us.min()), "max_us": float(us.max()),
            "between_invocation_spread_factor": float(us.max() / us.min()),
            "iqr_us": float(np.percentile(us, 75) - np.percentile(us, 25)),
            "speedup_median": float(np.median(sp)),
            "speedup_min": float(sp.min()), "speedup_max": float(sp.max()),
            "n_meeting_target": int((sp >= a.target).sum()),
            "n_invocations": len(rs),
            "median_ghz": float(np.median(kh) / 1e6) if kh else None,
            "max_distinct_cores_in_an_invocation":
                max(r["n_distinct_cores"] for r in rs),
            "verdict": ("PASS at every invocation" if sp.min() >= a.target else
                        "FAIL at every invocation" if sp.max() < a.target else
                        "STRADDLES the threshold"),
        }

    tightest = min(summary, key=lambda k:
                   summary[k]["between_invocation_spread_factor"])
    res = {
        "hypothesis": "H23: the between-invocation spread in clause-2 cost is "
                      "core and frequency state, not machine load. The load "
                      "hypothesis was falsified by the idle-box run, which came "
                      "back SLOWER (median 575.3 us at loadavg 20) than the "
                      "loaded run (526.0 us at loadavg 420).",
        "predictions_registered_before_the_run": [
            "rampup moves the median more than pin does",
            "pin reduces the between-invocation spread more than it moves the median",
            "pinned_ramp has the tightest spread of the four cells",
            "recorded frequency rises with rampup and carries the median difference",
        ],
        "adoption_rule_registered_before_the_run":
            "the cell with the SMALLEST between-invocation spread becomes the "
            "repo's clause-2 protocol, and its median is the clause-2 number "
            "even if that number is worse than what this repo has published. "
            "Selecting on speed is not available.",
        "config": a.config,
        "build": spec["build"],
        "box": {
            "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/"
                             "scaling_governor").read_text().strip(),
            "numa_nodes": 4, "cpus": 192,
            "model": "INTEL(R) XEON(R) PLATINUM 8558",
        },
        "protocol": {
            "invocations_per_cell": a.invocations,
            "rounds_per_invocation": a.rounds,
            "rampup_rollouts": a.rampup,
            "pin_core": a.pin_core,
            "interleaved": "invocation i of every cell before i+1 of any",
            "denominator_cpu_s": solver,
            "denominator_source": f"{a.workload} [{a.denominator}], "
                                  "workload-matched, held fixed",
            "speedup_target": a.target,
        },
        "cells": rows,
        "summary": summary,
        "tightest_cell": tightest,
        "clause2_number_under_adoption_rule": {
            "cell": tightest,
            "speedup_median": summary[tightest]["speedup_median"],
            "verdict": summary[tightest]["verdict"],
        },
    }
    Path(out).write_text(json.dumps(res, indent=2))

    print(f"\n{'cell':18s} {'median us':>10s} {'spread':>8s} {'GHz':>6s} "
          f"{'med x':>9s} {'meet':>6s}")
    for name, s in summary.items():
        print(f"{name:18s} {s['median_us']:10.1f} "
              f"{s['between_invocation_spread_factor']:7.2f}x "
              f"{(s['median_ghz'] or float('nan')):6.2f} "
              f"{s['speedup_median']:8.1f}x "
              f"{s['n_meeting_target']:3d}/{s['n_invocations']}")
    print(f"\ntightest spread: {tightest} -> clause-2 number "
          f"{summary[tightest]['speedup_median']:.1f}x "
          f"({summary[tightest]['verdict']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
