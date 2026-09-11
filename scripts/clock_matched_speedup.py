"""H24: clause 2 is a ratio of two timings taken at different CPU clocks, and
nobody has ever measured either clock.

    python scripts/clock_matched_speedup.py --rounds 6

**What forced this.** `runs/cost_pinning_specprop_m4_ma64_K10.json` established
that the operator's cost reading is driven by the core clock the governor
happens to give it: across 56 invocations of ONE unchanged architecture the
reading spans **2.55x** (353.7-901.0 us), the clock spans **1.91x**
(2.10-4.00 GHz), and **R^2 of cost on 1/clock is 0.784**. Machine load
correlates NEGATIVELY (r = -0.36) -- the idle-box run was slower than the
loaded one -- which falsified the load hypothesis this repo had registered.

The consequence nobody has checked: **clause 2 is a ratio**, and its two sides
have opposite duration profiles.

* the numerator (solver) is a single **~0.5-second** compute-bound C++ apply.
  A run that long drives the governor to a high clock and stays there.
* the denominator (operator) is **~0.5 milliseconds** of tiny dispatches. It is
  a thousand times too short to ramp anything, so it is measured at whatever
  clock the core was already in -- which `cost_pinning` shows is 2.10 GHz about
  half the time and 4.00 GHz the other half.

If the two sides sit at systematically different clocks, every speedup in this
repo is a hardware-state artefact of unknown sign and up to 1.9x in size. That
is the same class of error as the wrong-workload denominator this repo already
had to correct, and it is currently unmeasured.

**Predictions, registered before the run:**

1. The solver sustains **>= 3.5 GHz**, because 0.5 s is long enough to ramp.
2. The operator's clock is **lower and far more variable** than the solver's.
3. Therefore the published raw speedups **UNDERSTATE** the cycle-matched one:
   a fast solver clock shrinks the numerator and a slow operator clock inflates
   the denominator, and both push the ratio down.
4. The cycle-matched ratio has a **smaller spread** than the raw ratio, because
   removing the clock removes the dominant noise term.

**Prediction 3 says the correction makes the clause EASIER, and that is exactly
the direction in which I am least trustworthy.** Three guards, fixed here
before any number exists:

* the raw ratio is computed, reported and never deleted. It is the
  DEPLOYMENT reading and it stays the headline, because a real caller also
  issues short bursts and also gets whatever clock the governor gives it.
  The cycle-matched ratio is the ARCHITECTURE reading -- how much less work the
  operator does, clock removed -- and it is the one that transfers to another
  machine. Two questions, two numbers, both labelled. Neither replaces the other.
* the clause-2 verdict is taken from the **raw** reading. The cycle-matched
  number may not be used to declare the clause met.
* cost is measured in a pristine pass with NO monitor thread (the published
  protocol, byte for byte), and the clock in an immediately adjacent pass with
  one. The monitor pass reports its own cost too; if the two passes disagree by
  more than `--monitor-tol` the clock reading is marked untrustworthy in the
  JSON rather than used. `time.process_time()` is process-wide, so a monitor
  thread inside the timed process would inflate the very number it is there to
  explain.

Writes `runs/clock_matched_speedup.json`.
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

RECIPE_KEYS = ["ion_flux", "etchant_flux", "oxygen_flux", "ion_energy",
               "trench_width", "mask_height"]

# THE MONITOR LIVES IN THE PARENT PROCESS, and the first version of this
# script got that wrong. A sampler thread inside the timed process inflated the
# operator's 0.5 ms reading by 25% -- `time.process_time()` is process-wide, so
# the sampler's own CPU landed in the number it existed to explain, and the run
# tripped this script's own trustworthiness guard. Sampling from the parent
# cannot touch the child's process_time at all.
#
# The child therefore publishes the CLOCK_MONOTONIC bounds of its timed region
# and the parent timestamps every sample, keeping afterwards only those inside
# that window. This matters: the child spends seconds importing torch and
# microseconds in the timed region, so an unfiltered parent-side median would
# be reporting the clock during `import torch`.
_PRE = "import time, json, sys, os\nsys.path.insert(0, %r)\n" % str(ROOT)
_T0 = "t0_mono = time.clock_gettime(time.CLOCK_MONOTONIC)\n"
_T1 = "t1_mono = time.clock_gettime(time.CLOCK_MONOTONIC)\n"


def _watch(pid):
    """(core, kHz) for whichever core is currently running `pid`."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            c = int(f.read().rsplit(")", 1)[1].split()[36])
        with open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq") as f:
            return c, int(f.read().strip())
    except (OSError, IndexError, ValueError):
        return None


def _run(body, monitor):
    """Run a timing body in a fresh subprocess.

    The child is byte-for-byte identical in both passes, so the cost reading
    cannot depend on whether it was watched -- which is what makes the
    pristine/monitored agreement check meaningful.
    """
    full = (_PRE + _T0 + body + _T1 +
            'print(json.dumps({"cpu": cpu, "wall": wall, '
            '"t0": t0_mono, "t1": t1_mono}))\n')
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if not monitor:
        out = subprocess.run([sys.executable, "-c", full], capture_output=True,
                             text=True, env=env, timeout=3600, cwd=ROOT)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-3000:])
        r = json.loads(out.stdout.strip().splitlines()[-1])
        return {**r, "khz": [], "n_cores": 0}

    p = subprocess.Popen([sys.executable, "-c", full], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
    samples = []
    while p.poll() is None:
        w = _watch(p.pid)
        if w:
            samples.append((time.clock_gettime(time.CLOCK_MONOTONIC), *w))
        time.sleep(0.001)
    so, se = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(se[-3000:])
    r = json.loads(so.strip().splitlines()[-1])
    inwin = [(c, k) for t, c, k in samples if r["t0"] <= t <= r["t1"]]
    return {**r, "khz": [k for _, k in inwin],
            "n_cores": len(set(c for c, _ in inwin)),
            "n_samples_total": len(samples)}


# Reused, not reimplemented: the solver body must be byte-for-byte the one
# bench_workload.py times, and a second copy of the recipe serialiser is the
# obvious way for the two to drift apart.
from scripts.bench_workload import _rec_literal  # noqa: E402


def solver_body(rows, dts, n_steps, grid_delta, n_discard=1):
    """Byte-for-byte the timed region of bench_workload.solver_terminal."""
    recs = "\n".join(f"JOBS.append(({_rec_literal(r)}, {float(d)!r}))"
                     for r, d in zip(rows, dts))
    return f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu = [], []
for rec, dt in JOBS:
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {n_steps} * dt).apply()
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
wall = wall[{n_discard}:]; cpu = cpu[{n_discard}:]
'''


def operator_body(spec, n_rep, n_warm, H=128, sustained_s=0.0):
    """The operator's timed region, in one of two duration regimes.

    `burst` (sustained_s=0) is byte-for-byte cost_floor.time_model's, which
    every published number in this repo used: ~16 calls, ~5 ms of work in total.

    `sustained` answers the asymmetry this script found. The solver is one
    ~0.5 s run and the operator is a ~0.35 ms burst, and the governor treats
    those differently -- so the ratio carries a clock term that has nothing to
    do with either implementation. Sustained mode issues back-to-back
    **batch-one, sequential** calls until `sustained_s` of wall time has passed
    and divides by the number completed. That matches the DURATION profile
    without handing the operator a batching advantage: every call still
    produces exactly one wafer, one after another, which is the workload the
    KPI names. Proposed by `codex` at rung 4, and it beats my arithmetic
    normalisation because it MEASURES the common-frequency ratio rather than
    estimating it from a frequency that memory stalls need not obey.
    """
    if sustained_s <= 0:
        loop = (
            "    for _ in range(%d):\n"
            "        w0 = time.perf_counter(); c0 = time.process_time()\n"
            "        %s\n"
            "        wall.append(time.perf_counter() - w0)\n"
            "        cpu.append(time.process_time() - c0)\n"
            % (n_rep, spec["call"]))
    else:
        loop = (
            "    _n = 0\n"
            "    _w0 = time.perf_counter(); _c0 = time.process_time()\n"
            "    while time.perf_counter() - _w0 < %r:\n"
            "        %s\n"
            "        _n += 1\n"
            "    _w = time.perf_counter() - _w0\n"
            "    _c = time.process_time() - _c0\n"
            "    wall.append(_w / _n); cpu.append(_c / _n)\n"
            "    n_calls = _n\n"
            % (sustained_s, spec["call"]))
    head = (
        "\nimport torch\n"
        "torch.set_num_threads(1)\n"
        "%s\n"
        "if hasattr(M, 'eval'):\n"
        "    M.eval()\n"
        "torch.manual_seed(0)\n"
        "phi = torch.randn(1, 1, %d, %d); cond = torch.randn(1, 7)\n"
        "wall, cpu = [], []\n"
        "with torch.no_grad():\n"
        "    for _ in range(%d):\n"
        "        %s\n"
        % (spec["build"], H, H, n_warm, spec["call"]))
    return head + loop


def paired(body, tol, label):
    """Cost and clock from ONE monitored pass, plus a pristine bias check.

    The first version of this function took cost from a pristine pass and clock
    from an adjacent monitored one, and flagged the clock untrustworthy when the
    two costs disagreed by more than `tol`. That guard was mis-specified and it
    threw away 3 of 5 readings: the operator's INTRINSIC between-invocation
    spread is 2.55x (`runs/cost_pinning_specprop_m4_ma64_K10.json`), so two
    adjacent invocations routinely differ by far more than a 5% tolerance and
    the disagreement says nothing about the monitor. Worse, the monitored pass
    came out FASTER in every flagged case, which a monitor overhead cannot do.

    So cost and clock now come from the SAME pass and are self-consistent by
    construction. The pristine pass is still run and still recorded, but it is
    now used for a global question -- does watching a process bias its cost? --
    answered once over all repeats by a paired test, not per-reading by a
    tolerance that the noise floor makes meaningless.
    """
    clean = _run(body, monitor=False)
    mon = _run(body, monitor=True)
    c_clean = float(np.median(clean["cpu"]))
    c_mon = float(np.median(mon["cpu"]))
    khz = np.array(mon["khz"], dtype=float)
    row = {
        "median_cpu_s": c_mon,
        "median_cpu_s_pristine": c_clean,
        "monitored_over_pristine": c_mon / c_clean if c_clean > 0 else None,
        "n_clock_samples": int(len(khz)),
        "median_ghz": float(np.median(khz) / 1e6) if len(khz) else None,
        "min_ghz": float(khz.min() / 1e6) if len(khz) else None,
        "max_ghz": float(khz.max() / 1e6) if len(khz) else None,
        "clock_constant": bool(len(khz) and khz.min() == khz.max()),
        "distinct_cores_during_run": mon["n_cores"],
        "have_clock": bool(len(khz)),
        "rounds": len(mon["cpu"]),
        "n_calls": mon.get("n_calls"),
    }
    print("  %-12s %10.1f us  %5.2f GHz  (pristine %8.1f us, %d samples)"
          % (label, c_mon * 1e6,
             row["median_ghz"] if row["median_ghz"] else float("nan"),
             c_clean * 1e6, len(khz)), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="specprop_m4_ma64_K10")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rounds", type=int, default=6,
                    help="solver wafers priced (plus one discarded warm-up)")
    ap.add_argument("--op-rounds", type=int, default=16)
    ap.add_argument("--sustained-s", type=float, default=0.5,
                    help="wall seconds of back-to-back batch-1 operator calls "
                         "in the duration-matched arm; 0.5 matches the solver")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--monitor-tol", type=float, default=0.05)
    ap.add_argument("--target", type=float, default=1000.0)
    ap.add_argument("--out", default="runs/clock_matched_speedup.json")
    a = ap.parse_args()

    spec = SPECS[a.config]
    runlock.acquire(a.out, what="clock_matched_speedup")

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]
    d = np.load(Path(a.data) / f"{a.split}.npz")
    n_take = a.rounds + 1
    rows, dts = d["recipe"][:n_take], d["dt"][:n_take]

    sbody = solver_body(rows, dts, n_steps, grid_delta)
    burst = operator_body(spec, a.op_rounds, 3)
    sust = operator_body(spec, a.op_rounds, 3, sustained_s=a.sustained_s)

    pairs = []
    for r in range(a.repeats):
        print("[pair %d/%d]  loadavg %.1f" % (r + 1, a.repeats,
                                              os.getloadavg()[0]), flush=True)
        sol = paired(sbody, a.monitor_tol, "solver")
        ob = paired(burst, a.monitor_tol, "op burst")
        os_ = paired(sust, a.monitor_tol, "op sustained")

        def _ratios(op):
            raw = sol["median_cpu_s"] / op["median_cpu_s"]
            cyc = None
            if sol["median_ghz"] and op["median_ghz"]:
                cyc = raw * (sol["median_ghz"] / op["median_ghz"])
            return raw, cyc

        raw_b, cyc_b = _ratios(ob)
        raw_s, cyc_s = _ratios(os_)
        pairs.append({
            "repeat": r, "loadavg": os.getloadavg()[0], "solver": sol,
            "operator_burst": ob, "operator_sustained": os_,
            "speedup_raw_burst": raw_b, "speedup_cycle_matched_burst": cyc_b,
            "speedup_raw_sustained": raw_s,
            "speedup_cycle_matched_sustained": cyc_s,
            "clock_ratio_solver_over_burst":
                (sol["median_ghz"] / ob["median_ghz"])
                if (sol["median_ghz"] and ob["median_ghz"]) else None,
            "clock_ratio_solver_over_sustained":
                (sol["median_ghz"] / os_["median_ghz"])
                if (sol["median_ghz"] and os_["median_ghz"]) else None,
        })
        print("  -> burst %8.1fx (cyc %s)   sustained %8.1fx (cyc %s)\n"
              % (raw_b, ("%.1f" % cyc_b) if cyc_b else "n/a",
                 raw_s, ("%.1f" % cyc_s) if cyc_s else "n/a"), flush=True)

    def _sum(x, target=a.target):
        x = np.array([v for v in x if v is not None], dtype=float)
        if not len(x):
            return None
        out = {"median": float(np.median(x)), "min": float(x.min()),
               "max": float(x.max()), "n": int(len(x)),
               "n_meeting_target": int((x >= target).sum())}
        # A spread needs at least two readings. The previous version printed
        # "spread 1.00x" off a single surviving value, which reads as perfect
        # reproducibility and is the exact opposite of what n=1 means.
        out["spread_factor"] = (float(x.max() / x.min()) if len(x) > 1
                                else None)
        return out

    S = {
        "speedup_raw_burst": _sum([p["speedup_raw_burst"] for p in pairs]),
        "speedup_raw_sustained": _sum([p["speedup_raw_sustained"] for p in pairs]),
        "speedup_cycle_matched_burst":
            _sum([p["speedup_cycle_matched_burst"] for p in pairs]),
        "speedup_cycle_matched_sustained":
            _sum([p["speedup_cycle_matched_sustained"] for p in pairs]),
        "solver_ghz": _sum([p["solver"]["median_ghz"] for p in pairs], 0),
        "operator_burst_ghz":
            _sum([p["operator_burst"]["median_ghz"] for p in pairs], 0),
        "operator_sustained_ghz":
            _sum([p["operator_sustained"]["median_ghz"] for p in pairs], 0),
    }

    # Does watching a process bias its cost? Asked once over all readings with
    # an exact paired sign test, instead of per-reading with a tolerance that
    # this workload's 2.55x noise floor makes meaningless.
    ratios = [p[k]["monitored_over_pristine"] for p in pairs
              for k in ("solver", "operator_burst", "operator_sustained")
              if p[k]["monitored_over_pristine"]]
    n_up = sum(1 for x in ratios if x > 1)
    n = len(ratios)
    from math import comb
    p_two = min(1.0, 2 * sum(comb(n, k) for k in range(n_up, n + 1)) / 2 ** n) \
        if n else None
    S["monitor_bias_check"] = {
        "question": "does parent-side clock sampling change the cost it "
                    "measures? Answered globally by an exact paired sign test, "
                    "not per-reading by a tolerance.",
        "n_readings": n, "n_monitored_slower": n_up,
        "median_monitored_over_pristine": float(np.median(ratios)) if n else None,
        "sign_test_p_two_sided": p_two,
        "reading": "no detectable bias" if (p_two or 1) > 0.05 else
                   "monitoring biases the cost; the clock readings are suspect",
    }

    def _verdict(k):
        s = S[k]
        if not s:
            return "[not measured]"
        if s["min"] >= a.target:
            return "PASS at every repeat"
        if s["max"] < a.target:
            return "FAIL at every repeat"
        return "STRADDLES the threshold"

    res = {
        "hypothesis": "H24: clause 2 is a ratio of two timings taken at "
                      "different CPU clocks, and neither clock had ever been "
                      "measured. The solver is a ~0.5 s sustained run and the "
                      "operator a ~0.35 ms burst, so the governor treats them "
                      "differently and the ratio carries a clock term that "
                      "belongs to neither implementation.",
        "predictions_registered_before_the_run": [
            "the solver sustains >= 3.5 GHz",
            "the operator's clock is lower and more variable than the solver's",
            "the published raw speedups UNDERSTATE the cycle-matched one",
            "the cycle-matched ratio has a smaller spread than the raw ratio",
        ],
        "prediction_outcomes_first_run": {
            "1_solver_ge_3.5GHz": "FALSIFIED -- the solver ran at 2.10 GHz, "
                                  "exactly base, in 4 of 5 repeats",
            "2_operator_lower_and_more_variable": "FALSIFIED in direction, "
                                                  "confirmed in variability: "
                                                  "the operator was HIGHER "
                                                  "(median 3.02 vs 2.10 GHz) "
                                                  "and did vary more "
                                                  "(2.10-4.00 vs 2.10-3.90)",
            "3_raw_understates": "FALSIFIED -- raw OVERSTATES, because the "
                                 "fast side is the operator, not the solver",
            "4_cycle_matched_tighter": "[not measured] -- the mis-specified "
                                       "guard left only one surviving "
                                       "cycle-matched value, and a spread "
                                       "needs two",
        },
        "sustained_arm_registered_prediction":
            "matching the DURATION profile should collapse the clock "
            "asymmetry: the operator, run as 0.5 s of back-to-back batch-1 "
            "calls, should be measured at the solver's clock rather than at "
            "single-core turbo. If so the sustained raw ratio is the honest "
            "common-frequency reading, and it should land BELOW the burst raw "
            "ratio and near the cycle-matched estimate.",
        "which_reading_is_the_verdict":
            "the SUSTAINED raw ratio, if the sustained arm does equalise the "
            "clocks, because it is the only reading that measures both sides "
            "in the same governor regime without estimating anything. The "
            "burst raw ratio is retained as the deployment reading -- a real "
            "caller issuing one wafer at a time genuinely does get turbo -- "
            "and the cycle-matched ratio as the portable architecture reading. "
            "Three questions, three numbers, none deleted. Note that the "
            "pre-registration in the first version of this file named the "
            "BURST raw ratio as the verdict, on the expectation that it would "
            "be the conservative one; it turned out to be the flattering one, "
            "and honouring a pre-registration that has been falsified in its "
            "premise would be using the registration to launder a number.",
        "config": a.config,
        "box": {
            "governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/"
                             "scaling_governor").read_text().strip(),
            "model": "INTEL(R) XEON(R) PLATINUM 8558",
            "base_ghz": 2.1, "cpus": 192, "numa_nodes": 4,
            "turbo_control": "unavailable -- /sys/.../intel_pstate/no_turbo is "
                             "root-owned and this loop has no sudo, so the "
                             "clock cannot be pinned directly and the "
                             "duration-matched arm is the available substitute",
        },
        "protocol": {
            "solver_wafers_priced": a.rounds, "warmup_wafers_discarded": 1,
            "operator_rounds_burst": a.op_rounds,
            "operator_sustained_s": a.sustained_s,
            "sustained_is_not_batching": "batch size stays 1 and calls stay "
                                         "sequential; only the number of "
                                         "consecutive calls changes",
            "repeats": a.repeats,
            "denominator": "terminal_one_apply, measured in THIS invocation on "
                           "the test split's own recipes and dt values",
            "estimator": "CPU-seconds, one verified thread, fresh subprocess "
                         "per pass",
            "clock_method": "parent-process sampler at 1 ms, filtered to the "
                            "child's own CLOCK_MONOTONIC timed window. An "
                            "in-process sampler inflated the operator 25% "
                            "because process_time() is process-wide.",
            "speedup_target": a.target,
        },
        "pairs": pairs,
        "summary": S,
        "verdicts": {k: _verdict(k) for k in
                     ("speedup_raw_burst", "speedup_raw_sustained",
                      "speedup_cycle_matched_burst",
                      "speedup_cycle_matched_sustained")},
    }
    Path(a.out).write_text(json.dumps(res, indent=2))

    print("\n%-34s %9s %9s %8s %7s" % ("reading", "median", "range", "spread",
                                       "meet"))
    for k in ("speedup_raw_burst", "speedup_raw_sustained",
              "speedup_cycle_matched_burst", "speedup_cycle_matched_sustained"):
        s = S[k]
        if not s:
            print("%-34s %9s" % (k, "[not measured]")); continue
        print("%-34s %8.1fx %4.0f-%4.0f %7s %4d/%d"
              % (k, s["median"], s["min"], s["max"],
                 ("%.2fx" % s["spread_factor"]) if s["spread_factor"] else "n=1",
                 s["n_meeting_target"], s["n"]))
    for k in ("solver_ghz", "operator_burst_ghz", "operator_sustained_ghz"):
        s = S[k]
        if s:
            print("%-34s %8.2f  %4.2f-%4.2f" % (k, s["median"], s["min"],
                                                s["max"]))
    print("\nmonitor bias: %s (p=%s, n=%d)"
          % (S["monitor_bias_check"]["reading"],
             S["monitor_bias_check"]["sign_test_p_two_sided"],
             S["monitor_bias_check"]["n_readings"]))
    for k, v in res["verdicts"].items():
        print("  %-34s %s" % (k, v))
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
