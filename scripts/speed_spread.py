"""Run-to-run spread of the speedup measurement itself.

    python scripts/speed_spread.py --repeats 8

**Why.** Two invocations of `scripts/bench_symmetric.py`, identical arguments,
disagreed by up to 1.8x: the K=10 `marginal_warm` speedup read 3.83x and then
2.14x, and `runs/seed1`'s operator cost 0.9506 and then 1.2585 CPU-s per wafer.
Yet *within* either invocation the spread over 8 rounds was 1.03-1.17x. So the
noise lives between invocations, not between rounds -- a per-process factor
(core assignment, frequency, NUMA placement, cache pressure from the other
projects on this box), which is the CPU analogue of the H100 clock-ramp artefact
that made `uq-surrogate-kit` report 40.8x for a 23.5x quantity.

Averaging more rounds inside one invocation cannot see this and would only
tighten a confidence interval around the wrong centre. `.overnight/RULES.md`'s
seed-count lesson applies unchanged to a timing measurement: measure your own
run-to-run spread by repeating one configuration, and if a verdict depends on it,
repeat it enough times to use an exact test.

There is one asymmetry worth noting about which way this cuts. The solver's own
between-invocation spread is small (0.2758 and 0.2767 CPU-s warm across the two
invocations, 0.3%), while the operator's is large. The ratio is therefore mostly
inheriting the operator's instability, and reporting a single invocation's ratio
as the clause reading would be reporting one draw from a distribution 1.8x wide.

Writes `runs/speed_spread.json`, and each invocation's own file to
`runs/speed_sym_rep<N>.json` so nothing is aggregated that cannot be traced back.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402

READINGS = ("marginal_warm", "cold_single_wafer")


def summarise(xs):
    xs = np.asarray(xs, dtype=float)
    return {
        "per_invocation": [float(x) for x in xs],
        "n": int(xs.size),
        "median": float(np.median(xs)),
        "min": float(xs.min()), "max": float(xs.max()),
        "max_over_min": float(xs.max() / xs.min()),
        # the spread a reader needs in order to know whether a difference
        # between two arms is larger than the harness's own noise
        "iqr": [float(np.percentile(xs, 25)), float(np.percentile(xs, 75))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--runs", nargs="+",
                    default=["runs/seed1", "runs/kcurve/K2_nv_s1",
                             "runs/kcurve/K5_nv_s1", "runs/kcurve/K10_nv_s1"])
    ap.add_argument("--out", default="runs/speed_spread.json")
    ap.add_argument("--py", default=sys.executable)
    a = ap.parse_args()

    runlock.acquire(a.out, what="speed_spread")
    reps = []
    for i in range(a.repeats):
        out = Path(f"runs/speed_sym_rep{i}.json")
        cmd = [a.py, "scripts/bench_symmetric.py", "--runs", *a.runs,
               "--rounds", str(a.rounds), "--out", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            print(f"repeat {i} FAILED\n{r.stderr[-1500:]}", file=sys.stderr)
            continue
        reps.append(json.loads(out.read_text()))
        print(f"repeat {i}: best "
              f"{reps[-1]['kpi_clause']['best_value']:.2f}x", flush=True)

    if len(reps) < 2:
        raise SystemExit("need at least 2 successful repeats to state a spread")

    res = {
        "question": "how much does the speedup measurement move between "
                    "identical invocations, and is that larger than the "
                    "differences being reported?",
        "protocol": {
            "repeats": len(reps), "rounds_per_repeat": a.rounds,
            "arms": a.runs,
            "estimator": "median CPU-seconds per wafer within an invocation, "
                         "then the spread of that median across invocations",
            "per_invocation_files": [f"runs/speed_sym_rep{i}.json"
                                     for i in range(len(reps))],
        },
        "solver": {rd: summarise([r["solver"][rd]["median_cpu"] for r in reps])
                   for rd in READINGS},
        "arms": {},
    }
    for arm in a.runs:
        res["arms"][arm] = {
            "applications_per_wafer": reps[0]["arms"][arm]["applications_per_wafer"],
            **{rd: {
                "cpu_s_per_wafer": summarise(
                    [r["arms"][arm][rd]["median_cpu"] for r in reps]),
                "speedup_cpu": summarise(
                    [r["arms"][arm][rd]["speedup_cpu"] for r in reps]),
            } for rd in READINGS},
        }

    best = max(
        ((res["arms"][arm][rd]["speedup_cpu"], arm, rd)
         for arm in a.runs for rd in READINGS),
        key=lambda t: t[0]["median"])
    s, arm, rd = best
    res["kpi_clause"] = {
        "clause": "simulation speedup >= 1000x",
        "target": 1000.0,
        "best_arm": arm, "best_reading": rd,
        "value_median_over_invocations": s["median"],
        "value_range_over_invocations": [s["min"], s["max"]],
        "n_invocations": s["n"],
        "met": bool(s["max"] >= 1000.0),
        "met_note": "met only if the BEST invocation clears the target; it does "
                    "not, so no choice among invocations can pass this clause",
        "shortfall_factor_at_median": 1000.0 / s["median"],
        "shortfall_factor_at_best_invocation": 1000.0 / s["max"],
        "harness_noise_max_over_min": s["max_over_min"],
        "verdict_robust_to_harness_noise": bool(s["max"] < 1000.0),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["kpi_clause"], indent=2))


if __name__ == "__main__":
    main()
