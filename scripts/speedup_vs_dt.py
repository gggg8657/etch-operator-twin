"""H27: how much of clause 2 is the operator, and how much is the chosen dt?

    python scripts/speedup_vs_dt.py --run runs/specprop/m4_ma64_s7

A 'wafer' is `n_steps * dt` of simulated etching. The solver pays for that
duration; a K-step operator applied once per wafer does not -- its cost is set
by tensor shape. So the speedup ratio can be moved by choosing `dt`, with
nothing about either implementation changing.

This repo already has two live denominators that differ by 1.99x for exactly
this reason: `bench_symmetric.py` prices every wafer at a fixed dt=0.2, while
`clock_matched_speedup.py` uses the test split's own dt values, whose mean is
0.3544. That is a 1.77x difference in physics per wafer, and it is most of the
distance between this repo's `NOT MET` and its `PASS`.

Both sides are timed through the SAME functions `bench_symmetric` uses, imported
rather than copied, because a second copy of a timed region is how the cold
solver and the warm operator got divided by each other in the first place.

Writes `runs/speedup_vs_dt.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.bench_symmetric import (N_DISCARD, operator_run,  # noqa: E402
                                     solver_warm, _stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/specprop/m4_ma64_s7")
    ap.add_argument("--dts", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8])
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/speedup_vs_dt.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="speedup_vs_dt")

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]
    norm_p = Path(a.data) / "norm.json"
    args = json.loads((Path(a.run) / "args.json").read_text())
    stride = args.get("eval_stride") or args.get("stride") or 1
    n_apply = n_steps // stride

    # The dt distribution the project is actually defined over, for the
    # reference lines. Measured here, not quoted from elsewhere.
    dt_test = np.load(Path(a.data) / "test.npz", allow_pickle=True)["dt"]
    dt_stats = {"n": int(dt_test.size), "min": float(dt_test.min()),
                "median": float(np.median(dt_test)), "mean": float(dt_test.mean()),
                "max": float(dt_test.max())}

    load0 = os.getloadavg()[0]
    cells = []
    for dt in a.dts:
        sw = solver_warm(a.rounds, n_steps, dt, grid_delta, a.seed,
                         n_discard=N_DISCARD)
        s = _stats(sw["wall"], sw["cpu"])
        # The operator is timed once per cell even though nothing about it
        # depends on dt: asserting that would be the whole hypothesis assumed.
        o = operator_run(a.run, norm_p, n_apply, n_rep=a.rounds, n_warm=3)
        op = _stats(o["wall"], o["cpu"])
        cells.append({
            "dt": dt, "sim_time_per_wafer": n_steps * dt,
            "solver_cpu_s": s["median_cpu"],
            "solver_single_threaded": s["single_threaded_verified"],
            "operator_cpu_s": op["median_cpu"],
            "operator_single_threaded": op["single_threaded_verified"],
            "speedup_cpu": s["median_cpu"] / op["median_cpu"],
            "applications_per_wafer": n_apply,
        })
        print(f"  dt={dt:<5} solver {s['median_cpu']*1e3:8.2f} ms  "
              f"operator {op['median_cpu']*1e3:7.3f} ms  "
              f"speedup {cells[-1]['speedup_cpu']:8.1f}x", flush=True)

    dts = np.array([c["dt"] for c in cells])
    sol = np.array([c["solver_cpu_s"] for c in cells])
    opc = np.array([c["operator_cpu_s"] for c in cells])
    # log-log slope: 1.0 means cost is proportional to simulated duration.
    slope_solver = float(np.polyfit(np.log(dts), np.log(sol), 1)[0])
    slope_op = float(np.polyfit(np.log(dts), np.log(opc), 1)[0])
    op_spread = float(opc.max() / opc.min())

    # Where the ratio crosses the clause, on the fitted solver line and the
    # operator's median cost. Reported as a protocol fact, not an achievement.
    op_med = float(np.median(opc))
    cA, cB = np.polyfit(np.log(dts), np.log(sol), 1)
    dt_cross = float(np.exp((np.log(1000.0 * op_med) - cB) / cA))

    p1 = slope_solver > 0.8
    p2 = op_spread <= 1.15
    res = {
        "hypothesis": "H27: clause 2's ratio is ~linear in dt, because a K-step "
                      "operator applied once per wafer costs the same whatever "
                      "duration that wafer represents, while the solver pays for "
                      "the duration.",
        "predictions_registered_before_run": {
            "1_solver_loglog_slope_gt_0.8": p1,
            "2_operator_flat_within_15pct": p2,
            "3_falsifier": "operator moves >15% with dt, or solver slope < 0.8",
        },
        "falsifier_fired": bool(not (p1 and p2)),
        "protocol": {
            "estimator": "median CPU-seconds per wafer, solver / operator",
            "both_sides": "the same functions bench_symmetric.py uses, imported",
            "solver_reading": "marginal_warm: long-lived process, "
                              f"{N_DISCARD} wafer discarded",
            "operator_reading": "warm, 3 untimed rollouts",
            "recipe_stream": f"seeded {a.seed}, identical across dt cells",
            "rounds": a.rounds, "n_steps_per_wafer": n_steps,
            "grid_delta": grid_delta, "run": a.run,
            "loadavg_start": load0, "loadavg_end": os.getloadavg()[0],
        },
        "test_split_dt": dt_stats,
        "cells": cells,
        "solver_loglog_slope": slope_solver,
        "operator_loglog_slope": slope_op,
        "operator_spread_over_dt": op_spread,
        "dt_where_speedup_crosses_1000x": dt_cross,
        "dt_fixed_in_bench_symmetric": 0.2,
        "interpretation": (
            "speedup is a function of the simulated duration a wafer is defined "
            "to represent; quoting it without stating dt is not a property of "
            "the operator"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["solver_loglog_slope", "operator_spread_over_dt",
                       "dt_where_speedup_crosses_1000x", "falsifier_fired"]},
                     indent=2))


if __name__ == "__main__":
    main()
