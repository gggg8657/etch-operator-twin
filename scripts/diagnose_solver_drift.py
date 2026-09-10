"""Why does the same solver cost 1.1 s/wafer here and 8.5 s/wafer there?

    python scripts/diagnose_solver_drift.py

The like-for-like row of the speedup clause is a ratio whose denominator is
ViennaPS at one OpenMP thread on one wafer. Three committed measurements of that
denominator, same code, same grid (0.2 um, 128), same 10 timesteps:

    1.538 s/wafer   runs/speed.json            load 11.56
    8.457 s/wafer   runs/speed_seed1_cpu.json  load 40.65
    8.390 s/wafer   runs/speed_K10_cpu.json    load 22.32

and `scripts/bench_paired.py`, which times one wafer per fresh subprocess,
measures **1.11 CPU-s at load 11.3 and 1.26 CPU-s at load 41.0** -- 1.15x apart
across a 3.6x load range. So the load explanation, which is the one I wrote into
`bench_paired.py`'s docstring before running this, does not survive: load moves
this denominator by ~15%, not by 5.5x.

The remaining structural difference is **process reuse**. `bench_speed.py`'s
`time_solver` runs one child process that loops over `n_traj` trajectories, and
for each one first times 10 *stepped* applies and then times one full-duration
apply. By its last trajectory that process has built 16 domains and run 88
applies. If ViennaPS accumulates cost across domains in a process -- retained
level-set data, a growing allocator, mesh refinement that is never released --
then the `single` timings inflate with trajectory index, and the reported median
over 8 trajectories is not the cost of a wafer.

This script separates the two explanations by recording each trajectory's time
*individually* under three conditions:

* `reuse_stepped_then_single` -- exactly `bench_speed.py`'s loop, per-trajectory
  times kept rather than reduced to a median.
* `reuse_single_only`         -- the same long-lived process, but without the
  stepped applies, so domain count rises without the 10x apply count. Separates
  "many applies" from "many domains".
* `fresh_process_per_wafer`   -- `bench_paired.py`'s condition, the control.

A rising trend within a condition is accumulation. A flat trend with a level
offset between conditions is a fixed cost of the harness. A flat trend at the
same level everywhere would mean load after all, and this script is wrong.

Writes `runs/solver_drift.json`.
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

_PRE = f'''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, numpy as np, sys
sys.path.insert(0, {str(ROOT)!r})
from eot import solver as S
'''


def _child(body: str):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _PRE + body],
                         capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def reuse(n_traj, n_steps, dt, grid_delta, seed, stepped: bool):
    """One process, n_traj trajectories, per-trajectory times retained."""
    body = f'''
rng = np.random.default_rng({seed})
single, stepped_t, cpu = [], [], []
for i in range({n_traj}):
    rec = S.sample_recipe(rng)
    if {stepped}:
        dom = S.build_domain(rec, {grid_delta})
        t0 = time.perf_counter()
        for _ in range({n_steps}):
            S.make_process(rec, dom, {dt}).apply()
        stepped_t.append(time.perf_counter() - t0)
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {n_steps} * {dt}).apply()
    single.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"single": single, "stepped": stepped_t, "cpu": cpu}}))
'''
    return _child(body)


def fresh(n_traj, n_steps, dt, grid_delta, seed):
    """A fresh process per wafer -- bench_paired.py's condition."""
    single, cpu = [], []
    for i in range(n_traj):
        body = f'''
# Replay the reuse loop's recipe stream and take its i-th draw, so the three
# conditions time the SAME eight wafers and a difference between them cannot be
# a difference in the geometry sampled.
rng = np.random.default_rng({seed})
for _ in range({i} + 1):
    rec = S.sample_recipe(rng)
dom = S.build_domain(rec, {grid_delta})
w0 = time.perf_counter(); c0 = time.process_time()
S.make_process(rec, dom, {n_steps} * {dt}).apply()
print(json.dumps({{"wall": time.perf_counter() - w0, "cpu": time.process_time() - c0}}))
'''
        r = _child(body)
        single.append(r["wall"])
        cpu.append(r["cpu"])
    return {"single": single, "stepped": [], "cpu": cpu}


def trend(xs):
    """Slope per trajectory and the last/first ratio -- accumulation or not."""
    if len(xs) < 2:
        return {"slope_per_traj": None, "last_over_first": None}
    i = np.arange(len(xs), dtype=float)
    slope = float(np.polyfit(i, np.asarray(xs, dtype=float), 1)[0])
    return {"slope_per_traj": slope,
            "last_over_first": float(xs[-1] / xs[0]),
            "first": float(xs[0]), "last": float(xs[-1]),
            "median": float(np.median(xs)), "max_over_min": float(max(xs) / min(xs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=8,
                    help="8 is what bench_speed.py used")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/solver_drift.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="solver_drift")
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]

    load_before = os.getloadavg()
    conds = {}
    conds["reuse_stepped_then_single"] = reuse(a.n_traj, n_steps, a.dt, grid_delta,
                                               a.seed, stepped=True)
    conds["reuse_single_only"] = reuse(a.n_traj, n_steps, a.dt, grid_delta,
                                       a.seed, stepped=False)
    conds["fresh_process_per_wafer"] = fresh(a.n_traj, n_steps, a.dt, grid_delta,
                                             a.seed)
    load_after = os.getloadavg()

    res = {
        "question": "is the 1.5 -> 8.5 s/wafer spread in the committed solver "
                    "denominators caused by box load, or by reusing one process "
                    "across trajectories?",
        "protocol": {
            "n_traj": a.n_traj, "n_steps_per_wafer": n_steps, "dt": a.dt,
            "grid_delta": grid_delta, "threads": 1,
            "conditions": {
                "reuse_stepped_then_single": "bench_speed.py's exact loop; the "
                                             "condition every committed solver "
                                             "denominator was measured under",
                "reuse_single_only": "same long-lived process, no stepped "
                                     "applies: domain count rises, apply count "
                                     "does not",
                "fresh_process_per_wafer": "bench_paired.py's condition, the control",
            },
        },
        "load": {"before": load_before, "after": load_after,
                 "note": "one invocation, so all three conditions see the same load; "
                         "a difference between them cannot be a load difference"},
        "conditions": {},
    }
    for name, c in conds.items():
        res["conditions"][name] = {
            "single_wall_per_traj": c["single"],
            "single_cpu_per_traj": c["cpu"],
            "stepped_wall_per_traj": c["stepped"],
            "wall_trend": trend(c["single"]),
            "cpu_trend": trend(c["cpu"]),
        }

    fp = res["conditions"]["fresh_process_per_wafer"]["wall_trend"]["median"]
    ru = res["conditions"]["reuse_stepped_then_single"]["wall_trend"]["median"]
    res["verdict"] = {
        "fresh_median_s_per_wafer": fp,
        "reuse_median_s_per_wafer": ru,
        "reuse_over_fresh": ru / fp,
        "reuse_last_over_first": res["conditions"]["reuse_stepped_then_single"]["wall_trend"]["last_over_first"],
        "reading": ("accumulation within a reused process" if ru / fp > 1.5
                    else "no material difference between the conditions; the "
                         "committed spread is NOT explained by process reuse and "
                         "this script's hypothesis is wrong"),
        "same_load_for_all_conditions": True,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"load": res["load"], "verdict": res["verdict"]}, indent=2))
    for n, c in res["conditions"].items():
        print(f"\n{n}\n  wall per traj: "
              + ", ".join(f"{x:.3f}" for x in c["single_wall_per_traj"]))


if __name__ == "__main__":
    main()
