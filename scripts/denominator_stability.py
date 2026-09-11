"""Is the solver denominator reproducible across invocations on this box?

    python scripts/denominator_stability.py --repeats 12

Every speedup this repo has ever published is a ratio whose numerator is a
solver timing. `runs/solver_drift.json` established that CPU-seconds is nearly
load-invariant -- 1.11 CPU-s at load 11.3 against 1.26 at load 41.0, 1.15x
across a 3.6x load range -- and that is the entire justification for using
CPU-seconds on a box this track cannot quiet.

Four consecutive runs of `speedup_vs_dt.py`, identical seed and identical eight
wafers, put the dt=0.1 solver at 129.0, 129.7, 130.3 and then 348.9 ms. The last
was taken while the box load was FALLING, 400 -> 129. If that is real, the
justification above does not hold and no single-invocation speedup in this repo
is a verdict.

This repeats ONE fixed configuration in a fresh subprocess N times and reports
the distribution, with load and CPU frequency recorded per repeat so the
obvious confounders can be checked rather than assumed.

Writes `runs/denominator_stability.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.bench_symmetric import N_DISCARD, solver_warm, _stats  # noqa: E402


def cpu_mhz():
    """Mean current core frequency, from /proc/cpuinfo. None if unavailable."""
    try:
        v = [float(l.split(":")[1]) for l in
             Path("/proc/cpuinfo").read_text().splitlines()
             if l.startswith("cpu MHz")]
        return float(np.mean(v)) if v else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--wafers", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/denominator_stability.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="denominator_stability")

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]

    reps = []
    for i in range(a.repeats):
        load = os.getloadavg()[0]
        mhz0 = cpu_mhz()
        t0 = time.perf_counter()
        sw = solver_warm(a.wafers, n_steps, a.dt, grid_delta, a.seed,
                         n_discard=N_DISCARD)
        st = _stats(sw["wall"], sw["cpu"])
        reps.append({
            "i": i, "median_cpu_s": st["median_cpu"],
            "median_wall_s": st["median_wall"],
            "cpu_over_wall": st["cpu_over_wall_median"],
            "single_threaded": st["single_threaded_verified"],
            "loadavg": load, "cpu_mhz_before": mhz0,
            "cpu_mhz_after": cpu_mhz(), "elapsed_s": time.perf_counter() - t0,
            "per_wafer_cpu_s": st["cpu_s_per_wafer"],
        })
        print(f"  rep {i:2d}  median {st['median_cpu']*1e3:8.2f} ms  "
              f"load {load:6.1f}  MHz {mhz0 or float('nan'):7.1f}", flush=True)

    med = np.array([r["median_cpu_s"] for r in reps])
    load = np.array([r["loadavg"] for r in reps])
    mhz = np.array([r["cpu_mhz_before"] or np.nan for r in reps])
    spread = float(med.max() / med.min())

    def corr(x, y):
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 3 or np.std(x[m]) == 0:
            return None
        return float(np.corrcoef(x[m], y[m])[0, 1])

    res = {
        "question": "does the solver denominator reproduce across invocations, "
                    "given that CPU-seconds was adopted precisely because it "
                    "was believed load-invariant",
        "protocol": {
            "identical_every_repeat": f"{a.wafers} wafers, seed {a.seed}, "
                                      f"dt {a.dt}, {N_DISCARD} discarded, "
                                      "fresh subprocess, OMP_NUM_THREADS=1",
            "n_steps_per_wafer": n_steps, "grid_delta": grid_delta,
            "repeats": a.repeats,
            "prior_claim": "runs/solver_drift.json: 1.15x across a 3.6x load "
                           "range, which is what justified CPU-seconds here",
        },
        "repeats": reps,
        "median_cpu_s": {"min": float(med.min()), "median": float(np.median(med)),
                         "max": float(med.max()), "spread_factor": spread},
        "loadavg": {"min": float(load.min()), "max": float(load.max())},
        "cpu_mhz": {"min": float(np.nanmin(mhz)), "max": float(np.nanmax(mhz))},
        "r_cost_vs_load": corr(load, med),
        "r_cost_vs_mhz": corr(mhz, med),
        "all_single_threaded": bool(all(r["single_threaded"] for r in reps)),
        "verdict": (f"the denominator moves {spread:.2f}x across identical "
                    "invocations; a single-invocation speedup is not a verdict"
                    if spread > 1.3 else
                    f"reproducible to {spread:.2f}x across identical invocations"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["median_cpu_s", "r_cost_vs_load", "r_cost_vs_mhz",
                       "verdict"]}, indent=2))


if __name__ == "__main__":
    main()
