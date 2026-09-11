"""Clause 2 priced on the wafer population clause 1 is measured on.

    python scripts/clause2_on_test_split.py --run runs/specprop/m4_ma64_s7

`runs/speedup_vs_dt.json` showed the speedup is a function of `dt`: the solver
pays for the simulated duration a wafer represents and a K-step operator applied
once does not, so the ratio crosses 1000x at dt ~= 0.257 and reads 738x at 0.2
and 4189x at 0.8. That makes a single speedup number meaningless unless the
wafer population is stated.

There is one population that is not a convention: **the test split**. Clause 1's
accuracy is measured on it, so pricing clause 2 anywhere else compares a cost on
one set of wafers against an accuracy on another. This script prices every test
wafer at its OWN recipe and its OWN dt, and reports the distribution rather than
a point -- this repo's standing rule is that a clause verdict needs the whole
distribution on one side of the threshold.

Both sides are timed through the same helpers `bench_symmetric.py` uses. The
solver is warm (long-lived process, first wafer discarded) and the operator is
warm, which is the `marginal_warm` reading; the cold reading is not attempted
here because the point is the per-wafer spread, not the one-time cost.

Writes `runs/clause2_test_split.json`.
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
from scripts.bench_symmetric import _child, _stats, operator_run  # noqa: E402
from scripts.bench_workload import _rec_literal  # noqa: E402


def solver_per_wafer(rows, dts, n_steps, grid_delta, n_discard=1):
    """Time each wafer at its own recipe and its own dt, in one warm process."""
    recs = "\n".join(f"JOBS.append(({_rec_literal(r)}, {float(d)!r}))"
                     for r, d in zip(rows, dts))
    body = f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu = [], []
for rec, dt in JOBS:
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {n_steps} * dt).apply()
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:]}}))
'''
    return _child(body, timeout=7200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/specprop/m4_ma64_s7")
    ap.add_argument("--n-wafers", type=int, default=60,
                    help="test wafers priced, taken in file order from index 0")
    ap.add_argument("--op-rounds", type=int, default=32)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/clause2_test_split.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="clause2_test_split")

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]
    norm_p = Path(a.data) / "norm.json"
    args = json.loads((Path(a.run) / "args.json").read_text())
    stride = args.get("eval_stride") or args.get("stride") or 1
    n_apply = n_steps // stride

    d = np.load(Path(a.data) / "test.npz", allow_pickle=True)
    rows, dts = d["recipe"][:a.n_wafers], d["dt"][:a.n_wafers]

    load0 = os.getloadavg()[0]
    sv = solver_per_wafer(rows, dts, n_steps, grid_delta, n_discard=1)
    sol = _stats(sv["wall"], sv["cpu"])
    dts_priced = [float(x) for x in dts[1:]]  # the discarded wafer is not priced

    # The operator is warmed and timed many times: it does the same work for
    # every wafer, so its cost is a single distribution, not a per-wafer one.
    # A cell that is not verified single-threaded is void by this repo's rule.
    op = operator_run(a.run, norm_p, n_apply, n_rep=a.op_rounds, n_warm=3)
    ops = _stats(op["wall"], op["cpu"])
    if not ops["single_threaded_verified"]:
        raise SystemExit("operator cell is not single-threaded "
                         f"(cpu/wall {ops['cpu_over_wall_median']:.3f}); void")

    op_med = ops["median_cpu"]
    per = [{"dt": dt, "solver_cpu_s": s, "speedup_cpu": s / op_med}
           for dt, s in zip(dts_priced, sol["cpu_s_per_wafer"])]
    sp = np.array([p["speedup_cpu"] for p in per])
    n_meet = int((sp >= 1000).sum())

    res = {
        "question": "clause 2 priced on the wafer population clause 1 is "
                    "measured on: each test wafer at its own recipe and its own "
                    "dt, rather than at a fixed dt convention",
        "protocol": {
            "estimator": "median CPU-seconds, solver / operator, both 1 thread",
            "solver_reading": "marginal_warm, one long-lived process, wafer 0 "
                              "discarded as warm-up",
            "operator_reading": f"warm, 3 untimed rollouts, {a.op_rounds} timed",
            "operator_is_dt_independent": "a K-step operator applied "
                                          f"{n_apply}x per wafer does the same "
                                          "work whatever duration the wafer is",
            "n_wafers_priced": len(per), "n_steps_per_wafer": n_steps,
            "grid_delta": grid_delta, "run": a.run,
            "solver_single_threaded": sol["single_threaded_verified"],
            "operator_single_threaded": ops["single_threaded_verified"],
            "loadavg_start": load0, "loadavg_end": os.getloadavg()[0],
        },
        "operator": {"median_cpu_s": op_med, "n": ops["n"],
                     "cpu_over_wall_median": ops["cpu_over_wall_median"],
                     "spread": float(max(ops["cpu_s_per_wafer"])
                                     / min(ops["cpu_s_per_wafer"]))},
        "solver": {"median_cpu_s": sol["median_cpu"],
                   "cpu_over_wall_median": sol["cpu_over_wall_median"]},
        "dt_priced": {"min": float(np.min(dts_priced)),
                      "median": float(np.median(dts_priced)),
                      "mean": float(np.mean(dts_priced)),
                      "max": float(np.max(dts_priced))},
        "speedup": {
            "min": float(sp.min()), "p10": float(np.percentile(sp, 10)),
            "median": float(np.median(sp)), "p90": float(np.percentile(sp, 90)),
            "max": float(sp.max()), "mean": float(sp.mean()),
        },
        "n_wafers_meeting_1000x": n_meet,
        "frac_wafers_meeting_1000x": n_meet / len(per),
        "per_wafer": per,
        "verdict_rule": "this repo requires the whole distribution on one side "
                        "of the threshold for a clause verdict, not a point",
        "verdict": ("MET: every priced wafer clears 1000x" if sp.min() >= 1000
                    else "NOT MET: no wafer clears 1000x" if sp.max() < 1000
                    else f"STRADDLES: {n_meet}/{len(per)} wafers clear 1000x"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["speedup", "n_wafers_meeting_1000x",
                       "frac_wafers_meeting_1000x", "verdict"]}, indent=2))


if __name__ == "__main__":
    main()
