"""The denominator, matched to the output the operator actually produces.

    python scripts/bench_workload.py --rounds 8

**Why this file exists.** `runs/speed_symmetric.json` times the solver as ONE
`make_process(rec, dom, 10 * 0.2).apply()` -- a single continuous 2.0-minute
etch, at a dt fixed by the benchmark -- and every speedup this repo has
published divides by it. Two mismatches against the workload the operator is
scored on, one found by an adversarial reviewer (`codex`, rung 4) and one found
while checking it:

1. **The duration is wrong for this dataset.** The benchmark etches for a fixed
   10 x 0.2 = 2.0 minutes. Trajectories carry a per-recipe dt (median 0.275,
   quartiles 0.177-0.483, range 0.060-1.000 on the train split), so the median
   trajectory etches for 2.75 minutes and the upper quartile for 4.8. Solver cost
   grows with etch time, so a fixed 2.0 understates the median wafer.

2. **The number of applies is wrong for a K=1 operator, and right for a K=10
   one.** `eot.solver.simulate` accumulates `Trajectory.seconds` over **ten**
   separate `apply()` calls, one per emitted frame -- and that recorded figure,
   stored in the dataset as `solver_s`, has a median of **17.2 s per
   trajectory**, sixty-two times the benchmark's 0.277 CPU-s. The gap is not an
   error in either number; they price different outputs. A stride-1 operator
   emits all ten intermediate states, so its denominator is ten applies. A
   stride-10 operator emits only the terminal state, so its denominator is one
   apply of the full duration, and nobody needs the intermediate frames to get
   there.

So there is no single honest denominator: there is one per output, and this
script measures them side by side on the same wafers in the same invocation.
**Reporting both, labelled, is rung 1 of the ladder. Silently adopting the
larger one would be the protocol-loosening the rules forbid**, which is why the
existing fixed-duration reading is measured here too, as a third column, rather
than deleted.

Every row prices the SAME wafers -- the recipes and dt values recorded in the
dataset split, replayed in order -- so a difference between columns is never a
difference in the geometry sampled. Solver cost varies ~4x across recipes
(`runs/solver_drift.json`), which is larger than some of the gaps here.

Writes `runs/bench_workload.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.bench_symmetric import _child  # noqa: E402

RECIPE_KEYS = ["ion_flux", "etchant_flux", "oxygen_flux", "ion_energy",
               "trench_width", "mask_height"]


def _rec_literal(row):
    kw = ", ".join(f"{k}={float(v)!r}" for k, v in zip(RECIPE_KEYS, row))
    return f"S.Recipe({kw})"


def solver_terminal(rows, dts, n_steps, grid_delta, n_discard=1):
    """One apply of the FULL duration per wafer: the terminal state only.

    The matched denominator for a stride-10 operator, which emits exactly this
    and nothing else. `build_domain` is outside the timed region, matching
    bench_symmetric, and is timed separately by `solver_domain_build`.
    """
    recs = "\n".join(
        f"JOBS.append(({_rec_literal(r)}, {float(d)!r}))" for r, d in zip(rows, dts))
    body = f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu, dur = [], [], []
for rec, dt in JOBS:
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {n_steps} * dt).apply()
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
    dur.append({n_steps} * dt)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:],
                  "duration": dur[{n_discard}:]}}))
'''
    return _child(body)


def solver_allframes(rows, dts, n_steps, grid_delta, n, n_discard=1):
    """Ten applies of dt each, plus the rasterisation to a 128x128 field.

    The matched denominator for a stride-1 operator, which emits ten states as
    fields. This is what `eot.solver.simulate` does and what the dataset's
    `solver_s` recorded, except that `simulate` explicitly excludes
    rasterisation from its timing while the operator's output IS the field --
    so rasterisation is timed here and reported both in and out.
    """
    recs = "\n".join(
        f"JOBS.append(({_rec_literal(r)}, {float(d)!r}))" for r, d in zip(rows, dts))
    body = f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu, cpu_raster = [], [], []
for rec, dt in JOBS:
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    raster_c = 0.0
    for _ in range({n_steps}):
        S.make_process(rec, dom, dt).apply()
        r0 = time.process_time()
        S.rasterise(S.surface_polyline(dom), {n})
        raster_c += time.process_time() - r0
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
    cpu_raster.append(raster_c)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:],
                  "cpu_raster": cpu_raster[{n_discard}:]}}))
'''
    return _child(body)


def solver_fixed_duration(rows, total, grid_delta, n_discard=1):
    """The existing benchmark's reading: one apply at a duration the benchmark
    picks rather than the dataset's, on these same wafers. Kept so the
    correction is visible as a difference rather than asserted."""
    recs = "\n".join(f"JOBS.append({_rec_literal(r)})" for r in rows)
    body = f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu = [], []
for rec in JOBS:
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {total!r}).apply()
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:]}}))
'''
    return _child(body)


def solver_domain_build(rows, grid_delta, n_discard=1):
    """What `build_domain` costs, since every reading above excludes it and the
    operator does not pay it -- the operator is handed the initial field. If it
    is material it belongs in the denominator and its absence is a bias against
    the operator."""
    recs = "\n".join(f"JOBS.append({_rec_literal(r)})" for r in rows)
    body = f'''
from eot import solver as S
JOBS = []
{recs}
wall, cpu = [], []
for rec in JOBS:
    w0 = time.perf_counter(); c0 = time.process_time()
    S.build_domain(rec, {grid_delta})
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:]}}))
'''
    return _child(body)


def _stats(cpu, wall=None):
    out = {"median_cpu_s": float(np.median(cpu)), "min_cpu_s": float(np.min(cpu)),
           "max_cpu_s": float(np.max(cpu)),
           "spread_factor": float(np.max(cpu) / max(np.min(cpu), 1e-12)),
           "n": len(cpu)}
    if wall:
        cw = [c / w for c, w in zip(cpu, wall) if w > 0]
        out["cpu_over_wall_median"] = float(np.median(cw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--fixed-total", type=float, default=2.0,
                    help="the duration speed_symmetric.json used: 10 * 0.2")
    ap.add_argument("--out", default="runs/bench_workload.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="bench_workload")
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta, n = gen["steps"], gen["grid_delta"], gen["grid_n"]

    d = np.load(Path(a.data) / f"{a.split}.npz")
    n_take = a.rounds + 1                       # one discarded as warm-up
    rows, dts = d["recipe"][:n_take], d["dt"][:n_take]
    recorded = d["solver_s"][1:n_take]          # the same wafers, minus warm-up

    term = solver_terminal(rows, dts, n_steps, grid_delta)
    allf = solver_allframes(rows, dts, n_steps, grid_delta, n)
    fixed = solver_fixed_duration(rows, a.fixed_total, grid_delta)
    dom = solver_domain_build(rows, grid_delta)

    allf_no_raster = [c - r for c, r in zip(allf["cpu"], allf["cpu_raster"])]

    res = {
        "question": "what does the simulator cost to produce exactly what the "
                    "operator produces, on the wafers the operator is scored on",
        "protocol": {
            "split": a.split, "wafers_priced": a.rounds,
            "warmup_wafers_discarded": 1,
            "same_wafers": "every column replays the recipes and dt values "
                           "stored in the split, in order, so a difference "
                           "between columns is not a difference in geometry",
            "estimator": "CPU-seconds, one verified thread, one long-lived "
                         "process per column",
            "n_steps": n_steps, "grid_delta": grid_delta, "grid_n": n,
            "dt_source": "the dataset's own per-recipe dt",
            "build_domain": "outside the timed region in every column except "
                            "solver_domain_build, matching bench_symmetric",
        },
        "dt_of_priced_wafers": {
            "values": [float(x) for x in dts[1:]],
            "median": float(np.median(dts[1:])),
            "total_etch_time_median": float(np.median(dts[1:]) * n_steps),
        },
        "columns": {
            "terminal_one_apply": {
                **_stats(term["cpu"], term["wall"]),
                "matches_operator_at_stride": n_steps,
                "output": "terminal surface only, 1 apply of 10*dt",
                "note": "the matched denominator for a stride-10 operator, "
                        "which emits the terminal state in one application",
            },
            "all_frames_ten_applies": {
                **_stats(allf["cpu"], allf["wall"]),
                "matches_operator_at_stride": 1,
                "output": "10 states, rasterised to 128x128 fields",
                "cpu_raster_median_s": float(np.median(allf["cpu_raster"])),
                "median_cpu_s_excluding_raster": float(np.median(allf_no_raster)),
                "note": "the matched denominator for a stride-1 operator. "
                        "eot.solver.simulate excludes rasterisation from its "
                        "timing; the operator's output IS the field, so the "
                        "figure including it is the like-for-like one and both "
                        "are given",
            },
            "fixed_duration_one_apply": {
                **_stats(fixed["cpu"], fixed["wall"]),
                "output": f"terminal surface only, 1 apply of {a.fixed_total}",
                "note": "what speed_symmetric.json measures, on these wafers. "
                        "Its duration is the benchmark's choice, not the "
                        "dataset's, and this column exists so the correction "
                        "is a measured difference rather than an assertion",
            },
            "domain_build": {
                **_stats(dom["cpu"], dom["wall"]),
                "output": "the initial level set",
                "note": "excluded from every reading above and from "
                        "bench_symmetric. The operator is handed the initial "
                        "field, so if this is material its exclusion is a bias "
                        "against the operator",
            },
        },
        "dataset_recorded_solver_s": {
            "median": float(np.median(recorded)),
            "values": [float(x) for x in recorded],
            "what_it_is": "Trajectory.seconds as stored at generation: the sum "
                          "of ten apply() calls, wall-clock, measured while 8 "
                          "generation workers shared this box. Not comparable "
                          "to the CPU-second columns above and not used as a "
                          "denominator anywhere; recorded because it is the "
                          "number that exposed the mismatch.",
        },
    }
    r = res["columns"]
    res["ratios"] = {
        "terminal_vs_fixed_duration":
            r["terminal_one_apply"]["median_cpu_s"]
            / r["fixed_duration_one_apply"]["median_cpu_s"],
        "all_frames_vs_terminal":
            r["all_frames_ten_applies"]["median_cpu_s"]
            / r["terminal_one_apply"]["median_cpu_s"],
        "budget_per_wafer_at_1000x": {
            k: r[k]["median_cpu_s"] / 1000.0
            for k in ("terminal_one_apply", "all_frames_ten_applies",
                      "fixed_duration_one_apply")
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    for k, v in r.items():
        print(f"{k:26s} median {v['median_cpu_s']*1e3:9.2f} ms  "
              f"spread {v['spread_factor']:.2f}x  n={v['n']}")
    print("\nbudget per wafer at 1000x (us):")
    for k, v in res["ratios"]["budget_per_wafer_at_1000x"].items():
        print(f"  {k:26s} {v*1e6:9.1f} us")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
