"""The speedup clause, measured so box load cannot set the answer.

    python scripts/bench_paired.py --runs runs/seed1 runs/kcurve/K10_nv_s1

**Why this script exists.** `runs/speed.json`, `runs/speed_seed1_cpu.json` and
`runs/speed_K10_cpu.json` report the same like-for-like quantity -- ViennaPS at
one OpenMP thread against the operator on CPU at one torch thread, same grid
(0.2 um, 128) and the same 10 timesteps -- as **1.47x, 6.57x and 119.12x**. The
operator changed between them (modes 16 -> 20, and 10 applications -> 1), but
the *solver* is identical code on identical input in all three, and it reported

    1.538 s/wafer  at load average 11.56   (runs/speed.json)
    8.457 s/wafer  at load average 40.65   (runs/speed_seed1_cpu.json)
    8.390 s/wafer  at load average 22.32   (runs/speed_K10_cpu.json)

A 5.5x spread on a fixed computation. So the published ratios are substantially
a measurement of what else this box was doing, and the largest of them, 119x,
is the one taken while the repo's own training queue occupied the machine.

`bench_speed.py` carries a `load_note` claiming the ratio is defensible because
"both sides are timed in this same invocation and under the same load". **That
defence is wrong and this script is the correction.** The two sides are timed in
separate *blocks* minutes apart, and the load on this box moves on the timescale
of a training arm starting or finishing -- so "same invocation" does not imply
"same load", and a ratio of two block medians has the drift between the blocks
baked into it.

Three structural changes, none of them a loosening:

1. **Interleaved and paired.** One round times one solver wafer and one
   operator wafer *per arm*, back to back. Drift between rounds moves both sides
   of a round together, so the per-round ratio is the estimator and its spread
   across rounds is the honest error bar. The block-median ratio is also
   reported, so the two can be compared.
2. **Every arm in one invocation.** Comparing 6.57x from one invocation with
   119.12x from another is not a comparison, because the shared denominator was
   re-measured under different load. K=1 and K=10 are timed against the *same*
   solver rounds here, which is what makes the across-K statement valid.
3. **The thread count is verified, not configured.** Each side reports
   CPU-seconds / wall-seconds for its own timed region. A row labelled "1
   thread" whose ratio is 3.4 was never a 1-thread row, and a like-for-like
   claim resting on it is void. `OMP_NUM_THREADS=1` and
   `torch.set_num_threads(1)` are settings; this is the measurement.

**And the primary estimator is CPU-seconds, not wall-seconds.** Pairing cancels
drift that moves both sides *together*, but load does not move them together:
between the 11.56 and 40.65 rows above the solver slowed 5.5x while the operator
slowed only 1.23x. A wall-clock ratio between two processes that respond to
contention differently is therefore not a property of the two implementations at
all, and no amount of pairing repairs it. CPU-seconds consumed by a *verified*
single-threaded process measures work performed rather than time spent waiting
for a core, so it is very nearly load-invariant -- which is exactly the property
a like-for-like claim needs. `speedup_cpu_time` is the clause reading;
`speedup_paired_median` (wall) is reported beside it as the reading a user
experiences, and the two disagreeing is informative rather than a defect.

This box cannot be made quiet: it runs two other projects' jobs that this track
must not touch, and the lowest load average ever recorded here is 11.56. So
`quiet_box` is expected to be false and is recorded for the reader rather than
used as a gate -- gating on an unachievable threshold would only mean never
reporting a number. The load average is sampled before and after every round so
the CPU-time reading's load-invariance can be checked against the spread rather
than assumed.

Writes `runs/speed_paired.json`.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eot import runlock  # noqa: E402


def solver_round(n_steps: int, dt: float, grid_delta: float, seed: int, threads: int = 1):
    """One wafer, one process of the full duration, in a fresh subprocess.

    A subprocess because OpenMP reads OMP_NUM_THREADS once at first use: setting
    it after ViennaPS is imported does nothing, and the '1 thread' row would
    quietly be a 192-thread row. The child reports its own CPU time so that
    claim is checked rather than trusted.
    """
    code = f"""
import os
os.environ["OMP_NUM_THREADS"] = "{threads}"
import time, json, numpy as np, sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from eot import solver as S
rng = np.random.default_rng({seed})
rec = S.sample_recipe(rng)
dom = S.build_domain(rec, {grid_delta})
w0 = time.perf_counter(); c0 = time.process_time()
S.make_process(rec, dom, {n_steps} * {dt}).apply()
wall = time.perf_counter() - w0; cpu = time.process_time() - c0
print(json.dumps({{"wall_s": wall, "cpu_s": cpu}}))
"""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def warm_operator(model, n_apply: int, H: int = 128, n: int = 3):
    """Pay lazy initialisation outside the timed region.

    Without this, a K=10 arm (one application per wafer) charges the whole of
    MKL's FFT plan creation to its single application while a K=1 arm (ten
    applications) amortises it over ten -- which is how a smoke run of this
    script produced 0.0773 s per application at K=10 against 0.1238 s at K=1
    for *bit-identical* architectures (width 64, modes 20, layers 4, 26,248,025
    parameters both). One application of one architecture costs what it costs;
    a per-application cost that depends on K is a defect in the harness, and
    this was it.
    """
    import torch

    phi = torch.randn(1, 1, H, H)
    cond = torch.randn(1, 7)
    with torch.no_grad():
        for _ in range(n):
            model.rollout(phi, cond, n_apply)


def operator_round(model, n_apply: int, H: int = 128):
    """One wafer on CPU at one torch thread, with its own CPU/wall accounting."""
    import torch

    phi = torch.randn(1, 1, H, H)
    cond = torch.randn(1, 7)
    with torch.no_grad():
        w0 = time.perf_counter()
        c0 = time.process_time()
        model.rollout(phi, cond, n_apply)
        wall = time.perf_counter() - w0
        cpu = time.process_time() - c0
    return {"wall_s": wall, "cpu_s": cpu}


def load_now():
    return os.getloadavg()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs to time; each arm's stride sets its applications per wafer")
    ap.add_argument("--out", default="runs/speed_paired.json")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="default: data/gen_report.json's step count, as bench_speed.py uses")
    ap.add_argument("--dt", type=float, default=0.2,
                    help="representative per-step duration, matching bench_speed.py")
    ap.add_argument("--grid-delta", type=float, default=None)
    ap.add_argument("--max-load", type=float, default=4.0,
                    help="1-minute load average above which the box is not quiet")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    import torch

    from eot.operator import EtchOperator

    runlock.acquire(a.out, what="bench_paired")
    torch.set_num_threads(1)

    # Take the geometry and the step count from the same place bench_speed.py
    # does, so the two scripts are timing the same wafer and their numbers can
    # be compared directly.
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps = a.n_steps if a.n_steps is not None else gen["steps"]
    grid_delta = a.grid_delta if a.grid_delta is not None else gen["grid_delta"]
    dt = a.dt
    norm = json.loads((Path(a.data) / "norm.json").read_text())

    arms = {}
    for r in a.runs:
        rd = Path(r)
        args = json.loads((rd / "args.json").read_text())
        stride = int(args.get("stride", 1) or 1)
        if n_steps % stride:
            raise SystemExit(f"{r}: stride {stride} does not divide n_steps {n_steps}")
        model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=args["width"],
                             modes=args["modes"], n_layers=args["layers"])
        model.load_state_dict(torch.load(rd / "best.pt", map_location="cpu"))
        model.eval()
        arms[r] = {
            "run": r, "stride": stride,
            "applications_per_wafer": n_steps // stride,
            "arch": {k: args.get(k) for k in ("width", "modes", "layers")},
            "params": sum(p.numel() for p in model.parameters()),
            "model": model, "wall": [], "cpu": [],
        }

    # Warm every arm before the first timed round, not inside it.
    for arm in arms.values():
        warm_operator(arm["model"], arm["applications_per_wafer"])

    solver_wall, solver_cpu, loads = [], [], []
    for i in range(a.rounds):
        loads.append(load_now())
        s = solver_round(n_steps, dt, grid_delta, seed=7000 + i)
        solver_wall.append(s["wall_s"])
        solver_cpu.append(s["cpu_s"])
        for arm in arms.values():
            o = operator_round(arm["model"], arm["applications_per_wafer"])
            arm["wall"].append(o["wall_s"])
            arm["cpu"].append(o["cpu_s"])
        loads.append(load_now())

    res = {
        "protocol": {
            "estimator": "median over rounds of (solver_wall / operator_wall) "
                         "within the same round",
            "why": "a ratio of two block medians taken minutes apart carries the "
                   "load drift between the blocks; see this file's module docstring "
                   "and the 1.538/8.457/8.390 s solver spread it was written for",
            "rounds": a.rounds, "n_steps_per_wafer": n_steps, "dt": dt,
            "grid_delta": grid_delta, "grid_n": gen.get("grid_n", 128),
            "solver": "ViennaPS, OMP_NUM_THREADS=1, one process of the full duration",
            "operator": "torch CPU, torch.set_num_threads(1), batch 1",
            "excluded_from_both": "rasterisation",
            "operator_warmup": "3 untimed rollouts per arm before the first "
                               "round, so lazy FFT-plan creation is not charged "
                               "to whichever arm has the fewest applications",
            "solver_warmup": "none, deliberately: the timed region is one "
                             "process of the full duration, which is what an "
                             "engineer actually queues, and its setup cost is "
                             "part of that",
        },
        "box": {
            "cpu": subprocess.run(["bash", "-lc", "lscpu | grep 'Model name'"],
                                  capture_output=True, text=True).stdout.strip(),
            "n_cpu": os.cpu_count(),
            "load_1min_per_sample": loads,
            "load_min": min(loads), "load_max": max(loads),
            "max_load_allowed": a.max_load,
            "quiet_box": bool(max(loads) <= a.max_load),
            "quiet_note": "false means something else was running and every "
                          "absolute second here is inflated; the paired ratio is "
                          "still the best available estimator but is not clean",
        },
        "solver_1_thread_single": {
            "wall_s_per_wafer": solver_wall,
            "median": float(np.median(solver_wall)),
            "cpu_over_wall": [c / w for c, w in zip(solver_cpu, solver_wall)],
            "cpu_over_wall_median": float(np.median(
                [c / w for c, w in zip(solver_cpu, solver_wall)])),
            "single_threaded_verified": bool(np.median(
                [c / w for c, w in zip(solver_cpu, solver_wall)]) < 1.15),
        },
        "arms": {},
    }

    for name, arm in arms.items():
        cw = [c / w for c, w in zip(arm["cpu"], arm["wall"])]
        paired = [sw / ow for sw, ow in zip(solver_wall, arm["wall"])]
        paired_cpu = [sc / oc for sc, oc in zip(solver_cpu, arm["cpu"]) if oc > 0]
        res["arms"][name] = {
            "run": arm["run"], "stride": arm["stride"],
            "applications_per_wafer": arm["applications_per_wafer"],
            "arch": arm["arch"], "params": arm["params"],
            "wall_s_per_wafer": arm["wall"],
            "median_s_per_wafer": float(np.median(arm["wall"])),
            "median_s_per_application": float(np.median(arm["wall"])) / arm["applications_per_wafer"],
            "cpu_over_wall": cw,
            "cpu_over_wall_median": float(np.median(cw)),
            "single_threaded_verified": bool(np.median(cw) < 1.15),
            "cpu_s_per_wafer": arm["cpu"],
            "median_cpu_s_per_wafer": float(np.median(arm["cpu"])),
            "speedup_cpu_time_median": float(np.median(paired_cpu)),
            "speedup_cpu_time_per_round": paired_cpu,
            "speedup_cpu_time_min": float(np.min(paired_cpu)),
            "speedup_cpu_time_max": float(np.max(paired_cpu)),
            "speedup_cpu_time_note": "PRIMARY. CPU-seconds of work, both sides "
                                     "verified single-threaded, so contention on "
                                     "this shared box moves it far less than it "
                                     "moves the wall-clock ratio.",
            "speedup_paired_median": float(np.median(paired)),
            "speedup_paired_per_round": paired,
            "speedup_paired_min": float(np.min(paired)),
            "speedup_paired_max": float(np.max(paired)),
            "speedup_block_median_ratio": float(np.median(solver_wall)) / float(np.median(arm["wall"])),
            "block_vs_paired_note": "these two differ by exactly the within-"
                                    "invocation drift the block estimator ignores",
        }

    best = max(res["arms"].values(), key=lambda v: v["speedup_cpu_time_median"])
    threads_ok = (res["solver_1_thread_single"]["single_threaded_verified"]
                  and best["single_threaded_verified"])
    res["kpi_reading"] = {
        "clause": "simulation speedup >= 1000x",
        "target": 1000.0,
        "value": best["speedup_cpu_time_median"],
        "value_basis": "CPU-seconds, the load-invariant reading",
        "value_wall_clock": best["speedup_paired_median"],
        "value_wall_clock_range": [best["speedup_paired_min"], best["speedup_paired_max"]],
        "arm": best["run"],
        "basis": "like-for-like: operator CPU 1 thread batch 1 vs ViennaPS CPU 1 "
                 "thread single process, interleaved per round in one invocation; "
                 "ratio of CPU-seconds",
        "met": bool(best["speedup_cpu_time_median"] >= 1000.0),
        "shortfall_factor": 1000.0 / best["speedup_cpu_time_median"],
        "single_thread_checks_passed": threads_ok,
        "usable_as_headline": bool(threads_ok),
        "usable_note": "gated on the cpu/wall single-thread check for BOTH sides, "
                       "not on box quiet: a like-for-like claim is void if either "
                       "side was secretly multi-threaded, whereas load is handled "
                       "by using CPU-seconds. quiet_box is reported for context.",
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("box", "kpi_reading")}, indent=2))


if __name__ == "__main__":
    main()
