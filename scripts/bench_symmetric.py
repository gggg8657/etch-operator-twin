"""The speedup clause under a protocol that is symmetric between the two sides.

    python scripts/bench_symmetric.py --runs runs/seed1 runs/kcurve/K10_nv_s1

**The defect this script exists to fix is mine, and it flattered the clause.**

`runs/solver_drift.json` measured ViennaPS in a long-lived process, recording
each wafer separately instead of a median: the first wafer costs 1.257 s and
every later one 0.10-0.40 s. So the solver charges a large one-time
initialisation *inside* the timed `.apply()` region. `scripts/bench_paired.py`
then timed the solver in a **fresh process per wafer** -- paying that init on
every wafer -- while giving the operator **3 untimed warm-up rollouts** so it
paid its own one-time cost (MKL FFT plan creation) on none of them. Solver cold,
operator warm, ratio quoted as "like-for-like". That is not a like-for-like
comparison, and its 1.53x / 13.84x are withdrawn.

There is no single correct choice between cold and warm, so this script reports
**both, symmetrically**, and neither is allowed to borrow the other's
convenience:

* `marginal_warm` -- the cost of one *additional* wafer once the tool is up.
  Solver: a long-lived process, first wafer discarded. Operator: warmed, first
  rollouts discarded. This is the honest reading for a sweep, an inverse-design
  loop, or dataset generation, which is what this repo actually does with both.
  Both readings time the **same eight wafers**: the warm reading discards recipe
  0 as its warm-up, so the cold reading skips recipe 0 too. It did not, in the
  first version of this script, and since solver cost varies ~4x across recipes
  (`runs/solver_drift.json`: 0.10-0.40 s warm) the cold/warm factor it reported
  was partly a difference in geometry rather than in initialisation. Found by
  `codex` when asked how the clause could be made to pass.
* `cold_single_wafer` -- the latency of one wafer from nothing. Solver: fresh
  process, init inside the timed region. Operator: fresh process, no warm-up,
  plan creation inside the timed region. This is the honest reading for a
  one-shot query.

Both sides of both readings are timed in the same invocation, in **sequential
blocks rather than interleaved** -- and that is only acceptable because the
estimator is **CPU-seconds** rather than wall-seconds. Interleaving exists to
cancel load drift between blocks; CPU-seconds of a *verified* single-threaded
process is already close to load-invariant, so it removes the need rather than
satisfying it. (An earlier draft of this docstring claimed interleaving that the
code does not do; `codex` caught the contradiction against the block structure
in `main`.) A wall-clock ratio between processes with different threading
behaviour would not be a property of the implementations at any interleaving. `runs/solver_drift.json` measured the
solver at 1.11 CPU-s under load 11.3 and 1.26 CPU-s under load 41.0 -- 1.15x
across a 3.6x load range -- which is what makes CPU-seconds usable on a box this
track cannot quiet (it carries two other projects' jobs; the lowest load average
ever recorded here is 11.56).

**One number in this repo's history is still unreproduced and every speedup
resting on it is suspect.** `runs/speed_seed1_cpu.json` and
`runs/speed_K10_cpu.json` report the solver denominator at 8.457 and 8.390
s/wafer. Under three conditions at comparable load, `solver_drift.json` gets
1.00 s/wafer cold and 0.296 s/wafer warm -- 8.4x and 28x away. Neither box load
nor process reuse accounts for it, and no explanation is asserted here. Until it
is reproduced or explained, 1.47x, 6.57x and 119.12x are all withdrawn rather
than reinterpreted.

Writes `runs/speed_symmetric.json`.
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

# Wafers the warm reading spends on warm-up. The cold reading skips the same
# ones, so both readings price the identical eight wafers.
N_DISCARD = 1

_PRE = f'''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, numpy as np, sys
sys.path.insert(0, {str(ROOT)!r})
'''


def _child(body: str, timeout=3600):
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", _PRE + body], capture_output=True,
                         text=True, env=env, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def _recipe_stream(seed, i):
    """Draw the i-th recipe of the stream seeded at `seed`.

    Every condition and every arm times the same wafers, so a difference
    between them is never a difference in the geometry sampled -- solver cost
    varies ~4x across recipes (solver_drift.json), which is larger than several
    of the effects being compared.
    """
    return f'''
rng = np.random.default_rng({seed})
for _ in range({i} + 1):
    rec = S.sample_recipe(rng)
'''


def solver_warm(n_traj, n_steps, dt, grid_delta, seed, n_discard=1):
    """Marginal cost of one more wafer: one process, first `n_discard` dropped."""
    body = f'''
from eot import solver as S
rng = np.random.default_rng({seed})
wall, cpu = [], []
for i in range({n_traj} + {n_discard}):
    rec = S.sample_recipe(rng)
    dom = S.build_domain(rec, {grid_delta})
    w0 = time.perf_counter(); c0 = time.process_time()
    S.make_process(rec, dom, {n_steps} * {dt}).apply()
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall[{n_discard}:], "cpu": cpu[{n_discard}:],
                  "discarded_wall": wall[:{n_discard}]}}))
'''
    return _child(body)


def solver_cold(n_steps, dt, grid_delta, seed, i):
    """One wafer from a fresh process, initialisation inside the timed region.

    `i` indexes the same seeded recipe stream `solver_warm` walks, so callers
    must offset it past whatever that discards -- otherwise the two readings
    time different wafers and their ratio mixes initialisation with geometry.
    """
    body = f'''
from eot import solver as S
{_recipe_stream(seed, i)}
dom = S.build_domain(rec, {grid_delta})
w0 = time.perf_counter(); c0 = time.process_time()
S.make_process(rec, dom, {n_steps} * {dt}).apply()
print(json.dumps({{"wall": time.perf_counter() - w0, "cpu": time.process_time() - c0}}))
'''
    return _child(body)


_OP_BODY = '''
import torch
torch.set_num_threads(1)
from eot.operator import EtchOperator
norm = json.load(open({norm!r}))
args = json.load(open({args!r}))
model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=args["width"],
                     modes=args["modes"], n_layers=args["layers"])
model.load_state_dict(torch.load({ckpt!r}, map_location="cpu"))
model.eval()
torch.manual_seed(0)
phi = torch.randn(1, 1, {H}, {H}); cond = torch.randn(1, 7)
wall, cpu = [], []
with torch.no_grad():
    for _ in range({n_warm}):
        model.rollout(phi, cond, {n_apply})
    for _ in range({n_rep}):
        w0 = time.perf_counter(); c0 = time.process_time()
        model.rollout(phi, cond, {n_apply})
        wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall, "cpu": cpu}}))
'''


def operator_run(run_dir, norm_p, n_apply, n_rep, n_warm, H=128):
    """Time the operator in a fresh process too, so 'cold' means cold for it.

    The model is rebuilt from the checkpoint in the child, so with n_warm=0 the
    timed rollout pays FFT-plan creation exactly as the cold solver pays its
    initialisation.
    """
    rd = Path(run_dir)
    body = _OP_BODY.format(norm=str(norm_p), args=str(rd / "args.json"),
                           ckpt=str(rd / "best.pt"), H=H, n_apply=n_apply,
                           n_rep=n_rep, n_warm=n_warm)
    return _child(body)


def _stats(wall, cpu):
    cw = [c / w for c, w in zip(cpu, wall)]
    return {
        "wall_s_per_wafer": wall, "cpu_s_per_wafer": cpu,
        "median_wall": float(np.median(wall)), "median_cpu": float(np.median(cpu)),
        "cpu_over_wall_median": float(np.median(cw)),
        "single_threaded_verified": bool(np.median(cw) < 1.15),
        "n": len(wall),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="runs/speed_symmetric.json")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    runlock.acquire(a.out, what="bench_symmetric")
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]
    norm_p = Path(a.data) / "norm.json"
    norm = json.loads(norm_p.read_text())

    arm_meta = {}
    for r in a.runs:
        args = json.loads((Path(r) / "args.json").read_text())
        stride = int(args.get("stride", 1) or 1)
        if n_steps % stride:
            raise SystemExit(f"{r}: stride {stride} does not divide {n_steps}")
        arm_meta[r] = {
            "run": r, "stride": stride,
            "applications_per_wafer": n_steps // stride,
            "arch": {k: args.get(k) for k in ("width", "modes", "layers")},
        }

    loads = [os.getloadavg()[0]]

    # ---- marginal_warm: both sides warmed, both report an extra wafer's cost
    sw = solver_warm(a.rounds, n_steps, a.dt, grid_delta, a.seed, n_discard=N_DISCARD)
    loads.append(os.getloadavg()[0])
    warm_arms = {}
    for r, m in arm_meta.items():
        o = operator_run(r, norm_p, m["applications_per_wafer"],
                         n_rep=a.rounds, n_warm=3)
        warm_arms[r] = _stats(o["wall"], o["cpu"])
        loads.append(os.getloadavg()[0])

    # ---- cold_single_wafer: both sides pay their own one-time cost
    cw_wall, cw_cpu = [], []
    for i in range(N_DISCARD, N_DISCARD + a.rounds):  # the same wafers marginal_warm timed
        s = solver_cold(n_steps, a.dt, grid_delta, a.seed, i)
        cw_wall.append(s["wall"])
        cw_cpu.append(s["cpu"])
        loads.append(os.getloadavg()[0])
    cold_arms = {}
    for r, m in arm_meta.items():
        w, c = [], []
        for _ in range(a.rounds):  # a fresh process each, n_warm=0, one timed rollout
            o = operator_run(r, norm_p, m["applications_per_wafer"], n_rep=1, n_warm=0)
            w += o["wall"]
            c += o["cpu"]
        cold_arms[r] = _stats(w, c)
        loads.append(os.getloadavg()[0])

    res = {
        "protocol": {
            "estimator": "ratio of median CPU-seconds per wafer, solver / operator",
            "why_cpu_seconds": "contention does not move the two sides together, "
                               "so a wall-clock ratio is not a property of the "
                               "implementations; solver_drift.json measured 1.11 "
                               "CPU-s at load 11.3 and 1.26 at load 41.0",
            "readings": {
                "marginal_warm": "cost of one ADDITIONAL wafer: solver in a "
                                 "long-lived process with the first wafer "
                                 "discarded, operator warmed with 3 untimed "
                                 "rollouts. The reading for a sweep or an "
                                 "inverse-design loop.",
                "cold_single_wafer": "latency of one wafer from nothing: fresh "
                                     "process on BOTH sides, one-time cost inside "
                                     "the timed region for both. The reading for a "
                                     "one-shot query.",
            },
            "symmetry": "the defect being corrected is that bench_paired.py used "
                        "a cold solver against a warm operator and called the "
                        "ratio like-for-like",
            "rounds": a.rounds, "n_steps_per_wafer": n_steps, "dt": a.dt,
            "grid_delta": grid_delta, "threads_each_side": 1,
            "same_wafers": f"every condition replays one seeded recipe stream and "
                           f"prices its wafers {N_DISCARD}..{N_DISCARD + a.rounds - 1}, "
                           f"so the cold reading skips the wafer the warm reading "
                           f"spent on warm-up; solver cost varies ~4x across recipes, "
                           f"which is larger than the cold/warm factor being measured",
            "operator_input": "a random field of the deployed shape. FNO cost is set "
                              "by tensor shape, not by values, so this times the same "
                              "work a real SDF would -- but it is not a real SDF and "
                              "no accuracy claim may be read off this file",
            "excluded_from_both": "rasterisation",
        },
        "box": {"n_cpu": os.cpu_count(), "load_1min_samples": loads,
                "load_min": min(loads), "load_max": max(loads),
                "quiet_note": "this box runs two other projects' jobs that this "
                              "track must not touch; it cannot be quieted, which "
                              "is why the estimator is CPU-seconds"},
        "solver": {"marginal_warm": _stats(sw["wall"], sw["cpu"]),
                   "cold_single_wafer": _stats(cw_wall, cw_cpu)},
        "solver_first_wafer_discarded_wall": sw["discarded_wall"],
        "arms": {},
    }
    res["solver"]["cold_over_warm"] = (
        res["solver"]["cold_single_wafer"]["median_cpu"]
        / res["solver"]["marginal_warm"]["median_cpu"])
    res["solver"]["cold_over_warm_note"] = (
        "the one-time initialisation ViennaPS charges inside .apply(); this factor "
        "is what bench_paired.py handed to the operator for free")

    for r, m in arm_meta.items():
        e = dict(m)
        for reading, side, sv in (("marginal_warm", warm_arms[r], res["solver"]["marginal_warm"]),
                                  ("cold_single_wafer", cold_arms[r], res["solver"]["cold_single_wafer"])):
            e[reading] = dict(side)
            e[reading]["speedup_cpu"] = sv["median_cpu"] / side["median_cpu"]
            e[reading]["speedup_wall"] = sv["median_wall"] / side["median_wall"]
            e[reading]["median_cpu_per_application"] = (
                side["median_cpu"] / m["applications_per_wafer"])
            e[reading]["both_sides_single_threaded"] = bool(
                side["single_threaded_verified"] and sv["single_threaded_verified"])
        res["arms"][r] = e

    rows = []
    for r, e in res["arms"].items():
        for reading in ("marginal_warm", "cold_single_wafer"):
            rows.append((e[reading]["speedup_cpu"], r, reading,
                         e[reading]["both_sides_single_threaded"]))
    rows.sort(reverse=True)
    best_v, best_r, best_reading, best_ok = rows[0]
    res["kpi_clause"] = {
        "clause": "simulation speedup >= 1000x",
        "target": 1000.0,
        "best_value": best_v, "best_arm": best_r, "best_reading": best_reading,
        "met": bool(best_v >= 1000.0),
        "shortfall_factor": 1000.0 / best_v,
        "both_sides_single_threaded": best_ok,
        "all_rows": [{"speedup_cpu": v, "arm": r, "reading": rd,
                      "single_threaded_ok": ok} for v, r, rd, ok in rows],
        "reading_note": "the clause is reported under BOTH readings for every arm; "
                        "the best is named here so the gap is explicit, not so the "
                        "others can be dropped",
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"box": res["box"], "solver_cold_over_warm": res["solver"]["cold_over_warm"],
                      "kpi_clause": res["kpi_clause"]}, indent=2))


if __name__ == "__main__":
    main()
