"""The speedup clause, measured so it cannot flatter itself.

    python scripts/bench_speed.py --run runs/base

A batched H100 forward pass against a single-threaded C++ solver is not a
speedup, it is a comparison of two different machines. This script therefore
reports a grid, not a number:

* the solver at 1, 8 and 16 OpenMP threads,
* the operator on **CPU at 1 thread** -- the only strictly like-for-like cell in
  the table, same device, same thread count,
* the operator on one H100, per-sample (batch 1) and batched.

"Seconds per simulated wafer" means one full trajectory: `n_steps` timesteps
from the initial trench to the final profile, which is the unit a process
engineer would actually queue. Rasterisation is excluded from both sides.

The box load average at measurement time is recorded in the JSON. These numbers
are meaningless if something else is running -- the `solver_s` recorded during
dataset generation, taken under 90-way contention, is deliberately not used
here.
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


def time_solver(threads: int, n_traj: int, n_steps: int, dt: float, grid_delta: float, seed: int):
    """Time the solver in a subprocess so OMP_NUM_THREADS actually takes effect.

    OpenMP reads the variable once at first use; setting it after ViennaPS has
    been imported in this process would silently do nothing, and the 1-thread
    row would secretly be a 192-thread row.
    """
    code = f"""
import os, time, json, numpy as np
os.environ["OMP_NUM_THREADS"] = "{threads}"
import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from eot import solver as S
rng = np.random.default_rng({seed})
ts = []
for _ in range({n_traj}):
    rec = S.sample_recipe(rng)
    dom = S.build_domain(rec, {grid_delta})
    t0 = time.perf_counter()
    for _ in range({n_steps}):
        S.make_process(rec, dom, {dt}).apply()
    ts.append(time.perf_counter() - t0)
print(json.dumps(ts))
"""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def time_solver_parallel(workers: int, n_traj: int, n_steps: int, dt: float,
                         grid_delta: float, seed: int = 991):
    """Wall-clock per wafer when the solver is run at its best parallel setting."""
    code = f"""
import os, time, json
os.environ["OMP_NUM_THREADS"] = "1"
import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
import numpy as np
from multiprocessing import Pool
from eot import solver as S

def one(seed):
    rng = np.random.default_rng(seed)
    rec = S.sample_recipe(rng)
    dom = S.build_domain(rec, {grid_delta})
    t0 = time.perf_counter()
    S.make_process(rec, dom, {n_steps} * {dt}).apply()
    return time.perf_counter() - t0

if __name__ == "__main__":
    t0 = time.perf_counter()
    with Pool({workers}) as p:
        ts = p.map(one, range({seed}, {seed} + {n_traj}))
    wall = time.perf_counter() - t0
    print(json.dumps({{"wall_s": wall, "n": {n_traj}, "per_wafer": wall / {n_traj}}}))
"""
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    r = json.loads(out.stdout.strip().splitlines()[-1])
    return {"seconds_per_wafer_median": r["per_wafer"], "seconds_per_wafer_mean": r["per_wafer"],
            "n": r["n"], "threads": f"{workers} procs x 1", "device": "cpu",
            "mode": "single, parallel throughput",
            "note": "wall clock / n_wafers with the solver at its measured best worker count"}


def time_operator(model, device, n_steps, batch, n_rep, H=128, include_transfer=False):
    import torch

    model = model.to(device).eval()
    if include_transfer:
        phi_h = torch.randn(batch, 1, H, H, pin_memory=(device.type == "cuda"))
        cond_h = torch.randn(batch, 7, pin_memory=(device.type == "cuda"))
    phi = torch.randn(batch, 1, H, H, device=device)
    cond = torch.randn(batch, 7, device=device)
    with torch.no_grad():
        for _ in range(3):  # warm up kernels / autotune
            model.rollout(phi, cond, n_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            if include_transfer:
                p_ = phi_h.to(device, non_blocking=True)
                c_ = cond_h.to(device, non_blocking=True)
                out = model.rollout(p_, c_, n_steps)
                out[:, -1].cpu()
            else:
                model.rollout(phi, cond, n_steps)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) / batch)  # per wafer
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-traj", type=int, default=8)
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--par-workers", type=int, default=8,
                    help="solver's measured best worker count (runs/worker_scaling.json)")
    ap.add_argument("--par-n", type=int, default=32)
    ap.add_argument("--out", default="runs/speed.json")
    a = ap.parse_args()

    import torch

    from eot.operator import EtchOperator

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    n_steps, grid_delta = gen["steps"], gen["grid_delta"]
    dt = 0.2  # representative; the solver cost is set by geometry, not by dt alone

    run = Path(a.run)
    cfg = json.loads((run / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                         modes=cfg["modes"], n_layers=cfg["layers"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))

    load1, load5, load15 = os.getloadavg()
    res = {
        "load_average_at_measurement": [load1, load5, load15],
        "n_steps_per_wafer": n_steps,
        "grid_delta": grid_delta,
        "grid_n": gen["grid_n"],
        "params": model.param_count(),
        "cpu": subprocess.run(["bash", "-lc", "lscpu | grep 'Model name' | head -1"],
                              capture_output=True, text=True).stdout.strip(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "solver": {},
        "operator": {},
    }

    for th in (1, 8, 16):
        d = time_solver(th, a.n_traj, n_steps, dt, grid_delta, seed=777)
        for mode in ("single", "stepped"):
            ts = d[mode]
            res["solver"][f"{th}_thread_{mode}"] = {
                "seconds_per_wafer_mean": float(np.mean(ts)),
                "seconds_per_wafer_median": float(np.median(ts)),
                "seconds_per_wafer_std": float(np.std(ts)),
                "n": len(ts), "threads": th, "device": "cpu", "mode": mode,
            }
    st = res["solver"]["1_thread_stepped"]["seconds_per_wafer_median"]
    sg = res["solver"]["1_thread_single"]["seconds_per_wafer_median"]
    res["stepped_vs_single_overhead"] = {
        "ratio": st / sg,
        "note": ("Timing the solver as n_steps separate processes rather than one "
                 "process of the full duration inflates it by this factor, and would "
                 "inflate any speedup quoted against it by the same. 'single' is used "
                 "as the reference everywhere below."),
    }

    # Solver throughput under its best parallel configuration. A batched GPU
    # forward is a throughput number, so comparing it to a *latency* number for a
    # single-threaded solver is a category error on top of a hardware mismatch.
    # runs/worker_scaling.json measured the peak; re-measure it here under the
    # same clean conditions as everything else in this table.
    try:
        thr = time_solver_parallel(a.par_workers, a.par_n, n_steps, dt, grid_delta)
        res["solver"]["parallel_best_throughput"] = thr
    except Exception as e:  # never let the fair row's failure hide the unfair one
        res["solver"]["parallel_best_throughput"] = {"error": str(e)}

    torch.set_num_threads(1)
    ts = time_operator(model, torch.device("cpu"), n_steps, 1, max(a.n_rep // 4, 3))
    res["operator"]["cpu_1_thread_batch1"] = {
        "seconds_per_wafer_mean": float(np.mean(ts)),
        "seconds_per_wafer_median": float(np.median(ts)),
        "n": len(ts), "threads": 1, "device": "cpu", "batch": 1,
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        # A row that pays for host->device staging and the result coming back.
        # The pure-kernel rows below time neither, which is fine for an
        # algorithmic comparison and wrong for a deployment claim.
        try:
            ts = time_operator(model, dev, n_steps, 64, max(a.n_rep // 2, 3),
                               include_transfer=True)
            res["operator"]["h100_batch64_with_host_transfer"] = {
                "seconds_per_wafer_mean": float(np.mean(ts)),
                "seconds_per_wafer_median": float(np.median(ts)),
                "n": len(ts), "device": "cuda", "batch": 64,
                "note": "includes H2D of the input field and D2H of the result",
            }
        except Exception as e:
            res["operator"]["h100_batch64_with_host_transfer"] = {"error": str(e)}
        for b in (1, 16, 64, 256):
            try:
                ts = time_operator(model, dev, n_steps, b, a.n_rep)
            except torch.cuda.OutOfMemoryError:
                continue
            res["operator"][f"h100_batch{b}"] = {
                "seconds_per_wafer_mean": float(np.mean(ts)),
                "seconds_per_wafer_median": float(np.median(ts)),
                "n": len(ts), "device": "cuda", "batch": b,
            }

    base = res["solver"]["1_thread_single"]["seconds_per_wafer_median"]
    fair = res["operator"]["cpu_1_thread_batch1"]["seconds_per_wafer_median"]
    par = (res["solver"].get("parallel_best_throughput") or {}).get("seconds_per_wafer_median")
    res["speedup"] = {
        k: base / v["seconds_per_wafer_median"] for k, v in res["operator"].items()
    }
    res["speedup_like_for_like_cpu1_vs_cpu1"] = base / fair
    gpu_best = min((v["seconds_per_wafer_median"] for k, v in res["operator"].items()
                    if v.get("device") == "cuda"), default=None)
    res["kpi_clause"] = {
        "target": 1000.0,
        # The KPI verdict is keyed on the like-for-like number, NOT on the best
        # cell. Keying it on max() would award the clause to a batched H100
        # measured against a single-threaded C++ solver -- two different machines
        # doing two different things -- which is the accidental cheat this clause
        # invites and which an adversarial review of this file caught.
        "value": res["speedup_like_for_like_cpu1_vs_cpu1"],
        "basis": "operator CPU 1 thread batch 1 vs solver CPU 1 thread, single process",
        "met": bool(res["speedup_like_for_like_cpu1_vs_cpu1"] >= 1000.0),
        "context_throughput_speedup": (par / gpu_best) if (par and gpu_best) else None,
        "context_throughput_basis": (
            "solver at its best parallel worker count vs operator batched on one "
            "H100 -- throughput against throughput, still different hardware"),
        "context_naive_best_cell": max(res["speedup"].values()),
        "context_naive_configuration": max(res["speedup"], key=res["speedup"].get),
        "context_naive_basis": (
            "largest cell in the grid: batched H100 against a single-threaded CPU "
            "solver. Reported for completeness, not used for the verdict."),
        "stepped_vs_single_overhead": res["stepped_vs_single_overhead"]["ratio"],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
