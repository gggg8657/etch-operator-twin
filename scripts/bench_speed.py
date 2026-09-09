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


def time_operator(model, device, n_steps, batch, n_rep, H=128):
    import torch

    model = model.to(device).eval()
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
        ts = time_solver(th, a.n_traj, n_steps, dt, grid_delta, seed=777)
        res["solver"][f"{th}_thread"] = {
            "seconds_per_wafer_mean": float(np.mean(ts)),
            "seconds_per_wafer_median": float(np.median(ts)),
            "seconds_per_wafer_std": float(np.std(ts)),
            "n": len(ts), "threads": th, "device": "cpu",
        }

    torch.set_num_threads(1)
    ts = time_operator(model, torch.device("cpu"), n_steps, 1, max(a.n_rep // 4, 3))
    res["operator"]["cpu_1_thread_batch1"] = {
        "seconds_per_wafer_mean": float(np.mean(ts)),
        "seconds_per_wafer_median": float(np.median(ts)),
        "n": len(ts), "threads": 1, "device": "cpu", "batch": 1,
    }

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
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

    base = res["solver"]["1_thread"]["seconds_per_wafer_median"]
    fair = res["operator"]["cpu_1_thread_batch1"]["seconds_per_wafer_median"]
    res["speedup"] = {
        k: base / v["seconds_per_wafer_median"] for k, v in res["operator"].items()
    }
    res["speedup_like_for_like_cpu1_vs_cpu1"] = base / fair
    res["kpi_clause"] = {
        "target": 1000.0,
        "best_reported_speedup": max(res["speedup"].values()),
        "best_configuration": max(res["speedup"], key=res["speedup"].get),
        "like_for_like_speedup": res["speedup_like_for_like_cpu1_vs_cpu1"],
        "note": (
            "best_reported_speedup compares an H100 batched forward against a "
            "single-threaded CPU solver and is a hardware comparison as much as "
            "an algorithmic one. like_for_like is the same device at the same "
            "thread count."
        ),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
