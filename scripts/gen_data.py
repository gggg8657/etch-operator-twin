"""Generate the ViennaPS trajectory dataset.

    python scripts/gen_data.py --n-train 1800 --n-val 300 --n-test 400

Each sample is one recipe rolled forward `--steps` timesteps. The timestep is
chosen per recipe (see `solver.simulate_adaptive`) so that every trajectory
covers a comparable etch depth; `dt` is stored and handed to the operator, so
the model is never asked to guess how far a step goes.

Splits are disjoint in recipe by construction -- one recipe per trajectory, each
drawn from its own seeded stream -- so there is no split leakage to check for.
Solver wall-clock is recorded per sample at one OpenMP thread; that is the
number the speedup table's "solver, 1 thread" column comes from, and it is
measured here rather than re-timed later under different conditions.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402

RECIPE_KEYS = ["ion_flux", "etchant_flux", "oxygen_flux", "ion_energy",
               "trench_width", "mask_height"]


def one(job):
    idx, seed, steps, grid_delta, n, dt_mode, dt_lo, dt_hi = job
    rng = np.random.default_rng(seed)
    rec = S.sample_recipe(rng)
    t0 = time.perf_counter()
    if dt_mode == "independent":
        # dt drawn without looking at the recipe. This breaks the coupling that
        # makes every adaptive trajectory advance the same 0.4-1.0 um per step,
        # so per-step displacement varies with the recipe by the full ~20x of the
        # rate law. It is the split that can tell a model which learned the rate
        # law from one which learned "move about 0.7 um".
        dt = float(np.exp(rng.uniform(np.log(dt_lo), np.log(dt_hi))))
        target = float("nan")
        tr = S.simulate(rec, n_steps=steps, dt=dt, grid_delta=grid_delta, n=n)
    else:
        tr, dt, target = S.simulate_adaptive(rec, rng, n_steps=steps,
                                             grid_delta=grid_delta, n=n)
    wall = time.perf_counter() - t0
    return {
        "idx": idx,
        "sdf": tr.sdf,
        "recipe": np.array([getattr(rec, k) for k in RECIPE_KEYS], dtype=np.float32),
        "dt": np.float32(dt),
        "target_depth": np.float32(target),
        "solver_s": np.float32(tr.seconds),
        "wall_s": np.float32(wall),
        "ok": bool(tr.steps_ok),
    }


def build_split(name, n_samples, seed0, steps, grid_delta, n, workers, out_dir,
                dt_mode="adaptive", dt_lo=0.01, dt_hi=1.0):
    jobs = [(i, seed0 + i, steps, grid_delta, n, dt_mode, dt_lo, dt_hi)
            for i in range(n_samples)]
    t0 = time.perf_counter()
    res = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            res.append(r)
            if len(res) % 25 == 0:
                el = time.perf_counter() - t0
                print(f"[{name}] {len(res)}/{n_samples}  {el:.0f}s  "
                      f"{len(res)/el:.3f} traj/s  eta {(n_samples-len(res))/(len(res)/el):.0f}s",
                      flush=True)
    wall = time.perf_counter() - t0
    res.sort(key=lambda r: r["idx"])
    kept = [r for r in res if r["ok"]]
    out = out_dir / f"{name}.npz"
    np.savez_compressed(
        out,
        sdf=np.stack([r["sdf"] for r in kept]),
        recipe=np.stack([r["recipe"] for r in kept]),
        dt=np.array([r["dt"] for r in kept]),
        target_depth=np.array([r["target_depth"] for r in kept]),
        solver_s=np.array([r["solver_s"] for r in kept]),
    )
    return {
        "split": name,
        "requested": n_samples,
        "kept": len(kept),
        "rejected_out_of_window": n_samples - len(kept),
        "seed0": seed0,
        "dt_mode": dt_mode,
        "generate_wall_s": wall,
        "workers": workers,
        "solver_s_per_trajectory_mean_1thread": float(np.mean([float(r["solver_s"]) for r in kept])),
        "solver_s_per_trajectory_median_1thread": float(np.median([float(r["solver_s"]) for r in kept])),
        "file": str(out),
        "bytes": out.stat().st_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=900)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--grid-delta", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=128)
    # Measured, not guessed: throughput peaks near 8 workers (0.70 traj/s) and is
    # *lower* at 90 (0.63) -- see runs/worker_scaling.json. ViennaPS's flux solver
    # is a Monte Carlo ray trace and is bandwidth-bound, so extra workers buy
    # nothing and take the box away from the other tracks sharing it.
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data")
    ap.add_argument("--dt-mode", default="adaptive", choices=["adaptive", "independent"],
                    help="adaptive: dt from a per-recipe probe so every trajectory "
                         "covers a comparable depth. independent: dt drawn without "
                         "reference to the recipe, so per-step depth varies with the "
                         "rate law -- the crossed test split.")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="subset of train/val/test to build")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "steps": a.steps,
        "grid_delta": a.grid_delta,
        "grid_n": a.n,
        "window": {"x": [S.X_MIN, S.X_MAX], "y": [S.Y_MIN, S.Y_MAX]},
        "recipe_box": {k: list(v) for k, v in S.RECIPE_BOX.items()},
        "geom_box": {k: list(v) for k, v in S.GEOM_BOX.items()},
        "recipe_keys": RECIPE_KEYS,
        "target_depth_range": list(S.TARGET_DEPTH_RANGE),
        "splits": [],
    }
    # Seed blocks are far apart so no two splits can draw the same recipe.
    plan = [("train", a.n_train, 1_000_000), ("val", a.n_val, 2_000_000),
            ("test", a.n_test, 3_000_000)]
    if a.splits:
        plan = [x for x in plan if x[0] in a.splits]
    report["dt_mode"] = a.dt_mode
    for name, n_s, seed0 in plan:
        # separate seed block per dt-mode so the crossed split cannot reuse a
        # recipe the adaptive splits already spent
        seed0 = seed0 + (5_000_000 if a.dt_mode == "independent" else 0)
        info = build_split(name, n_s, seed0, a.steps, a.grid_delta, a.n, a.workers, out_dir,
                           dt_mode=a.dt_mode)
        report["splits"].append(info)
        print(json.dumps(info, indent=2), flush=True)

    Path(out_dir / "gen_report.json").write_text(json.dumps(report, indent=2))
    print("wrote", out_dir / "gen_report.json")


if __name__ == "__main__":
    main()
