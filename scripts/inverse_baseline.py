"""What inverse design is worth, measured in simulator calls.

    python scripts/inverse_baseline.py --n-targets 8 --budget 48

The gradient method spends **zero** simulator calls during its search and one to
verify. The honest question is therefore not "is 5% shape error good" -- there is
nothing to compare that to -- but *how many simulator calls does a search that
does not have the surrogate need in order to do as well*.

So: random search over the same recipe box, on the same targets, evaluating each
candidate in ViennaPS, recording the best-so-far error against the number of
calls spent. Reporting the whole curve rather than one number is deliberate; a
single-budget comparison can always be hidden in the budget.

Random search, not "random placement". It is a real optimiser here: the box is
4-dimensional and the objective is smooth, so a few dozen draws is not a straw
man. If it matches the gradient method within a handful of calls, that is the
finding and the operator has not earned its place in the loop.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.inverse import DESIGN_KEYS  # noqa: E402
from eot.metrics import shape_error  # noqa: E402


def _sample(rng):
    v = []
    for k in DESIGN_KEYS:
        lo, hi, mode = S.RECIPE_BOX[k]
        v.append(float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if mode == "log"
                 else float(rng.uniform(lo, hi)))
    return v


def _one(job):
    vals, geom, dt, n_steps, grid_delta, n, target, phi0 = job
    rec = S.Recipe(**dict(zip(DESIGN_KEYS, vals)),
                   trench_width=geom["trench_width"], mask_height=geom["mask_height"])
    t0 = time.perf_counter()
    tr = S.simulate(rec, n_steps=n_steps, dt=dt, grid_delta=grid_delta, n=n)
    gx, gy = S.grid_axes(n)
    e = shape_error(tr.sdf[-1].astype(np.float64), target, phi0, gx, gy)
    return {"values": vals, "area_error": e["area_error_vs_removed"],
            "hausdorff_um": e["hausdorff_um"], "solver_s": float(tr.seconds),
            "wall_s": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--design", default="runs/base/design.json")
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--budget", type=int, default=48)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/inverse_baseline.json")
    a = ap.parse_args()

    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    dsn = json.loads(Path(a.design).read_text())
    d = np.load(Path(a.data) / "test.npz")
    n_steps, grid_delta, n = gen["steps"], gen["grid_delta"], gen["grid_n"]

    # Same targets the gradient method was scored on, so the comparison is paired.
    targets = dsn["targets"][: a.n_targets]
    rng = np.random.default_rng(a.seed)
    out_targets = []
    t_start = time.perf_counter()

    for rank, t in enumerate(targets):
        i = t["target_index"]
        target = d["sdf"][i, -1].astype(np.float64)
        phi0 = d["sdf"][i, 0].astype(np.float64)
        geom, dt = t["geometry"], t["dt"]
        cands = [_sample(rng) for _ in range(a.budget)]
        jobs = [(c, geom, dt, n_steps, grid_delta, n, target, phi0) for c in cands]
        with Pool(a.workers) as pool:
            res = pool.map(_one, jobs)
        errs = [r["area_error"] for r in res]
        best_so_far = np.minimum.accumulate(errs).tolist()
        grad_err = t["operator_gd"]["area_error_vs_removed"]
        # first budget at which random search matches the gradient method
        reached = next((k + 1 for k, v in enumerate(best_so_far) if v <= grad_err), None)
        out_targets.append({
            "target_index": i,
            "budget": a.budget,
            "best_so_far_area_error": best_so_far,
            "final_best_area_error": best_so_far[-1],
            "gradient_area_error": grad_err,
            "calls_to_match_gradient": reached,
            "gradient_better": bool(grad_err < best_so_far[-1]),
            "total_solver_s": float(sum(r["solver_s"] for r in res)),
        })
        print(f"[{rank+1}/{len(targets)}] random-search best {best_so_far[-1]:.4f} "
              f"after {a.budget} sims | gradient {grad_err:.4f} | "
              f"calls to match: {reached}", flush=True)

    curves = np.array([t["best_so_far_area_error"] for t in out_targets])
    matched = [t["calls_to_match_gradient"] for t in out_targets]
    out = {
        "n_targets": len(out_targets),
        "budget": a.budget,
        "seed": a.seed,
        "wall_s": time.perf_counter() - t_start,
        "mean_best_so_far_curve": curves.mean(axis=0).tolist(),
        "median_best_so_far_curve": np.median(curves, axis=0).tolist(),
        "random_search_final_mean": float(curves[:, -1].mean()),
        "gradient_mean": float(np.mean([t["gradient_area_error"] for t in out_targets])),
        "n_targets_gradient_better": int(sum(t["gradient_better"] for t in out_targets)),
        "calls_to_match_gradient": {
            "per_target": matched,
            "n_never_matched": int(sum(m is None for m in matched)),
            "median_when_matched": (float(np.median([m for m in matched if m]))
                                    if any(m for m in matched) else None),
        },
        "per_target": out_targets,
        "note": ("The gradient method spends 0 simulator calls searching and 1 "
                 "verifying. 'calls_to_match_gradient' is how many the "
                 "surrogate-free search needed to reach the same shape error on "
                 "the same target; None means it did not within the budget."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_target"}, indent=2))


if __name__ == "__main__":
    main()
