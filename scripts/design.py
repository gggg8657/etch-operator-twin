"""Inverse design, scored in the simulator.

    python scripts/design.py --run runs/base --n-targets 20

For each target, a profile actually produced by ViennaPS from a recipe inside the
training box -- so a solution provably exists and is reachable, and a failure is
the method's, not the target's.

Four rows per target, and the comparison between them is the point:

* `true_resim` -- re-simulate the recipe that *made* the target. Shape error must
  be ~0. If it is not, the simulator is not deterministic under this pipeline and
  every other row is noise; this row is a tripwire, not a result.
* `operator_gd` -- gradient descent on the recipe through the differentiable
  operator. **Scored in the simulator**, not on the surrogate.
* `random_search` -- best of N random recipes ranked by the same operator, with a
  comparable number of operator evaluations. Gradient descent has to beat this to
  have earned the word "design"; if it does not, the result is that the operator
  is a usable *ranker* but its gradients are not informative.
* `surrogate_opinion` -- what the operator thought its own answer achieved. The
  gap between this and `operator_gd` is the surrogate-reality gap, and it is the
  number that decides whether this method is usable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.data import TrajDataset, cond_vector  # noqa: E402
from eot.inverse import DESIGN_KEYS, RecipeParam, build_cond, design, recipe_from_values  # noqa: E402
from eot.metrics import shape_error  # noqa: E402
from eot import runlock  # noqa: E402
from eot.operator import EtchOperator  # noqa: E402


# ---------------------------------------------------------------------------
# ViennaPS must not run in this process.
#
# Measured, reproducibly: a single S.simulate() call in a process that has a live
# CUDA context leaves cuFFT permanently broken -- every subsequent backward pass
# through the spectral layers dies with CUFFT_EXEC_FAILED. A backward that
# succeeds before the call fails after it, with nothing else changed. ViennaPS
# 4.6.2 ships its own GPU path (viennaps.d2.gpu, GpuMode) and evidently disturbs
# the context or the loaded CUDA libraries.
#
# So every simulation goes to a *spawn* worker with a fresh address space. Spawn,
# not fork: a forked child inherits the parent's CUDA context, which is the thing
# being protected. The pool is created before the model touches the GPU.
# ---------------------------------------------------------------------------
# A persistent multiprocessing.Pool deadlocks here. Observed: the parent blocked in
# futex_wait while an idle spawn worker sat in pipe_read, neither accumulating CPU,
# zero targets completed in ten minutes. Torch's CUDA context and thread pools do
# not coexist with a long-lived Pool in this process.
#
# Each simulation therefore gets a fresh short-lived subprocess -- the pattern
# gen_data.py and bench_speed.py already use without trouble. Interpreter startup
# costs ~2 s a call, a few calls per target, which is nothing next to the gradient
# descent, and it buys a mechanism that demonstrably works.
_SIM_SRC = """
import os, sys, base64, pickle
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, {root!r})
from eot import solver as S
job = pickle.loads(base64.b64decode(sys.argv[1]))
tr = S.simulate(S.Recipe(**job["kw"]), n_steps=job["n_steps"], dt=job["dt"],
                grid_delta=job["grid_delta"], n=job["n"])
sys.stdout.write(base64.b64encode(pickle.dumps(
    {{"last": tr.sdf[-1], "first": tr.sdf[0], "ok": bool(tr.steps_ok)}})).decode())
"""


def simulate_recipe(rec: S.Recipe, dt: float, n_steps: int, grid_delta: float, n: int):
    import base64
    import pickle
    import subprocess

    kw = {k: float(getattr(rec, k)) for k in DESIGN_KEYS}
    kw["trench_width"] = float(rec.trench_width)
    kw["mask_height"] = float(rec.mask_height)
    job = {"kw": kw, "dt": float(dt), "n_steps": int(n_steps),
           "grid_delta": float(grid_delta), "n": int(n)}
    src = _SIM_SRC.format(root=str(Path(__file__).resolve().parents[1]))
    # Do NOT hide the GPU from the child: ViennaPS initialises a CUDA context
    # even for CPU-only work and dies with CUDA_ERROR_NO_DEVICE without one.
    # That is also the likely mechanism behind the in-process cuFFT corruption --
    # two CUDA contexts, ViennaPS's and torch's, in one process. A separate
    # process is exactly the isolation that makes both work.
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        r = subprocess.run(
            [sys.executable, "-c", src, base64.b64encode(pickle.dumps(job)).decode()],
            capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired:
        return None, None, {"failed": "timeout", "recipe": kw}
    if r.returncode != 0:
        # Inverse design proposes recipes the solver may not survive -- a
        # geometry that self-intersects, an etch that clears the domain. That is
        # a property of the proposed recipe and therefore a *result*: the recipe
        # is not simulable, which is a failure of the design, not of the script.
        # Crashing here would throw away every target already computed and would
        # also quietly bias the reported mean toward recipes that happen to run.
        return None, None, {"failed": f"returncode {r.returncode}",
                            "stderr_tail": r.stderr[-400:], "recipe": kw}
    out = pickle.loads(base64.b64decode(r.stdout.strip()))
    return out["last"], out["first"], {"failed": None, "steps_ok": out["ok"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--random-budget", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--optimise-dt", action="store_true",
                    help="Search total etch time alongside the recipe. Off (the "
                         "default) pins T = n_steps*dt to the target's own value, "
                         "which is the easy protocol -- dt was derived from a probe "
                         "of the true recipe's rate, so it hands over the degree of "
                         "freedom that sets depth. Both variants get reported.")
    a = ap.parse_args()

    run = Path(a.run)
    # Two locks, because two design jobs on ONE model are redundant even when
    # their --out paths differ, and that is exactly what happened: two instances
    # ran the identical inverse design on runs/seed1 for eight minutes, one
    # writing design.json and the other design_Tfixed.json, before both were
    # killed. The out-file lock alone would not have caught it.
    runlock.acquire(run / ".design.lock", what="design-on-this-run")
    runlock.acquire(Path(a.out or (run / "design.json")), what="design-output")
    cfg = json.loads((run / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    grid_delta, n_grid = gen["grid_delta"], gen["grid_n"]
    scale = norm["sdf_scale_um"]
    device = torch.device(a.device)

    ds = TrajDataset(Path(a.data) / "test.npz", norm)
    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                         modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gx, gy = S.grid_axes(n_grid)
    rng = np.random.default_rng(a.seed)
    idxs = rng.choice(len(ds), size=min(a.n_targets, len(ds)), replace=False)
    rows = []

    for ti, idx in enumerate(idxs):
        idx = int(idx)
        phi0_np = ds.sdf[idx, 0]
        tgt_np = ds.sdf[idx, -1]
        n_steps = ds.sdf.shape[1] - 1
        dt = float(ds.dt[idx])
        true_rec_vec = ds.recipe[idx]
        geom = {"trench_width": float(true_rec_vec[4]), "mask_height": float(true_rec_vec[5])}
        phi0 = torch.from_numpy(phi0_np)[None, None].float().to(device) / scale
        tgt = torch.from_numpy(tgt_np)[None, None].float().to(device) / scale

        row = {"target_index": idx, "dt": dt, "n_steps": n_steps, "geometry": geom,
               "true_recipe": {k: float(v) for k, v in zip(DESIGN_KEYS, true_rec_vec[:4])}}

        # --- tripwire: the recipe that made the target, re-simulated
        t0 = time.perf_counter()
        rs_final, rs_init, st = simulate_recipe(
            recipe_from_values(true_rec_vec[:4], geom), dt, n_steps, grid_delta, n_grid)
        row["true_resim"] = (shape_error(rs_final, tgt_np, phi0_np, gx, gy)
                             if rs_final is not None else dict(st))
        row["true_resim"]["sim_status"] = st
        row["true_resim"]["sim_seconds"] = time.perf_counter() - t0

        # --- gradient descent through the operator, several restarts
        best = None
        for r in range(a.restarts):
            init = None
            if r > 0:
                lo = np.array([S.RECIPE_BOX[k][0] for k in DESIGN_KEYS])
                hi = np.array([S.RECIPE_BOX[k][1] for k in DESIGN_KEYS])
                u = rng.uniform(0.15, 0.85, size=len(DESIGN_KEYS))
                init = np.where([S.RECIPE_BOX[k][2] == "log" for k in DESIGN_KEYS],
                                np.exp(np.log(np.maximum(lo, 1e-6)) + u * (np.log(hi) - np.log(np.maximum(lo, 1e-6)))),
                                lo + u * (hi - lo))
            d = design(model, tgt, phi0, geom, dt, norm, n_steps, init=init,
                       iters=a.iters, device=device, seed=a.seed + r,
                       optimise_dt=a.optimise_dt)
            if best is None or d["best_surrogate_loss"] < best["best_surrogate_loss"]:
                best = d
        row["operator_gd_surrogate"] = {
            "best_surrogate_loss": best["best_surrogate_loss"],
            "box_margin": best["box_margin"],
            "recipe": dict(zip(DESIGN_KEYS, best["recipe_values"])),
            "restarts": a.restarts, "iters": a.iters,
            "dt_was_optimised": best["dt_was_optimised"],
            "dt_found": best["dt"], "dt_true": best["dt_given"],
            "dt_rel_err": abs(best["dt"] - best["dt_given"]) / max(best["dt_given"], 1e-12),
        }
        gd_rec = recipe_from_values(best["recipe_values"], geom)
        # If total etch time was searched, the verification must use the time the
        # optimiser chose -- verifying with the target's own dt would silently
        # hand back the degree of freedom the whole variant exists to remove.
        gd_dt = best["dt"]
        # what the surrogate believed it achieved, in the same shape metric
        with torch.no_grad():
            cond = build_cond(torch.tensor(best["recipe_values"], dtype=torch.float32, device=device),
                              geom, gd_dt, norm, device)
            phi = phi0
            for _ in range(n_steps):
                phi = model(phi, cond)
        row["surrogate_opinion"] = shape_error(
            phi[0, 0].cpu().numpy() * scale, tgt_np, phi0_np, gx, gy)
        t0 = time.perf_counter()
        gd_final, _, st = simulate_recipe(gd_rec, gd_dt, n_steps, grid_delta, n_grid)
        row["operator_gd"] = (shape_error(gd_final, tgt_np, phi0_np, gx, gy)
                              if gd_final is not None else dict(st))
        row["operator_gd"]["sim_status"] = st
        row["operator_gd"]["sim_seconds"] = time.perf_counter() - t0

        # --- random search over the same operator, matched budget
        cands = np.stack([
            np.array([getattr(S.sample_recipe(np.random.default_rng(a.seed + 10_000 * ti + j),
                                              vary_geometry=False), k) for k in DESIGN_KEYS])
            for j in range(a.random_budget)
        ])
        cvec = cond_vector(
            np.concatenate([cands, np.tile([[geom["trench_width"], geom["mask_height"]]],
                                           (len(cands), 1))], axis=1),
            np.full(len(cands), dt))
        cstd = (cvec - np.array(norm["cond_mean"], np.float32)) / np.array(norm["cond_std"], np.float32)
        band = (tgt.abs() * scale < norm["band_um"]).float()
        losses = []
        with torch.no_grad():
            for s in range(0, len(cands), 256):
                cb = torch.from_numpy(cstd[s:s + 256]).float().to(device)
                p = phi0.expand(cb.shape[0], -1, -1, -1)
                for _ in range(n_steps):
                    p = model(p, cb)
                num = (((p - tgt) ** 2) * band).flatten(1).sum(1).sqrt()
                den = ((tgt ** 2) * band).flatten(1).sum(1).sqrt().clamp_min(1e-8)
                losses.append((num / den).cpu().numpy())
        losses = np.concatenate(losses)
        jbest = int(np.argmin(losses))
        rs_rec = recipe_from_values(cands[jbest], geom)
        t0 = time.perf_counter()
        rand_final, _, st = simulate_recipe(rs_rec, dt, n_steps, grid_delta, n_grid)
        row["random_search"] = (shape_error(rand_final, tgt_np, phi0_np, gx, gy)
                                if rand_final is not None else dict(st))
        row["random_search"]["sim_status"] = st
        row["random_search"]["sim_seconds"] = time.perf_counter() - t0
        row["random_search"]["surrogate_loss"] = float(losses[jbest])
        row["random_search"]["budget"] = a.random_budget
        row["random_search"]["recipe"] = dict(zip(DESIGN_KEYS, cands[jbest].tolist()))

        rows.append(row)

        def _v(key):
            x = row[key].get("area_error_vs_removed")
            return f"{x:.4f}" if isinstance(x, float) else "SIMFAIL"

        print(f"[{ti+1}/{len(idxs)}] idx={idx} true_resim={_v('true_resim')} "
              f"gd_sim={_v('operator_gd')} gd_surrogate={_v('surrogate_opinion')} "
              f"rand_sim={_v('random_search')} "
              f"margin={row['operator_gd_surrogate']['box_margin']:.3f}", flush=True)

    def agg(name, field="area_error_vs_removed"):
        """Aggregate over targets whose simulation completed, and say how many did not.

        A recipe the solver cannot survive is a design failure, not a missing
        datum. Averaging only over the survivors without reporting the count
        would flatter the method exactly in proportion to how often it proposes
        something unphysical."""
        v = np.array([r[name][field] for r in rows if r[name].get(field) is not None], float)
        n_failed = sum(1 for r in rows if (r[name].get("sim_status") or {}).get("failed"))
        if v.size == 0:
            return {"mean": None, "n": 0, "n_failed_simulation": n_failed}
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "n": int(v.size), "n_failed_simulation": n_failed,
                "n_targets": len(rows)}

    summary = {
        "n_targets": len(rows),
        "dt_optimised": bool(a.optimise_dt),
        "protocol": ("total etch time searched (T unknown, honest)" if a.optimise_dt
                     else "total etch time pinned to the target (T known, optimistic bound)"),
        "area_error_vs_removed": {k: agg(k) for k in
                                  ["true_resim", "operator_gd", "surrogate_opinion", "random_search"]},
        "hausdorff_um": {k: agg(k, "hausdorff_um") for k in
                         ["true_resim", "operator_gd", "surrogate_opinion", "random_search"]},
        "box_margin": {
            "min": float(min(r["operator_gd_surrogate"]["box_margin"] for r in rows)),
            "n_pinned_at_wall": int(sum(r["operator_gd_surrogate"]["box_margin"] < 0.01 for r in rows)),
        },
    }
    gd = summary["area_error_vs_removed"]["operator_gd"]["mean"]
    ae_gd = [r["operator_gd"]["area_error_vs_removed"] for r in rows]
    summary["kpi_clause_shape_error"] = {
        "target": 0.05,
        "frac_targets_under_5pct": float(np.mean([v <= 0.05 for v in ae_gd])),
        "p90": float(np.percentile(ae_gd, 90)),
        "max": float(np.max(ae_gd)),
        "dt_optimised": bool(a.optimise_dt),
        "headline_metric": "normalised area error vs target-removed area, "
                           "measured in ViennaPS on the recipe the operator proposed",
        "value": gd,
        "met": bool(gd <= 0.05),
        "surrogate_reality_gap": gd - summary["area_error_vs_removed"]["surrogate_opinion"]["mean"],
        "beats_random_search": bool(gd < summary["area_error_vs_removed"]["random_search"]["mean"]),
        "simulator_determinism_check": summary["area_error_vs_removed"]["true_resim"]["max"],
    }
    out = Path(a.out or (run / "design.json"))
    out.write_text(json.dumps({"summary": summary, "targets": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
