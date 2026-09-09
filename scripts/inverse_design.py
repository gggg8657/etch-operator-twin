"""Inverse design, scored in the simulator.

    python scripts/inverse_design.py --run runs/base --n-targets 20

Each target is the final profile of a held-out **test** trajectory, so a recipe
that reaches it is known to exist and to lie inside the training box. The
operator is descended on the recipe; the recipe it proposes is then run through
ViennaPS and the shape error is measured *there*.

Three numbers are reported per target and the gap between them is the point:

* `surrogate_area_error` -- how well the operator thinks it did.
* `simulator_area_error` -- how well it actually did, re-simulated. **This is the
  KPI number.** A surrogate that reports 2% while the simulator says 20% has not
  solved the design problem, it has found an adversarial example of itself.
* `recovered_recipe_rel_err` -- distance to the recipe that generated the target.
  Reported for information only, never as the KPI: the map from recipe to profile
  need not be injective, and a different recipe reaching the same profile is a
  success, not a failure.

`box_margin` is the distance to the nearest edge of the training box in units of
box width. A solution at margin ~0 is pinned against a wall: the optimiser wanted
to leave the region the simulator was sampled in, and the answer is a clipped
one. Those are counted separately and never silently averaged in.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.data import TrajDataset, cond_vector  # noqa: E402
from eot.inverse import DESIGN_KEYS, design, recipe_from_values  # noqa: E402
from eot.metrics import shape_error  # noqa: E402
from eot.operator import EtchOperator  # noqa: E402


def stat(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()), "n": int(v.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/inverse.json")
    a = ap.parse_args()

    run = Path(a.run)
    cfg = json.loads((run / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    scale = norm["sdf_scale_um"]
    device = torch.device(a.device)

    ds = TrajDataset(Path(a.data) / "test.npz", norm)
    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                         modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_steps = gen["steps"]
    gx, gy = S.grid_axes(gen["grid_n"])
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(ds), size=min(a.n_targets, len(ds)), replace=False)

    results = []
    t_start = time.perf_counter()
    for rank, i in enumerate(idxs):
        phi0 = torch.from_numpy(ds.sdf[i, 0][None, None]).float().to(device) / scale
        target = torch.from_numpy(ds.sdf[i, -1][None, None]).float().to(device) / scale
        true_recipe = ds.recipe[i]
        dt = float(ds.dt[i])
        geom = {"trench_width": float(true_recipe[4]), "mask_height": float(true_recipe[5])}

        # multi-restart: the recipe->profile loss is not convex, and a single
        # start would report the optimiser's luck as the method's accuracy
        best = None
        for r in range(a.restarts):
            init = None if r == 0 else np.array([
                np.exp(rng.uniform(np.log(S.RECIPE_BOX[k][0]), np.log(S.RECIPE_BOX[k][1])))
                if S.RECIPE_BOX[k][2] == "log"
                else rng.uniform(S.RECIPE_BOX[k][0], S.RECIPE_BOX[k][1])
                for k in DESIGN_KEYS])
            d = design(model, target, phi0, geom, dt, norm, n_steps,
                       init=init, iters=a.iters, lr=a.lr, device=device, seed=r)
            if best is None or d["best_surrogate_loss"] < best["best_surrogate_loss"]:
                best = d

        found = recipe_from_values(best["recipe_values"], geom)

        # what the surrogate thinks it achieved
        with torch.no_grad():
            c = torch.from_numpy(
                (cond_vector(np.array([[found.ion_flux, found.etchant_flux,
                                        found.oxygen_flux, found.ion_energy,
                                        found.trench_width, found.mask_height]]),
                             np.array([dt])) - np.array(norm["cond_mean"], np.float32))
                / np.array(norm["cond_std"], np.float32)).float().to(device)
            phi = phi0
            for _ in range(n_steps):
                phi = model(phi, c)
        sur = shape_error(phi[0, 0].cpu().numpy() * scale,
                          target[0, 0].cpu().numpy() * scale,
                          phi0[0, 0].cpu().numpy() * scale, gx, gy)

        # what the simulator says -- the KPI number
        tr = S.simulate(found, n_steps=n_steps, dt=dt,
                        grid_delta=gen["grid_delta"], n=gen["grid_n"])
        sim = shape_error(tr.sdf[-1].astype(np.float64),
                          target[0, 0].cpu().numpy() * scale,
                          phi0[0, 0].cpu().numpy() * scale, gx, gy)

        tv = np.array([true_recipe[j] for j in range(4)], dtype=float)
        fv = np.array(best["recipe_values"], dtype=float)
        results.append({
            "test_index": int(i),
            "dt": dt,
            "geom": geom,
            "true_recipe": {k: float(v) for k, v in zip(DESIGN_KEYS, tv)},
            "found_recipe": {k: float(v) for k, v in zip(DESIGN_KEYS, fv)},
            "recovered_recipe_rel_err": float(np.mean(np.abs(fv - tv) / np.abs(tv))),
            "box_margin": best["box_margin"],
            "pinned_to_box_edge": bool(best["box_margin"] < 0.01),
            "surrogate_loss": best["best_surrogate_loss"],
            "surrogate_area_error": sur["area_error_vs_removed"],
            "surrogate_hausdorff_um": sur["hausdorff_um"],
            "simulator_area_error": sim["area_error_vs_removed"],
            "simulator_hausdorff_um": sim["hausdorff_um"],
            "simulator_mean_surface_dist_um": sim["mean_surface_dist_um"],
            "simulator_trajectory_in_window": bool(tr.steps_ok),
        })
        print(f"[{rank+1}/{len(idxs)}] surrogate {sur['area_error_vs_removed']:.4f} "
              f"simulator {sim['area_error_vs_removed']:.4f} "
              f"margin {best['box_margin']:.3f}", flush=True)

    interior = [r for r in results if not r["pinned_to_box_edge"]]
    summary = {
        "simulator_area_error": stat([r["simulator_area_error"] for r in results]),
        "surrogate_area_error": stat([r["surrogate_area_error"] for r in results]),
        "simulator_hausdorff_um": stat([r["simulator_hausdorff_um"] for r in results]),
        "recovered_recipe_rel_err": stat([r["recovered_recipe_rel_err"] for r in results]),
        "box_margin": stat([r["box_margin"] for r in results]),
    }
    val = summary["simulator_area_error"]["mean"] if summary["simulator_area_error"] else None
    out = {
        "run": str(run),
        "n_targets": len(results),
        "iters": a.iters,
        "restarts": a.restarts,
        "lr": a.lr,
        "wall_s": time.perf_counter() - t_start,
        "n_pinned_to_box_edge": len(results) - len(interior),
        "summary": summary,
        "summary_interior_only": {
            "simulator_area_error": stat([r["simulator_area_error"] for r in interior])},
        "per_target": results,
        "protocol_note": (
            "Shape error is the symmetric difference of the achieved and target "
            "solid regions divided by the area the target etch removed, computed "
            "sub-cell. The achieved profile comes from re-running the proposed "
            "recipe in ViennaPS, not from the operator."),
        "kpi_clause": {
            "target": 0.05,
            "value": val,
            "met": bool(val <= 0.05) if val is not None else None,
            "protocol": (f"mean normalised area error over {len(results)} held-out "
                         "targets, achieved profile re-simulated in ViennaPS"),
            "surrogate_optimism": (
                summary["simulator_area_error"]["mean"] - summary["surrogate_area_error"]["mean"]
                if summary["simulator_area_error"] and summary["surrogate_area_error"] else None),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["kpi_clause"], indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
