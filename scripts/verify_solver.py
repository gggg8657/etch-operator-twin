"""Verify the ground truth before any operator is trained against it.

Three checks, each with a closed-form answer, written to `runs/verify_solver.json`:

1. **SDF exactness.** On a flat plane at y = h the signed distance at (x, y) is
   exactly y - h. This checks our rasterisation, not ViennaPS.
2. **Isotropic plane recession.** `IsotropicProcess(rate=r)` for t minutes moves
   a flat surface down by exactly r*t. This checks the solver's time
   integration against its own definition.
3. **Grid convergence of the SF6O2 etch.** The physical model has no closed form,
   so the check is that refining the level-set grid makes the answer stop
   moving. Reported as the surface displacement between successive refinements.

Nothing here is a fit. If check 1 or 2 fails the pipeline is wrong and no
downstream number means anything.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402


def check_sdf_exact(n=128, height=0.0):
    vps = S._lazy_vps()
    d2 = vps.d2
    dom = d2.Domain(gridDelta=0.2, xExtent=S.X_MAX - S.X_MIN, yExtent=60.0)
    d2.MakePlane(domain=dom, height=height, material=vps.Material.Si).apply()
    poly = S.surface_polyline(dom)
    sdf = S.rasterise(poly, n)
    gx, gy = S.grid_axes(n)
    analytic = (gy[:, None] - height) * np.ones((1, n))
    err = np.abs(sdf - analytic)
    return {
        "max_abs_err_um": float(err.max()),
        "mean_abs_err_um": float(err.mean()),
        "plane_height_um": height,
        "grid": n,
    }


def check_isotropic(rate=-0.8, durations=(0.5, 1.0, 2.0)):
    vps = S._lazy_vps()
    d2 = vps.d2
    out = []
    for t in durations:
        dom = d2.Domain(gridDelta=0.1, xExtent=S.X_MAX - S.X_MIN, yExtent=60.0)
        d2.MakePlane(domain=dom, height=0.0, material=vps.Material.Si).apply()
        p = d2.Process()
        p.setDomain(dom)
        p.setProcessModel(d2.IsotropicProcess(rate=rate))
        p.setProcessDuration(t)
        p.apply()
        poly = S.surface_polyline(dom)
        # ignore the two boundary nodes, which sit on the domain edge
        interior = poly[2:-2, 1]
        measured = float(np.median(interior))
        expected = rate * t
        out.append(
            {
                "duration_min": t,
                "expected_um": expected,
                "measured_um": measured,
                "abs_err_um": abs(measured - expected),
                "rel_err": abs(measured - expected) / abs(expected),
                "surface_spread_um": float(interior.max() - interior.min()),
            }
        )
    return out


def check_grid_convergence(deltas=(0.4, 0.2, 0.1), duration=1.4, n=256):
    """Refine the level-set grid; report how far the surface still moves."""
    rec = S.Recipe()
    curves = {}
    times = {}
    for d in deltas:
        dom = S.build_domain(rec, d)
        p = S.make_process(rec, dom, duration)
        t0 = time.perf_counter()
        p.apply()
        times[d] = time.perf_counter() - t0
        curves[d] = S.rasterise(S.surface_polyline(dom), n)
    out = []
    ds = list(deltas)
    for a, b in zip(ds[:-1], ds[1:]):
        diff = np.abs(curves[a] - curves[b])
        band = np.minimum(np.abs(curves[a]), np.abs(curves[b])) < 1.0
        out.append(
            {
                "coarse_delta": a,
                "fine_delta": b,
                "max_sdf_shift_um": float(diff[band].max()),
                "mean_sdf_shift_um": float(diff[band].mean()),
                "coarse_solver_s": times[a],
                "fine_solver_s": times[b],
            }
        )
    return {"pairs": out, "solver_seconds_by_delta": {str(k): v for k, v in times.items()}}


def main():
    out = {
        "viennaps": __import__("viennaps").__version__,
        "viennals": __import__("viennals").__version__,
        "sdf_exactness": check_sdf_exact(),
        "isotropic_recession": check_isotropic(),
        "grid_convergence": check_grid_convergence(),
    }
    # A pass/fail the report can key on, with thresholds stated here not inferred.
    sdf_tol = 1e-3
    iso_tol = 0.02
    out["pass"] = {
        "sdf_exactness": out["sdf_exactness"]["max_abs_err_um"] < sdf_tol,
        "sdf_tol_um": sdf_tol,
        "isotropic_recession": all(c["rel_err"] < iso_tol for c in out["isotropic_recession"]),
        "isotropic_rel_tol": iso_tol,
    }
    p = Path("runs/verify_solver.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
