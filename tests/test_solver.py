"""ViennaPS itself: it imports, it steps, and its isotropic mode is exact.

Kept small enough for CI (seconds, one thread). The full verification with grid
convergence is `scripts/verify_solver.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402


def test_import_and_version_pin():
    import viennals
    import viennaps

    assert viennaps.__version__.startswith("4.6"), viennaps.__version__
    # 5.9.0 breaks `import viennaps` outright; if this ever passes on 5.9 the
    # pin in requirements.txt can be revisited, not before.
    assert viennals.__version__.startswith("5.8"), (
        f"viennals {viennals.__version__}: viennaps 4.6.2 needs 5.8.x"
    )


def test_isotropic_recession_is_exact():
    """rate x time, to the solver's own definition."""
    vps = S._lazy_vps()
    d2 = vps.d2
    rate, t = -0.8, 1.0
    dom = d2.Domain(gridDelta=0.2, xExtent=S.X_MAX - S.X_MIN, yExtent=60.0)
    d2.MakePlane(domain=dom, height=0.0, material=vps.Material.Si).apply()
    p = d2.Process()
    p.setDomain(dom)
    p.setProcessModel(d2.IsotropicProcess(rate=rate))
    p.setProcessDuration(t)
    p.apply()
    y = float(np.median(S.surface_polyline(dom)[2:-2, 1]))
    assert abs(y - rate * t) < 1e-3, y


def test_trench_etch_moves_the_surface_monotonically():
    rec = S.Recipe()
    dom = S.build_domain(rec, 0.3)
    depths = [S.surface_polyline(dom)[:, 1].min()]
    for _ in range(3):
        S.make_process(rec, dom, 0.2).apply()
        depths.append(S.surface_polyline(dom)[:, 1].min())
    assert all(b < a for a, b in zip(depths[:-1], depths[1:])), depths


def test_surface_is_a_single_open_chain():
    """The walk must cover every node; a partial walk mis-signs the SDF.

    The chain spans the window left to right, overhanging by at most one cell
    when gridDelta does not divide the extent (0.3 gives +-10.2). The overhang
    is harmless because the closing ring is padded relative to the endpoints,
    not to the window.
    """
    for gd in (0.3, 0.2):
        poly = S.surface_polyline(S.build_domain(S.Recipe(), gd))
        assert poly.ndim == 2 and poly.shape[1] == 2
        assert poly[0, 0] <= S.X_MIN + 1e-6 and poly[-1, 0] >= S.X_MAX - 1e-6
        assert poly[0, 0] > S.X_MIN - 2 * gd and poly[-1, 0] < S.X_MAX + 2 * gd


def test_simulate_returns_stacked_frames():
    tr = S.simulate(S.Recipe(), n_steps=2, dt=0.2, grid_delta=0.3, n=48)
    assert tr.sdf.shape == (3, 48, 48)
    assert tr.seconds > 0
    assert (tr.sdf[0] < 0).mean() > 0.5, "most of the window starts as solid"




def test_runaway_etch_stops_at_the_window_and_is_flagged():
    """A recipe that clears the window must terminate quickly and report it.

    Without the guard the level set keeps growing past the window and cost per
    step climbs without bound; an unguarded inverse-design verification hung for
    >13 min on one trajectory. The trajectory must still have the full frame
    count so downstream shape code needs no special case, and `steps_ok` must be
    False so a truncated run cannot be read as a completed one.
    """
    import time

    fast = S.Recipe(ion_flux=30.0, etchant_flux=5000.0, oxygen_flux=400.0,
                    ion_energy=200.0)
    t0 = time.perf_counter()
    tr = S.simulate(fast, n_steps=10, dt=1.0, grid_delta=0.3, n=64)
    elapsed = time.perf_counter() - t0
    assert tr.sdf.shape == (11, 64, 64), tr.sdf.shape
    assert tr.steps_ok is False, "a runaway etch must not be reported as ok"
    assert elapsed < 120, f"guard did not bound the cost: {elapsed:.0f}s"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
