"""The SDF rasteriser, against geometry whose distance field is known in closed form.

No ViennaPS needed: these feed `_segment_sdf` polylines written by hand, so a
failure here is ours and not the simulator's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S


def test_flat_plane_exact():
    """A flat surface at y=h: signed distance is exactly y-h everywhere."""
    for h in (0.0, -3.0, 1.5):
        poly = np.stack([np.linspace(S.X_MIN, S.X_MAX, 101),
                         np.full(101, h)], axis=1)
        n = 64
        sdf = S.rasterise(poly, n)
        gx, gy = S.grid_axes(n)
        analytic = (gy[:, None] - h) * np.ones((1, n))
        err = np.abs(sdf - analytic).max()
        assert err < 1e-9, f"h={h}: max err {err}"


def test_sign_convention():
    """Negative inside solid, positive in the gas above it."""
    poly = np.stack([np.linspace(S.X_MIN, S.X_MAX, 51), np.zeros(51)], axis=1)
    sdf = S.rasterise(poly, 32)
    gx, gy = S.grid_axes(32)
    assert (sdf[gy > 0.5] > 0).all(), "gas region must be positive"
    assert (sdf[gy < -0.5] < 0).all(), "solid region must be negative"


def test_right_edge_column_signed():
    """Regression: grid points on x = X_MAX used to be mis-signed.

    The closing polygon's vertical edge fell exactly on the last grid column, so
    the crossing test tied and every point in that column took the wrong sign
    (115 of 16384 at n=128). The ring is now padded outside the window.
    """
    poly = np.stack([np.linspace(S.X_MIN, S.X_MAX, 51), np.zeros(51)], axis=1)
    n = 128
    sdf = S.rasterise(poly, n)
    gx, gy = S.grid_axes(n)
    below = gy < -0.5
    assert (sdf[below, -1] < 0).all(), "right edge column mis-signed"
    assert (sdf[below, 0] < 0).all(), "left edge column mis-signed"


def test_step_profile_distance():
    """A vertical step: distance to the corner is Euclidean, not axis-aligned."""
    poly = np.array([[S.X_MIN, 0.0], [0.0, 0.0], [0.0, -5.0], [S.X_MAX, -5.0]])
    n = 128
    sdf = S.rasterise(poly, n)
    gx, gy = S.grid_axes(n)
    # a point up and to the right of the top corner (0,0): distance = |p|
    j = int(np.argmin(np.abs(gx - 3.0)))
    i = int(np.argmin(np.abs(gy - 3.0)))
    p = np.array([gx[j], gy[i]])
    expected = np.linalg.norm(p - np.array([0.0, 0.0]))
    assert abs(abs(sdf[i, j]) - expected) < 0.2, (sdf[i, j], expected)


def test_polyline_ordering_rejects_branching():
    """A branching line soup must raise, not silently return a partial walk."""
    nodes = np.array([[0.0, 0.0, 0], [1.0, 0.0, 0], [2.0, 0.0, 0], [1.0, 1.0, 0]])
    lines = np.array([[0, 1], [1, 2], [1, 3]])
    poly = S._ordered_polyline(nodes, lines)
    assert len(poly) < len(nodes), "branching walk should be short, and caught upstream"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
