"""Shape metrics against geometry with a hand-computable answer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.metrics import hausdorff_um, shape_error, zero_contour_points  # noqa: E402

N = 128
GX, GY = S.grid_axes(N)


def _plane(h):
    return (GY[:, None] - h) * np.ones((1, N))


def test_contour_sits_on_the_zero_set():
    p = zero_contour_points(_plane(-3.0), GX, GY)
    assert len(p) > 50
    assert np.abs(p[:, 1] - (-3.0)).max() < 0.2


def test_hausdorff_of_two_planes_is_their_separation():
    d = hausdorff_um(_plane(-3.0), _plane(-5.0), GX, GY)
    assert abs(d["hausdorff_um"] - 2.0) < 0.05, d
    assert abs(d["mean_surface_dist_um"] - 2.0) < 0.05, d


def test_hausdorff_is_zero_for_identical_fields():
    d = hausdorff_um(_plane(-4.0), _plane(-4.0), GX, GY)
    assert d["hausdorff_um"] < 1e-6


def test_area_error_matches_hand_computation():
    """Planes at -3 and -5 over a 20 um wide window: mismatch is 2 x 20 = 40 um^2.

    Target removed from an initial surface at 0 is 3 x 20 = 60 um^2, so the
    normalised error is 40/60 = 0.667 -- while the same mismatch against the
    whole solid area is ~0.13. The gap between those two is exactly why the
    denominator is stated.
    """
    r = shape_error(_plane(-5.0), _plane(-3.0), _plane(0.0), GX, GY)
    # the hard-threshold area is a whole cell row out; that gap is the reason
    # the sub-cell occupancy exists
    assert abs(r["mismatch_area_um2_hard_threshold"] - 40.0) > 0.9, r
    assert abs(r["mismatch_area_um2"] - 40.0) < 0.5, r
    assert abs(r["target_removed_area_um2"] - 60.0) < 0.5, r
    assert abs(r["area_error_vs_removed"] - 2.0 / 3.0) < 0.05, r
    assert r["area_error_vs_solid"] < 0.2, r


def test_perfect_match_is_zero_error():
    r = shape_error(_plane(-4.0), _plane(-4.0), _plane(0.0), GX, GY)
    assert r["area_error_vs_removed"] == 0.0
    assert r["hausdorff_um"] < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
