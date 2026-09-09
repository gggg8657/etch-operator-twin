"""Geometric agreement between two profiles, both given as SDFs on the window.

`rel-L2` answers "is the field close"; the inverse-design clause asks "is the
*shape* close", which is a different question and needs a metric with a length
or an area in it. Two are reported:

* **normalised area difference** (primary). The symmetric difference of the two
  solid regions, divided by the area of material the *target* etch removed.
  The denominator is the removed area, not the whole solid area: normalising by
  the wafer would divide a real mismatch by a number ten times larger and turn
  any profile into a 1% match. `area_error_vs_solid` is reported alongside
  purely so the difference between the two conventions is visible.
* **Hausdorff distance** in um (secondary), between the two zero contours.

Contours come from linear interpolation of zero crossings on grid edges -- the
same construction in both directions -- so the distance is symmetric by
construction rather than by convention.
"""
from __future__ import annotations

import numpy as np


def zero_contour_points(sdf: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Sub-cell zero crossings on horizontal and vertical grid edges, in um."""
    pts = []
    a, b = sdf[:, :-1], sdf[:, 1:]
    m = (a * b) < 0
    if m.any():
        i, j = np.nonzero(m)
        t = a[i, j] / (a[i, j] - b[i, j])
        pts.append(np.stack([gx[j] + t * (gx[j + 1] - gx[j]), gy[i]], axis=1))
    a, b = sdf[:-1, :], sdf[1:, :]
    m = (a * b) < 0
    if m.any():
        i, j = np.nonzero(m)
        t = a[i, j] / (a[i, j] - b[i, j])
        pts.append(np.stack([gx[j], gy[i] + t * (gy[i + 1] - gy[i])], axis=1))
    if not pts:
        return np.zeros((0, 2))
    return np.concatenate(pts, axis=0)


def _bilinear(sdf: np.ndarray, gx: np.ndarray, gy: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Sample sdf at arbitrary points. gy is descending (row 0 = top)."""
    n = len(gx)
    fx = (P[:, 0] - gx[0]) / (gx[-1] - gx[0]) * (n - 1)
    fy = (P[:, 1] - gy[0]) / (gy[-1] - gy[0]) * (len(gy) - 1)
    fx = np.clip(fx, 0, n - 1 - 1e-9)
    fy = np.clip(fy, 0, len(gy) - 1 - 1e-9)
    j0, i0 = fx.astype(int), fy.astype(int)
    tx, ty = fx - j0, fy - i0
    j1, i1 = np.minimum(j0 + 1, n - 1), np.minimum(i0 + 1, len(gy) - 1)
    return (
        sdf[i0, j0] * (1 - tx) * (1 - ty)
        + sdf[i0, j1] * tx * (1 - ty)
        + sdf[i1, j0] * (1 - tx) * ty
        + sdf[i1, j1] * tx * ty
    )


def hausdorff_um(sdf_a: np.ndarray, sdf_b: np.ndarray, gx, gy) -> dict:
    """Symmetric Hausdorff, using each field to measure the other's contour."""
    pa = zero_contour_points(sdf_a, gx, gy)
    pb = zero_contour_points(sdf_b, gx, gy)
    if len(pa) == 0 or len(pb) == 0:
        return {"hausdorff_um": None, "mean_surface_dist_um": None,
                "note": "a profile had no zero contour inside the window"}
    da = np.abs(_bilinear(sdf_b, gx, gy, pa))
    db = np.abs(_bilinear(sdf_a, gx, gy, pb))
    return {
        "hausdorff_um": float(max(da.max(), db.max())),
        "mean_surface_dist_um": float(0.5 * (da.mean() + db.mean())),
        "n_contour_pts": [int(len(pa)), int(len(pb))],
    }


def _occupancy(sdf: np.ndarray, h: float) -> np.ndarray:
    """Sub-cell solid fraction per cell, from the SDF.

    Hard thresholding (`sdf < 0`) quantises every area to a whole cell. On this
    window that is ~3% of a typical removed area -- the same order as the 5% the
    inverse-design clause is judged against, so a hard-threshold area cannot
    resolve the KPI it is being used to test. Because the field is a true
    distance function, `clip(0.5 - phi/h, 0, 1)` is the exact area fraction of a
    cell cut by a straight contour, and near-exact for a gently curved one.
    """
    return np.clip(0.5 - sdf / h, 0.0, 1.0)


def shape_error(sdf_achieved: np.ndarray, sdf_target: np.ndarray,
                sdf_initial: np.ndarray, gx, gy) -> dict:
    """Normalised area difference plus Hausdorff. `sdf_initial` sets the scale.

    The denominator is the area the target etch actually removed. Without it the
    metric has no scale: an identical mismatch is 3% or 30% depending only on how
    much of the window happens to be solid.
    """
    dx, dy = abs(gx[1] - gx[0]), abs(gy[1] - gy[0])
    cell = dx * dy
    h = 0.5 * (dx + dy)
    oa, ot, o0 = (_occupancy(f, h) for f in (sdf_achieved, sdf_target, sdf_initial))
    mismatch = float(np.abs(oa - ot).sum() * cell)
    removed = float(np.clip(o0 - ot, 0, 1).sum() * cell)
    solid_area = float(ot.sum() * cell)
    # the hard-threshold reading, reported so the quantisation gap is visible
    hard = float(np.logical_xor(sdf_achieved < 0, sdf_target < 0).sum() * cell)
    out = {
        "mismatch_area_um2": mismatch,
        "target_removed_area_um2": removed,
        "area_error_vs_removed": mismatch / removed if removed > 0 else None,
        "area_error_vs_solid": mismatch / solid_area if solid_area > 0 else None,
        "mismatch_area_um2_hard_threshold": hard,
        "cell_um2": cell,
    }
    out.update(hausdorff_um(sdf_achieved, sdf_target, gx, gy))
    return out
