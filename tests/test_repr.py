"""The surface-representation reconstruction, which was wrong five ways.

`scripts/repr_floor.py` measures how much of the field a compact interface
representation throws away, and it is only a measurement of the representation
if the reconstruction is right. Mine was not, five times, and every one of them
produced a plausible-looking floor that was really my own error:

* the sign convention was inverted (the dataset is positive in the void);
* the phase at the top of the window was hard-coded as void, which inverted the
  parity of every masked column and scored 2.0-4.2 band rel-L2;
* the field's non-unit gradient (0.787) was ignored, worth ~0.27 on its own;
* the fine grid was sampled at cell centres while the dataset defines its grid
  points at `j*delta`, offsetting every value by half a cell (0.1 um) and
  turning an exactly-representable flat front into 0.131;
* the analytic version skipped points it judged could not be in the band, which
  dropped genuine band points beside steep sidewalls and read 9.24.

Each test below pins one of those. The pattern is the argument for round-trip
tests over care: four of the five were invisible in the output and only showed
up against a case whose answer is known in advance.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.repr_floor import (  # noqa: E402
    band_rel_l2, crossings_from_sdf, gradient_scale, n_crossings_per_column,
    rebuild_multi_fine, top_phase_positive,
)

DELTA = 0.2
H = W = 64


def _flat_front(y0_um: float, scale: float = 1.0):
    """A flat horizontal interface at depth `y0_um`, exact signed distance.

    Positive above (void), negative below (solid) -- the dataset's convention.
    """
    ys = np.arange(H, dtype=np.float64)[:, None] * DELTA
    return np.repeat((y0_um - ys) * scale, W, axis=1)


def test_the_dataset_sign_convention_is_positive_in_the_void():
    f = _flat_front(6.4)
    assert f[0, 0] > 0, "the top of the window must be void (positive)"
    assert f[-1, 0] < 0, "the bottom must be solid (negative)"
    assert top_phase_positive(f).all()


def test_gradient_scale_recovers_a_known_scaling():
    """The stored field is `s` times the Euclidean distance. If this returns
    something other than `s`, every floor is off by the same factor -- ignoring
    the real dataset's 0.787 cost ~0.27 band rel-L2 on its own."""
    for s in (1.0, 0.787, 2.0):
        got = gradient_scale(_flat_front(6.4, scale=s), DELTA, 1.5 * s)
        assert abs(got - s) < 1e-6, (s, got)


def test_a_flat_front_reconstructs_almost_exactly():
    """The reconstruction is piecewise constant in x, which is EXACT for a flat
    interface, so a flat front must round-trip to near zero.

    It scored 0.131 until the fine grid's sampling offset was fixed. This is the
    test that found that bug, and it is why the floor it measures is quoted as an
    upper bound rather than as the representation's loss.
    """
    true = _flat_front(6.3)
    cross = crossings_from_sdf(true, DELTA)
    err = band_rel_l2(
        rebuild_multi_fine(cross, top_phase_positive(true), true.shape, DELTA, 8),
        true, 1.5)
    assert err < 0.02, f"a flat front should round-trip exactly, got {err}"


def test_a_solid_top_column_keeps_its_phase():
    """A column under the mask has a SOLID top row, so its phase parity starts
    inverted. Hard-coding a void top here inverted every such column and was the
    single largest of the three bugs."""
    true = _flat_front(6.4)
    true[:, :W // 2] *= -1.0  # left half: solid at the top, void below
    tp = top_phase_positive(true)
    assert not tp[:W // 2].any() and tp[W // 2:].all()

    rec = rebuild_multi_fine(crossings_from_sdf(true, DELTA), tp, true.shape,
                             DELTA, 8)
    # the reconstruction must agree with the truth about which phase is where
    agree = (np.signbit(rec) == np.signbit(true)).mean()
    assert agree > 0.98, f"phase disagreement {1 - agree:.3f}"


def test_ignoring_the_top_phase_is_detectably_worse():
    """The guard has teeth: reconstructing the same field while asserting a void
    top everywhere must score far worse than using the real per-column phase."""
    true = _flat_front(6.4)
    true[:, :W // 2] *= -1.0
    cross = crossings_from_sdf(true, DELTA)
    s = gradient_scale(true, DELTA, 1.5)
    good = band_rel_l2(rebuild_multi_fine(cross, top_phase_positive(true),
                                          true.shape, DELTA, 8) * s, true, 1.5)
    bad = band_rel_l2(rebuild_multi_fine(cross, np.ones(W, bool), true.shape,
                                         DELTA, 8) * s, true, 1.5)
    assert bad > 5 * good, (good, bad)


def test_multi_interface_columns_are_counted_not_collapsed():
    """A masked column crosses more than once (measured: up to 6 on real data),
    so a one-height representation is ambiguous by construction."""
    ys = np.arange(H, dtype=np.float64)[:, None] * DELTA
    col = np.where((ys > 2.0) & (ys < 5.0), -1.0, 1.0)  # void / solid / void
    f = np.repeat(col, W, axis=1)
    nc = n_crossings_per_column(f)
    assert nc.min() == 2 and nc.max() == 2
    assert all(c.size == 2 for c in crossings_from_sdf(f, DELTA))


def test_the_polyline_reconstruction_is_exact_for_a_flat_front():
    """`rebuild_polyline` computes exact point-to-segment distance with no
    reconstruction grid, so a flat interface must round-trip to EXACTLY zero --
    not merely close, as the grid-based reconstruction does.

    This is also the test that caught a band-restriction bug: an earlier version
    skipped points it thought could not be in the band, which dropped genuine
    band points near steep sidewalls and read 9.24 instead of ~0.08.
    """
    from scripts.repr_floor import rebuild_polyline

    true = _flat_front(6.3)
    cross = crossings_from_sdf(true, DELTA)
    field, broken = rebuild_polyline(cross, top_phase_positive(true),
                                     true.shape, DELTA, 1.5)
    assert broken == 0, f"a flat front has no ambiguous joins, got {broken}"
    assert band_rel_l2(field, true, 1.5) < 1e-6


def test_the_polyline_reports_the_joins_it_could_not_make():
    """Connectivity is a heuristic: the i-th crossing of a column joins the i-th
    of the next column only when the counts match. Where an overhang starts the
    counts differ and the polyline breaks, and the count of such breaks has to
    reach the output -- on real data it is 4-6 per trajectory, and those gaps are
    why the analytic reconstruction scores worse than the grid one despite being
    exact in x."""
    from scripts.repr_floor import interface_segments

    # two columns with 1 crossing, then one with 3: two joins, one impossible
    cross = [np.array([4.0]), np.array([4.1]), np.array([1.0, 2.0, 4.2])]
    segs, broken = interface_segments(cross, DELTA)
    assert broken == 1, broken
    assert segs.shape == (1, 2, 2), segs.shape


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
