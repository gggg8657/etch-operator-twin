"""The seed-level permutation test, and the p-floor it has to report honestly.

This test exists because the script's first version contradicted itself: it
returned p = 0.0222 for `K10_sm` while claiming the smallest attainable
two-sided p was 0.0444 -- a p-value below its own stated minimum. The floor was
hard-coded as 2/total, which is only right when the two groups are the same
size.
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seed_level_test import exact_two_sample  # noqa: E402


def test_the_p_floor_is_never_above_an_attainable_p():
    """The invariant the first version broke. For perfectly separated groups the
    returned p IS the floor, so any overstatement of the floor shows up here."""
    for n_a, n_b in [(2, 8), (3, 8), (6, 8), (8, 8), (3, 3)]:
        a = np.full(n_a, 10.0)          # perfectly separated
        b = np.full(n_b, 1.0)
        t = exact_two_sample(a, b)
        assert t["exact"], (n_a, n_b)
        assert t["p"] >= t["smallest_attainable_two_sided_p"] - 1e-12, t
        assert t["p"] <= t["smallest_attainable_two_sided_p"] + 1e-12, (
            "perfect separation must land exactly on the floor", t)


def test_the_floor_is_two_over_total_only_for_equal_groups():
    """Equal groups enumerate each split's complement, which has the same
    |difference of means|, so two splits tie at the extreme. Unequal groups do
    not: the complement of a 2-subset is an 8-subset and is not enumerated."""
    eq = exact_two_sample(np.full(4, 5.0), np.full(4, 0.0))
    assert eq["smallest_attainable_two_sided_p"] == 2 / comb(8, 4)
    ne = exact_two_sample(np.full(2, 5.0), np.full(8, 0.0))
    assert ne["smallest_attainable_two_sided_p"] == 1 / comb(10, 2)


def test_identical_groups_cannot_be_distinguished():
    t = exact_two_sample(np.ones(3), np.ones(8))
    assert t["p"] == 1.0, t


def test_a_null_is_only_informative_when_the_floor_clears_the_threshold():
    """With 2 seeds against 8 the floor is 0.0222, so a null at 0.05 is
    meaningful; with 2 against 2 the floor is 0.333 and no arrangement could
    ever reject, which the caller must be able to see."""
    tiny = exact_two_sample(np.full(2, 9.0), np.full(2, 0.0))
    assert tiny["smallest_attainable_two_sided_p"] > 0.05
    ok = exact_two_sample(np.full(2, 9.0), np.full(8, 0.0))
    assert ok["smallest_attainable_two_sided_p"] <= 0.05


def test_the_interval_excludes_zero_exactly_when_the_test_rejects():
    """The duality that makes the interval trustworthy: inverting the test at
    level alpha must exclude 0 if and only if the test of no difference rejects
    at alpha. If these disagree, one of the two is wrong and the interval cannot
    be quoted beside the p-value.
    """
    from seed_level_test import invert_to_interval

    rng = np.random.default_rng(0)
    for _ in range(12):
        n_a, n_b = int(rng.integers(3, 6)), int(rng.integers(3, 7))
        a = rng.normal(0.02, 0.002, n_a)
        b = rng.normal(0.02 + rng.choice([0.0, 0.004]), 0.002, n_b)
        t = exact_two_sample(a, b)
        iv = invert_to_interval(a, b)
        if t["smallest_attainable_two_sided_p"] > 0.05:
            continue  # no interval can exist at this level; covered elsewhere
        rejects = t["p"] <= 0.05
        assert iv["empty"] is False, iv
        assert iv["contains_zero"] == (not rejects), (t["p"], iv)


def test_the_interval_contains_its_own_point_estimate():
    from seed_level_test import invert_to_interval

    a = np.array([0.0250, 0.0261, 0.0244])
    b = np.array([0.0189, 0.0181, 0.0184, 0.0193, 0.0198, 0.0189, 0.0188, 0.0187])
    iv = invert_to_interval(a, b)
    assert iv["lo"] <= iv["point"] <= iv["hi"], iv


def test_a_truncated_interval_is_flagged_rather_than_reported_as_finite():
    """A search range that clips the interval must say so, or a truncated bound
    gets read as a real one -- which would understate the uncertainty in exactly
    the direction that makes a null look strong."""
    from seed_level_test import invert_to_interval

    a = np.array([0.02, 0.02, 0.02])          # zero spread -> tiny search span
    b = np.array([0.02, 0.02, 0.02, 0.02])
    iv = invert_to_interval(a, b, span=0.0)
    assert iv["empty"] or iv["truncated_at_search_edge"], iv


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
