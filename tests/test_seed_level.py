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


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
