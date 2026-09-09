"""The bootstrap that decides clause 1c. It overturned a committed claim, so it
gets tests of its own."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from coverage_verdict import boot_ci  # noqa: E402


def test_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.04, 0.01, size=(4, 200))
    sel = np.ones(200, bool)
    ci = boot_ci(vals, sel, n_boot=2000)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n"] == 200


def test_resampling_is_over_trajectories_not_seeds():
    """The seeds share trajectories. If a seed were an independent sample the CI
    would shrink with more seeds; it must not, because adding a seed adds no new
    trajectories. Duplicating a seed must barely move the interval."""
    rng = np.random.default_rng(1)
    base = rng.normal(0.04, 0.01, size=(2, 300))
    sel = np.ones(300, bool)
    a = boot_ci(base, sel, n_boot=4000, seed=3)
    b = boot_ci(np.concatenate([base, base]), sel, n_boot=4000, seed=3)
    wa, wb = a["hi"] - a["lo"], b["hi"] - b["lo"]
    assert abs(wa - wb) / wa < 0.02, (wa, wb)


def test_subset_selection_is_respected():
    vals = np.zeros((2, 10))
    vals[:, :5] = 1.0          # first half bad, second half perfect
    ci_bad = boot_ci(vals, np.array([True] * 5 + [False] * 5), n_boot=500)
    ci_good = boot_ci(vals, np.array([False] * 5 + [True] * 5), n_boot=500)
    assert ci_bad["point"] == 1.0 and ci_good["point"] == 0.0


def test_empty_selection_returns_none():
    assert boot_ci(np.zeros((2, 5)), np.zeros(5, bool)) is None


def test_worst_seed_is_the_max_not_the_mean():
    vals = np.array([[0.01] * 10, [0.09] * 10])
    ci = boot_ci(vals, np.ones(10, bool), n_boot=200)
    assert abs(ci["worst_seed_point"] - 0.09) < 1e-9
    assert abs(ci["point"] - 0.05) < 1e-9


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); n += 1; print(f"  ok  {k}")
    print(f"{n} passed")
