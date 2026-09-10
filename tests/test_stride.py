"""The K-step (stride) dataset, and the one property the K-curve rests on.

The K-curve compares arms that differ only in how many dataset timesteps one
application of the operator advances. Two things have to hold or the comparison
is not a comparison:

  * stride=1 must reproduce the original one-step dataset *exactly*, so the K=1
    arm is the historical arm and not a re-implementation of it;
  * every arm must end at the same physical time, so the terminal-step reading
    is the same field for every K.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import PairDataset, TrajDataset, cond_vector, pair_starts, standardise

DATA = Path(__file__).resolve().parents[1] / "data"
HAVE_DATA = (DATA / "train.npz").exists() and (DATA / "norm.json").exists()


def _norm():
    return json.loads((DATA / "norm.json").read_text())


def raises(exc, fn, match=None):
    try:
        fn()
    except exc as e:
        assert match is None or match in str(e), f"{e!r} does not mention {match!r}"
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_pair_starts_non_overlapping_partition_the_trajectory():
    assert pair_starts(10, 1, False) == list(range(10))
    assert pair_starts(10, 2, False) == [0, 2, 4, 6, 8]
    assert pair_starts(10, 5, False) == [0, 5]
    assert pair_starts(10, 10, False) == [0]


def test_pair_starts_overlapping_restore_the_pair_count():
    # the data-matched control: ~T starts at every stride, not T/K
    for k in (1, 2, 5, 10):
        assert len(pair_starts(10, k, True)) == 10 - k + 1


def test_pair_starts_rejects_a_stride_longer_than_the_trajectory():
    raises(ValueError, lambda: pair_starts(10, 11, False))


def test_stride_1_reproduces_the_original_one_step_dataset():
    """Guards the K=1 arm's identity with the runs already on disk.

    The original PairDataset indexed i -> (n, t) = divmod(i, T) and conditioned on
    log(dt). If stride=1 diverged from that in either the ordering or the
    conditioning, the K=1 point of the curve would not be the arm that produced
    runs/seed1..8 and the curve would have no anchor.
    """
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    ds = PairDataset(DATA / "train.npz", norm, stride=1, overlap=False)
    d = np.load(DATA / "train.npz")
    T = d["sdf"].shape[1] - 1
    assert len(ds) == d["sdf"].shape[0] * T
    ref_cond = (cond_vector(d["recipe"], d["dt"])
                - np.array(norm["cond_mean"], np.float32)) / np.array(
                    norm["cond_std"], np.float32)
    for i in (0, 1, T - 1, T, T + 3, len(ds) - 1):
        n, t = divmod(i, T)
        a, c, b = ds[i]
        np.testing.assert_allclose(a.numpy()[0], d["sdf"][n, t] / norm["sdf_scale_um"])
        np.testing.assert_allclose(b.numpy()[0], d["sdf"][n, t + 1] / norm["sdf_scale_um"])
        np.testing.assert_allclose(c.numpy(), ref_cond[n], rtol=1e-6, atol=1e-6)


def test_stride_k_target_is_k_steps_ahead_of_its_input():
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    d = np.load(DATA / "train.npz")
    for k in (2, 5):
        ds = PairDataset(DATA / "train.npz", norm, stride=k, overlap=False)
        starts = ds.starts
        a, _, b = ds[0]  # trajectory 0, first start
        np.testing.assert_allclose(a.numpy()[0], d["sdf"][0, starts[0]] / ds.scale)
        np.testing.assert_allclose(b.numpy()[0], d["sdf"][0, starts[0] + k] / ds.scale)


def test_conditioning_shifts_by_log_k_and_nothing_else():
    """The only difference between two arms' conditioning is the horizon channel.

    log_dt is the last channel and is standardised, so a stride of K must move it
    by exactly log(K)/cond_std[log_dt] and leave the six recipe channels alone.
    If a K arm differed in any other channel the curve would be confounded.
    """
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    d = np.load(DATA / "train.npz")
    c1 = standardise(d["recipe"], d["dt"], norm, 1)
    for k in (2, 5, 10):
        ck = standardise(d["recipe"], d["dt"], norm, k)
        np.testing.assert_allclose(ck[:, :6], c1[:, :6], rtol=1e-5, atol=1e-5)
        shift = (ck[:, 6] - c1[:, 6])
        expect = np.log(k) / norm["cond_std"][6]
        np.testing.assert_allclose(shift, expect, rtol=1e-4, atol=1e-4)


def test_every_stride_arm_ends_at_the_same_physical_time():
    """The property the terminal-step reading of the K-curve depends on."""
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    ends = set()
    for k in (1, 2, 5, 10):
        ds = TrajDataset(DATA / "val.npz", norm, stride=k)
        assert len(ds.times) == 10 // k, f"stride {k} emits {len(ds.times)} states"
        ends.add(ds.times[-1])
        _, _, tgt = ds[0]
        assert tgt.shape[0] == 10 // k
    assert ends == {10}, f"arms end at different times: {ends}"


def test_a_stride_that_does_not_divide_the_trajectory_is_refused():
    """stride=3 on a 10-step trajectory would stop at t=9, not t=10, so its
    terminal error would be measured at a different physical time from every
    other arm. Refuse rather than silently compare unlike things."""
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    raises(ValueError, lambda: TrajDataset(DATA / "val.npz", norm, stride=3),
           match="same physical time")


def test_stride_k_trajectory_targets_are_the_k_multiples():
    if not HAVE_DATA:
        print("  SKIP (no dataset on this host)"); return
    norm = _norm()
    d = np.load(DATA / "val.npz")
    ds = TrajDataset(DATA / "val.npz", norm, stride=5)
    _, _, tgt = ds[7]
    np.testing.assert_allclose(tgt.numpy()[0, 0], d["sdf"][7, 5] / ds.scale)
    np.testing.assert_allclose(tgt.numpy()[1, 0], d["sdf"][7, 10] / ds.scale)


def test_mixed_strides_cover_every_start_offset_and_label_each_horizon():
    """H10's dataset. Two properties have to hold or the hypothesis is untestable.

    First, mixing strides {1,2,5,10} must supply the input-state diversity a
    stride-10 arm cannot have: on a T=10 trajectory stride 10 admits exactly one
    start offset (`T-K+1 = T/K = 1`), so every K=10 arm in this repo has only
    ever seen the initial trench as an input. The union over strides must cover
    all ten offsets.

    Second, each stride's pairs must be labelled with their OWN horizon, because
    one network serves all of them and the only thing distinguishing a 1-step
    sample from a 10-step one is the `log_dt` conditioning channel carrying
    log(K*dt). If two strides shared a conditioning vector the mixed arm would be
    trained on contradictory targets for the same input.
    """
    from eot.data import PairDataset

    root = Path(__file__).resolve().parents[1]
    norm = json.loads((root / "data" / "norm.json").read_text())
    npz = root / "data" / "train.npz"
    strides = [1, 2, 5, 10]
    parts = {k: PairDataset(npz, norm, stride=k, overlap=False) for k in strides}

    covered = set()
    for k, d in parts.items():
        covered |= set(d.starts)
        assert d.starts == list(range(0, d.T - d.T % k, k)), (k, d.starts)
    assert covered == set(range(10)), f"mixed starts cover {sorted(covered)}"
    assert parts[10].starts == [0], "stride 10 must have exactly one start offset"

    # the log_dt channel must differ between strides for the same trajectory
    idx = norm["cond_keys"].index("log_dt")
    vals = {k: float(parts[k].cond[0][idx]) for k in strides}
    assert len(set(round(v, 6) for v in vals.values())) == len(strides), vals
    # and it must be ordered, since log(K*dt) is increasing in K
    assert [vals[k] for k in strides] == sorted(vals[k] for k in strides), vals


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); n += 1; print(f"  ok  {k}")
    print(f"{n} passed")
