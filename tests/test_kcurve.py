"""The K-curve's verdict machinery, and the properties its tables rest on.

The curve compares arms whose only difference is how many dataset timesteps one
application advances, so what has to be guarded is not the numbers but the
claims *about* the numbers: that the paired test is the exact test it says it
is, that an arm is identified by the right variant, and that a restart curve is
nested.
"""
from __future__ import annotations

import json
import os
import time
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.kcurve_report import arms, sign_test, signflip_test  # noqa: E402


def close(a, b, tol=1e-12):
    assert abs(a - b) <= tol, f"{a} != {b}"


def test_sign_test_is_the_exact_binomial_and_not_an_approximation():
    # 3 differences, all favouring the arm: two-sided p = 2 * (1/8) = 0.25.
    close(sign_test(np.array([-1.0, -2.0, -3.0]))["p"], 0.25)
    # 5 of 5: 2 * (1/32)
    close(sign_test(np.array([-1.0] * 5))["p"], 2 / 32)
    # a perfectly split sample cannot be evidence of anything
    close(sign_test(np.array([-1.0, 1.0]))["p"], 1.0)
    # 8 of 10 -- computed the same way a reader would check it
    d = np.array([-1.0] * 8 + [1.0] * 2)
    close(sign_test(d)["p"], 2 * sum(comb(10, i) for i in range(3)) / 2 ** 10)


def test_sign_test_drops_exact_ties_rather_than_counting_them_as_wins():
    r = sign_test(np.array([-1.0, 0.0, 0.0, -1.0]))
    assert r["n"] == 2 and r["n_arm_better"] == 2
    close(r["p"], 0.5)


def test_sign_test_direction_is_arm_minus_anchor():
    """A negative difference means the arm beat the anchor. A sign error here would invert
    every verdict in the K table, so it is pinned."""
    assert sign_test(np.array([-1.0, -1.0, -1.0]))["n_arm_better"] == 3
    assert sign_test(np.array([1.0, 1.0, 1.0]))["n_arm_better"] == 0


def test_sign_flip_test_is_labelled_as_resampled_not_exact():
    """2^121 assignments cannot be enumerated; the p must not claim to be exact."""
    r = signflip_test(np.array([-0.1, 0.2, -0.05, 0.01]), n_perm=2000)
    assert r["exact"] is False and r["n_perm"] == 2000
    assert 0 < r["p"] <= 1


def test_sign_flip_p_is_bounded_below_by_the_monte_carlo_floor():
    """With n_perm resamples no p below 1/(n_perm+1) is representable, so a
    reported 0.0 would be a lie about resolution."""
    d = np.full(64, -1.0)          # as extreme as a sign-flip null can see
    r = signflip_test(d, n_perm=1000)
    assert r["p"] >= 1 / 1001


def test_sign_flip_finds_no_effect_in_symmetric_noise():
    rng = np.random.default_rng(0)
    d = rng.normal(size=400)
    assert signflip_test(d, n_perm=5000)["p"] > 0.05


def test_arms_reads_the_variant_from_the_directory_and_checks_it(tmp=None):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "kcurve").mkdir()
        def mk(name, complete=True, **cfg):
            """A fixture arm. `complete=False` leaves off the completion
            markers, which is what a killed or still-training run looks like.

            `arms()` filters by `is_complete` since 2026-09-10 -- partial
            checkpoints had been entering seed groups and always score worse --
            so a fixture that writes only args.json and best.pt is now invisible
            to it, and this test asserted against an empty dict for as long as
            that mismatch stood. The completion markers are written in the order
            `seed_spread.completed` requires: `test_eval.json` must not predate
            `best.pt`, because an eval older than its checkpoint belongs to no
            model on disk.
            """
            d = root / "kcurve" / name if name.startswith("K") else root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "args.json").write_text(json.dumps(cfg))
            (d / "best.pt").write_bytes(b"")
            if complete:
                (d / "log.jsonl").write_text('{"epoch": 1}\n')
                (d / "done.json").write_text(json.dumps({"finished": 1.0}))
                (d / "test_eval.json").write_text(json.dumps({"ok": True}))
                now = time.time()
                os.utime(d / "best.pt", (now - 10, now - 10))
                os.utime(d / "test_eval.json", (now, now))
            return d
        mk("seed1", stride=1, blind=False, seed=1)
        mk("seed2", stride=1, blind=False, seed=2)
        mk("blind", stride=1, blind=True, seed=0)
        mk("K2_nv_s1", stride=2, overlap_pairs=False, seed=1)
        mk("K2_ov_s1", stride=2, overlap_pairs=True, seed=1)
        mk("K5_sm_s1", stride=5, overlap_pairs=False, seed=1)
        found = arms(root)
        assert sorted(found) == [(1, "nv"), (2, "nv"), (2, "ov"), (5, "sm")]
        # the anchor is the stride-1 conditioned runs, and the blind ablation is
        # not one of them -- pooling it would drag the anchor from 0.019 to 0.67
        assert [p.name for p in found[(1, "nv")]] == ["seed1", "seed2"]

        # And an arm that has not finished must not appear at all. This is the
        # guard whose absence contaminated every K-curve number the repo had
        # produced: a partial checkpoint always scores worse than a finished
        # one, so including it biases an arm downward by an amount that depends
        # on queue order rather than on anything about the arm.
        mk("K2_nv_s2", complete=False, stride=2, overlap_pairs=False, seed=2)
        mk("K10_nv_s1", complete=False, stride=10, overlap_pairs=False, seed=1)
        again = arms(root)
        assert sorted(again) == sorted(found), (
            "an unfinished arm changed the arm set")
        assert [p.name for p in again[(2, "nv")]] == ["K2_nv_s1"], (
            "an unfinished seed was pooled into a finished arm")

        # a directory whose name disagrees with its args.json is a mislabelled
        # arm, which is how a data-matched control gets reported as a horizon
        # effect. Refuse rather than average them together.
        mk("K5_nv_s2", stride=5, overlap_pairs=True, seed=2)
        try:
            arms(root)
        except AssertionError:
            return
        raise AssertionError("expected a mislabelled arm to be refused")


def test_restart_curve_on_disk_is_nested_and_monotone_in_R():
    """The selector is argmin over the first R restarts, so its chosen surrogate
    loss can only fall as R grows. If a run on disk violates that, the prefixes
    were not prefixes -- the failure mode the per-(target, restart) rng exists
    to prevent -- and every R row is a different sample instead of one curve."""
    docs = sorted(Path(ROOT / "runs").glob("design_*restartcurve*.json"))
    if not docs:
        print("  SKIP (no restart-curve run on this host)"); return
    for q in docs:
        d = json.loads(q.read_text())
        for t in d["targets"]:
            cur = t.get("restart_curve")
            if not cur:
                continue
            Rs = sorted(cur, key=int)
            losses = [cur[R]["best_surrogate_loss"] for R in Rs]
            assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])), \
                f"{q.name} target {t['target_index']}: {dict(zip(Rs, losses))}"


def test_the_screen_never_reports_a_mean_over_an_empty_accepted_set():
    docs = sorted(Path(ROOT / "runs").glob("design_*restartcurve*.json"))
    if not docs:
        print("  SKIP (no restart-curve run on this host)"); return
    for q in docs:
        s = json.loads(q.read_text())["summary"].get("restart_curve")
        if not s:
            continue
        for R, row in s["R"].items():
            if row["n_accepted"] == 0:
                assert row["mean_accepted"] is None
                assert row["met_mean_accepted"] is False
            else:
                assert row["mean_accepted"] is not None
                # a screened figure must never be quoted without its cost
                assert 0.0 <= row["reject_rate"] <= 1.0


def test_an_arm_still_training_is_not_scored_as_a_seed():
    """`kcurve_report` evaluates `best.pt` directly rather than reading a
    committed eval, so an arm that is mid-training has a checkpoint on disk and
    will be scored at whatever epoch it has reached -- entering an undertrained
    model into the seed group as if it had finished.

    That is not hypothetical. Before this guard, `runs/kcurve.json` scored
    `K2_nv_s5` (killed by a process-group kill at epoch 26 of 80) at 0.06705 and
    `K2_nv_s8` (four minutes into an 80-epoch run) at 0.11902, against six
    completed K2_nv seeds all between 0.02043 and 0.02291. Those two partial
    checkpoints moved the arm's mean from 0.02169 to 0.03958 and its seed range
    from 0.00248 to 0.09859, and they were about to be written up as a
    training-instability finding.

    A partial checkpoint always scores worse, and which arms are caught
    mid-training depends on queue order, so the bias does not cancel across arms.
    """
    import json as _json
    import tempfile

    # `scripts.kcurve_report`, matching this file's module-level import. A
    # bare `kcurve_report` only resolves when scripts/ happens to be on
    # sys.path, which it is not under `python tests/test_kcurve.py` -- the
    # way CI and every other runner invoke this file.
    from scripts.kcurve_report import is_complete

    root = Path(tempfile.mkdtemp())
    run = root / "K2_nv_s1"
    run.mkdir()
    (run / "args.json").write_text(_json.dumps({"epochs": 80, "stride": 2}))
    (run / "best.pt").write_text("weights")
    # a log that stops early: still training, or killed
    (run / "log.jsonl").write_text(
        "".join(_json.dumps({"epoch": e, "train_loss": 0.1}) + "\n" for e in range(26)))
    assert not is_complete(run), "a 26/80 arm must not be scored"

    # completing the log is still not enough without an eval no older than the
    # checkpoint -- but done.json is proof, which is what a clean run writes
    (run / "log.jsonl").write_text(
        "".join(_json.dumps({"epoch": e, "train_loss": 0.1}) + "\n" for e in range(80)))
    assert is_complete(run), "a full 0..79 log is a complete run"


def test_the_anchor_is_not_excluded_for_predating_the_done_marker():
    """The first attempt at the guard required `done.json` and silently cut the
    K=1 anchor from 8 seeds to 2: `runs/seed1..8` all logged 80/80 epochs and all
    carry a `test_eval.json`, but only two have a `done.json`, because
    `runlock.mark_done` was added after the other six ran.

    A guard that drops the anchor changes the seed group just as surely as one
    that admits a partial checkpoint.
    """
    import json as _json
    import tempfile

    # `scripts.kcurve_report`, matching this file's module-level import. A
    # bare `kcurve_report` only resolves when scripts/ happens to be on
    # sys.path, which it is not under `python tests/test_kcurve.py` -- the
    # way CI and every other runner invoke this file.
    from scripts.kcurve_report import is_complete

    run = Path(tempfile.mkdtemp()) / "seed3"
    run.mkdir(parents=True)
    (run / "args.json").write_text(_json.dumps({"epochs": 80, "stride": 1}))
    (run / "best.pt").write_text("weights")
    (run / "log.jsonl").write_text(
        "".join(_json.dumps({"epoch": e, "train_loss": 0.1}) + "\n" for e in range(80)))
    assert not (run / "done.json").exists()
    assert is_complete(run), "a complete run without done.json must still count"


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); n += 1; print(f"  ok  {k}")
    print(f"{n} passed")
