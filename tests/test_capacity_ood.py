"""The capacity-versus-OOD comparison, and the ways it could mislead.

This comparison is the first crossed-split win in the repo, so the claim is
load-bearing and the things that would make it wrong are pinned here:

1. **Stride must match.** The terminal state is at the same physical time for
   every stride, but arms at different strides also differ in
   applications-per-wafer, so a stride-10 arm compared against the stride-1
   anchor would attribute a horizon effect to capacity.
2. **Sign convention.** `mean_diff_arm_minus_anchor < 0` means the SMALLER model
   is better. Getting this backwards would invert the headline.
3. **A detection needs an interval that excludes zero**, not a small p-value.
   That rule was earned: every crossed comparison before this one had a p-value
   and bounded nothing.
4. **The anchor's parameter count must be constructed, not typed**, since
   `kcurve.json` predates the field and a hand-typed 26M would be a number in a
   document that no run produced.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("runs/capacity_ood.json")


def _res():
    return json.loads(OUT.read_text()) if OUT.exists() else None


def test_every_compared_arm_is_stride_1():
    r = _res()
    if not r:
        return
    sh = json.loads(Path("runs/shrink.json").read_text())
    for name in r["arms"]:
        assert sh["configs"][name]["config"]["stride"] == 1, \
            f"{name} is not stride 1; capacity is not the only difference"


def test_anchor_params_match_a_constructed_model():
    """Not a literal: build the deployed architecture and count."""
    r = _res()
    if not r:
        return
    from eot.operator import EtchOperator
    norm = json.loads(Path("data/norm.json").read_text())
    n = EtchOperator(cond_dim=len(norm["cond_keys"]), width=64, modes=20,
                     n_layers=4).param_count()
    assert r["protocol"]["anchor_params"] == n, (r["protocol"]["anchor_params"], n)


def test_sign_convention_matches_the_points():
    r = _res()
    if not r:
        return
    for name, row in r["arms"].items():
        for split in ("in_distribution", "crossed_in_coverage"):
            s = row[split]
            d = s["arm_point"] - s["anchor_point"]
            assert abs(d - s["mean_diff_arm_minus_anchor"]) < 1e-9, (name, split)


def test_interval_brackets_the_observed_difference():
    """A 95% interval from inverting the test must contain the point estimate;
    if it does not, the inversion grid was too narrow and the interval is
    meaningless."""
    r = _res()
    if not r:
        return
    for name, row in r["arms"].items():
        for split in ("in_distribution", "crossed_in_coverage"):
            s = row[split]
            iv, d = s["interval"], s["mean_diff_arm_minus_anchor"]
            if iv.get("lo") is None or iv.get("hi") is None:
                continue
            assert iv["lo"] <= d <= iv["hi"], (name, split, iv, d)


def test_the_crossed_win_is_a_detection_not_a_null():
    """The specific claim made in critique_log and on the board: w16m8L4_K1 is
    better out of distribution, with an interval that excludes zero. If a future
    re-run moves that interval across zero, this test fails and the claim must
    be withdrawn rather than quietly kept."""
    r = _res()
    if not r or "w16m8L4_K1" not in r["arms"]:
        return
    s = r["arms"]["w16m8L4_K1"]["crossed_in_coverage"]
    assert s["mean_diff_arm_minus_anchor"] < 0, "the smaller model is not better"
    iv = s["interval"]
    assert iv["hi"] < 0, (
        f"interval {iv['lo']:.5f}..{iv['hi']:.5f} does not exclude zero, so this "
        "is a null and the crossed-win claim must be withdrawn")


def test_clause1_crossed_is_not_claimed_met():
    """Two of three readings pass and the third does not. The bootstrap upper
    bound is the one that fails, and nothing in this repo may report clause 1 as
    met on the crossed split while it does."""
    r = _res()
    if not r or "w16m8L4_K1" not in r["arms"]:
        return
    s = r["arms"]["w16m8L4_K1"]["crossed_in_coverage"]
    assert s["arm_met_point"] and s["arm_met_every_seed"], \
        "the two readings that do pass have stopped passing"
    assert not s["arm_met_bootstrap_upper"], (
        "the bootstrap upper bound now passes, so clause 1 may be MET on the "
        "crossed split -- update the board and the paper deliberately rather "
        "than letting this test keep asserting the old verdict")


def test_smaller_is_not_claimed_monotone():
    """The 10,897-param arm must be worse than the anchor on the crossed split;
    that is what makes the story a capacity optimum rather than 'smaller is
    better', and the distinction is in three documents."""
    r = _res()
    if not r or "w8m4L2_K1" not in r["arms"]:
        return
    s = r["arms"]["w8m4L2_K1"]["crossed_in_coverage"]
    assert s["mean_diff_arm_minus_anchor"] > 0, (
        "the smallest arm now beats the anchor out of distribution, so the "
        "U-shape claim is wrong and 'reduce capacity' may be monotone")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
