"""The comparison with SEEDS as the replication unit, which the K-curve lacked.

    python scripts/seed_level_test.py

`kcurve_report.py` pairs on the **test trajectory**: it averages each arm's
per-trajectory error over that arm's seeds, subtracts the anchor's seed-averaged
per-trajectory error, and runs an exact sign test over the 250 shared
trajectories. That is a sound test of a real question -- do these two
seed-averaged error fields differ across trajectories -- but it is **not** the
question a claim like "K=2 at matched steps is indistinguishable from K=1" is
making, and I used it for exactly that claim.

Two problems, both of which push toward a false null:

1. **Seed noise is averaged into the point estimate instead of propagated into
   the test.** The trajectory test's uncertainty is trajectory-to-trajectory
   variation at fixed seed-averaged predictions. Whether a *differently seeded*
   run of the same configuration would land elsewhere is not in it.
2. **The arms have unequal seed counts.** The anchor averages 8 seeds, `K2_sm`
   averages 3. Averaging fewer seeds leaves more residual seed noise in the arm's
   term of the difference, which inflates the variance of the paired difference
   and makes the sign test *harder to reject* -- conservative in the direction
   that manufactures "indistinguishable".

So this script re-asks the comparison with the seed as the unit: each arm
contributes its per-seed arm means (already in `runs/kcurve.json` as
`per_seed`), and the two sets are compared by an **exact two-sample permutation
test** over every way of splitting the pooled seeds, when that is enumerable.

**Resolution is reported beside every p, because at these seed counts it is the
binding constraint.** With 8 anchor seeds against 3 the smallest attainable
two-sided p is 1 / C(11,3) = 0.0061, and against 2 it is 1 / C(10,2) = 0.0222 --
so an arm can only be distinguished if its seeds separate *perfectly* from the
anchor's, and a p exactly equal to that floor means precisely that and nothing
stronger. `.overnight/RULES.md` already says
3 seeds is a screen; this quantifies what the screen can and cannot see, so a
null result is reported as "cannot tell" rather than as "the same".

Writes `runs/seed_level_test.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402


def exact_two_sample(a: np.ndarray, b: np.ndarray, max_splits: int = 200_000,
                     seed: int = 0) -> dict:
    """Two-sided permutation test on the difference of means, seeds as units.

    Enumerates every split when C(n, k) is small enough, which it is for the
    seed counts here; otherwise samples and says so. The statistic is the
    difference of means, and the p-value counts splits at least as extreme in
    absolute value.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n_a, n_b = a.size, b.size

    def floor_p(total: int) -> float:
        """Smallest two-sided p this enumeration can return.

        The observed split always counts itself, so the floor is 1/total --
        EXCEPT when the groups are the same size, where a split's complement is
        also enumerated and has the same |difference of means|, making the floor
        2/total. Reporting 2/total unconditionally was wrong and showed up as an
        internal contradiction: `K10_sm` returned p = 0.0222 = 1/45 against a
        claimed floor of 0.0444, i.e. a p-value below its own stated minimum.
        """
        return (2.0 if n_a == n_b else 1.0) / total
    pooled = np.concatenate([a, b])
    obs = float(a.mean() - b.mean())
    total = comb(n_a + n_b, n_a)
    idx = np.arange(n_a + n_b)
    if total <= max_splits:
        hits = 0
        for pick in combinations(idx, n_a):
            m = np.zeros(n_a + n_b, bool)
            m[list(pick)] = True
            if abs(pooled[m].mean() - pooled[~m].mean()) >= abs(obs) - 1e-15:
                hits += 1
        return {"statistic": "difference of per-seed arm means",
                "observed": obs, "n_a": n_a, "n_b": n_b,
                "n_splits": int(total), "exact": True,
                "p": float(hits / total),
                "smallest_attainable_two_sided_p": floor_p(total)}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(max_splits):
        perm = rng.permutation(pooled)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= abs(obs) - 1e-15:
            hits += 1
    return {"statistic": "difference of per-seed arm means",
            "observed": obs, "n_a": n_a, "n_b": n_b,
            "n_splits": int(max_splits), "exact": False,
            "p": float((hits + 1) / (max_splits + 1)),
            "smallest_attainable_two_sided_p": float(1 / (max_splits + 1))}


def invert_to_interval(arm: np.ndarray, anc: np.ndarray, alpha: float = 0.05,
                       n_grid: int = 801, span: float = 6.0) -> dict:
    """95% CI for the arm-minus-anchor mean difference, by inverting the test.

    **Why this exists.** A p-value at n=3 answers "could these seeds have been
    shuffled to look this extreme", which is a statement about resolution. It
    does not say what effect size the test could have detected, and I used a
    p-floor below 0.05 as though it did. The interval is the object that answers
    the question a null raises: the set of shifts delta for which subtracting
    delta from the arm leaves the test unrejected.

    Assumption-free -- no normality, no equal variances -- because it reuses the
    same exact permutation enumeration. Reported as an interval over the mean
    difference in band rel-L2, in the same units as every other number here.

    `span` sets the search range as a multiple of the pooled spread, and the
    result records whether the interval hit that range rather than closing, so a
    truncated interval can never be read as a finite one.
    """
    arm = np.asarray(arm, float)
    anc = np.asarray(anc, float)
    obs = float(arm.mean() - anc.mean())
    spread = float(max(arm.std(ddof=0) + anc.std(ddof=0), 1e-9))
    lo_edge, hi_edge = obs - span * spread - abs(obs), obs + span * spread + abs(obs)
    deltas = np.linspace(lo_edge, hi_edge, n_grid)
    kept = []
    for d in deltas:
        t = exact_two_sample(arm - d, anc)
        if t["p"] > alpha:
            kept.append(float(d))
    if not kept:
        return {"alpha": alpha, "point": obs, "lo": None, "hi": None,
                "empty": True,
                "note": "no shift was left unrejected anywhere on the search "
                        "range; with these seed counts the floor may exceed "
                        "alpha, in which case NO interval exists at this level "
                        "and the test cannot bound the effect at all"}
    lo, hi = min(kept), max(kept)
    return {"alpha": alpha, "point": obs, "lo": lo, "hi": hi,
            "width": hi - lo, "empty": False,
            "contains_zero": bool(lo <= 0.0 <= hi),
            "truncated_at_search_edge": bool(
                lo <= deltas[0] + 1e-12 or hi >= deltas[-1] - 1e-12),
            "method": "inversion of the exact two-sample permutation test: the "
                      "set of shifts delta with p(arm - delta, anchor) > alpha",
            "n_grid": n_grid}


def seeds_needed(arm: np.ndarray, anc: np.ndarray, target: float,
                 observed_width: float) -> dict:
    """How many seeds per arm would make the interval narrower than `target`.

    **This exists to settle a contradiction between two sections of this repo's
    own critique log**, written by two instances of this loop from the same JSON:
    one concluded the crossed split "needs more seeds -- not a different test",
    the other that adding seeds "will not fix it". Both are assertions where a
    number is available, so here is the number.

    The interval's width scales with the standard error of the mean difference,
    `sqrt(sd_arm^2/n_a + sd_anc^2/n_b)`. Rather than assume the normal
    approximation -- the permutation interval at n=3 is coarse and measurably
    wider than it -- this calibrates against the width actually observed and
    scales from there, so the answer inherits the real enumeration's
    conservatism instead of a textbook constant.

    Both arms are scaled together, because the DOMINANT term is usually the
    anchor's own spread: an arm seeded to infinity against an 8-seed anchor
    cannot close the interval by itself.
    """
    arm = np.asarray(arm, float)
    anc = np.asarray(anc, float)
    sd_a = float(arm.std(ddof=1)) if arm.size > 1 else 0.0
    sd_b = float(anc.std(ddof=1)) if anc.size > 1 else 0.0
    se_obs = float(np.sqrt(sd_a ** 2 / max(arm.size, 1)
                           + sd_b ** 2 / max(anc.size, 1)))
    if se_obs <= 0 or observed_width <= 0:
        return {"n_per_arm": None, "note": "no spread to extrapolate from"}
    # width(n) = observed_width * SE(n)/SE(observed), with both arms at n
    var_sum = sd_a ** 2 + sd_b ** 2
    k = observed_width * float(np.sqrt(var_sum)) / se_obs  # width(n) = k/sqrt(n)
    n_req = (k / target) ** 2
    return {
        "sd_arm": sd_a, "sd_anchor": sd_b,
        "se_observed": se_obs,
        "target_width": target,
        "n_per_arm": float(n_req),
        "n_per_arm_rounded_up": int(np.ceil(n_req)),
        "width_at_8_per_arm": float(k / np.sqrt(8)),
        "width_at_32_per_arm": float(k / np.sqrt(32)),
        "dominant_term": "anchor" if sd_b > sd_a else "arm",
        "method": "calibrated from the observed interval width, scaling as "
                  "1/sqrt(n) with both arms at n; the dominant variance term is "
                  "named because scaling only the arm cannot close the interval",
    }


def seeds_needed(arm: np.ndarray, anc: np.ndarray, target_width: float) -> dict:
    """How many seeds per arm would make the interval narrower than `target_width`.

    **This settles a contradiction between two sections of this repo's own
    critique log.** One said the crossed split "needs more seeds -- not a
    different test"; the other said adding seeds "will not fix it". Both were
    unquantified, and the disagreement is resolvable from per-seed spreads that
    are already measured.

    A normal-approximation PROJECTION, not a measurement, and labelled as one:
    for equal n per arm the 95% interval half-width is about
    `1.96 * sigma * sqrt(2/n)`, so

        n = 2 * (1.96 * sigma / (target_width / 2))**2

    with `sigma` the larger of the two measured per-seed standard deviations,
    since the interval inherits the noisier arm. The measured inputs are run
    products; the extrapolation is arithmetic and is not a number this repo has
    observed.

    Also reported: the **floor** imposed by the anchor alone. The anchor's seed
    count is shared by every comparison, so if `1.96 * sigma_anchor /
    sqrt(n_anchor)` already exceeds `target_width / 2`, then no number of *arm*
    seeds can close the interval and the anchor is the binding constraint.
    """
    arm = np.asarray(arm, float)
    anc = np.asarray(anc, float)
    sd_arm = float(arm.std(ddof=1)) if arm.size > 1 else float("nan")
    sd_anc = float(anc.std(ddof=1)) if anc.size > 1 else float("nan")
    sigma = float(np.nanmax([sd_arm, sd_anc]))
    z, half = 1.96, target_width / 2.0
    n_req = 2.0 * (z * sigma / half) ** 2 if half > 0 else float("inf")
    anchor_only_half = z * sd_anc / np.sqrt(anc.size) if anc.size else float("inf")
    return {
        "method": "normal-approximation projection from MEASURED per-seed SDs; "
                  "an extrapolation, not an observation",
        "target_width": target_width,
        "sd_per_seed_arm": sd_arm, "sd_per_seed_anchor": sd_anc,
        "sigma_used": sigma,
        "seeds_per_arm_required": float(n_req),
        "seeds_per_arm_now": [int(arm.size), int(anc.size)],
        "anchor_alone_half_width_at_current_n": float(anchor_only_half),
        "anchor_is_the_binding_constraint": bool(anchor_only_half > half),
        "reading": (
            f"the anchor's own {anc.size} seeds already give a half-width of "
            f"{anchor_only_half:.5f} against a target half-width of {half:.5f}, "
            f"so NO number of arm seeds can close this interval -- the anchor "
            f"must be re-run too" if anchor_only_half > half else
            f"about {n_req:.0f} seeds per arm would bring the interval under "
            f"{target_width:.5f}"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcurve", default="runs/kcurve.json")
    ap.add_argument("--anchor", default="K1_nv")
    ap.add_argument("--out", default="runs/seed_level_test.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="seed_level_test")
    kc = json.loads(Path(a.kcurve).read_text())
    arms = kc["arms"]
    if a.anchor not in arms:
        raise SystemExit(f"anchor {a.anchor} not in {a.kcurve}")

    res = {
        "question": "with the SEED as the replication unit, can this dataset "
                    "distinguish each K-curve arm from the K=1 anchor?",
        "why": "kcurve_report.py pairs on the test trajectory with seeds "
               "averaged into the point estimate, so its p-values do not carry "
               "seed uncertainty and are conservative for arms with fewer seeds "
               "than the anchor -- conservative in the direction that "
               "manufactures 'indistinguishable'",
        "protocol": {
            "unit": "one seed of one arm; the observation is that seed's "
                    "terminal-step band rel-L2 averaged over test trajectories",
            "source": f"{a.kcurve}: arms[*][split].terminal_step.per_seed",
            "test": "exact two-sample permutation over every split of the pooled "
                    "seeds, statistic = difference of means, two-sided",
            "anchor": a.anchor,
            "note": "an underpowered null here is 'cannot tell', not 'the same'. "
                    "The smallest attainable p is reported beside every result "
                    "so a null can be read against what the test could have "
                    "detected at all.",
        },
        "arms": {},
    }

    for name, arm in arms.items():
        if name == a.anchor:
            continue
        row = {"n_seeds": arm["n_seeds"],
               "applications_per_wafer": arm["applications_per_wafer"],
               "approx_gradient_steps": arm["approx_gradient_steps"]}
        for split in ("in_distribution", "crossed_in_coverage"):
            arm_seeds = arm[split]["terminal_step"]["per_seed"]
            anc_seeds = arms[a.anchor][split]["terminal_step"]["per_seed"]
            t = exact_two_sample(np.asarray(arm_seeds), np.asarray(anc_seeds))
            t["arm_per_seed"] = list(map(float, arm_seeds))
            t["anchor_per_seed"] = list(map(float, anc_seeds))
            t["arm_mean"] = float(np.mean(arm_seeds))
            t["anchor_mean"] = float(np.mean(anc_seeds))
            # A null is only informative if the test could have rejected.
            t["interval"] = invert_to_interval(np.asarray(arm_seeds),
                                              np.asarray(anc_seeds))
            t["can_reject_at_0.05"] = bool(
                t["smallest_attainable_two_sided_p"] <= 0.05)
            iv = t["interval"]
            if not t["can_reject_at_0.05"]:
                t["reading"] = (
                    "CANNOT TELL: with these seed counts the smallest attainable "
                    "two-sided p is above 0.05, so no arrangement of these seeds "
                    "could have produced a significant result")
            elif t["p"] <= 0.05:
                t["reading"] = (
                    f"distinguishable, and the difference is somewhere in "
                    f"[{iv['lo']:.5f}, {iv['hi']:.5f}]" if not iv["empty"] else
                    "distinguishable, but the interval is empty at this level")
            else:
                t["reading"] = (
                    f"no difference detected; the 95% interval on the mean "
                    f"difference is [{iv['lo']:.5f}, {iv['hi']:.5f}], so effects "
                    f"up to {max(abs(iv['lo']), abs(iv['hi'])):.5f} are NOT ruled "
                    f"out" if not iv["empty"] else
                    "no difference detected and no interval exists at this level")
            row[split] = t
        res["arms"][name] = row

    # A yardstick, so an interval width can be read against something the
    # K-curve actually treats as a real effect rather than against zero. The
    # K=1 -> K=5 step-matched gap is the smallest structure in the curve that
    # this repo has called a difference.
    try:
        yard = abs(arms["K5_sm"]["in_distribution"]["terminal_step"]["point"]
                   - arms[a.anchor]["in_distribution"]["terminal_step"]["point"])
    except KeyError:
        yard = None
    res["yardstick"] = {
        "value": yard,
        "what": "|K5_sm - K1_nv| terminal-step in-distribution, the smallest gap "
                "this repo has treated as a real effect in the K-curve",
        "why": "an interval is only interpretable against an effect size that "
               "matters; a null whose interval covers this yardstick has not "
               "ruled out a difference of the same order as the curve's own "
               "structure",
    }
    for name, row in res["arms"].items():
        for split in ("in_distribution", "crossed_in_coverage"):
            iv = row[split]["interval"]
            row[split]["interval_covers_yardstick"] = bool(
                yard is not None and not iv["empty"]
                and iv["lo"] <= yard <= iv["hi"])
            if yard and not iv["empty"]:
                row[split]["seeds_needed_for_width_below_yardstick"] = (
                    seeds_needed(np.asarray(row[split]["arm_per_seed"]),
                                 np.asarray(row[split]["anchor_per_seed"]),
                                 yard, iv["width"]))

    # The claim this script was written to audit.
    k2sm = res["arms"].get("K2_sm", {}).get("in_distribution")
    res["audit_of_the_K2_sm_claim"] = {
        "claim_made": "K=2 at matched gradient steps is statistically "
                      "indistinguishable from K=1 in-distribution",
        "trajectory_paired_p_reported": 0.411,
        "trajectory_paired_source": "runs/kcurve.json tests_vs_K1.K2_sm."
                                    "in_distribution.sign_test.p",
        "seed_level_p": k2sm["p"] if k2sm else None,
        "seed_level_n": [k2sm["n_a"], k2sm["n_b"]] if k2sm else None,
        "smallest_attainable_p": (k2sm["smallest_attainable_two_sided_p"]
                                  if k2sm else None),
        "verdict": (k2sm["reading"] if k2sm else "K2_sm absent from the K-curve"),
        "how_the_claim_should_read": (
            "the trajectory-paired test finds no difference across trajectories "
            "at seed-averaged predictions, and the seed-level test at these seed "
            "counts is reported beside it with its own resolution stated. The "
            "word 'indistinguishable' was doing work the trajectory test cannot "
            "do, and is withdrawn in favour of whichever of the two readings "
            "applies."),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"audit": res["audit_of_the_K2_sm_claim"]}, indent=2))
    print(f"\nyardstick |K5_sm - K1_nv| in-dist = "
          f"{res['yardstick']['value']:.5f}" if res["yardstick"]["value"]
          else "\nyardstick unavailable")
    for n, r in res["arms"].items():
        for split in ("in_distribution", "crossed_in_coverage"):
            t = r[split]
            iv = t["interval"]
            ivs = ("empty" if iv["empty"]
                   else f"[{iv['lo']:+.5f},{iv['hi']:+.5f}] w={iv['width']:.5f}")
            print(f"{n:8} {split[:9]:9} n={t['n_a']}v{t['n_b']} "
                  f"d={t['observed']:+.5f} p={t['p']:.4f} "
                  f"floor={t['smallest_attainable_two_sided_p']:.4f} "
                  f"CI={ivs} yard={t.get('interval_covers_yardstick')}")


if __name__ == "__main__":
    main()
