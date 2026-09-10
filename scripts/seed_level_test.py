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
            t["can_reject_at_0.05"] = bool(
                t["smallest_attainable_two_sided_p"] <= 0.05)
            t["reading"] = (
                "distinguishable" if t["p"] <= 0.05 else
                ("no difference detected, and the test HAD the resolution to "
                 "detect one" if t["can_reject_at_0.05"] else
                 "CANNOT TELL: with these seed counts the smallest attainable "
                 "two-sided p is above 0.05, so no arrangement of these seeds "
                 "could have produced a significant result"))
            row[split] = t
        res["arms"][name] = row

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
    for n, r in res["arms"].items():
        for split in ("in_distribution", "crossed_in_coverage"):
            t = r[split]
            print(f"{n:8} {split[:9]:9} n={t['n_a']}v{t['n_b']} "
                  f"arm={t['arm_mean']:.5f} anc={t['anchor_mean']:.5f} "
                  f"p={t['p']:.4f} min_p={t['smallest_attainable_two_sided_p']:.4f} "
                  f"{t['reading'][:34]}")


if __name__ == "__main__":
    main()
