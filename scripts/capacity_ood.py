"""H17: is the crossed-split failure a capacity problem rather than a physics one?

    python scripts/capacity_ood.py

**The observation this exists to test.** Clause 1 has failed on the crossed-dt
split for this whole project, and every attempt to move it treated the failure
as a property of the data (displacement coverage) or of the horizon (the
K-curve). `runs/shrink.json` shows something neither predicts: `w16m8L4_K1`, a
266,729-parameter FNO -- **98x smaller** than the deployed 26,248,025-parameter
one -- scores a *lower* crossed-in-coverage terminal error than the deployed
anchor does. It was produced by making the model smaller.

**Why this needs its own script rather than `seed_level_test.py`.** That file
compares arms inside `runs/kcurve.json`, and the arms being compared here live
in two different files: the anchor is `K1_nv` in `runs/kcurve.json` (the
deployed architecture, 8 seeds) and the challenger is a config in
`runs/shrink.json`. The statistics are the same and are imported rather than
reimplemented -- `exact_two_sample` and `invert_to_interval`, seeds as the unit,
the interval obtained by inverting the exact permutation enumeration.

**Reported as an interval, not a p-value.** That rule was earned in this repo:
`runs/seed_level_test.json` measured every crossed interval 0.022-0.042 wide *at
every seed count including 8*, so a crossed p-value near 1 can bound nothing at
all, and a 3-seed crossed reading in this repo has reversed its sign three
separate times. Every row here carries the interval and whether it covers the
yardstick.

**Both arms must be stride-1** or the comparison mixes horizons: the terminal
state is at the same physical time for every stride, but the arms would then
differ in applications-per-wafer as well as in capacity, and the effect could
not be attributed. Enforced, not assumed.

Writes `runs/capacity_ood.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.seed_level_test import (exact_two_sample,  # noqa: E402
                                     invert_to_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcurve", default="runs/kcurve.json")
    ap.add_argument("--shrink", default="runs/shrink.json")
    ap.add_argument("--anchor", default="K1_nv")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--out", default="runs/capacity_ood.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="capacity_ood")
    kc = json.loads(Path(a.kcurve).read_text())
    sh = json.loads(Path(a.shrink).read_text())
    anc = kc["arms"][a.anchor]
    assert anc["stride"] == 1, f"anchor {a.anchor} is stride {anc['stride']}"

    # The deployed architecture's parameter count is not recorded in kcurve.json
    # (it predates the field), so it is constructed rather than typed in.
    from eot.operator import EtchOperator
    norm = json.loads(Path("data/norm.json").read_text())
    anc_params = EtchOperator(cond_dim=len(norm["cond_keys"]), width=64,
                              modes=20, n_layers=4).param_count()

    # The yardstick: the smallest structure this repo treats as a real effect,
    # so an interval width can be read against something rather than admired.
    yard = None
    try:
        yard = abs(kc["arms"]["K5_sm"]["in_distribution"]["terminal_step"]["point"]
                   - kc["arms"]["K1_nv"]["in_distribution"]["terminal_step"]["point"])
    except KeyError:
        pass

    rows = {}
    for name, cfg in sorted(sh["configs"].items()):
        if cfg["config"]["stride"] != 1:
            continue                      # see module docstring
        row = {
            "params": cfg["params_trained"],
            "params_ratio_vs_anchor": anc_params / cfg["params_trained"],
            "n_seeds_arm": cfg["n_seeds"], "n_seeds_anchor": anc["n_seeds"],
            "verdict_strength": "verdict" if min(cfg["n_seeds"],
                                                 anc["n_seeds"]) >= 8 else "screen",
        }
        for split in ("in_distribution", "crossed_in_coverage"):
            arm_seeds = np.array(cfg[split]["terminal_step"]["per_seed"], float)
            anc_seeds = np.array(anc[split]["terminal_step"]["per_seed"], float)
            test = exact_two_sample(arm_seeds, anc_seeds)
            iv = invert_to_interval(arm_seeds, anc_seeds)
            row[split] = {
                "arm_point": float(arm_seeds.mean()),
                "anchor_point": float(anc_seeds.mean()),
                "arm_seed_range": float(np.ptp(arm_seeds)),
                # The "every seed" reading needs the worst seed, and a document
                # that quotes it must read it from here rather than have it
                # typed in -- which is exactly what happened once in
                # scripts/weekend.py and is why this field exists.
                "arm_worst_seed": float(arm_seeds.max()),
                "anchor_worst_seed": float(anc_seeds.max()),
                "anchor_seed_range": float(np.ptp(anc_seeds)),
                # Negative = the SMALLER model is better.
                "mean_diff_arm_minus_anchor": float(arm_seeds.mean() - anc_seeds.mean()),
                "exact_test": test,
                "interval": iv,
                "arm_met_point": bool(arm_seeds.mean() <= a.target),
                "arm_met_every_seed": bool(arm_seeds.max() <= a.target),
                "arm_met_bootstrap_upper": bool(
                    cfg[split]["terminal_step"].get("hi", 1.0) <= a.target),
                "anchor_met_point": bool(anc_seeds.mean() <= a.target),
                "bootstrap_upper_arm": cfg[split]["terminal_step"].get("hi"),
                "bootstrap_upper_anchor": anc[split]["terminal_step"].get("hi"),
            }
            if yard:
                lo, hi = iv["lo"], iv["hi"]
                row[split]["interval_covers_yardstick"] = bool(
                    lo is not None and hi is not None
                    and lo <= yard <= hi or (lo is not None and hi is not None
                                             and lo <= -yard <= hi))
        rows[name] = row

    res = {
        "hypothesis": "H17: the deployed operator's crossed-dt (out-of-"
                      "distribution) failure is partly overfitting, so reducing "
                      "capacity improves crossed generalisation. Falsified if "
                      "the added seeds pull the small model's crossed mean above "
                      "the anchor's, which is what happened to the last three "
                      "3-seed crossed readings in this repo.",
        "protocol": {
            "anchor": a.anchor,
            "anchor_arch": "EtchOperator width=64 modes=20 layers=4 (deployed)",
            "anchor_params": anc_params,
            "unit_of_replication": "seed; per-seed arm means over the same "
                                   "trajectories",
            "test": "exact two-sided permutation on the difference of per-seed "
                    "means, imported from seed_level_test.py",
            "interval": "95% CI on the mean difference by inverting that test",
            "sign_convention": "mean_diff_arm_minus_anchor < 0 means the "
                               "SMALLER model is better",
            "stride_restriction": "stride-1 arms only, so capacity is the only "
                                  "thing that differs from the anchor",
            "target": a.target,
            "why_interval_not_p": "runs/seed_level_test.json measures every "
                                  "crossed interval 0.022-0.042 wide at every "
                                  "seed count including 8, so a crossed "
                                  "p-value can bound nothing",
        },
        "yardstick": {"value": yard,
                      "what": "|K5_sm - K1_nv| in-distribution, the smallest "
                              "gap this repo treats as a real effect"},
        "arms": rows,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"anchor {a.anchor}: {anc_params:,} params, {anc['n_seeds']} seeds")
    print(f"yardstick = {yard:.5f}" if yard else "yardstick unavailable")
    for k, v in rows.items():
        for split in ("in_distribution", "crossed_in_coverage"):
            s = v[split]
            iv = s["interval"]
            print(f"{k:14s} {split:20s} n={v['n_seeds_arm']} "
                  f"arm={s['arm_point']:.5f} anc={s['anchor_point']:.5f} "
                  f"diff={s['mean_diff_arm_minus_anchor']:+.5f} "
                  f"p={s['exact_test']['p']:.4f} "
                  f"CI=[{iv['lo']:+.5f},{iv['hi']:+.5f}] "
                  f"covers_yard={s.get('interval_covers_yardstick')}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
