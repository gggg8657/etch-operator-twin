"""H13's pre-check: can the independent-dt data cover the crossed split at all?

    python scripts/coverage_gap.py

Clause 1c localised the crossed-split failure to **displacement coverage**: the
operator degrades where per-step displacement leaves the range the training data
covers. `data/train_indep.npz` (735 trajectories, `dt_mode: independent`) was
generated for exactly that reason and has never been trained on. Before spending
GPU on it, this measures whether it would widen coverage — if it does not, the
route is dead and no run is needed.

Note what is *not* different: the marginal dt ranges nearly coincide (adaptive
0.0599–1.0000, independent 0.0652–0.9865). The adaptive protocol chooses dt as a
function of the recipe, so displacement ≈ rate(recipe) × dt is confined; drawing
dt independently lets a fast recipe meet a long timestep and produces
displacements the adaptive protocol cannot generate. So the widening, if any, is
in *displacement*, not in dt, and that is what gets measured.

**The protocol point this file exists to fix in advance.** The coverage rule is
derived from the training set, so an arm trained on mixed data has a *wider* rule
and would be scored on a larger subset of the crossed split. Comparing
"crossed-in-coverage under the adaptive rule" against "crossed-in-coverage under
the mixed rule" is not a comparison — it is the same number measured on
different sets, and improving it that way would be exactly the silent
test-loosening the rules forbid. Every arm must therefore be reported on:

* the **adaptive-coverage subset**, which is what every crossed figure in this
  repo currently refers to, so old and new arms are directly comparable;
* the **whole crossed split**, which is protocol-free and comparable by
  construction;

with each arm's own coverage fraction reported as context and never as the
verdict set.

Writes `runs/coverage_gap.json`.
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
from scripts.analyse_confound import per_step_displacement  # noqa: E402


def disp(npz: Path) -> np.ndarray:
    """Mean absolute per-step displacement per trajectory, in um."""
    return np.abs(per_step_displacement(np.load(npz)["sdf"]).mean(axis=1))


def summarise(v: np.ndarray) -> dict:
    return {"n": int(v.size), "p1": float(np.percentile(v, 1)),
            "p99": float(np.percentile(v, 99)), "min": float(v.min()),
            "max": float(v.max()), "median": float(np.median(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/coverage_gap.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="coverage_gap")
    d = Path(a.data)
    sets = {k: disp(d / f"{k}.npz") for k in
            ("train", "train_indep", "test", "test_crossed")}
    mixed = np.concatenate([sets["train"], sets["train_indep"]])

    rules = {
        "adaptive_p1_p99": (float(np.percentile(sets["train"], 1)),
                            float(np.percentile(sets["train"], 99))),
        "independent_p1_p99": (float(np.percentile(sets["train_indep"], 1)),
                               float(np.percentile(sets["train_indep"], 99))),
        "mixed_p1_p99": (float(np.percentile(mixed, 1)),
                         float(np.percentile(mixed, 99))),
    }
    cov = {}
    for name, (lo, hi) in rules.items():
        c = sets["test_crossed"]
        t = sets["test"]
        cov[name] = {
            "lo": lo, "hi": hi,
            "frac_crossed_inside": float(((c >= lo) & (c <= hi)).mean()),
            "n_crossed_inside": int(((c >= lo) & (c <= hi)).sum()),
            "n_crossed_total": int(c.size),
            "frac_test_inside": float(((t >= lo) & (t <= hi)).mean()),
        }

    res = {
        "question": "would training on the independent-dt data widen displacement "
                    "coverage of the crossed split enough to matter? Measured "
                    "before any GPU is spent, because if it would not, the route "
                    "is dead.",
        "protocol": {
            "quantity": "mean absolute per-step displacement per trajectory (um), "
                        "the quantity clause 1c localised the crossed failure to",
            "dt_ranges_nearly_coincide": {
                "note": "the independent set does NOT widen the marginal dt range; "
                        "it decouples dt from the recipe, and displacement "
                        "= rate(recipe) x dt is what widens",
                "adaptive": [float(np.load(d / "train.npz")["dt"].min()),
                             float(np.load(d / "train.npz")["dt"].max())],
                "independent": [float(np.load(d / "train_indep.npz")["dt"].min()),
                                float(np.load(d / "train_indep.npz")["dt"].max())],
            },
            "comparability_rule": (
                "the coverage rule is derived from the training set, so a "
                "mixed-data arm has a WIDER rule and would be scored on a larger "
                "subset of the crossed split. Every arm must be reported on (a) "
                "the adaptive-coverage subset, which is what every crossed figure "
                "in this repo currently refers to, and (b) the whole crossed "
                "split. An arm's own coverage is context, never the verdict set."),
        },
        "displacement": {k: summarise(v) for k, v in sets.items()},
        "displacement_mixed": summarise(mixed),
        "coverage_of_crossed_split": cov,
    }
    ad = cov["adaptive_p1_p99"]["frac_crossed_inside"]
    mx = cov["mixed_p1_p99"]["frac_crossed_inside"]
    res["verdict"] = {
        "frac_crossed_covered_by_adaptive": ad,
        "frac_crossed_covered_by_mixed": mx,
        "gain": mx - ad,
        "route_is_worth_a_run": bool(mx - ad > 0.10),
        "reading": (
            f"The adaptive training set covers {ad:.1%} of the crossed split's "
            f"displacements; mixing in the independent-dt set covers {mx:.1%}. "
            f"The crossed failure was localised to displacement coverage, so this "
            f"is the direct attack on it and is worth the GPU."
            if mx - ad > 0.10 else
            f"Mixing moves crossed coverage only from {ad:.1%} to {mx:.1%}. That "
            f"cannot fix a coverage failure, so the route is dead and no arm is "
            f"trained on this data."),
        "what_it_does_not_show": (
            "that wider coverage will translate into a lower crossed error. It "
            "shows only that the training distribution would contain the "
            "displacements the crossed split asks for. Whether the operator then "
            "learns them is the experiment."),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"displacement": res["displacement"],
                      "coverage_of_crossed_split": cov,
                      "verdict": res["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
