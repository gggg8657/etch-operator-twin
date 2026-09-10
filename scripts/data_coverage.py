"""Does mixing the independent-dt set actually widen displacement coverage?

    python scripts/data_coverage.py

`data/train_indep.npz` (735 trajectories, `dt_mode: independent`) and
`scripts/mix_data.py` were both built for H2 — the claim that the crossed-split
failure is a *coverage* failure and that mixing in trajectories whose timestep
was drawn without reference to the recipe would widen the covered interval. That
claim has sat in this repo untested and the data unused: no run's `args.json`
mentions `indep`, and `data_mixed/` was never created.

This script is the **gate** on spending GPU time on it. `mix_data.py`'s docstring
asserts the widening; if the independent set's per-step displacement range does
not in fact extend past the adaptive one, mixing cannot fix a coverage failure
and the training arm is dead before it starts. Cheap to check, and no model is
involved.

Note what does *not* differ: the marginal **dt** ranges are nearly identical
(adaptive 0.0599–1.0, independent 0.0652–0.9865). What the adaptive protocol
constrains is not dt but the *product* — it chooses dt per recipe to hit a target
depth, so a fast recipe never gets a long timestep. Decoupling lets fast recipes
pair with long timesteps, and per-step displacement is where that shows up.

Writes `runs/data_coverage.json`.
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
    """Per-trajectory mean per-step displacement, µm. The quantity the coverage
    rule in `scripts/coverage_verdict.py` is defined on."""
    return np.abs(per_step_displacement(np.load(npz)["sdf"]).mean(axis=1))


def summ(v: np.ndarray) -> dict:
    return {"n": int(v.size), "min": float(v.min()), "max": float(v.max()),
            "p1": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)),
            "median": float(np.median(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/data_coverage.json")
    ap.add_argument("--n-per-source", type=int, default=450,
                    help="trajectories per source in the size-matched mix, so the "
                         "matched arm has the same trajectory count as the anchor")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runlock.acquire(a.out, what="data_coverage")
    d = Path(a.data)
    sets = {k: disp(d / f"{k}.npz") for k in
            ("train", "train_indep", "test", "test_crossed")}

    rng = np.random.default_rng(a.seed)
    ia = rng.permutation(sets["train"].size)[:a.n_per_source]
    ii = rng.permutation(sets["train_indep"].size)[:a.n_per_source]
    sets["mixed_all"] = np.concatenate([sets["train"], sets["train_indep"]])
    sets["mixed_matched"] = np.concatenate([sets["train"][ia],
                                            sets["train_indep"][ii]])

    crossed, in_dist = sets["test_crossed"], sets["test"]
    rules = {}
    for name in ("train", "train_indep", "mixed_all", "mixed_matched"):
        v = sets[name]
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        rules[name] = {
            "lo": lo, "hi": hi,
            "frac_crossed_inside": float(((crossed >= lo) & (crossed <= hi)).mean()),
            "n_crossed_inside": int(((crossed >= lo) & (crossed <= hi)).sum()),
            "n_crossed_total": int(crossed.size),
            "frac_test_inside": float(((in_dist >= lo) & (in_dist <= hi)).mean()),
        }

    ad, mx = rules["train"], rules["mixed_matched"]
    res = {
        "question": "mix_data.py asserts that adding independently-drawn "
                    "timesteps widens the covered per-step-displacement "
                    "interval. Does it, and by enough to cover the crossed "
                    "split?",
        "why_this_gates_a_run": "if the independent set's displacement range "
                                "does not extend past the adaptive one, mixing "
                                "cannot fix a coverage failure and the training "
                                "arm is dead before it starts. No model here.",
        "quantity": "per-trajectory mean per-step displacement (µm), the same "
                    "quantity scripts/coverage_verdict.py defines its coverage "
                    "rule on",
        "coverage_rule": "train-split p1..p99, as in coverage_verdict.py",
        "size_matched_mix": {"n_per_source": a.n_per_source, "seed": a.seed,
                             "note": "same trajectory count as the anchor's "
                                     "train split, so a mixed arm built this way "
                                     "differs from the anchor ONLY in the "
                                     "dt-recipe coupling, not in data volume"},
        "distributions": {k: summ(v) for k, v in sets.items()},
        "dt_ranges": {k: {"min": float(np.load(d / f"{k}.npz")["dt"].min()),
                          "max": float(np.load(d / f"{k}.npz")["dt"].max())}
                      for k in ("train", "train_indep")},
        "rules": rules,
        "verdict": {
            "adaptive_covers_crossed": ad["frac_crossed_inside"],
            "matched_mix_covers_crossed": mx["frac_crossed_inside"],
            "all_mix_covers_crossed": rules["mixed_all"]["frac_crossed_inside"],
            "gate_open": bool(mx["frac_crossed_inside"] > ad["frac_crossed_inside"] + 0.10),
            "reading": (
                "OPEN: a size-matched mix raises the fraction of crossed "
                "trajectories inside the trained displacement range from "
                f"{ad['frac_crossed_inside']:.3f} to {mx['frac_crossed_inside']:.3f}, "
                "so the coverage failure H2 named is addressable by the data "
                "already on disk and the training arm is worth running."
                if mx["frac_crossed_inside"] > ad["frac_crossed_inside"] + 0.10 else
                "CLOSED: mixing does not materially widen coverage of the "
                "crossed split, so mix_data.py's premise is wrong and no GPU "
                "time should be spent on the arm."),
            "in_distribution_cost": (
                "the matched mix still covers "
                f"{mx['frac_test_inside']:.3f} of the in-distribution test split "
                f"against the adaptive rule's {ad['frac_test_inside']:.3f} — "
                "widening the interval cannot lose in-distribution coverage, but "
                "it does spend half the training trajectories on a different "
                "regime, and whether THAT costs in-distribution accuracy is a "
                "question for the run, not for this file"),
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"distributions": res["distributions"],
                      "rules": res["rules"], "verdict": res["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
