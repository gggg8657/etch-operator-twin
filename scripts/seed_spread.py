"""Run-to-run spread, which decides whether any comparison in this repo is a verdict.

    python scripts/seed_spread.py

The weekend's standing lesson is that two identical invocations of one pipeline
can differ by more than the effects being reported. So before any configuration
is called better than another, this measures what the pipeline's own noise floor
is: the spread of the headline metric across runs that differ only in seed.

Writes `runs/seed_spread.json`. Runs that differ in more than seed are listed
separately -- pooling them would measure configuration and seed together and
report the sum as noise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*")
    ap.add_argument("--out", default="runs/seed_spread.json")
    a = ap.parse_args()

    runs = []
    for d in sorted(Path().glob(a.glob)):
        cfg, ev = d / "args.json", d / "test_eval.json"
        if not (cfg.exists() and ev.exists()):
            continue
        c = json.loads(cfg.read_text())
        e = json.loads(ev.read_text())
        k = e.get("kpi_clause_rel_l2", {})
        runs.append({
            "run": str(d), "seed": c.get("seed"), "modes": c.get("modes"),
            "width": c.get("width"), "layers": c.get("layers"),
            "epochs": c.get("epochs"), "params": c.get("params"),
            "rollout_steps": c.get("rollout_steps"),
            "band_rel_l2": k.get("value"), "terminal": k.get("value_terminal_step"),
            "one_step_band": e.get("one_step", {}).get("op", {}).get("band", {}).get("mean"),
        })
    # group by everything except seed
    groups = {}
    for r in runs:
        key = (r["modes"], r["width"], r["layers"], r["epochs"], r["rollout_steps"])
        groups.setdefault(key, []).append(r)

    out = {"n_runs": len(runs), "runs": runs, "groups": []}
    for key, g in groups.items():
        vals = [r["band_rel_l2"] for r in g if r["band_rel_l2"] is not None]
        entry = {
            "config": dict(zip(["modes", "width", "layers", "epochs", "rollout_steps"], key)),
            "n_seeds": len(g), "seeds": [r["seed"] for r in g],
            "band_rel_l2": vals,
        }
        if len(vals) >= 2:
            entry.update({
                "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "cv": float(np.std(vals, ddof=1) / np.mean(vals)),
            })
        out["groups"].append(entry)

    multi = [g for g in out["groups"] if g["n_seeds"] >= 2]
    allv = [r["band_rel_l2"] for r in runs if r["band_rel_l2"] is not None]
    out["verdict"] = {
        "largest_within_config_range": (max(g["range"] for g in multi) if multi else None),
        "max_seeds_in_any_config": max((g["n_seeds"] for g in out["groups"]), default=0),
        "across_all_runs_range": (float(max(allv) - min(allv)) if len(allv) > 1 else None),
        "across_all_runs_spread_note": (
            "This pools configurations, so it is an upper bound on seed noise and a "
            "lower bound on nothing. It is reported because if even the pooled range "
            "is tiny, no configuration in the set is distinguishable from any other."),
        "eight_seed_rule": (
            "The standing rule is 8 seeds per arm plus an exact test before a "
            "comparison is a verdict. Below that, a difference is a screen."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
