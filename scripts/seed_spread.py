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


def completed(d, cfg):
    """Is this run finished, written by exactly one process, and evaluated after
    the checkpoint it claims to describe?

    Counting lines in log.jsonl is not enough. `train.py` opens the log in
    APPEND mode, so two trainers sharing a directory interleave into one file
    and the line count reaches the epoch budget with neither model converged --
    which is how a 0.0400 rel-L2 belonging to no model got reported once. So:

      * `test_eval.json` must be at least as new as `best.pt`, always checked
        first, because an eval older than the checkpoint describes a model that
        is no longer on disk;
      * `done.json` (written only at the end of a clean run) is the proof;
      * failing that, the logged epochs must be exactly 0..epochs-1, each once.
        A duplicate epoch number is positive evidence of two writers.

    The staleness check is what caught `runs/base` on 2026-09-10. That directory
    had been raced by two trainers and excluded for logging 49/80 epochs; the
    surviving writer then finished the log to a clean 0..79 and kept training,
    so the duplicate-epoch and completeness tests both went green while
    `test_eval.json` still described a checkpoint from 14 minutes earlier.
    Admitting it moved the reported 4-seed range from 0.00114 to 0.03721 and the
    mean from 0.0126 to 0.0171 -- a 33x change in this repo's own noise floor,
    from a run whose number was already formally withdrawn (commit 7a9a66e).
    A guard that passes once is not a guard: the directory it protects can go
    stale afterwards.
    """
    epochs = cfg.get("epochs")
    best, ev = d / "best.pt", d / "test_eval.json"
    if best.exists() and ev.exists():
        lag = ev.stat().st_mtime - best.stat().st_mtime
        if lag < 0:
            return False, (f"test_eval.json is {-lag:.0f}s older than best.pt -- "
                           "it describes a checkpoint that has since been "
                           "overwritten, so the metric belongs to no model on disk")
    if (d / "done.json").exists():
        return True, "done.json"
    log = d / "log.jsonl"
    if not log.exists():
        return False, "no log.jsonl"
    eps = []
    for line in log.open():
        line = line.strip()
        if not line:
            continue
        try:
            eps.append(json.loads(line)["epoch"])
        except (json.JSONDecodeError, KeyError):
            return False, "unparseable log line (interleaved writers?)"
    if epochs is None:
        return False, "args.json has no epoch budget"
    if len(eps) != len(set(eps)):
        dup = sorted({e for e in eps if eps.count(e) > 1})
        return False, f"duplicate epoch records {dup[:5]} -- two writers shared this log"
    if sorted(eps) != list(range(epochs)):
        return False, f"logged {len(eps)}/{epochs} epochs, not a complete 0..{epochs - 1}"
    return True, "complete epoch sequence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*")
    ap.add_argument("--out", default="runs/seed_spread.json")
    a = ap.parse_args()

    runs, incomplete = [], []
    for d in sorted(Path().glob(a.glob)):
        cfg, ev = d / "args.json", d / "test_eval.json"
        if not (cfg.exists() and ev.exists()):
            continue
        c = json.loads(cfg.read_text())
        e = json.loads(ev.read_text())
        # Refuse runs that have not finished their epoch budget. A checkpoint
        # from a still-training job evaluates to whatever it happens to be worth
        # at that moment; pooled with converged runs it reads as a seed outlier.
        # This happened: a run 26 epochs into an 80-epoch budget scored 0.0491
        # against 0.0126-0.0131 for four converged seeds, and was very nearly
        # reported as a 3.8x seed effect.
        ok, why = completed(d, c)
        if not ok:
            incomplete.append({"run": str(d), "reason": why,
                               "epochs_requested": c.get("epochs")})
            continue
        k = e.get("kpi_clause_rel_l2", {})
        runs.append({
            "run": str(d), "seed": c.get("seed"), "modes": c.get("modes"),
            "width": c.get("width"), "layers": c.get("layers"),
            "epochs": c.get("epochs"), "params": c.get("params"),
            "rollout_steps": c.get("rollout_steps"),
            "blind": bool(c.get("blind", False)),
            "band_rel_l2": k.get("value"), "terminal": k.get("value_terminal_step"),
            "one_step_band": e.get("one_step", {}).get("op", {}).get("band", {}).get("mean"),
        })
    # group by everything except seed
    groups = {}
    for r in runs:
        # `blind` MUST be in the key. It was not, and the recipe-blind ablation
        # (0.405) was pooled with the conditioned runs (0.013) and reported as a
        # 31x seed outlier -- an ablation result read as instability.
        key = (r["modes"], r["width"], r["layers"], r["epochs"], r["rollout_steps"],
               r["blind"])
        groups.setdefault(key, []).append(r)

    out = {"n_runs": len(runs), "runs": runs, "groups": [],
           "excluded_incomplete": incomplete}
    for key, g in groups.items():
        vals = [r["band_rel_l2"] for r in g if r["band_rel_l2"] is not None]
        entry = {
            "config": dict(zip(["modes", "width", "layers", "epochs", "rollout_steps",
                                "blind"], key)),
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

    # the conditioning ablation, if both arms are present at matched config
    cond = {}
    for g in out["groups"]:
        c = dict(g["config"])
        b = c.pop("blind")
        cond.setdefault(tuple(sorted(c.items())), {})[b] = g
    for cfg, arms in cond.items():
        if True in arms and False in arms and arms[True]["band_rel_l2"] and arms[False]["band_rel_l2"]:
            import numpy as _np
            out.setdefault("conditioning_ablation", []).append({
                "config": dict(cfg),
                "conditioned_mean": float(_np.mean(arms[False]["band_rel_l2"])),
                "conditioned_seeds": arms[False]["seeds"],
                "blind_mean": float(_np.mean(arms[True]["band_rel_l2"])),
                "blind_seeds": arms[True]["seeds"],
                "ratio": float(_np.mean(arms[True]["band_rel_l2"])
                               / _np.mean(arms[False]["band_rel_l2"])),
                "note": ("Same architecture and capacity, recipe conditioning removed. "
                         "This is a far stronger test than beating the recipe-blind "
                         "*null*, because the blind model is free to fit everything "
                         "except the recipe."),
            })

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
