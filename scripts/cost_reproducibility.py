"""Is a clause-2 cost reading reproducible across invocations?

    python scripts/cost_reproducibility.py --config specprop_m4_ma64_K10 --invocations 12

**Why this exists.** `specprop_m4_ma64` was priced three times on 2026-09-10 and
read 426.4, 401.8 and 530.6 us/wafer -- 1.32x on an unchanged architecture,
giving 1187.7x, 1283.8x and 958.7x against the same clause. Two of those pass
clause 2 and one fails it, so the verdict is currently a property of which
invocation is quoted.

**What the three readings already rule out, and what they do not.**

* NOT the denominator. The solver cost is measured in the same invocation and
  read 0.5064, 0.5158 and 0.5087 CPU-s -- a 1.9% spread. The load does not hit
  both sides of the ratio, so the ratio inherits the operator's full variance.
  A first version of this analysis said "both move with load"; that was wrong
  and the measurement above is what corrects it.
* NOT within-round noise. Each row carries a range over its rounds, and h22's
  [858-1057x] is DISJOINT from h21's [1169-1401x]. Something moves between
  invocations that does not move within one.
* NOT explained by loadavg, which is recorded and is not monotone with the
  reading: loadavg 481 gave 426.4 us and loadavg 430 gave 530.6 us. Load is a
  candidate cause -- a workload of four tiny dispatches is far more exposed to
  cache and memory-bandwidth contention than a 0.5-second compute-bound solve --
  but three points with a non-monotone ordering do not establish it, and this
  script does not assume it.

So this measures the between-invocation distribution directly rather than
arguing about its cause: N fully independent invocations of the SAME pricing,
each a fresh subprocess with its own warmup, each recording its own loadavg, and
the output is the distribution and its relationship to load. A clause-2 verdict
needs a distribution whose whole range sits on one side of the threshold.

**A denominator error I made in this very script, kept as the reason for the
guard.** Its first version defaulted to `runs/speed_symmetric.json`'s 0.2767
CPU-s because that is the flag `cost_floor.py` takes. That is the retired
fixed-2.0-minute reading, not the workload the operator is scored on (0.5158
CPU-s), and it made every row read 1.86x too slow -- reporting "0/14 invocations
meet 1000x, FAIL at every invocation" when the corrected range straddles the
threshold. The between-invocation SPREAD was unaffected, because a constant
denominator cancels out of a ratio of two readings; the VERDICT was entirely an
artefact. The script now refuses that value by assertion rather than by comment.

Writes `runs/cost_repro_<config>.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.arch_cost import SPECS  # noqa: E402
from scripts.cost_floor import time_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="specprop_m4_ma64_K10",
                    help="a key of scripts.arch_cost.SPECS")
    ap.add_argument("--invocations", type=int, default=12,
                    help="independent pricings; each is a fresh subprocess")
    ap.add_argument("--rounds", type=int, default=16,
                    help="timed rounds WITHIN each invocation, so the "
                         "within-invocation spread is measured alongside the "
                         "between-invocation one")
    ap.add_argument("--workload", default="runs/arch_cost_h21.json",
                    help="an arch_cost JSON, whose "
                         "protocol.budget_provenance.denominators_cpu_s carries "
                         "the WORKLOAD-MATCHED solver cost")
    ap.add_argument("--denominator", default="terminal_one_apply",
                    choices=("terminal_one_apply",
                             "all_frames_ten_applies_excl_raster"),
                    help="`fixed_duration` is deliberately NOT offered; see the "
                         "module docstring")
    ap.add_argument("--target", type=float, default=1000.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.config not in SPECS:
        raise SystemExit(f"unknown config {a.config!r}; "
                         f"choose from {sorted(SPECS)[:8]} ...")
    spec = SPECS[a.config]
    out = a.out or f"runs/cost_repro_{a.config}.json"
    runlock.acquire(out, what="cost_reproducibility")

    # The denominator is held FIXED at the stored value on purpose. This script
    # asks one question -- how reproducible is the operator reading -- and
    # re-measuring the solver each time would fold its own variance in and make
    # the answer a property of two things at once. runs/arch_cost_h2*.json
    # already establishes the solver side is stable to 1.9%.
    dens = (json.loads(Path(a.workload).read_text())
            ["protocol"]["budget_provenance"]["denominators_cpu_s"])
    solver = dens[a.denominator]
    # THE GUARD, AND WHY IT IS AN ASSERTION RATHER THAN A COMMENT. The first
    # version of this script defaulted to
    # `runs/speed_symmetric.json -> solver.marginal_warm.median_cpu` = 0.2767
    # CPU-s, because that is the flag `cost_floor.py` carries. That is the
    # FIXED-2.0-MINUTE denominator, which this repo retired: the dataset's
    # trajectories carry a per-recipe dt whose median is 0.262, so the median
    # test wafer etches for 2.62 minutes and the like-for-like cost of what one
    # operator application produces is 0.5158 CPU-s. Dividing by 0.2767 made
    # every reading read 1.86x too slow and turned a straddling verdict into
    # "FAIL at every invocation". `tests/test_workload.py` exists to stop that
    # denominator being reinstated and I reinstated it anyway by copying a flag.
    assert abs(solver - 0.2767124970000001) > 1e-6, (
        "that is the fixed-duration denominator, which is not the workload the "
        "operator is scored on")
    assert solver > 0.4, (
        f"denominator {solver} is implausibly small for a workload-matched "
        f"solver cost; the retired fixed-duration reading is 0.2767")

    invocations = []
    for i in range(a.invocations):
        before = os.getloadavg()[0]
        t0 = time.time()
        r = time_model(spec, n_rep=a.rounds, n_warm=3)
        cpu = np.asarray(r["cpu"], dtype=float)
        after = os.getloadavg()[0]
        invocations.append({
            "i": i,
            "loadavg_before": before, "loadavg_after": after,
            "wall_s": time.time() - t0,
            "params": r["params"],
            "median_per_wafer_s": float(np.median(cpu)),
            "min_s": float(cpu.min()), "max_s": float(cpu.max()),
            "within_spread_factor": float(cpu.max() / cpu.min()),
            "speedup": float(solver / np.median(cpu)),
        })
        print(f"  [{i + 1}/{a.invocations}] load {before:6.1f}  "
              f"{np.median(cpu) * 1e6:7.1f} us  "
              f"{solver / np.median(cpu):8.1f}x", flush=True)

    sp = np.array([v["speedup"] for v in invocations])
    us = np.array([v["median_per_wafer_s"] for v in invocations]) * 1e6
    ld = np.array([v["loadavg_before"] for v in invocations])
    # Spearman without scipy: Pearson on the ranks.
    def _rank(x):
        order = np.argsort(x)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(x))
        return r
    rho = (float(np.corrcoef(_rank(us), _rank(ld))[0, 1])
           if len(us) > 2 and ld.std() > 0 else None)

    verdict = ("PASS at every invocation" if sp.min() >= a.target else
               "FAIL at every invocation" if sp.max() < a.target else
               "STRADDLES the threshold: the verdict depends on which "
               "invocation is quoted")

    res = {
        "question": "Is the clause-2 cost reading for one architecture "
                    "reproducible across independent invocations?",
        "config": a.config,
        "build": spec["build"],
        "protocol": {
            "invocations": a.invocations, "rounds_per_invocation": a.rounds,
            "estimator": "CPU-seconds per wafer, one verified thread, fresh "
                         "subprocess per invocation, 3 untimed warm rollouts",
            "denominator_cpu_s": solver,
            "denominator_source":
                f"{a.workload} budget_provenance.denominators_cpu_s"
                f"[{a.denominator}] = {solver:.6f} CPU-s -- the "
                "WORKLOAD-MATCHED cost of producing what one operator "
                "application produces, held FIXED so this measures the "
                "operator side alone. NOT the retired fixed-duration 0.2767.",
            "speedup_target": a.target,
        },
        "invocations": invocations,
        "summary": {
            "median_us": float(np.median(us)),
            "min_us": float(us.min()), "max_us": float(us.max()),
            "between_invocation_spread_factor": float(us.max() / us.min()),
            "worst_within_invocation_spread_factor":
                float(max(v["within_spread_factor"] for v in invocations)),
            "speedup_median": float(np.median(sp)),
            "speedup_min": float(sp.min()), "speedup_max": float(sp.max()),
            "n_invocations_meeting_target": int((sp >= a.target).sum()),
            "spearman_us_vs_loadavg": rho,
            "loadavg_range": [float(ld.min()), float(ld.max())],
        },
        "verdict": verdict,
    }
    Path(out).write_text(json.dumps(res, indent=2))
    print(f"\n{a.config}: {us.min():.1f}-{us.max():.1f} us "
          f"({us.max() / us.min():.2f}x), {sp.min():.1f}-{sp.max():.1f}x, "
          f"{(sp >= a.target).sum()}/{len(sp)} invocations meet {a.target:g}x")
    print(f"spearman(us, loadavg) = {rho}")
    print(verdict)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
