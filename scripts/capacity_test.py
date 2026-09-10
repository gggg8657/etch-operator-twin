"""H17: is out-of-distribution error non-monotone in capacity? Exact test.

    python scripts/capacity_test.py --device cuda:0

**The observation.** `runs/shrink.json` puts `w16m8L4_K1` -- 266,729 parameters,
98x smaller than the deployed 26,248,025-parameter operator -- at **0.04289**
crossed-in-coverage terminal band rel-L2 against the anchor's **0.05433**, both
at 8 seeds. It is the only sub-0.05 crossed reading this repo has produced, and
it survived the seed extension that reversed the last two 3-seed crossed
readings.

**And the pattern is not "smaller is better", which is what H17 guessed.** At
stride 1 the three capacities read 10,897 -> 0.08213, 266,729 -> 0.04289,
26,248,025 -> 0.05433. Non-monotone, with the optimum in the middle. That is a
more specific claim than H17's and needs a test rather than three point
estimates, because the crossed split's per-seed spread is an order of magnitude
larger than the in-distribution one (`runs/seed_level_test.json`: crossed
intervals are 0.022-0.042 wide at every seed count including 8).

**So this script reports intervals, not p-values.** The rule this repo settled
on after a 3-seed null was read as evidence: a p-value at these seed counts
measures resolution, and only the interval says whether a null bounds anything.
Both are emitted, and every row carries `interval_covers_yardstick`.

**The confound is stated rather than hidden.** The three arms differ in width,
modes AND layers together (w64/m20/L4, w16/m8/L4, w8/m4/L2), so "capacity" here
is a single axis through a three-dimensional space and this cannot attribute the
effect to any one of them. Establishing non-monotonicity does not require
disentangling them; attributing it would, and no attribution is claimed.

Writes `runs/capacity_test.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from eot.data import TrajDataset  # noqa: E402
from eot.operator import build_from_cfg  # noqa: E402
from scripts.analyse_confound import per_step_displacement  # noqa: E402
from scripts.coverage_verdict import boot_ci, per_traj_both_readings  # noqa: E402
from scripts.kcurve_report import is_complete, sign_test  # noqa: E402
from scripts.seed_level_test import exact_two_sample, invert_to_interval  # noqa: E402


def collect(root: Path):
    """{name: (runs, params)} for every completed stride-1 arm, anchor included.

    The anchor is `runs/seed[0-9]*`, the same set `kcurve_report.arms` uses, so
    the number this script compares against is the same 0.05433 the K-curve
    reports rather than a re-fit of it.
    """
    out = {}
    anchor = [p for p in sorted(root.glob("seed[0-9]*"))
              if is_complete(p)
              and json.loads((p / "args.json").read_text()).get("stride", 1) == 1
              and not json.loads((p / "args.json").read_text()).get("blind")]
    if anchor:
        cfg = json.loads((anchor[0] / "args.json").read_text())
        out["anchor_w64m20L4"] = (anchor, cfg["params"])
    for p in sorted((root / "shrink").glob("*_K1_s*")):
        if not is_complete(p):
            continue
        cfg = json.loads((p / "args.json").read_text())
        key = f"w{cfg['width']}m{cfg['modes']}L{cfg['layers']}"
        out.setdefault(key, ([], cfg["params"]))
        out[key][0].append(p)
    return {k: (sorted(v[0]), v[1]) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--anchor", default="anchor_w64m20L4")
    ap.add_argument("--out", default="runs/capacity_test.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="capacity_test")
    root, data = Path(a.runs_root), Path(a.data)
    norm = json.loads((data / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    device = torch.device(a.device)

    tr = np.load(data / "train.npz")["sdf"]
    disp_tr = np.abs(per_step_displacement(tr).mean(axis=1))
    lo, hi = float(np.percentile(disp_tr, 1)), float(np.percentile(disp_tr, 99))
    cr = np.load(data / "test_crossed.npz")["sdf"]
    sel = (np.abs(per_step_displacement(cr).mean(axis=1)) >= lo) & \
          (np.abs(per_step_displacement(cr).mean(axis=1)) <= hi)

    ds_in = TrajDataset(data / "test.npz", norm, stride=1)
    ds_cr = TrajDataset(data / "test_crossed.npz", norm, stride=1)

    found = collect(root)
    per = {}
    for name, (runs, params) in sorted(found.items(), key=lambda kv: kv[1][1]):
        rows = {"in": [], "crossed": []}
        for r in runs:
            cfg = json.loads((r / "args.json").read_text())
            model = build_from_cfg(cfg, len(norm["cond_keys"])).to(device)
            model.load_state_dict(torch.load(r / "best.pt", map_location=device))
            model.eval()
            for tag, ds in (("in", ds_in), ("crossed", ds_cr)):
                _m, t, _ = per_traj_both_readings(model, ds, device, scale,
                                                  band_um, stride=1)
                rows[tag].append(t)
        M_in, M_cr = np.stack(rows["in"]), np.stack(rows["crossed"])
        per[name] = {
            "params": params, "n_seeds": len(runs),
            "runs": [str(x) for x in runs],
            "seeds": sorted(int(json.loads((x / "args.json").read_text())["seed"])
                            for x in runs),
            "verdict_strength": "verdict" if len(runs) >= 8 else "screen",
            "_in": M_in, "_cr": M_cr,
        }
        for tag, M, s in (("in_distribution", M_in, np.ones(M_in.shape[1], bool)),
                          ("crossed_in_coverage", M_cr, sel)):
            e = {"per_seed": [float(x) for x in M[:, s].mean(axis=1)],
                 "point": float(M[:, s].mean()),
                 "seed_range": float(np.ptp(M[:, s].mean(axis=1))) if len(runs) > 1 else None,
                 **{k: v for k, v in (boot_ci(M, s) or {}).items()
                    if k in ("lo", "hi", "n")}}
            e["met_point"] = bool(e["point"] <= a.target)
            e["met_every_seed"] = bool(max(e["per_seed"]) <= a.target)
            e["met_upper_ci"] = bool(e.get("hi", 1.0) <= a.target)
            per[name][tag] = e

    anc = per.get(a.anchor)
    if anc is None:
        raise SystemExit(f"anchor {a.anchor} not found among {sorted(per)}")

    # The yardstick: the smallest gap this repo treats as a real effect, taken
    # from the K-curve rather than invented here.
    yardstick = 0.00440
    tests = {}
    for name, row in per.items():
        if name == a.anchor:
            continue
        t = {}
        for tag, key in (("in_distribution", "_in"), ("crossed_in_coverage", "_cr")):
            s = np.ones(row[key].shape[1], bool) if tag == "in_distribution" else sel
            arm_seeds = row[key][:, s].mean(axis=1)
            anc_seeds = anc[key][:, s].mean(axis=1)
            iv = invert_to_interval(arm_seeds, anc_seeds)
            per_traj = row[key][:, s].mean(axis=0) - anc[key][:, s].mean(axis=0)
            t[tag] = {
                "mean_diff_vs_anchor": float(arm_seeds.mean() - anc_seeds.mean()),
                "seed_level_exact": exact_two_sample(arm_seeds, anc_seeds),
                "interval_95": {k: iv[k] for k in ("lo", "hi") if k in iv},
                "interval_width": (float(iv["hi"] - iv["lo"])
                                   if "lo" in iv and "hi" in iv else None),
                "interval_covers_yardstick": (
                    bool(iv.get("lo", -9) <= yardstick <= iv.get("hi", 9))
                    if "lo" in iv and "hi" in iv else None),
                "trajectory_paired_sign_test": sign_test(per_traj),
                "n_trajectories": int(s.sum()),
            }
        tests[name] = t

    curve = [{"name": k, "params": v["params"], "n_seeds": v["n_seeds"],
              "in_distribution": v["in_distribution"]["point"],
              "crossed_in_coverage": v["crossed_in_coverage"]["point"]}
             for k, v in sorted(per.items(), key=lambda kv: kv[1]["params"])]
    cr_vals = [c["crossed_in_coverage"] for c in curve]
    non_monotone = bool(len(cr_vals) >= 3
                        and min(cr_vals) not in (cr_vals[0], cr_vals[-1]))

    res = {
        "hypothesis": "H17: the deployed operator's crossed-split failure is "
                      "partly overfitting, so reducing capacity improves "
                      "out-of-distribution error. Guessed monotone; the point "
                      "estimates are not.",
        "protocol": {
            "metric": "terminal-step band rel-L2, coverage_verdict."
                      "per_traj_both_readings, stride 1 for every arm",
            "crossed_rule": {"disp_lo_um": lo, "disp_hi_um": hi,
                             "n_in_coverage": int(sel.sum()),
                             "n_total": int(sel.size)},
            "yardstick": yardstick,
            "yardstick_source": "|K5_sm - K1_nv| in-distribution, the smallest "
                                "gap runs/kcurve.json treats as a real effect",
            "unit_of_replication": "seed, for the exact test; trajectory, for "
                                   "the paired sign test. Both reported "
                                   "because they answer different questions.",
            "confound": "the arms differ in width, modes AND layers together, "
                        "so this is one axis through a three-dimensional space. "
                        "Non-monotonicity does not need them disentangled; "
                        "attribution would, and none is claimed.",
            "interval_rule": "a p-value at these seed counts measures "
                             "resolution. The interval is what bounds an "
                             "effect, so interval_covers_yardstick is on every "
                             "row.",
        },
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in per.items()},
        "tests_vs_anchor": tests,
        "capacity_curve": curve,
        "crossed_non_monotone_in_capacity": non_monotone,
        "best_crossed": min(curve, key=lambda c: c["crossed_in_coverage"]),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"{'arm':20s} {'params':>10s} {'n':>2s} {'in-dist':>8s} {'crossed':>8s}")
    for c in curve:
        print(f"{c['name']:20s} {c['params']:10d} {c['n_seeds']:2d} "
              f"{c['in_distribution']:8.5f} {c['crossed_in_coverage']:8.5f}")
    print(f"\nnon-monotone in capacity (crossed): {non_monotone}")
    for name, t in tests.items():
        c = t["crossed_in_coverage"]
        print(f"{name:20s} crossed diff {c['mean_diff_vs_anchor']:+.5f} "
              f"p={c['seed_level_exact']['p']:.4f} "
              f"iv=[{c['interval_95'].get('lo', float('nan')):+.5f},"
              f"{c['interval_95'].get('hi', float('nan')):+.5f}] "
              f"covers_yardstick={c['interval_covers_yardstick']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
