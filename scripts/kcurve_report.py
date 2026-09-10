"""H5: the accuracy-versus-horizon curve, on the reading that is comparable.

    python scripts/kcurve_report.py --device cuda:0

One application of the operator advances K dataset timesteps, so a wafer takes
T/K applications and the rollout compounds T/K times instead of T. The addendum
of 2026-09-10 predicted this attacks clause 1 and clause 2 together: fewer
compoundings should lower the terminal-step error, and fewer applications lower
the cost by K.

**Only the terminal-step reading is comparable across K.** Every arm ends at the
same physical time (`tests/test_stride.py` enforces stride | T), so the terminal
field is the same target for every arm. The mean-over-emitted-states reading
averages T/K states -- a K=10 arm averages one state and a K=1 arm averages ten
-- so it is computed, printed, and never used for a verdict.

Two splits, because clause 1 already has two fates on them:

* `test.npz`      -- in distribution, where clause 1 was MET at K=1 (0.01256
                     over 8 seeds, terminal 0.0188).
* `test_crossed.npz` restricted to the trajectories inside the training
                     displacement range (rule B, train p1-p99), which is where
                     the terminal reading FAILED at K=1 (0.0531-0.0665). This is
                     the reading the K-curve exists to move.

Verdict machinery, per `.overnight/RULES.md`: trajectories are the pairing unit
(every arm sees the same ones), so the test against the K=1 anchor is paired per
trajectory and reported as an exact sign test plus a 100k sign-flip
permutation test. Seed spread is reported for every arm, and an arm whose gap to
the anchor is smaller than the anchor's own seed range is called a screen.

Writes `runs/kcurve.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import TrajDataset, pair_starts  # noqa: E402
from eot.operator import EtchOperator  # noqa: E402
from scripts.analyse_confound import per_step_displacement  # noqa: E402
from scripts.coverage_verdict import boot_ci, per_traj_both_readings  # noqa: E402


def is_complete(p: Path) -> bool:
    """Has this arm finished training?

    **This guard was missing until 2026-09-10 and it contaminated every K-curve
    number this repo produced.** `arms()` required only `args.json` and
    `best.pt`, and this script scores `best.pt` itself rather than reading a
    committed `test_eval.json` -- so an arm that was still training, or that had
    been killed mid-training, was scored at whatever partial checkpoint happened
    to be on disk and entered the seed group as if it were a finished run.

    Two concrete instances in `runs/kcurve.json` as generated at 13:26:
    `K2_nv_s5`, killed by a process-group kill at epoch 24, scored 0.06705; and
    `K2_nv_s8`, four minutes into an 80-epoch run, scored 0.11902. The other six
    completed K2_nv seeds all sit between 0.02043 and 0.02291. Those two
    partial checkpoints moved the arm's mean from 0.0217 to 0.03958 and its seed
    range from 0.0025 to 0.09859, and they are the whole of the "bimodal seed
    distribution" I was about to write up as a training-instability finding.

    The bias does not cancel: a partial checkpoint always scores *worse*, and
    which arms get caught mid-training depends on the queue order, so arms with
    more seeds queued later are penalised more than arms with fewer.

    `done.json` is written by `runlock.mark_done` only after the epoch loop
    completes, which is exactly the property needed.
    """
    return (p / "args.json").exists() and (p / "best.pt").exists() \
        and (p / "done.json").exists()


def arms(root: Path) -> dict:
    """{(K, variant): [run dirs]}. K=1 is runs/seed1..8, which stride=1 is pinned
    to reproduce element-for-element, so the anchor is not re-fitted.

    Only *completed* arms are collected -- see `is_complete`."""
    out = {}
    anchor = sorted(p for p in root.glob("seed[0-9]*") if is_complete(p)
                    and json.loads((p / "args.json").read_text()).get("stride", 1) == 1
                    and not json.loads((p / "args.json").read_text()).get("blind"))
    if anchor:
        out[(1, "nv")] = anchor
    for p in sorted((root / "kcurve").glob("K*_s*")):
        if not is_complete(p):
            continue
        cfg = json.loads((p / "args.json").read_text())
        # The variant is in the directory name because three variants share one
        # (stride, overlap) pair: nv and sm both use non-overlapping starts and
        # differ only in epoch count, so args.json alone cannot separate them.
        var = p.name.split("_")[1]
        assert var in ("nv", "ov", "sm"), f"unknown variant in {p.name}"
        assert (var == "ov") == bool(cfg.get("overlap_pairs")), \
            f"{p.name} says {var} but overlap_pairs={cfg.get('overlap_pairs')}"
        out.setdefault((int(cfg["stride"]), var), []).append(p)
    return out


def incomplete_arms(root: Path) -> list[dict]:
    """Arms present on disk but not finished, so a reader can see what was left
    out rather than having to infer it from a seed count."""
    out = []
    for p in sorted((root / "kcurve").glob("K*_s*")):
        if is_complete(p) or not (p / "args.json").exists():
            continue
        log = p / "log.jsonl"
        n_ep = sum(1 for ln in log.read_text().splitlines() if ln.strip()) \
            if log.exists() else 0
        cfg = json.loads((p / "args.json").read_text())
        out.append({"run": str(p), "epoch_lines": n_ep,
                    "epochs_requested": cfg.get("epochs"),
                    "has_checkpoint": (p / "best.pt").exists(),
                    "why_excluded": "no done.json: still training or killed. "
                                    "Scoring its partial best.pt would enter an "
                                    "undertrained model into the seed group, "
                                    "which is what contaminated every K-curve "
                                    "number before 2026-09-10."})
    return out


def sign_test(d: np.ndarray) -> dict:
    """Exact two-sided sign test on paired differences (arm - anchor)."""
    d = d[d != 0]
    n = d.size
    if n == 0:
        return {"n": 0, "p": 1.0, "n_arm_better": 0}
    k = int((d < 0).sum())          # arm better than anchor
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return {"n": n, "n_arm_better": k, "p": float(min(1.0, 2 * tail / 2 ** n))}


def signflip_test(d: np.ndarray, n_perm: int = 100_000, seed: int = 0) -> dict:
    """Two-sided sign-flip permutation test on the mean paired difference.

    2^n is astronomically large at n>=137, so this is a resampled exact test and
    is labelled as one: the reported p is a Monte-Carlo estimate with the +1
    correction, not an enumeration."""
    rng = np.random.default_rng(seed)
    obs = float(np.mean(d))
    flips = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (flips * d).mean(axis=1)
    hits = int((np.abs(null) >= abs(obs)).sum())
    return {"mean_diff": obs, "n_perm": n_perm, "exact": False,
            "p": float((hits + 1) / (n_perm + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--out", default="runs/kcurve.json")
    a = ap.parse_args()

    root, data = Path(a.runs_root), Path(a.data)
    norm = json.loads((data / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    device = torch.device(a.device)

    tr = np.load(data / "train.npz")["sdf"]
    disp_tr = np.abs(per_step_displacement(tr).mean(axis=1))
    lo, hi = float(np.percentile(disp_tr, 1)), float(np.percentile(disp_tr, 99))
    cr = np.load(data / "test_crossed.npz")["sdf"]
    disp_cr = np.abs(per_step_displacement(cr).mean(axis=1))
    sel_cr = (disp_cr >= lo) & (disp_cr <= hi)

    found = arms(root)
    per_arm, term_by_arm = {}, {}
    for (K, var), runs in sorted(found.items()):
        ds_in = TrajDataset(data / "test.npz", norm, stride=K)
        ds_cr = TrajDataset(data / "test_crossed.npz", norm, stride=K)
        rows = {"in": {"mean": [], "term": []}, "crossed": {"mean": [], "term": []}}
        cfgs = []
        for r in runs:
            cfg = json.loads((r / "args.json").read_text())
            cfgs.append(cfg)
            model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                                 modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
            model.load_state_dict(torch.load(r / "best.pt", map_location=device))
            model.eval()
            for tag, ds in (("in", ds_in), ("crossed", ds_cr)):
                m, t, _ = per_traj_both_readings(model, ds, device, scale, band_um,
                                                 stride=K)
                rows[tag]["mean"].append(m)
                rows[tag]["term"].append(t)
        M_in = np.stack(rows["in"]["term"]); Mm_in = np.stack(rows["in"]["mean"])
        M_cr = np.stack(rows["crossed"]["term"]); Mm_cr = np.stack(rows["crossed"]["mean"])
        term_by_arm[(K, var)] = {"in": M_in, "crossed": M_cr}
        # The K=1 anchor was trained before args.json recorded the pair count,
        # so derive it for those from the same function the loader uses rather
        # than leaving the field blank or, worse, typing 9000 in from memory.
        n_pairs_default = tr.shape[0] * len(pair_starts(tr.shape[1] - 1, K, False))
        pairs = sorted({int(c.get("n_train_pairs", n_pairs_default)) for c in cfgs})
        per_arm[f"K{K}_{var}"] = {
            "stride": K, "variant": var,
            "variant_note": {
                "nv": "non-overlapping starts: T/K pairs per trajectory, 80 "
                      "epochs -- so a K arm also takes K times fewer gradient "
                      "steps than the anchor",
                "ov": "data-matched control: T-K+1 overlapping starts at 80 "
                      "epochs, so the pair count and the gradient-step count are "
                      "both restored to ~T at every K. Its inputs include "
                      "timesteps the rollout never visits.",
                "sm": "step-matched control: non-overlapping starts (the same "
                      "pairs and the same input distribution as nv) trained for "
                      "80*K epochs, so the gradient-step count matches the "
                      "anchor while the input distribution does not change. This "
                      "is the control that separates 'the K map is harder' from "
                      "'the K arm got K times fewer updates'.",
            }[var],
            "epochs": sorted({int(c["epochs"]) for c in cfgs}),
            "approx_gradient_steps": sorted({
                int(c["epochs"]) * -(-int(c.get("n_train_pairs", n_pairs_default))
                                     // int(c["batch"])) for c in cfgs}),
            "runs": [str(r) for r in runs], "n_seeds": len(runs),
            "seeds": [int(c["seed"]) for c in cfgs],
            "n_train_pairs": pairs,
            "applications_per_wafer": len(ds_in.times),
            "in_distribution": {
                "terminal_step": {
                    "per_seed": [float(x) for x in M_in.mean(axis=1)],
                    "point": float(M_in.mean()),
                    "seed_range": float(np.ptp(M_in.mean(axis=1))) if len(runs) > 1 else None,
                    **{k: v for k, v in (boot_ci(M_in, np.ones(M_in.shape[1], bool)) or {}).items()
                       if k in ("lo", "hi", "n")},
                },
                "mean_over_emitted_states_NOT_COMPARABLE_ACROSS_K": float(Mm_in.mean()),
            },
            "crossed_in_coverage": {
                "n_trajectories": int(sel_cr.sum()),
                "terminal_step": {
                    "per_seed": [float(x) for x in M_cr[:, sel_cr].mean(axis=1)],
                    "point": float(M_cr[:, sel_cr].mean()),
                    "seed_range": float(np.ptp(M_cr[:, sel_cr].mean(axis=1))) if len(runs) > 1 else None,
                    **{k: v for k, v in (boot_ci(M_cr, sel_cr) or {}).items()
                       if k in ("lo", "hi", "n")},
                },
                "mean_over_emitted_states_NOT_COMPARABLE_ACROSS_K": float(Mm_cr[:, sel_cr].mean()),
            },
        }
        for tag, key in (("in_distribution", "in"), ("crossed_in_coverage", "crossed")):
            e = per_arm[f"K{K}_{var}"][tag]["terminal_step"]
            e["met_point"] = bool(e["point"] <= a.target)
            e["met_every_seed"] = bool(max(e["per_seed"]) <= a.target)
            e["met_upper_ci"] = bool(e.get("hi") is not None and e["hi"] <= a.target)

    anchor = term_by_arm.get((1, "nv"))
    tests = {}
    if anchor is not None:
        anc = {"in": anchor["in"].mean(axis=0), "crossed": anchor["crossed"].mean(axis=0)}
        anc_range = per_arm["K1_nv"]["in_distribution"]["terminal_step"]["seed_range"]
        for key, M in term_by_arm.items():
            if key == (1, "nv"):
                continue
            name = f"K{key[0]}_{key[1]}"
            t = {}
            for tag, seln in (("in_distribution", None), ("crossed_in_coverage", sel_cr)):
                src = "in" if tag == "in_distribution" else "crossed"
                arm = M[src].mean(axis=0)
                d = (arm - anc[src]) if seln is None else (arm - anc[src])[seln]
                t[tag] = {
                    "paired_unit": "test trajectory, shared by every arm",
                    "mean_diff_vs_K1": float(np.mean(d)),
                    "n_trajectories_arm_better": int((d < 0).sum()),
                    "sign_test": sign_test(d),
                    "sign_flip": signflip_test(d),
                    "smaller_than_anchor_seed_range": bool(
                        anc_range is not None and abs(np.mean(d)) < anc_range),
                }
            tests[name] = t

    verdict_arms = {k: v for k, v in per_arm.items() if v["n_seeds"] >= 8}
    out = {
        "hypothesis": ("H5 horizon collapse: training the operator to advance K "
                       "dataset timesteps per application divides both the "
                       "application count and the number of compoundings by K, "
                       "which the 2026-09-10 addendum predicted would lower the "
                       "terminal-step error (clause 1) and raise the throughput "
                       "reading by K (clause 2)."),
        "protocol": {
            "comparable_reading": "terminal_step (same physical time for every K)",
            "incomparable_reading_reported_anyway": "mean over the T/K emitted states",
            "coverage_rule": {"name": "B_train_p1_p99", "lo": lo, "hi": hi,
                              "n_in": int(sel_cr.sum()), "n_total": int(sel_cr.size)},
            "target": a.target,
            "anchor": "runs/seed1..8 (stride=1, pinned identical to the historical "
                      "one-step dataset by tests/test_stride.py)",
            "seed_rule": ("8 seeds per arm plus an exact paired test before a "
                          "comparison is a verdict; fewer is a screen"),
            "completeness_rule": ("an arm is scored only if it carries done.json. "
                                  "This script evaluates best.pt directly, so "
                                  "without that rule a run still training, or "
                                  "killed mid-training, is scored at its partial "
                                  "checkpoint and joins the seed group as if "
                                  "finished -- see kcurve_report.is_complete"),
        },
        "excluded_incomplete_arms": incomplete_arms(Path(a.runs_root)),
        "arms": per_arm,
        "tests_vs_K1": tests,
        "n_arms_at_verdict_strength": len(verdict_arms),
        "arms_at_verdict_strength": sorted(verdict_arms),
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: {
        "n_seeds": v["n_seeds"],
        "pairs": v["n_train_pairs"],
        "apps": v["applications_per_wafer"],
        "in_term": round(v["in_distribution"]["terminal_step"]["point"], 5),
        "crossed_term": round(v["crossed_in_coverage"]["terminal_step"]["point"], 5),
        "met_in": v["in_distribution"]["terminal_step"]["met_point"],
        "met_crossed": v["crossed_in_coverage"]["terminal_step"]["met_point"],
    } for k, v in per_arm.items()}, indent=2))


if __name__ == "__main__":
    main()
