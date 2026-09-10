"""H7: the accuracy cost of shrinking, joined to the measured inference cost.

    python scripts/shrink_report.py

**Why the two halves must be in one file.** `runs/cnn_cost.json` and
`runs/cost_floor.json` price architectures and claim no accuracy; `runs/shrink`
trains architectures and knows no cost. Clause 2 needs a model that is *both*
under 277 us/wafer and under 0.05 rel-L2, and a frontier assembled by eye from
two files is exactly where a cost row and an accuracy row from different widths
get read as one model. This script scores the trained checkpoints and prices the
same configs through `cost_floor.time_model` -- same estimator, same fresh
subprocess, same verified single thread -- and emits one row per config.

**Cost is per WAFER, not per application.** An arm at stride K applies the
operator T/K times to advance one wafer through the full 10 dataset timesteps,
so per-wafer cost is `per_apply * applications_per_wafer`. This is the entire
reason the stride-10 arms are interesting: they pay one forward pass where the
stride-1 anchor pays ten. Reporting per-application cost would flatter every
high-stride row by a factor of K.

Accuracy is the repo's canonical clause-1 quantity: band rel-L2 via
`coverage_verdict.per_traj_both_readings`, reported as the terminal-step reading
(the only one comparable across stride, since every arm ends at the same
physical time) and as the mean over emitted states, on the in-distribution test
set and on the displacement-covered subset of the crossed-dt set.

**3 seeds is a screen, not a verdict** -- the repo's own rule, earned twice this
weekend. Every row carries `n_seeds` and `verdict_strength`, and the seed range
is printed beside the point estimate so a gap smaller than the spread cannot be
read as an effect.

Writes `runs/shrink.json`.
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

# Set from argv in main(); module-level so the collectors stay single-purpose.
SUBDIR = ["shrink"]
GLOB = ["w*_K*_s*"]

from eot import runlock  # noqa: E402
from eot.data import TrajDataset  # noqa: E402
from eot.operator import build_from_cfg  # noqa: E402
from scripts.analyse_confound import per_step_displacement  # noqa: E402
from scripts.cost_floor import time_model  # noqa: E402
from scripts.coverage_verdict import boot_ci, per_traj_both_readings  # noqa: E402
from scripts.kcurve_report import is_complete  # noqa: E402


# Every args.json field that distinguishes one architecture from another. A key
# that omits one of these MERGES TWO ARCHITECTURES INTO ONE SEED GROUP, and the
# `len({config_key(c)}) == 1` guard below cannot catch it because the guard calls
# this same function. That is not hypothetical: the first version of this file
# had no `specprop` branch, so specprop runs fell through to the multiscale
# branch, `modes_a` was absent from the key, and `m4_ma4_s1` and `m4_ma16_s1`
# -- 9,344 and 71,744 parameters -- were pooled and reported as one two-seed
# arm at 0.76796. `tests/test_report_keys.py` now asserts that changing any
# field named here changes the key.
ARCH_FIELDS = {
    "fno": ("width", "modes", "layers", "stride"),
    "multiscale": ("width", "modes", "layers", "stride", "width_full",
                   "scale", "n_local", "act"),
    "specprop": ("modes", "modes_a", "state_modes", "stride"),
}

# Fields added to an architecture AFTER runs of it already existed on disk.
# `ARCH_FIELDS` raises on a missing field on purpose -- defaulting is what let a
# key blind to `modes_a` pool a 9,344-parameter model and a 71,744-parameter one
# into a single seed group and average their means. But a field that did not
# exist when a run was launched is a different case from a field that was
# dropped: the run has a definite value for it, namely the behaviour before the
# flag existed. Those values go here, ONE PER LINE WITH THE REASON, and nothing
# else may be defaulted.
#
# `state_modes = 0`: every specprop arm trained before 2026-09-10 22:45 had a
# recipe-only additive head, which is exactly what `state_modes=0` builds. Same
# reasoning as `hermitian_closed=False` in `eot.operator.build_from_cfg`.
LEGACY_DEFAULTS = {"state_modes": 0}


def config_key(cfg: dict) -> str:
    """The config's name, derived from args.json rather than the directory name.

    Deriving the key from args.json and then asserting it against the directory
    name catches a run launched with one set of flags into a directory named for
    another -- which a sweep driver with a copy-pasted line does silently, and
    which would merge two architectures into one seed group.
    """
    arch = cfg.get("arch", "fno")
    if arch not in ARCH_FIELDS:
        raise ValueError(f"unknown arch {arch!r}; add it to ARCH_FIELDS")
    missing = [f for f in ARCH_FIELDS[arch]
               if cfg.get(f) is None and f not in LEGACY_DEFAULTS]
    if missing:
        raise ValueError(f"{arch} run is missing {missing} in args.json")
    if arch == "fno":
        return f"w{cfg['width']}m{cfg['modes']}L{cfg['layers']}_K{cfg['stride']}"
    if arch == "specprop":
        # `_sm{n}` is appended only when n > 0, so the keys of every arm scored
        # before the field existed are byte-identical to what they were and the
        # published rows keep their names. A legacy run and a new run explicitly
        # launched with `--state-modes 0` are the same architecture and SHOULD
        # share a key; anything with a state head must not share one with them.
        sm = cfg.get("state_modes") or 0
        tail = f"sm{sm}_" if sm else ""
        return f"sp_m{cfg['modes']}ma{cfg['modes_a']}_{tail}K{cfg['stride']}"
    sc = cfg["scale"]
    body = "pw" if sc == 0 else f"ms_s{sc}_w{cfg['width']}m{cfg['modes']}L{cfg['layers']}"
    return f"{body}_wf{cfg['width_full']}n{cfg['n_local']}_{cfg['act']}_K{cfg['stride']}"


def shrink_arms(root: Path) -> dict:
    """{config_name: [run dirs]} over completed arms only.

    Completeness is `kcurve_report.is_complete`, i.e. the repo's canonical
    guard. It is not optional here: the K-curve's entire "monotone trend" was
    once eight partial checkpoints, and this sweep queues 21 cells on one GPU,
    so mid-training checkpoints are the default state of the directory while it
    runs.
    """
    out: dict[str, list[Path]] = {}
    for p in sorted((root / SUBDIR[0]).glob(GLOB[0])):
        if not is_complete(p):
            continue
        cfg = json.loads((p / "args.json").read_text())
        key = config_key(cfg)
        if cfg.get("arch", "fno") == "fno":
            assert p.name.startswith(key + "_s"), \
                f"{p.name} does not match its own args.json config {key}"
        out.setdefault(key, []).append(p)
    return out


def incomplete(root: Path) -> list[dict]:
    rows = []
    for p in sorted((root / SUBDIR[0]).glob(GLOB[0])):
        if is_complete(p) or not (p / "args.json").exists():
            continue
        cfg = json.loads((p / "args.json").read_text())
        log = p / "log.jsonl"
        rows.append({
            "run": str(p),
            "epoch_lines": sum(1 for ln in log.read_text().splitlines() if ln.strip())
            if log.exists() else 0,
            "epochs_requested": cfg.get("epochs"),
            "has_checkpoint": (p / "best.pt").exists(),
        })
    return rows


def build_string(cfg, cond_dim):
    """The source line that reconstructs the architecture `cfg` names.

    This MUST stay in lockstep with `eot.operator.build_from_cfg`: that function
    builds the model a checkpoint is SCORED with, this string builds the model
    the same config is PRICED with, and a divergence reports one model's cost
    beside another model's accuracy.

    The previous version was an `if arch == "fno" ... else MultiScaleOperator`,
    so every `specprop` run was priced as a `MultiScaleOperator` -- and priced
    without error, because argparse writes its defaults for `width`, `layers`,
    `scale` and `width_full` into every args.json whether the architecture reads
    them or not. Nothing published came from it (both scoring runs used
    `--no-cost`), and it surfaced only because `modes=0` finally made the wrong
    build crash. So: an explicit branch per architecture and a `raise` on the
    fall-through, never a default. `tests/test_price_build.py` pins that the
    string and `build_from_cfg` agree on parameter count for every architecture.
    """
    arch = cfg.get("arch", "fno")
    if arch == "fno":
        return (f"from eot.operator import EtchOperator\n"
                f"M = EtchOperator(cond_dim={cond_dim}, width={cfg['width']}, "
                f"modes={cfg['modes']}, n_layers={cfg['layers']})")
    if arch == "multiscale":
        return (f"from eot.operator import MultiScaleOperator\n"
                f"M = MultiScaleOperator(cond_dim={cond_dim}, "
                f"width={cfg['width']}, modes={cfg['modes']}, "
                f"n_layers={cfg['layers']}, width_full={cfg['width_full']}, "
                f"scale={cfg['scale']}, n_local={cfg.get('n_local', 1)}, "
                f"act={cfg.get('act', 'gelu')!r})")
    if arch == "specprop":
        return (f"from eot.operator import SpectralPropagator\n"
                f"M = SpectralPropagator(cond_dim={cond_dim}, "
                f"modes={cfg['modes']}, modes_a={cfg.get('modes_a', 16)}, "
                f"hermitian_closed={cfg.get('hermitian_closed', False)!r}, "
                f"state_modes={cfg.get('state_modes', 0)})")
    raise ValueError(f"cannot price unknown arch {arch!r}; add a branch here "
                     f"AND to eot.operator.build_from_cfg")


def price(cfg, cond_dim, n_apply, rounds, budget_w, budget_c,
          solver_w, solver_c):
    """Per-wafer CPU cost of this config, same protocol as cost_floor.py.

    Builds whatever architecture args.json names. Pricing an FNO for a
    multiscale run would report a cost the checkpoint never had.
    """
    build = build_string(cfg, cond_dim)
    spec = dict(build=build, call="M(phi, cond)")
    warm = time_model(spec, n_rep=rounds, n_warm=3)
    colds = [time_model(spec, n_rep=1, n_warm=0) for _ in range(rounds)]
    cold_cpu = [c for r in colds for c in r["cpu"]]
    out = {"params": warm["params"], "applications_per_wafer": n_apply,
           "rounds": rounds}
    for tag, cpu, budget, solver in (("warm", warm["cpu"], budget_w, solver_w),
                                     ("cold", cold_cpu, budget_c, solver_c)):
        per_apply = float(np.median(cpu))
        per_wafer = per_apply * n_apply
        out[tag] = {
            "per_apply_cpu_s": per_apply,
            "per_wafer_cpu_s": per_wafer,
            "spread_factor": float(np.max(cpu) / np.min(cpu)) if cpu else None,
            "over_budget_factor": per_wafer / budget,
            "under_budget": bool(per_wafer <= budget),
            "speedup_vs_solver": solver / per_wafer,
            "n": len(cpu),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--speedup-target", type=float, default=1000.0)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--sym", default="runs/speed_symmetric.json")
    ap.add_argument("--out", default="runs/shrink.json")
    ap.add_argument("--subdir", default="shrink",
                    help="directory under --runs-root holding the arms")
    ap.add_argument("--glob", default="w*_K*_s*")
    ap.add_argument("--no-cost", action="store_true",
                    help="skip the CPU pricing (accuracy only)")
    a = ap.parse_args()

    SUBDIR[0], GLOB[0] = a.subdir, a.glob
    runlock.acquire(a.out, what="shrink_report")

    root, data = Path(a.runs_root), Path(a.data)
    norm = json.loads((data / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    cond_dim = len(norm["cond_keys"])
    device = torch.device(a.device)

    sym = json.loads(Path(a.sym).read_text())
    solver_w = sym["solver"]["marginal_warm"]["median_cpu"]
    solver_c = sym["solver"]["cold_single_wafer"]["median_cpu"]
    budget_w, budget_c = solver_w / a.speedup_target, solver_c / a.speedup_target

    # Displacement coverage of the crossed set, defined exactly as kcurve_report
    # defines it (1st-99th percentile of the training displacement), so the
    # crossed columns of the two files are the same subset of trajectories.
    tr = np.load(data / "train.npz")["sdf"]
    disp_tr = np.abs(per_step_displacement(tr).mean(axis=1))
    lo, hi = float(np.percentile(disp_tr, 1)), float(np.percentile(disp_tr, 99))
    cr = np.load(data / "test_crossed.npz")["sdf"]
    disp_cr = np.abs(per_step_displacement(cr).mean(axis=1))
    sel_cr = (disp_cr >= lo) & (disp_cr <= hi)

    found = shrink_arms(root)
    rows = {}
    for key, runs in sorted(found.items()):
        cfgs = [json.loads((r / "args.json").read_text()) for r in runs]
        K = cfgs[0]["stride"]
        assert len({config_key(c) for c in cfgs}) == 1, f"{key} mixes configs"
        ds_in = TrajDataset(data / "test.npz", norm, stride=K)
        ds_cr = TrajDataset(data / "test_crossed.npz", norm, stride=K)
        acc = {"in": {"mean": [], "term": []}, "crossed": {"mean": [], "term": []}}
        for r, cfg in zip(runs, cfgs):
            model = build_from_cfg(cfg, cond_dim).to(device)
            model.load_state_dict(torch.load(r / "best.pt", map_location=device))
            model.eval()
            for tag, ds in (("in", ds_in), ("crossed", ds_cr)):
                m, t, _ = per_traj_both_readings(model, ds, device, scale, band_um,
                                                 stride=K)
                acc[tag]["mean"].append(m)
                acc[tag]["term"].append(t)
        M_in, Mm_in = np.stack(acc["in"]["term"]), np.stack(acc["in"]["mean"])
        M_cr, Mm_cr = np.stack(acc["crossed"]["term"]), np.stack(acc["crossed"]["mean"])
        n_apply = len(ds_in.times)

        row = {
            "config": {k: cfgs[0].get(k) for k in
                       ("arch", "width", "modes", "layers", "stride",
                        "width_full", "scale", "n_local", "act")},
            "params_trained": cfgs[0]["params"],
            "epochs": sorted({int(c["epochs"]) for c in cfgs}),
            "applications_per_wafer": n_apply,
            "runs": [str(r) for r in runs],
            "seeds": [int(c["seed"]) for c in cfgs],
            "n_seeds": len(runs),
            "verdict_strength": "screen" if len(runs) < 8 else "verdict",
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
        for tag in ("in_distribution", "crossed_in_coverage"):
            e = row[tag]["terminal_step"]
            e["met_point"] = bool(e["point"] <= a.target)
            e["met_every_seed"] = bool(max(e["per_seed"]) <= a.target)
            e["met_upper_ci"] = bool(e.get("hi", 1.0) <= a.target)
        if not a.no_cost:
            row["cost"] = price(cfgs[0], cond_dim, n_apply, a.rounds,
                                budget_w, budget_c, solver_w, solver_c)
            row["clause2_met_warm"] = bool(
                row["cost"]["warm"]["speedup_vs_solver"] >= a.speedup_target)
            row["joint_met"] = bool(row["clause2_met_warm"]
                                    and row["in_distribution"]["terminal_step"]["met_point"])
        rows[key] = row

    joint = [k for k, v in rows.items() if v.get("joint_met")]
    res = {
        "hypothesis": "H7: shrinking the FNO costs accuracy. The decision rule, "
                      "written before the sweep ran: if the smallest arm already "
                      "misses rel-L2 <= 0.05 badly, the frontier is closed below "
                      "1000x for this family; if it holds, capacity is cheap here "
                      "and the cost-admissible architectures are worth building.",
        "protocol": {
            "accuracy": "band rel-L2, coverage_verdict.per_traj_both_readings; "
                        "terminal-step reading is the cross-stride-comparable one",
            "cost": "CPU-seconds on one verified thread, fresh subprocess per "
                    "config, cost_floor.time_model; PER WAFER = per_apply * "
                    "applications_per_wafer",
            "budget_source": f"{a.sym}: solver marginal_warm {solver_w:.4f} and "
                             f"cold_single_wafer {solver_c:.4f} CPU-s, / "
                             f"{a.speedup_target:g}",
            "budget_warm_s": budget_w, "budget_cold_s": budget_c,
            "accuracy_target": a.target, "speedup_target": a.speedup_target,
            "seeds_caveat": "3 seeds per config is a screen. No row here is a "
                            "verdict, and every row carries its seed range.",
        },
        "crossed_coverage_rule": {"disp_lo_um": lo, "disp_hi_um": hi,
                                  "n_in_coverage": int(sel_cr.sum()),
                                  "n_total": int(sel_cr.size)},
        "configs": rows,
        "incomplete": incomplete(root),
        "joint_pass_configs": joint,
        "verdict": ("no config in this sweep meets both clauses"
                    if not joint else f"joint pass: {joint}"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["configs"], indent=2)[:200])
    print(f"wrote {a.out}: {len(rows)} configs, joint pass {joint}")


if __name__ == "__main__":
    main()
