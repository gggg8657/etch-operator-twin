"""Score a trained operator on the held-out test split.

    python scripts/eval.py --run runs/base

Writes `<run>/test_eval.json`. Every operator number is accompanied by the same
number for two nulls, because the KPI threshold alone cannot tell a working
operator from a lucky metric:

* **persistence** -- predict no change at all. An SDF over one timestep is mostly
  unchanged, so this scores far better than it deserves to.
* **uniform recession (oracle)** -- move the whole surface down by the mean depth
  change the ground truth made over the step. It is handed the correct amount of
  etch and knows only that; it has no shape information. This is deliberately
  *unfair to the operator*: the null is given a quantity the operator has to
  infer. Beating persistence is cheap; beating an oracle-offset null means the
  operator has learned the shape of the etch and not just its rate.

Also reports geometry (Hausdorff, normalised area error) on the final profile,
which is the quantity the inverse-design clause is judged in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.data import TrajDataset, band_mask  # noqa: E402
from eot.metrics import shape_error  # noqa: E402
from eot import runlock
from eot.operator import build_from_cfg, EtchOperator, band_rel_l2, rel_l2  # noqa: E402


def uniform_recession(prev_um, target_um):
    """ORACLE null: shift `prev` by the mean surface displacement `target` made.

    Reads the target to get the offset, so it knows the correct etch *rate* for
    free and only lacks the etch *shape*. Reported as an upper bar the operator
    should clear, not as a like-for-like competitor.

    Implemented as an SDF offset: adding c to a signed distance field moves its
    zero set by c along the normal, so this is exactly 'the right amount of etch,
    everywhere, with no shape information'.
    """
    band = (target_um.abs() < 1.5) | (prev_um.abs() < 1.5)
    w = band.flatten(1).float()
    diff = (target_um - prev_um).flatten(1)
    c = (diff * w).sum(1) / w.sum(1).clamp_min(1.0)
    return prev_um + c.view(-1, *([1] * (prev_um.dim() - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--indices-from", default=None,
                    help="JSON file with overlap.crossed_indices_inside: restrict "
                         "scoring to the subset of trajectories whose per-step "
                         "displacement lies inside the training range. This is what "
                         "separates an extrapolation failure from a learned shortcut.")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output file, so evaluating a second "
                         "split into the same run directory cannot overwrite the "
                         "first (it did once)")
    ap.add_argument("--blind", action="store_true",
                    help="zero the conditioning, to score a model trained with --blind")
    a = ap.parse_args()

    run = Path(a.run)
    runlock.acquire(run / f"{a.split}_eval{a.tag or ''}.json", what="eval")
    cfg = json.loads((run / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    device = torch.device(a.device)

    # A mixed-stride arm (--strides) trains on several horizons at once, so
    # cfg["stride"] is not the horizon it is meant to be judged at -- args.json
    # records `eval_stride` for exactly that reason. Prefer it when present, so a
    # mixed arm is scored at its deployment horizon rather than at the default
    # --stride it never really used.
    stride = int(cfg.get("eval_stride") or cfg.get("stride", 1))
    ds = TrajDataset(Path(a.data) / f"{a.split}.npz", norm, stride=stride)
    subset_note = None
    if a.indices_from:
        idx = json.loads(Path(a.indices_from).read_text())["overlap"]["crossed_indices_inside"]
        ds = torch.utils.data.Subset(ds, idx)
        subset_note = (f"restricted to {len(idx)} trajectories whose mean per-step "
                       f"displacement lies inside the training p1-p99 range, from "
                       f"{a.indices_from}")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = build_from_cfg(cfg, len(norm["cond_keys"])).to(device)
    model.load_state_dict(torch.load(run / a.ckpt, map_location=device))
    model.eval()

    keys = ["op", "persist", "blind", "uniform"]
    # constant advance fitted on train, ignoring recipe, dt and target
    # One application advances `stride` dataset timesteps, so the constant-advance
    # null must advance by that much too, or a K>1 arm would be scored against a
    # null that under-etches by K and would beat it for free.
    blind_c = stride * norm["mean_step_displacement_um"] / scale
    one = {k: {"full": [], "band": []} for k in keys}
    roll = {k: {"full": [], "band": []} for k in keys}
    per_step = None
    geo = []

    with torch.no_grad():
        for phi0, cond, traj in loader:
            phi0, cond, traj = phi0.to(device), cond.to(device), traj.to(device)
            if a.blind:
                cond = torch.zeros_like(cond)
            B, T = traj.shape[0], traj.shape[1]
            ins = torch.cat([phi0[:, None], traj[:, :-1]], dim=1)
            fi = ins.reshape(B * T, *ins.shape[2:])
            ft = traj.reshape(B * T, *traj.shape[2:])
            fc = cond[:, None].expand(-1, T, -1).reshape(B * T, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred1 = model(fi, fc).float()
            m1 = band_mask(ft * scale, band_um)
            cands = {"op": pred1, "persist": fi,
                     "blind": fi + blind_c,
                     "uniform": uniform_recession(fi * scale, ft * scale) / scale}
            for k, v in cands.items():
                one[k]["full"] += rel_l2(v, ft, "none").tolist()
                one[k]["band"] += band_rel_l2(v, ft, m1, "none").tolist()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rl = model.rollout(phi0, cond, T).float()
            mm = band_mask(traj * scale, band_um)
            pz = phi0[:, None].expand(-1, T, -1, -1, -1)
            uz = uniform_recession((pz * scale).flatten(0, 1), (traj * scale).flatten(0, 1)) / scale
            bz = pz + blind_c * torch.arange(
                1, T + 1, device=pz.device, dtype=pz.dtype).view(1, T, 1, 1, 1)
            rcands = {"op": rl.flatten(0, 1), "persist": pz.flatten(0, 1),
                      "blind": bz.flatten(0, 1), "uniform": uz}
            for k, v in rcands.items():
                roll[k]["full"] += rel_l2(v, traj.flatten(0, 1), "none").tolist()
                roll[k]["band"] += band_rel_l2(v, traj.flatten(0, 1), mm.flatten(0, 1), "none").tolist()

            # error as a function of rollout step, to see drift compound
            step_band = [
                band_rel_l2(rl[:, t], traj[:, t], mm[:, t], "none").tolist() for t in range(T)
            ]
            per_step = step_band if per_step is None else [
                a_ + b_ for a_, b_ in zip(per_step, step_band)
            ]

            # geometry of the final profile
            gx, gy = S.grid_axes(traj.shape[-1])
            for b in range(B):
                geo.append(
                    shape_error(rl[b, -1, 0].cpu().numpy() * scale,
                                traj[b, -1, 0].cpu().numpy() * scale,
                                phi0[b, 0].cpu().numpy() * scale, gx, gy)
                )

    def stat(v):
        v = np.asarray(v, dtype=float)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max()), "n": int(v.size)}

    out = {
        "run": str(run), "split": a.split, "ckpt": a.ckpt,
        "stride": stride,
        "trained_strides": cfg.get("trained_strides", [stride]),
        "applications_per_wafer": int(traj.shape[1]),
        "physical_steps_per_wafer": int(traj.shape[1]) * stride,
        "n_trajectories": len(ds), "steps": int(traj.shape[1]),
        "subset_note": subset_note,
        "band_um": band_um, "sdf_scale_um": scale,
        "one_step": {k: {kk: stat(vv) for kk, vv in v.items()} for k, v in one.items()},
        "rollout": {k: {kk: stat(vv) for kk, vv in v.items()} for k, v in roll.items()},
        "rollout_band_rel_l2_by_step": [float(np.mean(s)) for s in per_step],
        "final_profile_geometry": {
            "area_error_vs_removed": stat([g["area_error_vs_removed"] for g in geo]),
            "hausdorff_um": stat([g["hausdorff_um"] for g in geo if g["hausdorff_um"] is not None]),
            "mean_surface_dist_um": stat(
                [g["mean_surface_dist_um"] for g in geo if g["mean_surface_dist_um"] is not None]),
        },
    }
    kpi = 0.05
    op_b = out["rollout"]["op"]["band"]["mean"]
    # The mean over rollout steps is the permissive reading: it averages the
    # cheap early steps in with the expensive late ones. The terminal step is
    # the error you actually hold at the end of the etch, and it is the one the
    # inverse-design clause depends on, so it is reported beside the mean and
    # the clause is not called met unless both readings are.
    op_terminal = float(np.mean(per_step[-1]))
    out["kpi_clause_rel_l2"] = {
        "target": kpi,
        "headline_metric": ("rollout band rel-L2, mean over the states the "
                            "operator emits (T/stride of them), mean over test "
                            "trajectories"),
        "terminal_is_comparable_across_stride": True,
        "mean_reading_is_comparable_across_stride": False,
        "stride_comparability_note": (
            "The terminal-step value is at the same physical time (end of etch) "
            "for every stride, so it is the reading a K-curve may be built from. "
            "The mean-over-emitted-states value averages a different number of "
            "states at each stride and is NOT comparable across arms; it is kept "
            "because it is the historical headline at stride 1."),
        "value": op_b,
        "met": bool(op_b <= kpi),
        "value_terminal_step": op_terminal,
        "met_terminal_step": bool(op_terminal <= kpi),
        "met_both_readings": bool(op_b <= kpi and op_terminal <= kpi),
        "terminal_note": ("mean-over-steps averages the easy first step in with the "
                          "hardest last one. The terminal-step value is the error "
                          "standing at the end of the etch and is the stricter read."),
        "persistence_null": out["rollout"]["persist"]["band"]["mean"],
        "recipe_blind_null": out["rollout"]["blind"]["band"]["mean"],
        "beats_recipe_blind": bool(op_b < out["rollout"]["blind"]["band"]["mean"]),
        "recipe_blind_note": (
            "Constant advance by the train-set mean per-step displacement, ignoring "
            "recipe, dt and target. Because the dataset's timestep is chosen per "
            "recipe so every trajectory covers a comparable depth, per-step "
            "displacement is nearly recipe-independent by construction; this null "
            "therefore absorbs that confound, and beating it is the evidence that "
            "the conditioning carries shape information."),
        "uniform_recession_null": out["rollout"]["uniform"]["band"]["mean"],
        "beats_persistence": bool(op_b < out["rollout"]["persist"]["band"]["mean"]),
        "beats_uniform_recession": bool(op_b < out["rollout"]["uniform"]["band"]["mean"]),
        "full_window_reading": out["rollout"]["op"]["full"]["mean"],
        "note": ("full_window_reading is the looser reading of the same clause: the "
                 "far field of an SDF is a smooth ramp with a large norm, so it "
                 "dilutes interface error. The band value is the headline."),
    }
    (run / f"{a.split}_eval{a.tag or ''}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["kpi_clause_rel_l2"], indent=2))
    print(json.dumps(out["final_profile_geometry"], indent=2))


if __name__ == "__main__":
    main()
