"""Does the clause actually hold on the in-coverage subset? Computed, not asserted.

    python scripts/coverage_verdict.py --runs runs/seed1 runs/seed2 runs/seed5 runs/seed6

`analyse_crossed.py` established that crossed-split error tracks per-step
displacement, and `report.py` printed "every seed meets the clause in coverage".
An adversarial review (codex, turn 7) took that apart, and four of its objections
were right. This script exists to answer them with measurements:

1. **Two incompatible definitions of "in range" were being printed side by side.**
   `analyse_crossed.py` selects on the *test* split's displacement min/max
   (n=137 of 209); the paragraph next to it in RESULTS.md described the *train*
   split's 1st-99th percentile (76 of 225, a different dataset). Both rules are
   computed here, on one dataset, and reported separately. Rule B is the
   stricter and the one the prose had been claiming.

2. **The subset is defined using the ground-truth future**, so it is a diagnostic,
   not a deployable coverage rule -- you cannot select on a displacement you have
   not simulated yet. Recorded as a limitation; the a-priori variant (select on
   the *predicted* displacement, which needs no ground truth) is computed
   alongside so the gap between them is visible.

3. **A point estimate 0.0035 under a threshold, with a seed range of 0.0171, is
   not a pass.** Errors are bootstrapped over trajectories, paired across seeds
   (the seeds share the same trajectories, so resampling must resample
   trajectories, not seeds), and the verdict is `upper CI bound <= 0.05`.

4. **Both readings must pass.** The repo's clause-1 rule is mean-over-steps AND
   terminal-step. The subset analysis only ever reported the mean. Terminal-step
   is computed here and the verdict requires both.

Writes `runs/coverage_verdict.json`.
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
from eot.data import TrajDataset, band_mask  # noqa: E402
from eot.operator import EtchOperator, band_rel_l2  # noqa: E402
from scripts.analyse_confound import per_step_displacement  # noqa: E402


def per_traj_both_readings(model, ds, device, scale, band_um):
    """Per-trajectory band rel-L2 as mean-over-steps and as terminal step, and
    the model's own predicted per-step displacement (no ground truth used)."""
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    mean_r, term_r, pred_disp = [], [], []
    with torch.no_grad():
        for phi0, cond, traj in loader:
            phi0, cond, traj = phi0.to(device), cond.to(device), traj.to(device)
            T = traj.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rl = model.rollout(phi0, cond, T).float()
            m = band_mask(traj * scale, band_um)
            e = band_rel_l2(rl.flatten(0, 1), traj.flatten(0, 1),
                            m.flatten(0, 1), "none").view(traj.shape[0], T)
            mean_r += e.mean(dim=1).tolist()
            term_r += e[:, -1].tolist()
            # Displacement the MODEL predicts, so this selector needs no oracle.
            # Must use the SAME definition as per_step_displacement(): band-weighted
            # signed normal advance per step, in micron. A different definition here
            # made the a-priori and oracle selectors agree only 44% of the time,
            # which measured the mismatch and not the selector.
            seq = torch.cat([phi0.unsqueeze(1), rl], dim=1) * scale
            prev, nxt = seq[:, :-1], seq[:, 1:]
            bw = ((nxt.abs() < band_um) | (prev.abs() < band_um)).flatten(2).double()
            df = (nxt - prev).flatten(2).double()
            step = (df * bw).sum(-1) / bw.sum(-1).clamp_min(1.0)
            pred_disp += step.abs().mean(dim=1).tolist()
    return np.asarray(mean_r), np.asarray(term_r), np.asarray(pred_disp)


def boot_ci(vals, sel, n_boot=10000, seed=0, alpha=0.05):
    """Bootstrap the subset mean by resampling TRAJECTORIES. `vals` is
    (n_seeds, n_traj); seeds share trajectories, so a resample draws a set of
    trajectory indices and applies it to every seed."""
    idx_pool = np.flatnonzero(sel)
    if idx_pool.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, idx_pool.size, size=(n_boot, idx_pool.size))
    picks = idx_pool[draws]                      # (n_boot, n_sub)
    means = vals[:, picks].mean(axis=2)          # (n_seeds, n_boot)
    pooled = means.mean(axis=0)                  # pooled across seeds per draw
    return {
        "point": float(vals[:, idx_pool].mean()),
        "lo": float(np.percentile(pooled, 100 * alpha / 2)),
        "hi": float(np.percentile(pooled, 100 * (1 - alpha / 2))),
        "worst_seed_point": float(vals[:, idx_pool].mean(axis=1).max()),
        "per_seed": [float(x) for x in vals[:, idx_pool].mean(axis=1)],
        "n": int(idx_pool.size), "n_boot": n_boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["runs/seed1", "runs/seed2", "runs/seed5", "runs/seed6"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--out", default="runs/coverage_verdict.json")
    a = ap.parse_args()

    norm = json.loads((Path(a.data) / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    device = torch.device(a.device)

    tr = np.load(Path(a.data) / "train.npz")["sdf"]
    te = np.load(Path(a.data) / "test.npz")["sdf"]
    cr = np.load(Path(a.data) / "test_crossed.npz")["sdf"]
    disp_tr = np.abs(per_step_displacement(tr).mean(axis=1))
    disp_te = np.abs(per_step_displacement(te).mean(axis=1))
    disp_cr = np.abs(per_step_displacement(cr).mean(axis=1))

    ds = TrajDataset(Path(a.data) / "test_crossed.npz", norm)
    mean_rows, term_rows, pred_rows, used = [], [], [], []
    for r in a.runs:
        run = Path(r)
        cfg = json.loads((run / "args.json").read_text())
        model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                             modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
        model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        model.eval()
        m, t, pd_ = per_traj_both_readings(model, ds, device, scale, band_um)
        mean_rows.append(m); term_rows.append(t); pred_rows.append(pd_)
        used.append(run.name)
    M = np.stack(mean_rows); T = np.stack(term_rows); P = np.stack(pred_rows)

    rules = {
        "A_test_minmax": {
            "lo": float(disp_te.min()), "hi": float(disp_te.max()),
            "note": "what analyse_crossed.py used: the adaptive TEST split's full "
                    "displacement min/max. Widest of the three, so most permissive.",
        },
        "B_train_p1_p99": {
            "lo": float(np.percentile(disp_tr, 1)), "hi": float(np.percentile(disp_tr, 99)),
            "note": "the TRAIN split's 1st-99th percentile -- the rule RESULTS.md's "
                    "prose had been describing while the table used rule A.",
        },
        "C_train_minmax": {
            "lo": float(disp_tr.min()), "hi": float(disp_tr.max()),
            "note": "train split full range; the literal reading of 'a displacement "
                    "the model saw in training'.",
        },
    }

    out = {"runs": used, "n_crossed": int(len(disp_cr)), "target": a.target,
           "readings": ["mean_over_steps", "terminal_step"], "rules": {}}
    for name, r in rules.items():
        sel = (disp_cr >= r["lo"]) & (disp_cr <= r["hi"])
        entry = {**r, "n_in": int(sel.sum()), "n_out": int((~sel).sum())}
        for reading, V in (("mean_over_steps", M), ("terminal_step", T)):
            ci_in = boot_ci(V, sel)
            ci_out = boot_ci(V, ~sel)
            entry[reading] = {"in_range": ci_in, "out_of_range": ci_out}
            if ci_in:
                entry[reading]["met_point"] = bool(ci_in["point"] <= a.target)
                entry[reading]["met_every_seed"] = bool(
                    max(ci_in["per_seed"]) <= a.target)
                entry[reading]["met_upper_ci"] = bool(ci_in["hi"] <= a.target)
                entry[reading]["margin_to_target"] = float(a.target - ci_in["hi"])
        both = all(entry[x].get("met_upper_ci") for x in
                   ("mean_over_steps", "terminal_step") if entry.get(x))
        entry["verdict_in_coverage"] = (
            "MET" if both else "NOT MET")
        entry["verdict_basis"] = ("upper bound of a 95% trajectory bootstrap on BOTH "
                                  "readings must be <= target")
        out["rules"][name] = entry

    # a-priori selector: the model's own predicted displacement, no ground truth
    sel_pred = (P.mean(axis=0) >= rules["B_train_p1_p99"]["lo"]) & \
               (P.mean(axis=0) <= rules["B_train_p1_p99"]["hi"])
    sel_true = (disp_cr >= rules["B_train_p1_p99"]["lo"]) & \
               (disp_cr <= rules["B_train_p1_p99"]["hi"])
    out["a_priori_selector"] = {
        "note": "Rule B applied to the displacement the MODEL predicts rather than "
                "the simulated truth. This one is deployable -- it needs no oracle. "
                "If its verdict matches the oracle selector's, the coverage rule is "
                "usable in practice and not merely diagnostic.",
        "n_in": int(sel_pred.sum()),
        "agreement_with_oracle_selector": float((sel_pred == sel_true).mean()),
        "mean_over_steps": boot_ci(M, sel_pred),
        "terminal_step": boot_ci(T, sel_pred),
    }
    for k in ("mean_over_steps", "terminal_step"):
        ci = out["a_priori_selector"][k]
        if ci:
            out["a_priori_selector"][k]["met_upper_ci"] = bool(ci["hi"] <= a.target)

    out["limitations"] = [
        "The subset is chosen by displacement, and displacement = rate(recipe, "
        "geometry) x dt. Conditioning on it can preferentially select recipe/dt "
        "pairs resembling the adaptive training relationship, so this localises "
        "the failure but does not prove displacement is the sole cause. A "
        "controlled test would hold recipe fixed and vary dt across the boundary.",
        "dt is inside its trained MARGINAL range on every crossed trajectory. "
        "That does not establish that the joint (recipe, geometry, dt) input is "
        "covered, and the claim is stated marginally for that reason.",
    ]
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rules"}, indent=2)[:800])
    for n, e in out["rules"].items():
        print(f"\n{n}: n_in={e['n_in']} n_out={e['n_out']} -> {e['verdict_in_coverage']}")
        for reading in ("mean_over_steps", "terminal_step"):
            ci = e[reading]["in_range"]
            co = e[reading]["out_of_range"]
            print(f"  {reading:15s} IN  {ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] "
                  f"worst seed {ci['worst_seed_point']:.4f}  "
                  f"met_point={e[reading]['met_point']} "
                  f"met_every_seed={e[reading]['met_every_seed']} "
                  f"met_upper_ci={e[reading]['met_upper_ci']}")
            print(f"  {'':15s} OUT {co['point']:.4f} [{co['lo']:.4f}, {co['hi']:.4f}]")
    ap_ = out["a_priori_selector"]
    print(f"\na-priori selector (no oracle): n_in={ap_['n_in']} "
          f"agreement={ap_['agreement_with_oracle_selector']:.3f}")
    for reading in ("mean_over_steps", "terminal_step"):
        ci = ap_[reading]
        print(f"  {reading:15s} {ci['point']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}] "
              f"met_upper_ci={ci['met_upper_ci']}")


if __name__ == "__main__":
    main()
