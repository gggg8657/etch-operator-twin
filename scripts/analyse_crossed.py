"""Why does the operator fail on the crossed split?

    python scripts/analyse_crossed.py --run runs/seed1

Three explanations were written down before this ran (critique_log, H2):
coverage (dt outside the trained range), capacity, or the crossed split being
intrinsically harder because its trajectories simply move further per step.

The third is testable without any new training: regress per-trajectory band
rel-L2 on per-step displacement and on dt separately, on both splits. If the
error is explained by displacement magnitude, "coverage" is the wrong word and
mixed-dt training will plateau. If it is explained by dt *outside the trained
range* at fixed displacement, coverage is the right word.

Writes `runs/crossed_analysis.json`.
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
from eot.metrics import hausdorff_um  # noqa: E402
from eot import solver as S  # noqa: E402
from eot.operator import EtchOperator, band_rel_l2  # noqa: E402
from scripts.analyse_confound import per_step_displacement  # noqa: E402


def per_traj_error(model, ds, device, scale, band_um):
    """Per-trajectory relative error AND an absolute one, in micron.

    rel-L2 divides by the norm of the target field in the band. On a trajectory
    that barely moves that denominator is small, so a *relative* metric reports a
    large error for a small absolute one. The crossed split contains such
    trajectories by construction (dt drawn independently of the recipe pairs slow
    recipes with short timesteps) and the adaptive split does not. Reporting an
    absolute surface distance alongside is the only way to tell a real accuracy
    collapse from that arithmetic.
    """
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    errs, absd = [], []
    with torch.no_grad():
        for phi0, cond, traj in loader:
            phi0, cond, traj = phi0.to(device), cond.to(device), traj.to(device)
            T = traj.shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rl = model.rollout(phi0, cond, T).float()
            m = band_mask(traj * scale, band_um)
            e = band_rel_l2(rl.flatten(0, 1), traj.flatten(0, 1), m.flatten(0, 1), "none")
            errs += e.view(traj.shape[0], T).mean(dim=1).tolist()
            # absolute: mean |phi_pred - phi_true| inside the band, in micron
            w = m.flatten(2).float()
            d = ((rl - traj).abs().flatten(2) * w).sum(-1) / w.sum(-1).clamp_min(1.0)
            absd += (d * scale).mean(dim=1).reshape(-1).tolist()
    return np.asarray(errs), np.asarray(absd)


def fit(x, y):
    """R^2 of a 1-D least-squares fit, plus Spearman rank correlation."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.stack([x, np.ones_like(x)], 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - (y - X @ b).var() / max(y.var(), 1e-12)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    sp = float(np.corrcoef(rx, ry)[0, 1])
    return {"r2": float(r2), "spearman": sp, "slope": float(b[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/seed1")
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/crossed_analysis.json")
    a = ap.parse_args()

    cfg = json.loads((Path(a.run) / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    device = torch.device(a.device)
    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                         modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(torch.load(Path(a.run) / "best.pt", map_location=device))
    model.eval()

    dt_lo, dt_hi = norm["dt_lo"], norm["dt_hi"]
    out = {"run": a.run, "trained_dt_range": [dt_lo, dt_hi], "splits": {}}
    for split in ("test", "test_crossed"):
        p = Path(a.data) / f"{split}.npz"
        if not p.exists():
            continue
        ds = TrajDataset(p, norm)
        d = np.load(p)
        err, absd = per_traj_error(model, ds, device, scale, band_um)
        disp = np.abs(per_step_displacement(d["sdf"]).mean(axis=1))
        dt = d["dt"]
        # dt outside the range the model was trained on
        outside = (dt < dt_lo) | (dt > dt_hi)
        out["splits"][split] = {
            "n": int(len(err)),
            "band_rel_l2_mean": float(err.mean()),
            "band_rel_l2_median": float(np.median(err)),
            "abs_band_err_um_mean": float(absd.mean()),
            "abs_band_err_um_median": float(np.median(absd)),
            "abs_band_err_um_p90": float(np.percentile(absd, 90)),
            "vs_displacement": fit(disp, err),
            "vs_dt": fit(dt, err),
            "vs_log_dt": fit(np.log(dt), err),
            "displacement_range": [float(disp.min()), float(disp.max())],
            "dt_range": [float(dt.min()), float(dt.max())],
            "frac_dt_outside_trained_range": float(outside.mean()),
            # the decisive slice: trajectories whose displacement matches the
            # in-distribution range, so only dt differs
            "err_on_displacement_matched_subset": None,
        }
    # Restrict the crossed split to displacements inside the adaptive split's range
    if "test" in out["splits"] and "test_crossed" in out["splits"]:
        dtest = np.load(Path(a.data) / "test.npz")
        dcross = np.load(Path(a.data) / "test_crossed.npz")
        disp_t = np.abs(per_step_displacement(dtest["sdf"]).mean(axis=1))
        disp_c = np.abs(per_step_displacement(dcross["sdf"]).mean(axis=1))
        lo, hi = disp_t.min(), disp_t.max()
        ds_c = TrajDataset(Path(a.data) / "test_crossed.npz", norm)
        err_c, absd_c = per_traj_error(model, ds_c, device, scale, band_um)
        sel = (disp_c >= lo) & (disp_c <= hi)
        out["displacement_matched"] = {
            "adaptive_displacement_range": [float(lo), float(hi)],
            "n_crossed_in_range": int(sel.sum()),
            "crossed_err_in_range_mean": float(err_c[sel].mean()) if sel.any() else None,
            "crossed_err_out_of_range_mean": (float(err_c[~sel].mean())
                                              if (~sel).any() else None),
            "adaptive_err_mean": out["splits"]["test"]["band_rel_l2_mean"],
            "crossed_abs_um_in_range_mean": float(absd_c[sel].mean()) if sel.any() else None,
            "crossed_abs_um_out_of_range_mean": (float(absd_c[~sel].mean())
                                                 if (~sel).any() else None),
            "adaptive_abs_um_mean": out["splits"]["test"]["abs_band_err_um_mean"],
            "crossed_displacement_below_range": int((disp_c < lo).sum()),
            "crossed_displacement_above_range": int((disp_c > hi).sum()),
            "err_of_below_range": (float(err_c[disp_c < lo].mean())
                                   if (disp_c < lo).any() else None),
            "err_of_above_range": (float(err_c[disp_c > hi].mean())
                                   if (disp_c > hi).any() else None),
            "abs_um_of_below_range": (float(absd_c[disp_c < lo].mean())
                                      if (disp_c < lo).any() else None),
            "abs_um_of_above_range": (float(absd_c[disp_c > hi].mean())
                                      if (disp_c > hi).any() else None),
            "note": ("If crossed error restricted to in-range displacements is close to "
                     "the adaptive error, the crossed failure is explained by trajectories "
                     "simply moving further per step, not by dt being unfamiliar. If it "
                     "stays high, the model is failing on dt values it never saw at "
                     "displacements it did."),
        }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
