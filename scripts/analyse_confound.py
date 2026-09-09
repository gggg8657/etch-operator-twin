"""How much did the adaptive timestep actually flatten the problem?

    python scripts/analyse_confound.py

An adversarial review pointed out that `dt = target_depth / (rate * n_steps)`
makes per-step displacement equal `target_depth / n_steps` for *every* recipe, so
a model that ignores the recipe entirely could predict the dominant variance.
That is an argument. This measures it, on the dataset that actually exists.

Three quantities, written to `runs/confound.json`:

1. The spread of per-step interface displacement across the corpus. If the
   coefficient of variation is small, the recipe-blind null is strong and the
   argument holds.
2. How much of that displacement the recipe explains — R² of a linear fit of
   displacement on the standardised conditioning vector. Near zero means the
   conditioning carries no *rate* information at all, only shape.
3. The same for the crossed split, once it exists, where dt is drawn without
   reference to the recipe and the spread should be much larger.

None of this says whether the operator is good. It says what a good number would
have to beat.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import BAND_UM, cond_vector  # noqa: E402


def per_step_displacement(sdf: np.ndarray, band_um: float = BAND_UM) -> np.ndarray:
    """(N, T) mean normal displacement of the interface per step, in um."""
    prev, nxt = sdf[:, :-1], sdf[:, 1:]
    band = (np.abs(nxt) < band_um) | (np.abs(prev) < band_um)
    w = band.reshape(band.shape[0], band.shape[1], -1).astype(np.float64)
    diff = (nxt - prev).reshape(w.shape)
    return (diff * w).sum(axis=2) / np.clip(w.sum(axis=2), 1.0, None)


def analyse(npz: Path, label: str) -> dict:
    d = np.load(npz)
    disp = per_step_displacement(d["sdf"])
    traj = disp.mean(axis=1)  # per-trajectory mean advance per step
    flat = disp.reshape(-1)

    # R^2 of a least-squares fit of per-trajectory displacement on conditioning
    c = cond_vector(d["recipe"], d["dt"])
    c = (c - c.mean(0)) / (c.std(0) + 1e-8)
    X = np.concatenate([c, np.ones((len(c), 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X, traj, rcond=None)
    resid = traj - X @ beta
    r2 = 1.0 - resid.var() / max(traj.var(), 1e-12)

    # the same, but conditioning WITHOUT dt -- how much the recipe alone explains
    X2 = np.concatenate([c[:, :-1], np.ones((len(c), 1))], axis=1)
    b2, *_ = np.linalg.lstsq(X2, traj, rcond=None)
    r2_recipe_only = 1.0 - (traj - X2 @ b2).var() / max(traj.var(), 1e-12)

    return {
        "split": label,
        "n_trajectories": int(d["sdf"].shape[0]),
        "displacement_um_per_step": {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "cv": float(flat.std() / abs(flat.mean())),
            "p5": float(np.percentile(flat, 5)), "p95": float(np.percentile(flat, 95)),
            "max_over_min": float(flat.max() / max(flat.min(), 1e-12)),
        },
        "per_trajectory_mean_displacement": {
            "mean": float(traj.mean()), "std": float(traj.std()),
            "cv": float(traj.std() / abs(traj.mean())),
            "max_over_min": float(traj.max() / max(traj.min(), 1e-12)),
        },
        "r2_displacement_on_full_conditioning": float(r2),
        "r2_displacement_on_recipe_without_dt": float(r2_recipe_only),
        "dt": {"min": float(d["dt"].min()), "max": float(d["dt"].max()),
               "max_over_min": float(d["dt"].max() / max(d["dt"].min(), 1e-12))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/confound.json")
    a = ap.parse_args()
    res = {"band_um": BAND_UM, "splits": []}
    for name in ("train", "val", "test", "test_crossed"):
        p = Path(a.data) / f"{name}.npz"
        if p.exists():
            res["splits"].append(analyse(p, name))
    tr = next((s for s in res["splits"] if s["split"] == "train"), None)
    cr = next((s for s in res["splits"] if s["split"] == "test_crossed"), None)
    if tr:
        res["verdict"] = {
            "adaptive_displacement_cv": tr["per_trajectory_mean_displacement"]["cv"],
            "adaptive_displacement_max_over_min":
                tr["per_trajectory_mean_displacement"]["max_over_min"],
            "crossed_displacement_max_over_min": (
                cr["per_trajectory_mean_displacement"]["max_over_min"] if cr else None),
            "note": ("If the adaptive spread is small, the recipe-blind null is strong "
                     "and the operator has to beat it to have shown anything. The "
                     "crossed split is the same recipe box with dt drawn independently, "
                     "where the spread is the rate law's own."),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
