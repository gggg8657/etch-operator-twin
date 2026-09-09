"""How much does per-step surface displacement actually vary, and where does the
crossed split sit relative to training?

Two different failures produce the same symptom -- an operator that collapses on
the crossed split -- and they have opposite implications:

* **Shortcut.** Training displacement is nearly constant, the model learned that
  constant, and the crossed split exposes it. Damning: the in-distribution
  number would then be an artefact of dataset construction.
* **Extrapolation.** Training displacement varies, the model learned the rate
  law over the range it saw, and the crossed split simply asks about steps
  larger or smaller than any it was trained on. Expected, and not a defect of
  the model -- only a limit on where it may be used.

The discriminator is the *overlap*: restrict the crossed split to trajectories
whose per-step displacement lies inside the training range, and re-score. If the
operator recovers on that subset, the collapse is extrapolation. If it stays
broken where the two distributions overlap, it is a shortcut.

Writes `runs/confound.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import BAND_UM  # noqa: E402


def displacement_per_step(sdf_um: np.ndarray, band_um: float = BAND_UM) -> np.ndarray:
    """(N, T) band-mean normal displacement between consecutive frames, in um.

    Adding a constant to an SDF translates its zero set by that constant along
    the normal, so the band-mean of (phi_{t+1} - phi_t) is the mean normal
    motion of the interface over the step.
    """
    prev, nxt = sdf_um[:, :-1], sdf_um[:, 1:]
    band = (np.abs(nxt) < band_um) | (np.abs(prev) < band_um)
    w = band.reshape(band.shape[0], band.shape[1], -1).astype(np.float64)
    diff = (nxt - prev).reshape(w.shape)
    return np.abs((diff * w).sum(axis=2) / np.clip(w.sum(axis=2), 1.0, None))


def describe(d: np.ndarray) -> dict:
    flat = d.reshape(-1)
    per_traj = d.mean(axis=1)
    return {
        "per_step": {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "min": float(flat.min()), "max": float(flat.max()),
            "p5": float(np.percentile(flat, 5)), "p95": float(np.percentile(flat, 95)),
            "cv": float(flat.std() / flat.mean()),
            "max_over_min": float(flat.max() / max(flat.min(), 1e-12)),
        },
        "per_trajectory_mean": {
            "mean": float(per_traj.mean()), "std": float(per_traj.std()),
            "min": float(per_traj.min()), "max": float(per_traj.max()),
            "cv": float(per_traj.std() / per_traj.mean()),
            "max_over_min": float(per_traj.max() / max(per_traj.min(), 1e-12)),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.npz")
    ap.add_argument("--test", default="data/test.npz")
    ap.add_argument("--crossed", default="data_crossed/test.npz")
    ap.add_argument("--out", default="runs/confound.json")
    a = ap.parse_args()

    out = {"band_um": BAND_UM, "splits": {}}
    disp = {}
    for name, path in [("train", a.train), ("test", a.test), ("crossed", a.crossed)]:
        p = Path(path)
        if not p.exists():
            out["splits"][name] = "[not measured] file absent"
            continue
        d = np.load(p)
        dd = displacement_per_step(d["sdf"])
        disp[name] = dd
        out["splits"][name] = {"n_trajectories": int(d["sdf"].shape[0]), **describe(dd)}

    if "train" in disp and "crossed" in disp:
        lo = float(np.percentile(disp["train"].reshape(-1), 1))
        hi = float(np.percentile(disp["train"].reshape(-1), 99))
        c_traj = disp["crossed"].mean(axis=1)
        inside = (c_traj >= lo) & (c_traj <= hi)
        out["overlap"] = {
            "train_p1_um": lo, "train_p99_um": hi,
            "crossed_trajectories_inside_train_range": int(inside.sum()),
            "crossed_trajectories_total": int(inside.size),
            "fraction_inside": float(inside.mean()),
            "crossed_indices_inside": np.nonzero(inside)[0].tolist(),
            "note": ("A crossed trajectory is 'inside' when its mean per-step "
                     "displacement lies within the 1st-99th percentile of the "
                     "training per-step displacement. Scoring the operator on this "
                     "subset separates extrapolation from a learned shortcut."),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    for k, v in out["splits"].items():
        if isinstance(v, dict):
            ps = v["per_step"]
            print(f"{k:8s} n={v['n_trajectories']:4d}  per-step disp "
                  f"mean {ps['mean']:.4f} um  cv {ps['cv']:.3f}  "
                  f"range {ps['min']:.4f}-{ps['max']:.4f}  max/min {ps['max_over_min']:.1f}")
    if "overlap" in out:
        o = out["overlap"]
        print(f"overlap: {o['crossed_trajectories_inside_train_range']}/"
              f"{o['crossed_trajectories_total']} crossed trajectories "
              f"({o['fraction_inside']:.1%}) inside train p1-p99 "
              f"[{o['train_p1_um']:.4f}, {o['train_p99_um']:.4f}] um")


if __name__ == "__main__":
    main()
