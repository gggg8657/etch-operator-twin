"""Achieved etch depth per trajectory, derived from the stored SDFs.

    python scripts/derive_depth.py

**Why this exists.** `runs/bench_workload.json` measures a ceiling on clause 2
that no architecture can beat: the operator takes `log_dt` as a conditioning
input, so a caller must know the timestep before querying it. If the query is a
*target depth* rather than a timestep, obtaining dt costs `eot.solver.probe_rate`
-- itself a solver run, measured at 39.9 ms -- and the speedup is bounded by
solver/probe = **12.8x** however fast the network gets.

The route that dissolves that ceiling instead of pricing it is to condition the
operator on **depth** rather than on dt, so a depth query needs no probe by
construction. This script produces the conditioning channel that route needs.

**It derives the depth rather than reading it, and that is deliberate.** The
dataset does store a `target_depth` per trajectory, but:

* it is the depth *requested* of `choose_dt`, not the depth achieved. `choose_dt`
  divides the target by the etch rate measured on the *initial* geometry
  (`eot/solver.py:326`), and the rate falls as the trench deepens, so requested
  and achieved differ systematically.
* it is **entirely NaN on `test_crossed`** -- all 209 trajectories -- because
  that split was generated in `independent` dt mode, which never computes a
  target (`scripts/gen_data.py:47`). Conditioning on the stored field would make
  the one split clause 1 already fails on impossible to evaluate.

Achieved depth is computable from the frames every split already has, with the
same definition everywhere, so it exists for `test_crossed` too.

**Definition.** The depth advanced between two frames is the drop in the lowest
point of the zero level set:

    depth(t) = y_deepest(frame 0) - y_deepest(frame t)

`y_deepest` is found by scanning each column for sign changes in the SDF and
linearly interpolating the crossing, then taking the most negative crossing over
all columns. This is sub-cell accurate; the row grid is 0.1575 um and the
interpolation resolves inside that. Taking the minimum over columns picks the
trench bottom rather than the mask top, which is what "etch depth" means and
what `choose_dt` used (`surface_polyline(dom)[:, 1].min()`).

**The check that makes this trustworthy** is that derived achieved depth must
track the stored requested depth on the splits that have one. They cannot be
equal -- the rate falls with depth -- but a weak correlation would mean the
extraction is wrong, not that the physics is interesting. The correlation and
the residual are written to the JSON and asserted in `tests/test_depth.py`.

Writes `data/depth_<split>.npy` (N, T+1) and `runs/depth_derivation.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from eot.data import fit_norm, norm_path  # noqa: E402
from eot.solver import Y_MAX, Y_MIN  # noqa: E402


def deepest_y(field: np.ndarray, gy: np.ndarray) -> float:
    """Lowest y on the zero level set of one (H, W) SDF frame, sub-cell.

    Scans every column for sign changes between vertically adjacent samples and
    interpolates the crossing. Returns the most negative crossing found.

    A column with no sign change contributes nothing -- that is the correct
    treatment, not a gap: it means the column is entirely solid or entirely void
    at this grid resolution, and neither locates a surface.
    """
    a, b = field[:-1, :], field[1:, :]
    cross = (np.sign(a) != np.sign(b)) & (a != b)
    if not cross.any():
        return float("nan")
    ya = gy[:-1, None] * np.ones((1, field.shape[1]))
    yb = gy[1:, None] * np.ones((1, field.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = a / (a - b)                  # 0 at row i, 1 at row i+1
    # a == b is already excluded from `cross`, so those entries are never read;
    # suppressing the warning rather than masking first keeps the arithmetic
    # vectorised and the exclusion in one place.
    ycross = ya + (yb - ya) * frac
    return float(np.nanmin(np.where(cross, ycross, np.inf)))


def depths(sdf: np.ndarray) -> np.ndarray:
    """(N, T+1) cumulative depth advanced from frame 0, in micron."""
    n_traj, n_frame = sdf.shape[0], sdf.shape[1]
    gy = np.linspace(Y_MAX, Y_MIN, sdf.shape[-2])
    out = np.zeros((n_traj, n_frame), dtype=np.float64)
    for i in range(n_traj):
        y0 = deepest_y(sdf[i, 0], gy)
        for t in range(n_frame):
            out[i, t] = y0 - deepest_y(sdf[i, t], gy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "test_crossed"])
    ap.add_argument("--out", default="runs/depth_derivation.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="derive_depth")
    res = {
        "definition": "depth(t) = y_deepest(frame 0) - y_deepest(frame t), "
                      "y_deepest = most negative zero crossing of the SDF over "
                      "all columns, linearly interpolated between vertically "
                      "adjacent samples (sub-cell; row grid is "
                      f"{(Y_MAX - Y_MIN) / 127:.4f} um)",
        "why_derived_not_read": "the stored target_depth is the depth REQUESTED "
                                "of choose_dt, not achieved, and it is NaN for "
                                "every test_crossed trajectory",
        "splits": {},
    }
    for sp in a.splits:
        p = Path(a.data) / f"{sp}.npz"
        if not p.exists():
            res["splits"][sp] = {"absent": True}
            continue
        d = np.load(p)
        dep = depths(d["sdf"])
        np.save(Path(a.data) / f"depth_{sp}.npy", dep.astype(np.float32))
        final = dep[:, -1]
        row = {
            "n_trajectories": int(dep.shape[0]),
            "n_frames": int(dep.shape[1]),
            "n_nan": int(np.isnan(dep).sum()),
            "final_depth_um": {
                "min": float(np.nanmin(final)), "max": float(np.nanmax(final)),
                "median": float(np.nanmedian(final)),
                "p25": float(np.nanpercentile(final, 25)),
                "p75": float(np.nanpercentile(final, 75)),
            },
            "monotone_fraction": float(
                np.mean(np.all(np.diff(dep, axis=1) >= -1e-6, axis=1))),
            "file": f"data/depth_{sp}.npy",
        }
        # The validation: does achieved track requested, where requested exists?
        if "target_depth" in d.files:
            tgt = d["target_depth"].astype(np.float64)
            ok = np.isfinite(tgt) & np.isfinite(final)
            if ok.sum() >= 3:
                r = float(np.corrcoef(tgt[ok], final[ok])[0, 1])
                resid = final[ok] - tgt[ok]
                row["vs_stored_target_depth"] = {
                    "n_comparable": int(ok.sum()),
                    "pearson_r": r,
                    "median_residual_um": float(np.median(resid)),
                    "median_abs_residual_um": float(np.median(np.abs(resid))),
                    "achieved_over_requested_median": float(
                        np.median(final[ok] / np.clip(tgt[ok], 1e-9, None))),
                    "reading": "achieved should track requested but sit BELOW "
                               "it: choose_dt sizes dt from the rate on the "
                               "initial flat geometry, and the rate falls as "
                               "the trench deepens, so the etch under-delivers. "
                               "A weak correlation would mean the extraction is "
                               "wrong, not that the physics is interesting.",
                }
            else:
                row["vs_stored_target_depth"] = {
                    "n_comparable": int(ok.sum()),
                    "note": "stored target_depth is all NaN on this split "
                            "(independent-dt generation), which is why depth is "
                            "derived rather than read",
                }
        res["splits"][sp] = row
        v = row["final_depth_um"]
        cmp_ = row.get("vs_stored_target_depth", {})
        print(f"{sp:13s} n={row['n_trajectories']:4d} final depth "
              f"med={v['median']:6.2f} [{v['min']:5.2f},{v['max']:6.2f}] um  "
              f"nan={row['n_nan']}  monotone={row['monotone_fraction']:.3f}"
              + (f"  r={cmp_['pearson_r']:.4f} achieved/requested="
                 f"{cmp_['achieved_over_requested_median']:.3f}"
                 if "pearson_r" in cmp_ else "  (no stored target)"))
    # The depth-mode normalisation lives beside the dt one and is fitted on
    # TRAIN ONLY, same as the dt constants, so val/test/crossed are scored under
    # constants they did not contribute to.
    if not res["splits"].get("train", {}).get("absent"):
        np_ = norm_path(a.data, "depth")
        np_.write_text(json.dumps(fit_norm(Path(a.data) / "train.npz",
                                           cond_mode="depth"), indent=2))
        res["norm_written"] = str(np_)
        print(f"wrote {np_}")

    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
