"""H8: can a height field represent these etch fronts at all?

    python scripts/surface_representable.py

`runs/cost_floor.json` falsified H6 and put a surface-output model at **5830x**
warm (8375x including rasterisation back to a field), against the deployed
field-output FNO's 3.2x and a 1000x clause. That makes changing the output
representation the cheapest route to clause 2 -- but only if the representation
can express the data.

**A height field h(x) is single-valued by construction.** A plasma etch that
undercuts a mask, or bows, produces a re-entrant front where one column of the
domain has the surface at two or more heights. No h(x) can represent that,
however good the model, so the route would be dead before any training and the
5830x would be a speedup at predicting something else.

`tests/test_solver.py::test_surface_is_a_single_open_chain` already establishes
the front is a single open chain, but a chain may still be multi-valued: an
undercut is one connected chain that doubles back.

**H8: the fronts in this dataset are single-valued in x, so a height
representation is lossless up to grid resolution, and the surface-output route
is open.**

The measurement is **void runs with solid both above and below**, per column.

A height field h(x) can describe a column exactly when its solid is a single
interval reaching the bottom of the domain. A column containing void with solid
above *and* below it is re-entrant, and no h(x) can express it.

**Sign-change count is not the criterion, though an earlier version of this
script used it, and that was a second error.** Counting sign changes and
thresholding at three misses the dominant case here: a column whose mask reaches
the top of the domain reads solid-void-solid, which is **two** changes, not
three. `runs/surface_representable.json` recorded `max_sign_changes: 2` beside an
84-cell trapped void in the same column, which is how the contradiction
surfaced. Sign changes depend on whether a column starts solid or void, so they
are kept as a diagnostic and the verdict rests on the trapped-void runs, which
do not.

**My first attempt at this measured the wrong thing and its numbers are
discarded.** It binned zero-contour points by column and flagged any column
whose points spanned more than a grid cell vertically. That flags every frame
including `t=0`, before any etch has run, with a worst span of 17.0 um: the
initial trench's *vertical sidewall* puts a whole column of contour points at
one x. But a vertical wall is a transition *between* adjacent columns, and the
solid in each of those columns is still one interval, so it is perfectly
height-representable. The extent metric conflated the wall with the undercut it
was meant to detect. Sign changes do not: a vertical wall gives one per column.

The distinguishing prediction. If H8 holds, no column anywhere carries a trapped
void thicker than the corner artefact, or such columns are confined to `t=0` and
do not grow. If it fails, they appear at the mask edge and their count **grows
with etch time**, which is the signature of a real undercut developing.

Writes `runs/surface_representable.json`. No model is involved and no accuracy
is claimed: this bounds what the representation can express, not what a network
would learn.
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
from eot.metrics import zero_contour_points  # noqa: E402

# A trapped void this thin or thinner is a level-set discretisation artefact at
# the trench corner rather than an undercut. Fixed at 2 because that is the
# observed thickness of the t=0 sliver -- 27.5% of t=0 frames carry one, at the
# symmetric sidewall pairs, before any etch physics has run. Excluded from the
# verdict; counted and reported so the exclusion can be checked.
ARTEFACT_CELLS = 2


def column_sign_changes(sdf):
    """Sign changes of the SDF down each column.

    DIAGNOSTIC ONLY -- not the re-entrancy criterion. A trapped void reads
    solid-void-solid, which is TWO changes when the column starts solid at the
    domain top and three when it starts void, so no single threshold on this
    count separates the cases. `trapped_void_runs` is the criterion.

    1 -> the solid is one interval reaching the domain edge. A vertical sidewall
         lands here, being a transition BETWEEN columns.
    0 -> no surface in the column at all.

    Zeros are treated as belonging to the side they are approached from by
    ignoring them in the sign product, so a sample sitting exactly on the
    interface does not manufacture a crossing.
    """
    s = np.sign(sdf)                      # (rows, cols), row 0 = top
    out = np.zeros(s.shape[1], dtype=int)
    for c in range(s.shape[1]):
        col = s[:, c]
        col = col[col != 0]
        if col.size < 2:
            continue
        out[c] = int(np.count_nonzero(col[1:] != col[:-1]))
    return out


def trapped_void_runs(sdf):
    """Thicknesses, in cells, of every void run with solid both above and below.

    This is the object of interest: a void run sandwiched in solid is exactly
    what no h(x) can express. Returned per column so the two populations can be
    told apart -- see `ARTEFACT_CELLS`.
    """
    solid = sdf < 0
    out = {}
    for c in range(solid.shape[1]):
        col = solid[:, c]
        runs, cur, n = [], col[0], 0
        for v in col:
            if v == cur:
                n += 1
            else:
                runs.append((bool(cur), n))
                cur, n = v, 1
        runs.append((bool(cur), n))
        th = [runs[i][1] for i in range(1, len(runs) - 1)
              if not runs[i][0] and runs[i - 1][0] and runs[i + 1][0]]
        if th:
            out[c] = th
    return out


def reentrant_depth_um(sdf, delta):
    """For each column, how much solid a height field would have to invent.

    A column with void trapped under solid is mis-described by h(x) = the top
    interface: everything below the first void run gets filled in. This returns
    the total thickness of the trapped void per column, in um -- the volumetric
    error the representation forces, which is the quantity clause 1's field
    rel-L2 and clause 3's area error would pay.
    """
    inside = sdf < 0                      # solid
    rows, cols = inside.shape
    err = np.zeros(cols)
    for c in range(cols):
        col = inside[:, c]
        if not col.any():
            continue
        first = int(np.argmax(col))       # topmost solid row
        # void rows below the first solid row are what h(x) cannot express
        err[c] = float(np.count_nonzero(~col[first:]) * delta)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test.npz")
    ap.add_argument("--out", default="runs/surface_representable.json")
    ap.add_argument("--n-traj", type=int, default=0,
                    help="0 = all trajectories in the split")
    a = ap.parse_args()

    runlock.acquire(a.out, what="surface_representable")
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    delta, n_grid = gen["grid_delta"], gen["grid_n"]
    z = np.load(Path(a.data) / a.split)
    sdf = z["sdf"]
    n_traj = sdf.shape[0] if a.n_traj == 0 else min(a.n_traj, sdf.shape[0])

    # The generator's grid, matching eot.metrics' convention (row 0 = top, so gy
    # descends). Only differences of gy are used here, so an offset is harmless.
    gx = np.arange(n_grid) * delta
    gy = (n_grid - 1 - np.arange(n_grid)) * delta

    frames, worst = [], {"reentrant_void_um": 0.0}
    all_thick, artefact_thick = [], []
    for ti in range(n_traj):
        for t in range(sdf.shape[1]):
            f = sdf[ti, t]
            sc = column_sign_changes(f)
            re_err = reentrant_depth_um(f, delta)
            tv = trapped_void_runs(f)
            crossed = sc > 0
            # A trapped void of 1-2 cells at a sidewall column is a level-set
            # corner artefact, not an undercut: it is present at t=0 before any
            # etch physics has run, at symmetric column pairs (49/50 and 77/78),
            # as a 2-row sliver of the wrong sign where the vertical wall meets
            # the trench floor. Real undercuts are counted separately.
            real = {c: [x for x in th if x > ARTEFACT_CELLS] for c, th in tv.items()}
            real = {c: th for c, th in real.items() if th}
            for th in tv.values():
                all_thick += th
                artefact_thick += [x for x in th if x <= ARTEFACT_CELLS]
            worst_um = max((max(th) for th in real.values()), default=0) * delta
            argmax_col = max(real, key=lambda c: max(real[c])) if real else -1
            frames.append({
                "traj": ti, "t": t,
                "n_columns_with_a_surface": int(crossed.sum()),
                "n_columns_reentrant": len(real),
                "n_columns_reentrant_incl_artefact": len(tv),
                "max_sign_changes": int(sc.max()),
                "max_reentrant_void_um": float(worst_um),
                "total_reentrant_void_um": float(re_err.sum()),
                "argmax_col": int(argmax_col),
            })
            if frames[-1]["max_reentrant_void_um"] > worst["reentrant_void_um"]:
                worst = {"reentrant_void_um": frames[-1]["max_reentrant_void_um"],
                         "traj": ti, "t": t, "col": frames[-1]["argmax_col"],
                         "sign_changes": frames[-1]["max_sign_changes"]}

    n_frames = len(frames)
    n_re_frames = sum(1 for r in frames if r["n_columns_reentrant"] > 0)
    re_trajs = sorted({r["traj"] for r in frames if r["n_columns_reentrant"] > 0})
    tot_cols = sum(r["n_columns_with_a_surface"] for r in frames)
    tot_re = sum(r["n_columns_reentrant"] for r in frames)
    by_t = {}
    for r in frames:
        by_t.setdefault(r["t"], []).append(r["n_columns_reentrant"])
    frac_by_t = {str(t): float(np.mean([x > 0 for x in v]))
                 for t, v in sorted(by_t.items())}
    maxsc = max(r["max_sign_changes"] for r in frames)
    at = np.asarray(all_thick)
    thick_hist = {str(int(k)): int(v) for k, v in
                  zip(*np.unique(at, return_counts=True))} if at.size else {}

    summary = {
        "n_frames": n_frames,
        "n_frames_with_a_reentrant_column": n_re_frames,
        "frac_frames_affected": n_re_frames / n_frames if n_frames else None,
        "n_trajectories_affected": len(re_trajs),
        "frac_trajectories_affected": len(re_trajs) / n_traj,
        "columns_with_a_surface_total": tot_cols,
        "columns_reentrant_total": tot_re,
        "frac_columns_reentrant": tot_re / tot_cols if tot_cols else None,
        "max_sign_changes_anywhere": maxsc,
        "worst_case": worst,
        "frac_frames_affected_by_timestep": frac_by_t,
        "grows_with_etch_time": (
            None if len(frac_by_t) < 3 else
            bool(list(frac_by_t.values())[-1] > list(frac_by_t.values())[0] + 0.05)),
    }
    summary["trapped_void_runs"] = {
        "n_total": int(at.size),
        "n_artefact_le_%d_cells" % ARTEFACT_CELLS: len(artefact_thick),
        "frac_artefact": (len(artefact_thick) / at.size) if at.size else None,
        "n_real": int(at.size) - len(artefact_thick),
        "max_cells": int(at.max()) if at.size else 0,
        "median_cells_of_real": (float(np.median([x for x in all_thick
                                                  if x > ARTEFACT_CELLS]))
                                 if at.size > len(artefact_thick) else None),
        "thickness_histogram_cells": thick_hist,
        "artefact_note": "runs of <= %d cells are the level-set corner artefact "
                         "described in column_sign_changes' caller: present at "
                         "t=0 before any etch, at symmetric sidewall column "
                         "pairs. Excluded from the verdict and counted here so "
                         "the exclusion is auditable." % ARTEFACT_CELLS,
    }
    per_eps = {"sign_changes": summary}

    strict = summary
    res = {
        "hypothesis": "H8: the fronts in this dataset are single-valued in x, so "
                      "a height representation is lossless up to grid resolution "
                      "and the surface-output route to clause 2 is open.",
        "why_it_matters": "runs/cost_floor.json puts a surface-output model at "
                          "5830x warm (8375x including rasterisation) against the "
                          "deployed field FNO's 3.2x and a 1000x clause. A height "
                          "field h(x) is single-valued by construction, so an "
                          "undercut or bowed front cannot be represented at all.",
        "protocol": {
            "split": a.split, "n_trajectories": n_traj,
            "timesteps_per_trajectory": int(sdf.shape[1]),
            "grid_delta_um": delta, "grid_n": n_grid,
            "measurement": "per column, void runs with solid both above and "
                           "below: exactly what no height field h(x) can "
                           "express. Runs of <= 2 cells are the corner artefact "
                           "and are excluded from the verdict but counted.",
            "sign_changes_are_diagnostic_only": "an earlier version thresholded "
                                                "sign changes at >=3. That misses "
                                                "solid-void-solid columns whose "
                                                "mask reaches the domain top, "
                                                "which read 2 -- the file recorded "
                                                "max_sign_changes 2 beside an "
                                                "84-cell trapped void, which is "
                                                "how the error surfaced. The "
                                                "verdict no longer uses them.",
            "superseded_measurement": "an earlier version binned zero-contour "
                                      "points by column and flagged spans over a "
                                      "grid cell. It flagged 100% of frames "
                                      "including t=0 with a worst span of 17.0 um "
                                      "-- the initial trench's vertical sidewall, "
                                      "which is height-representable. Those "
                                      "numbers are discarded, not reinterpreted.",
            "no_model_involved": True,
        },
        "readings": per_eps,
        "verdict": {
            "max_sign_changes_anywhere": maxsc,
            "frac_columns_reentrant": strict["frac_columns_reentrant"],
            "frac_frames_affected": strict["frac_frames_affected"],
            "worst_reentrant_void_um": strict["worst_case"]["reentrant_void_um"],
            "worst_in_grid_cells": strict["worst_case"]["reentrant_void_um"] / delta,
            "grows_with_etch_time": strict["grows_with_etch_time"],
            "H8_supported": bool(summary["trapped_void_runs"]["n_real"] == 0),
            "n_real_trapped_voids": summary["trapped_void_runs"]["n_real"],
            "criterion": "H8 holds only if NO column carries a trapped void "
                         "thicker than the 2-cell corner artefact",
            "reading": None,
        },
    }
    v = res["verdict"]
    v["reading"] = (
        f"SUPPORTED: no column in {strict['n_frames']} frames carries a trapped "
        f"void beyond the 2-cell corner artefact, so every front in this split is "
        f"single-valued in x and a height output is lossless up to grid resolution. "
        f"The surface route to clause 2 is open on representability -- which says "
        f"nothing about whether a model can learn it."
        if v["H8_supported"] else
        f"NOT SUPPORTED: {strict['frac_columns_reentrant']:.4f} of columns with a "
        f"surface carry a trapped void beyond the corner artefact, affecting "
        f"{strict['frac_frames_affected']:.4f} of frames and "
        f"{strict['frac_trajectories_affected']:.4f} of trajectories; the worst "
        f"trapped void a height field would have to fill in is "
        f"{v['worst_reentrant_void_um']:.4f} um "
        f"({v['worst_in_grid_cells']:.2f} grid cells). Growth with etch time: "
        f"{v['grows_with_etch_time']} -- true is a developing undercut, false with "
        f"a t=0 onset would be initial geometry.")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"verdict": v,
                      "summary": {k: vv for k, vv in summary.items()
                                  if k != "frac_frames_affected_by_timestep"}},
                     indent=2))
    print("\nfrac frames with a re-entrant column, by timestep:")
    print("  " + ", ".join(f"t{t}={f:.3f}" for t, f
                           in summary["frac_frames_affected_by_timestep"].items()))


if __name__ == "__main__":
    main()
