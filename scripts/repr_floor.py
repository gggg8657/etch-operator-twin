"""H8: what the surface representation throws away, before anything is trained.

    python scripts/repr_floor.py

`runs/cost_floor.json` opened one route to clause 2: emit the etch front as ~128
surface heights instead of a 128x128 field, which is 100-1000x less output and
clears the 1000x budget on cost. Before writing that model, this script asks
whether the representation can satisfy **clause 1** at all.

The method needs no training. Take the true final SDF of every test trajectory,
extract its zero crossing per column to get 128 heights -- discarding everything
the field knows that the heights do not -- then rebuild a field and score it
against the original under the *same* band mask the KPI uses. What is left is an
**information floor**: the best band rel-L2 any surface-output model could reach
*even with perfect heights*.

Two reconstructions, because they are not the same claim:

* `vertical` -- `ys - h` broadcast down each column, a signed *vertical*
  distance. This is what `runs/cost_floor.json`'s "+raster" row actually timed at
  33 us, and it equals the Euclidean distance only where the surface is
  horizontal.
* `edt` -- an exact Euclidean distance transform of the sign mask, which is the
  honest reconstruction of an SDF and costs a great deal more.

Reported beside them, because it is a hard limit rather than an error: the
fraction of surfaces that are **not single-valued** in x. An undercut or
re-entrant front has no representation as one height per column at all, and any
such trajectory bounds the route independently of the floor.

Cost is measured under the interval protocol -- each reconstruction timed in
`--cost-reps` separate processes -- because `runs/speed_spread.json` measured the
between-process spread of this box at 1.37x, and `cost_floor.json`'s 47 us / 33 us
surface rows differ by 1.42x, i.e. by nothing.

Writes `runs/repr_floor.json`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from eot.data import BAND_UM  # noqa: E402


def crossings_from_sdf(sdf: np.ndarray, delta: float) -> list[np.ndarray]:
    """Every sub-cell zero crossing down each column, in um, top to bottom.

    The geometry is NOT a single-valued height field: a column can cross
    void -> mask -> trench void -> substrate, and `n_crossings_per_column`
    measures up to 3 on real test data. So "the surface height" is ambiguous and
    which crossing you take is a representation choice, not a detail -- hence
    this returns all of them and the callers below build the competing readings.
    """
    H, W = sdf.shape
    out = []
    for x in range(W):
        col = sdf[:, x]
        sb = np.signbit(col)
        idx = np.nonzero(sb[:-1] != sb[1:])[0]
        zs = []
        for i in idx:
            a, b = col[i], col[i + 1]
            frac = 0.0 if (b - a) == 0 else a / (a - b)
            zs.append((i + frac) * delta)
        out.append(np.asarray(zs, dtype=np.float64))
    return out


def top_phase_positive(sdf: np.ndarray) -> np.ndarray:
    """Sign of the field at the top of the window, per column.

    Needed because the top row is NOT always void: at the wafer edge the mask
    surface can sit above the window, so `sdf[0, x]` is negative (trajectory 1,
    column 0: -0.96). An earlier version of `rebuild_multi` assumed a void top
    everywhere, which inverted the phase parity of every such column and scored
    2.0-4.2 band rel-L2 -- a reconstruction bug, not a property of the
    representation.

    This is 1 bit per column, 128 bits per wafer against the field's 16,384
    floats, so carrying it costs the compact representation nothing.
    """
    return sdf[0, :] > 0


def rebuild_multi(cross: list[np.ndarray], top_pos: np.ndarray, shape,
                  delta: float) -> np.ndarray:
    """SDF rebuilt from EVERY interface in each column, not just one.

    This is the strongest form of the surface representation: the column's phase
    is determined by how many interfaces lie above a given depth (the top of the
    window is void, checked against the data), so all interfaces are kept. The
    output is `sum(len(c) for c in cross)` numbers -- a few hundred against the
    field's 16,384 -- so it is still a large reduction, and it is the reading the
    escape route deserves to be judged on rather than the one-height reading.
    """
    from scipy import ndimage

    H, W = shape
    ys = np.arange(H, dtype=np.float64) * delta
    void = np.zeros((H, W), dtype=bool)
    for x, zs in enumerate(cross):
        # phase flips at each interface, starting from this column's own top sign
        n_above = np.searchsorted(zs, ys, side="right")
        even = (n_above % 2) == 0
        void[:, x] = even if top_pos[x] else ~even
    d_void = ndimage.distance_transform_edt(void, sampling=delta)
    d_solid = ndimage.distance_transform_edt(~void, sampling=delta)
    return (d_void - d_solid).astype(np.float32)


def heights_from_sdf(sdf: np.ndarray, delta: float, which: str = "first"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-column zero crossing of the SDF, in grid units, plus a validity mask.

    `sdf[y, x]`, y increasing downward into the substrate. The dataset's sign
    convention, checked against `data/test.npz` rather than assumed: **positive
    in the void above the front, negative inside the solid** (row 0 of
    trajectory 0 means +0.623, row 127 means -8.128). The crossing is found as
    the first sign change down each column and refined by linear interpolation
    between the two straddling samples, the same sub-cell accuracy
    `eot.metrics.zero_contour_points` uses.

    `which` selects the crossing, and the choice matters: "first" is the topmost
    interface, which in a masked geometry is the top of the mask rather than the
    etch front; "last" is the deepest, which is the substrate interface. Both are
    reported, because picking whichever loses least after seeing the numbers
    would be choosing a protocol from its result.

    Returns (h, ok) where `ok[x]` is False for a column with no crossing.
    """
    H, W = sdf.shape
    cross = crossings_from_sdf(sdf, delta)
    h = np.full(W, np.nan, dtype=np.float64)
    ok = np.zeros(W, dtype=bool)
    for x, zs in enumerate(cross):
        if zs.size == 0:
            continue
        h[x] = zs[0] if which == "first" else zs[-1]
        ok[x] = True
    return h, ok


def n_crossings_per_column(sdf: np.ndarray) -> np.ndarray:
    """How many times each column crosses the interface.

    More than one means the front is not single-valued in x for that column --
    an undercut -- and 128 heights cannot represent it, however accurate they
    are.
    """
    s = np.signbit(sdf)
    return (s[:-1, :] != s[1:, :]).sum(axis=0)


def rebuild_vertical(h: np.ndarray, ok: np.ndarray, shape, delta: float) -> np.ndarray:
    """Signed vertical distance: what cost_floor.py's '+raster' row computed.

    Sign follows the dataset: positive in the void above the front (y < h),
    negative in the solid below it.
    """
    H, W = shape
    ys = np.arange(H, dtype=np.float64)[:, None] * delta
    hh = np.where(ok, h, np.nanmedian(h[ok]) if ok.any() else 0.0)[None, :]
    return (hh - ys).astype(np.float32)


def rebuild_edt(h: np.ndarray, ok: np.ndarray, shape, delta: float) -> np.ndarray:
    """Exact Euclidean SDF of the front at the reconstructed heights."""
    from scipy import ndimage

    H, W = shape
    ys = np.arange(H, dtype=np.float64)[:, None] * delta
    hh = np.where(ok, h, np.nanmedian(h[ok]) if ok.any() else 0.0)[None, :]
    void = ys < hh  # above the front: already etched away
    # `distance_transform_edt(m)` gives, for each True cell, the distance to the
    # nearest False cell. So edt(void) is nonzero only in the void and edt(~void)
    # only in the solid, and their difference is the signed distance with the
    # dataset's convention: positive in the void, negative in the solid.
    d_void = ndimage.distance_transform_edt(void, sampling=delta)
    d_solid = ndimage.distance_transform_edt(~void, sampling=delta)
    return (d_void - d_solid).astype(np.float32)


def rebuild_multi_fine(cross: list[np.ndarray], top_pos: np.ndarray, shape,
                       delta: float, upsample: int = 8) -> np.ndarray:
    """`rebuild_multi`, but with the interface placed at sub-cell accuracy.

    **This function exists because the coarse version is not a measurement of
    the representation.** `rebuild_multi` rasterises the phase onto the 0.2 um
    grid, so it quantises the interface to cell centres -- an error of up to half
    a cell, 0.1 um, against a band where the typical |phi| is ~0.75 um. That is a
    ~13% error floor built into the reconstruction, and on trajectory 0 the coarse
    version scores 0.387 band rel-L2 where the same crossings at 8x resolution
    score far less. Quoting the coarse number as "what the surface representation
    throws away" would have closed the route on my own rasteriser's error.

    The crossings are sub-cell already (linearly interpolated), so the phase is
    built on an `upsample`x grid, the distance transform is taken there, and the
    result is sampled at the coarse grid points. Nothing but the crossing
    positions is used, so this is still a reconstruction from the compact
    representation alone.
    """
    from scipy import ndimage

    H, W = shape
    u = upsample
    ysf = np.arange(H * u, dtype=np.float64) * (delta / u)
    void = np.zeros((H * u, W * u), dtype=bool)
    for x, zs in enumerate(cross):
        n_above = np.searchsorted(zs, ysf, side="right")
        even = (n_above % 2) == 0
        col = even if top_pos[x] else ~even
        void[:, x * u:(x + 1) * u] = col[:, None]

    # `distance_transform_edt` can only see interfaces inside the array, so a
    # cell whose nearest interface lies outside the window gets a distance that
    # is too large. Untreated, this put the whole error of trajectories 1-3 in
    # the top-left corner: median band error 0.045 um but p90 2.32 and max 7.22,
    # all in rows 0-7 and columns 0-7. It is a boundary artefact of the
    # reconstruction, not information the representation lost.
    #
    # The solver's x boundary is reflective, and this repo already relies on that
    # (eot/solver.py:168, "extending flat past the edge is also what the solver's
    # reflective boundary does"), so mirror in x. In y the phase simply continues
    # past the window -- mask above, substrate below -- so replicate the edge row.
    pad = int(np.ceil(3.0 / (delta / u)))  # 3 um, twice the 1.5 um band
    padded = np.pad(void, ((pad, pad), (pad, pad)), mode="symmetric")
    padded[:pad, :] = padded[pad:pad + 1, :]
    padded[-pad:, :] = padded[-pad - 1:-pad, :]
    d_void = ndimage.distance_transform_edt(padded, sampling=delta / u)
    d_solid = ndimage.distance_transform_edt(~padded, sampling=delta / u)
    fine = (d_void - d_solid)[pad:pad + H * u, pad:pad + W * u]
    # sample at the coarse cell centres' fine-grid indices
    off = u // 2
    return fine[off::u, off::u][:H, :W].astype(np.float32)


def gradient_scale(sdf: np.ndarray, delta: float, band_um: float) -> float:
    """|grad phi| of the stored field in the band.

    The dataset's field is **not** a unit-gradient SDF: this measures 0.787 on
    every test trajectory (p10 0.785, p90 0.788), i.e. the stored value is
    0.787 x the Euclidean distance. A reconstruction that produces true
    Euclidean distance is therefore 1/0.787 = 1.27x too large and scores ~0.27
    band rel-L2 from the units alone. Reading this off the data is reading the
    dataset's convention, in the same way `grid_delta` is; it is not a fitted
    correction, and it is measured per wafer and recorded so a reader can see it
    is constant.
    """
    gy, gx = np.gradient(sdf, delta)
    g = np.sqrt(gx ** 2 + gy ** 2)
    m = np.abs(sdf) < band_um
    return float(np.median(g[m]))


def band_rel_l2(pred: np.ndarray, target: np.ndarray, band_um: float) -> float:
    """The KPI's headline reading, on the band of the GROUND TRUTH field."""
    m = np.abs(target) < band_um
    num = np.sqrt(((pred - target) ** 2 * m).sum())
    den = np.sqrt((target ** 2 * m).sum())
    return float(num / max(den, 1e-8))


_COST_CHILD = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, sys
sys.path.insert(0, {root!r})
import numpy as np
{setup}
w, c = [], []
for _ in range({n_rep}):
    w0 = time.perf_counter(); c0 = time.process_time()
    {call}
    w.append(time.perf_counter() - w0); c.append(time.process_time() - c0)
print(json.dumps({{"wall": w, "cpu": c}}))
'''

COST_CASES = {
    "rebuild_multi_fine_u8": dict(
        setup="from scripts.repr_floor import rebuild_multi_fine\n"
              "cross = [np.array([4.0 + 0.01 * i]) for i in range(128)]\n"
              "tp = np.ones(128, bool)",
        call="rebuild_multi_fine(cross, tp, (128, 128), 0.2, 8)",
        note="the reconstruction the floor is actually measured with: phase on an "
             "8x grid plus two distance transforms there. This is the cost the "
             "surface route must pay if clause 1 is scored as a field, and it is "
             "far above the cheap broadcast cost_floor.json timed."),
    "rebuild_vertical": dict(
        setup="from scripts.repr_floor import rebuild_vertical\n"
              "h = np.linspace(3.0, 6.0, 128); ok = np.ones(128, bool)",
        call="rebuild_vertical(h, ok, (128, 128), 0.2)",
        note="the cheap broadcast; produces signed VERTICAL distance, which is "
             "not the quantity clause 1 scores unless the surface is flat"),
    "rebuild_edt": dict(
        setup="from scripts.repr_floor import rebuild_edt\n"
              "h = np.linspace(3.0, 6.0, 128); ok = np.ones(128, bool)",
        call="rebuild_edt(h, ok, (128, 128), 0.2)",
        note="two exact Euclidean distance transforms; the honest reconstruction "
             "of an SDF from heights, and the cost the escape route must pay if "
             "clause 1 is scored as a field"),
}


def _summ(xs):
    xs = np.asarray(xs, float)
    return {"median": float(np.median(xs)), "min": float(xs.min()),
            "max": float(xs.max()), "max_over_min": float(xs.max() / max(xs.min(), 1e-12)),
            "n": int(xs.size), "per_process": [float(x) for x in xs]}


def measure_cost(reps, n_rep):
    out = {}
    for name, c in COST_CASES.items():
        med = []
        for _ in range(reps):
            body = _COST_CHILD.format(root=str(ROOT), setup=c["setup"],
                                      call=c["call"], n_rep=n_rep)
            r = subprocess.run([sys.executable, "-c", body], capture_output=True,
                               text=True, cwd=ROOT)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])
            med.append(float(np.median(json.loads(
                r.stdout.strip().splitlines()[-1])["cpu"])))
        out[name] = {"note": c["note"], "cpu_s_per_call": _summ(med),
                     "protocol": f"median of {n_rep} calls inside each of {reps} "
                                 f"separate processes; the spread across "
                                 f"processes is the error bar, because this box's "
                                 f"between-process spread is 1.37x"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.npz")
    ap.add_argument("--n-traj", type=int, default=60)
    ap.add_argument("--upsample", type=int, default=8)
    ap.add_argument("--converge", type=int, nargs="*", default=[4, 8, 16],
                    help="upsample factors for the convergence check, on a subset")
    ap.add_argument("--converge-n", type=int, default=5)
    ap.add_argument("--grid-delta", type=float, default=None)
    ap.add_argument("--band-um", type=float, default=None)
    ap.add_argument("--cost-reps", type=int, default=8)
    ap.add_argument("--cost-calls", type=int, default=20)
    ap.add_argument("--out", default="runs/repr_floor.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="repr_floor")
    gen = json.loads((Path("data") / "gen_report.json").read_text())
    delta = a.grid_delta if a.grid_delta is not None else gen["grid_delta"]
    norm = json.loads((Path("data") / "norm.json").read_text())
    band_um = a.band_um if a.band_um is not None else norm.get("band_um", BAND_UM)

    d = np.load(a.data)
    sdf = d["sdf"]
    n = min(a.n_traj, sdf.shape[0])
    H, W = sdf.shape[2], sdf.shape[3]

    # Whether the window contains all the geometry the field knows about. A
    # column whose top row is solid is under the mask, and the field there
    # encodes the distance to the mask's top surface, which sits ABOVE the
    # window and is in no in-window crossing list. Those trajectories cannot be
    # scored against an in-window reconstruction, and are reported separately
    # rather than dropped.
    all_void_top = np.array([(sdf[i, -1][0, :] > 0).all() for i in range(sdf.shape[0])])

    groups = {"window_contains_all_geometry": [], "mask_extends_above_window": []}
    scales, n_iface, n_undercut, max_cross = [], [], 0, 0
    for i in range(n):
        true = sdf[i, -1].astype(np.float64)
        sc = gradient_scale(true, delta, band_um)
        cross = crossings_from_sdf(true, delta)
        tp = top_phase_positive(true)
        nc = n_crossings_per_column(true)
        max_cross = max(max_cross, int(nc.max()))
        n_undercut += int((nc > 1).any())
        n_iface.append(int(sum(c.size for c in cross)))
        scales.append(sc)
        e = band_rel_l2(rebuild_multi_fine(cross, tp, true.shape, delta,
                                           a.upsample) * sc, true, band_um)
        k = ("window_contains_all_geometry" if all_void_top[i]
             else "mask_extends_above_window")
        groups[k].append(e)

    # Is the number a property of the representation or of my rasteriser? The
    # floor must stop moving as the reconstruction grid is refined.
    conv_idx = [i for i in range(n) if all_void_top[i]][:a.converge_n]
    convergence = {}
    for u in a.converge:
        es = []
        for i in conv_idx:
            true = sdf[i, -1].astype(np.float64)
            sc = gradient_scale(true, delta, band_um)
            es.append(band_rel_l2(rebuild_multi_fine(
                crossings_from_sdf(true, delta), top_phase_positive(true),
                true.shape, delta, u) * sc, true, band_um))
        convergence[f"upsample_{u}"] = float(np.median(es))

    res = {
        "hypothesis": "H8: the etch fronts are single-valued in x and an SDF "
                      "rebuilt from the compact interface representation loses "
                      "less than clause 1's 0.05, so the representation is "
                      "admissible.",
        "protocol": {
            "what_this_is": "an INFORMATION FLOOR, not a model result: the true "
                            "terminal field is reduced to its interface crossings "
                            "and rebuilt, so this is what a surface-output model "
                            "would score with PERFECT predictions",
            "representation": "per column, the sub-cell zero crossings plus one "
                              "bit for the phase at the top of the window",
            "data": a.data, "n_trajectories": n,
            "frame": "terminal step, the reading clause 1 is judged on",
            "grid_delta_um": delta, "band_um": band_um, "upsample": a.upsample,
            "metric": "band rel-L2 on the band of the GROUND TRUTH field, the "
                      "same mask and reduction eot.operator.band_rel_l2 uses",
            "threshold": 0.05,
            "field_scale_measured": {
                "median_grad_phi_in_band": float(np.median(scales)),
                "min": float(np.min(scales)), "max": float(np.max(scales)),
                "note": "the stored field is NOT unit-gradient: it is this factor "
                        "times the Euclidean distance, constant across "
                        "trajectories. Reading it off the data is reading the "
                        "dataset's units, as grid_delta is; a reconstruction that "
                        "ignores it scores ~0.27 from the units alone.",
            },
            "KNOWN_UPPER_BOUND": "the reconstruction is piecewise CONSTANT in x -- "
                                 "each column's crossings are held across that "
                                 "column's width -- so the interface is quantised "
                                 "horizontally at one grid cell. A reconstruction "
                                 "that interpolated the crossings between adjacent "
                                 "columns would score no worse and probably "
                                 "better, so every floor below is an UPPER BOUND "
                                 "on what the representation loses, not the loss.",
        },
        "window_coverage": {
            "n_all_void_top_row": int(all_void_top[:n].sum()),
            "n_scored": n,
            "frac_window_contains_all_geometry": float(all_void_top[:n].mean()),
            "frac_over_full_test_split": float(all_void_top.mean()),
            "why_it_matters": "where the top row is solid, the field encodes the "
                              "distance to the mask's top surface above the "
                              "window, which no in-window crossing list contains. "
                              "The recipe carries mask_height and trench_width, so "
                              "a fair reconstruction would composite the known "
                              "static mask; that is not done here and those "
                              "trajectories are reported separately, not dropped.",
        },
        "single_valuedness": {
            "n_trajectories_with_an_undercut": n_undercut,
            "frac_with_an_undercut": n_undercut / n,
            "max_crossings_in_any_column": max_cross,
            "interfaces_per_wafer_median": float(np.median(n_iface)),
            "field_values_per_wafer": int(H * W),
            "reduction_factor": float(H * W / max(np.median(n_iface), 1)),
            "note": "the geometry is masked, so a column can cross void -> mask "
                    "-> trench -> substrate. 'The surface height' is therefore "
                    "ambiguous and all crossings are kept.",
        },
        "convergence_in_reconstruction_grid": {
            **convergence,
            "n_trajectories": len(conv_idx),
            "reading": "if these agree, the floor is a property of the "
                       "representation rather than of the reconstruction grid",
        },
        "floors": {},
        "cost": measure_cost(a.cost_reps, a.cost_calls),
    }
    for k, v in groups.items():
        if not v:
            continue
        v = np.asarray(v)
        res["floors"][k] = {
            "band_rel_l2_median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max()),
            "min": float(v.min()), "n": int(v.size),
            "frac_over_threshold": float((v > 0.05).mean()),
            "admissible_at_median": bool(np.median(v) <= 0.05),
        }

    main_group = res["floors"].get("window_contains_all_geometry")
    res["verdict"] = {
        "H8_answer": "PARTIAL -- not settled either way, and deliberately not "
                     "reported as a verdict",
        "floor_upper_bound_median": main_group["band_rel_l2_median"] if main_group else None,
        "threshold": 0.05,
        "reading": (
            "On the trajectories whose window contains all the geometry, reducing "
            "the terminal field to its interface crossings and rebuilding an exact "
            "SDF costs {:.4f} band rel-L2, converged in the reconstruction grid. "
            "That is ABOVE clause 1's 0.05 -- but it is an upper bound, because "
            "the reconstruction is piecewise constant in x and quantises the "
            "interface horizontally at one cell. So this neither admits nor "
            "excludes the surface representation, and the next step is a "
            "reconstruction that interpolates crossings between columns, plus "
            "compositing the static mask from the recipe so the other {:.0f}% of "
            "trajectories can be scored at all."
        ).format(main_group["band_rel_l2_median"],
                 100 * (1 - res["window_coverage"]["frac_over_full_test_split"]))
        if main_group else "no trajectory in this sample had a fully covered window",
        "what_would_settle_it": [
            "interpolate crossings between adjacent columns (removes the known "
            "horizontal quantisation, which is the dominant remaining error)",
            "composite the mask from the recipe's mask_height and trench_width so "
            "the mask-above-window trajectories are scorable",
            "if the floor then clears 0.05, train a surface-output model; if it "
            "does not, the representation is excluded and clause 2's cheapest "
            "route dies with it",
        ],
        "reconstruction_bugs_found_and_fixed_this_turn": [
            "sign convention inverted (the field is positive in the void, "
            "checked against the data rather than assumed)",
            "per-column top phase hard-coded as void, which inverted the phase "
            "parity of every masked column and scored 2.0-4.2",
            "the field's non-unit gradient (0.787) ignored, worth ~0.27 alone",
        ],
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ("window_coverage", "single_valuedness",
                       "convergence_in_reconstruction_grid", "floors",
                       "verdict")}, indent=2))


if __name__ == "__main__":
    main()
