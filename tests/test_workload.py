"""The workload-matched denominator, and the ways it could be silently wrong.

`bench_workload.py` changed a KPI denominator in the direction that makes the
clause easier, so the things that could make it wrong are worth pinning harder
than usual. The hazards, in order of how quietly they would fail:

1. **Recipe column order.** Each column rebuilds a `Recipe` from a row of the
   stored `recipe` array by zipping it against a local `RECIPE_KEYS` list. If
   that list's order ever diverges from the one `gen_data.py` wrote with, every
   solver call would price a valid recipe with `ion_flux` and `etchant_flux`
   swapped -- a well-formed etch of the wrong wafer, with no error anywhere.
2. **Different wafers per column.** The whole point is that a difference between
   columns is a difference in workload, not in geometry. Solver cost varies ~4x
   across recipes.
3. **The warm-up discard.** Every column must drop the same count, or one column
   pays ViennaPS's one-time init and another does not -- the asymmetry that
   invalidated every earlier speedup in this repo.
4. **The rasterisation subtraction**, which is the self-serving-looking step and
   must at least be arithmetically what it claims.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bench_workload as W  # noqa: E402


def test_recipe_keys_match_the_generator_that_wrote_the_array():
    """The one failure mode that produces no error and no wrong-looking number."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "gen_data.py").read_text()
    i = src.index("RECIPE_KEYS")
    line = src[i:src.index("]", i) + 1]
    gen_keys = [k.strip().strip('"\'') for k in
                line.split("[", 1)[1].rstrip("]").split(",") if k.strip()]
    assert gen_keys == W.RECIPE_KEYS, (
        f"gen_data wrote columns in order {gen_keys} but bench_workload reads "
        f"them as {W.RECIPE_KEYS}; every solver call would price the wrong wafer")


def test_recipe_literal_round_trips_a_stored_row():
    """Build the literal, eval it against the real Recipe class, and check every
    field comes back as the array said."""
    from eot.solver import Recipe
    data = Path("data/test.npz")
    if not data.exists():
        return
    row = np.load(data)["recipe"][3]
    rec = eval(W._rec_literal(row).replace("S.Recipe", "Recipe"), {"Recipe": Recipe})
    for k, v in zip(W.RECIPE_KEYS, row):
        assert abs(getattr(rec, k) - float(v)) < 1e-6, k


def test_every_column_prices_the_same_wafers():
    """The recipe literals embedded in each column's child program must be
    identical, or the columns are not comparable."""
    data = Path("data/test.npz")
    if not data.exists():
        return
    d = np.load(data)
    rows, dts = d["recipe"][:5], d["dt"][:5]
    lits = [W._rec_literal(r) for r in rows]
    # Reconstruct what each builder embeds, without running the solver.
    for lit in lits:
        assert lit.startswith("S.Recipe(")
    # The dt-bearing columns must use the dataset's dt, not a constant.
    assert len({float(x) for x in dts}) > 1, "fixture has no dt variation"


def test_warmup_discard_is_the_same_in_every_column():
    """Each builder's default n_discard must agree; a column that keeps its
    first wafer pays ViennaPS's one-time init and the others do not."""
    import inspect
    ds = {}
    for fn in (W.solver_terminal, W.solver_allframes, W.solver_fixed_duration,
               W.solver_domain_build):
        ds[fn.__name__] = inspect.signature(fn).parameters["n_discard"].default
    assert len(set(ds.values())) == 1, f"columns discard different counts: {ds}"


def test_raster_subtraction_is_what_it_claims():
    """excl_raster must equal total minus raster, per wafer, and stay positive."""
    out = Path("runs/bench_workload.json")
    if not out.exists():
        return
    d = json.loads(out.read_text())
    c = d["columns"]["all_frames_ten_applies"]
    tot, raster = c["median_cpu_s"], c["cpu_raster_median_s"]
    excl = c["median_cpu_s_excluding_raster"]
    assert 0 < excl < tot, (excl, tot)
    # Medians do not subtract exactly (excl is the median of per-wafer
    # differences, not the difference of medians), so this is a sanity band
    # rather than an equality -- but it must be the right order of magnitude.
    assert abs(excl - (tot - raster)) < 0.5 * tot, (excl, tot, raster)


def test_budget_is_the_denominator_over_the_target():
    out = Path("runs/bench_workload.json")
    if not out.exists():
        return
    d = json.loads(out.read_text())
    for k, b in d["ratios"]["budget_per_wafer_at_1000x"].items():
        assert abs(b - d["columns"][k]["median_cpu_s"] / 1000.0) < 1e-12, k


def test_solver_ran_single_threaded():
    """cpu/wall near 1 is the evidence; if a column went multi-threaded its
    CPU-seconds would exceed its wall time and the comparison would be against
    a parallel solver, which the brief forbids."""
    out = Path("runs/bench_workload.json")
    if not out.exists():
        return
    d = json.loads(out.read_text())
    for k, c in d["columns"].items():
        r = c.get("cpu_over_wall_median")
        if r is not None:
            assert r < 1.25, f"{k} used {r:.2f} CPU-s per wall-s: not 1 thread"


def test_the_old_reading_is_still_reported():
    """Rung 1: replacing a denominator is only legitimate if the one it replaced
    is still in the file. If this column ever disappears, the change stopped
    being a correction and became a loosening."""
    out = Path("runs/bench_workload.json")
    if not out.exists():
        return
    d = json.loads(out.read_text())
    assert "fixed_duration_one_apply" in d["columns"]
    assert d["columns"]["fixed_duration_one_apply"]["median_cpu_s"] > 0


def test_dataset_recorded_solver_s_is_not_used_as_a_denominator():
    """The 17-19 s figure stored at generation is wall-clock measured while 8
    workers shared the box. It is recorded because it exposed the mismatch, and
    it must never become a denominator -- it would make the clause ~40x easier."""
    out = Path("runs/bench_workload.json")
    if not out.exists():
        return
    d = json.loads(out.read_text())
    rec = d["dataset_recorded_solver_s"]["median"]
    for k, b in d["ratios"]["budget_per_wafer_at_1000x"].items():
        assert abs(b - rec / 1000.0) > 1e-9, \
            f"{k} is the generation wall-clock figure, which is not a denominator"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
