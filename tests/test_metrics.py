"""Shape metrics against geometry with a hand-computable answer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.metrics import hausdorff_um, shape_error, zero_contour_points  # noqa: E402

N = 128
GX, GY = S.grid_axes(N)


def _plane(h):
    return (GY[:, None] - h) * np.ones((1, N))


def test_contour_sits_on_the_zero_set():
    p = zero_contour_points(_plane(-3.0), GX, GY)
    assert len(p) > 50
    assert np.abs(p[:, 1] - (-3.0)).max() < 0.2


def test_hausdorff_of_two_planes_is_their_separation():
    d = hausdorff_um(_plane(-3.0), _plane(-5.0), GX, GY)
    assert abs(d["hausdorff_um"] - 2.0) < 0.05, d
    assert abs(d["mean_surface_dist_um"] - 2.0) < 0.05, d


def test_hausdorff_is_zero_for_identical_fields():
    d = hausdorff_um(_plane(-4.0), _plane(-4.0), GX, GY)
    assert d["hausdorff_um"] < 1e-6


def test_area_error_matches_hand_computation():
    """Planes at -3 and -5 over a 20 um wide window: mismatch is 2 x 20 = 40 um^2.

    Target removed from an initial surface at 0 is 3 x 20 = 60 um^2, so the
    normalised error is 40/60 = 0.667 -- while the same mismatch against the
    whole solid area is ~0.13. The gap between those two is exactly why the
    denominator is stated.
    """
    r = shape_error(_plane(-5.0), _plane(-3.0), _plane(0.0), GX, GY)
    # the hard-threshold area is a whole cell row out; that gap is the reason
    # the sub-cell occupancy exists
    assert abs(r["mismatch_area_um2_hard_threshold"] - 40.0) > 0.9, r
    assert abs(r["mismatch_area_um2"] - 40.0) < 0.5, r
    assert abs(r["target_removed_area_um2"] - 60.0) < 0.5, r
    assert abs(r["area_error_vs_removed"] - 2.0 / 3.0) < 0.05, r
    assert r["area_error_vs_solid"] < 0.2, r


def test_perfect_match_is_zero_error():
    r = shape_error(_plane(-4.0), _plane(-4.0), _plane(0.0), GX, GY)
    assert r["area_error_vs_removed"] == 0.0
    assert r["hausdorff_um"] < 1e-6




def test_mean_step_displacement_recovers_a_known_advance():
    """A surface stepped down by a known amount must return that amount.

    This constant is the recipe-blind null. If it were wrong the null would be
    weak for the wrong reason and the operator would look better than it is.
    """
    from eot.data import mean_step_displacement

    step = 0.4
    T = 6
    frames = np.stack([_plane(-1.0 - step * t) for t in range(T + 1)])
    sdf = frames[None]  # (1, T+1, H, W)
    d = mean_step_displacement(sdf.astype(np.float32))
    # the surface moves down by `step`, so phi at a fixed point increases by step
    assert abs(d - step) < 0.02, d


def test_mean_step_displacement_is_zero_for_a_static_surface():
    from eot.data import mean_step_displacement

    sdf = np.stack([_plane(-2.0)] * 5)[None].astype(np.float32)
    assert abs(mean_step_displacement(sdf)) < 1e-6



def test_symmetric_bench_pairs_cold_with_cold_and_warm_with_warm():
    """The defect `bench_symmetric.py` exists to fix: a cold solver timed against
    a warm operator, with the ratio labelled like-for-like.

    ViennaPS charges a one-time initialisation inside .apply() (measured: first
    wafer 1.257 s, later wafers 0.10-0.40 s, runs/solver_drift.json), so which
    side pays its one-time cost decides the answer by ~3x. This pins the
    invariant that each reading uses the same warm-up policy on both sides.
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "bench_symmetric.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    # marginal_warm: the solver discards its first wafer, the operator warms up
    assert "n_discard" in [a.arg for a in fns["solver_warm"].args.args + fns["solver_warm"].args.kwonlyargs] \
        or fns["solver_warm"].args.defaults, "solver_warm must discard warm-up wafers"
    # cold_single_wafer: the operator is timed in a fresh process, like the solver
    assert "n_warm" in [a.arg for a in fns["operator_run"].args.args], \
        "operator_run must expose n_warm so a cold reading can ask for zero"

    main_src = ast.get_source_segment(src, fns["main"])
    assert "n_warm=3" in main_src and "n_warm=0" in main_src, \
        "both a warm and a cold operator reading must be taken"
    assert "solver_cold" in main_src and "solver_warm" in main_src, \
        "both a warm and a cold solver reading must be taken"


def test_no_test_is_defined_after_its_files_main_block():
    """Every test file runs itself with `python tests/test_x.py`, iterating
    globals() from inside `if __name__ == "__main__"`. A test function defined
    BELOW that block is never reached -- the runner has already finished by the
    time the interpreter defines it -- so it silently never executes, in CI
    either.

    This had happened in four files at once. `test_inverse_protocol.py` hid 3
    tests that way and `test_solver.py` 1, and the four had never run a single
    time; all four pass, so nothing was broken, but the guards they check were
    unguarded. Found by deriving the test count in scripts/weekend.py and
    noticing it said 72 where the runners printed 68.
    """
    from pathlib import Path

    bad = {}
    for t in sorted((Path(__file__).resolve().parent).glob("test_*.py")):
        lines = t.read_text().splitlines()
        main_at = next((i for i, ln in enumerate(lines)
                        if ln.startswith('if __name__ ==')), None)
        if main_at is None:
            continue
        after = [ln.split("(")[0][4:] for ln in lines[main_at:]
                 if ln.startswith("def test_")]
        if after:
            bad[t.name] = after
    assert not bad, (
        "these tests are defined after their file's __main__ block and can "
        f"never run: {bad}. Move the __main__ block to the end of the file.")

def test_reentrancy_criterion_is_trapped_void_not_sign_count():
    """A trapped void reads solid-void-solid, which is TWO sign changes when the
    column starts solid at the domain top and THREE when it starts void. So no
    threshold on sign-change count detects re-entrancy, and an earlier version of
    scripts/surface_representable.py used `>= 3` and missed the dominant case in
    this dataset (mask reaching the top of the domain).

    Builds both columns explicitly and pins that the trapped-void detector finds
    each while the sign counter disagrees with itself across them.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.surface_representable import column_sign_changes, trapped_void_runs

    # column starting VOID at the top: void, solid slab, cavity, solid -> 3 changes
    starts_void = np.array([[+1.], [+1.], [-1.], [-1.], [+1.], [+1.], [+1.], [-1.], [-1.]])
    # column starting SOLID at the top: solid slab, cavity, solid -> 2 changes
    starts_solid = np.array([[-1.], [-1.], [+1.], [+1.], [+1.], [-1.], [-1.], [-1.], [-1.]])

    assert column_sign_changes(starts_void)[0] == 3
    assert column_sign_changes(starts_solid)[0] == 2, (
        "this is the case a >=3 threshold silently misses")

    # the criterion that actually works finds the cavity in BOTH
    assert trapped_void_runs(starts_void) == {0: [3]}
    assert trapped_void_runs(starts_solid) == {0: [3]}

    # and a plain single interface is re-entrant under neither
    single = np.array([[+1.], [+1.], [+1.], [-1.], [-1.], [-1.]])
    assert column_sign_changes(single)[0] == 1
    assert trapped_void_runs(single) == {}


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
