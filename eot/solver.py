"""ViennaPS ground truth: 2-D plasma-etch surface evolution on a fixed window.

The simulator carries its own adaptive level-set grid whose bounding box grows as
the trench deepens. Learning wants the opposite -- one fixed frame for every
sample and every timestep -- so the state we store is not ViennaPS's level set but
an *exact geometric* signed distance field, recomputed on a fixed window from the
explicit surface polyline the simulator hands back. Sign is negative inside solid
material, positive in the gas above it. That keeps the representation
frame-stable, resolution-independent and free of the narrow-band bookkeeping.

Version pin that matters: viennaps 4.6.2 needs viennals **5.8.5**. The current
5.9.0 wheel changes the registered `viennahrle::BoundaryType` and `import
viennaps` dies at load with "could not convert default argument". `pip install
viennaps` resolves to 5.9.0 on its own, so the pin is not optional.
"""
from __future__ import annotations

import os

# ViennaPS 4.6.2 SEGFAULTs above ~96 OpenMP threads on this 192-core host.
# Measured, deterministic, bisected: exit 0 at 2/8/16/32/64/80/96 threads and
# exit 139 at 112/128/192 (the unset default is nproc = 192, so an unpinned
# import crashes here 3 times out of 3). This is not load-dependent flakiness --
# it reproduces at every thread count above the threshold and never below it.
#
# The pin has to happen before viennaps is imported, so it lives here rather
# than in each caller. Callers that want more threads set OMP_NUM_THREADS
# themselves and stay under the cliff; nothing in this repo needs more than 16.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "8"


import time
from dataclasses import dataclass, asdict, field

import numpy as np

# Fixed physical window every sample is rasterised onto (micron).
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -18.0, 2.0


def _lazy_vps():
    """Import ViennaPS once, set units once. Import is ~1s, so cache it."""
    global _VPS
    try:
        return _VPS
    except NameError:
        pass
    import viennaps as vps

    vps.Logger.setLogLevel(vps.LogLevel.ERROR)
    vps.Length.setUnit("um")
    vps.Time.setUnit("min")
    _VPS = vps
    return _VPS


@dataclass
class Recipe:
    """The four knobs inverse design is allowed to turn, plus the geometry.

    Ranges are the box the training set is sampled from; anything found outside
    it is an extrapolated recipe and is reported as such, never as a solution.
    """

    ion_flux: float = 12.0
    etchant_flux: float = 1.8e3
    oxygen_flux: float = 1.0e2
    ion_energy: float = 100.0
    # geometry, held fixed across the design problem but varied in training
    trench_width: float = 6.0
    mask_height: float = 2.0

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.ion_flux, self.etchant_flux, self.oxygen_flux, self.ion_energy],
            dtype=np.float64,
        )


# (lo, hi) for each design knob. Sampled log-uniform for the two fluxes that
# span more than a decade, uniform otherwise.
RECIPE_BOX = {
    "ion_flux": (5.0, 30.0, "log"),
    "etchant_flux": (5.0e2, 5.0e3, "log"),
    "oxygen_flux": (0.0, 4.0e2, "lin"),
    "ion_energy": (50.0, 200.0, "lin"),
}
GEOM_BOX = {
    "trench_width": (4.0, 9.0, "lin"),
    "mask_height": (1.0, 3.0, "lin"),
}


def sample_recipe(rng: np.random.Generator, vary_geometry: bool = True) -> Recipe:
    vals = {}
    boxes = dict(RECIPE_BOX)
    if vary_geometry:
        boxes.update(GEOM_BOX)
    for k, (lo, hi, mode) in boxes.items():
        if mode == "log":
            vals[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            vals[k] = float(rng.uniform(lo, hi))
    return Recipe(**vals)


def _ordered_polyline(nodes: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Walk the line soup into one ordered open polyline, left edge to right.

    ViennaPS hands back segments in no particular order. Chaining them matters:
    an unordered segment list rasterises to the same SDF but cannot be closed
    into a polygon for the inside test, and cannot be compared as a curve.

    This assumes a non-branching chain, which is why `surface_polyline` meshes
    the top level set rather than `Domain.getSurfaceMesh()`: the domain mesh
    overlays the mask outline on the substrate outline and the two share
    junction nodes, giving one connected component with four loose ends and
    degree-3 vertices that no single walk can traverse.
    """
    adj: dict[int, list[int]] = {}
    for a, b in lines:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    ends = [n for n, nb in adj.items() if len(nb) == 1]
    if not ends:  # closed loop; start anywhere
        start = next(iter(adj))
    else:  # start at whichever endpoint sits further left
        start = min(ends, key=lambda n: nodes[n, 0])
    order = [start]
    prev = -1
    cur = start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        order.append(cur)
    return nodes[np.asarray(order), :2]


def _segment_sdf(poly: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Exact distance from every grid point to the polyline, signed by inside-ness.

    Closed along the bottom of the window so "inside" means solid material.
    Vectorised point-to-segment; the polygon test is a ray cast on the same
    closed loop, so the zero set of the result is the polyline itself.
    """
    P = np.stack(np.meshgrid(gx, gy, indexing="xy"), axis=-1).reshape(-1, 2)
    A, B = poly[:-1], poly[1:]
    AB = B - A
    denom = np.einsum("ij,ij->i", AB, AB)
    denom = np.where(denom == 0.0, 1e-30, denom)
    AP = P[:, None, :] - A[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", AP, AB) / denom, 0.0, 1.0)
    closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
    dist = np.linalg.norm(P[:, None, :] - closest, axis=-1).min(axis=1)

    # Close the surface into a polygon by dropping to the window floor. The
    # closing edges are pushed 1 um outside the window on both sides: the
    # surface endpoints sit exactly on x = +-10, so a ring closed at the window
    # edge puts a vertical edge through the last grid column and the crossing
    # test becomes a coin flip there (it silently mis-signed 115 points, the
    # whole right column, before this). Extending flat past the edge is also
    # what the solver's reflective boundary does.
    pad = 1.0
    ring = np.concatenate(
        [
            np.array([[poly[0, 0] - pad, poly[0, 1]]]),
            poly,
            np.array(
                [
                    [poly[-1, 0] + pad, poly[-1, 1]],
                    [poly[-1, 0] + pad, Y_MIN - pad],
                    [poly[0, 0] - pad, Y_MIN - pad],
                ]
            ),
        ],
        axis=0,
    )
    inside = _point_in_poly(P, ring)
    return np.where(inside, -dist, dist).reshape(len(gy), len(gx))


def _point_in_poly(P: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Crossing-number test, vectorised over points."""
    x, y = P[:, 0], P[:, 1]
    x1, y1 = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(ring[:, 0], -1), np.roll(ring[:, 1], -1)
    cond = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1)[None, :] * (y[:, None] - y1[None, :]) / (y2 - y1)[None, :] + x1[None, :]
    crossings = (cond & (x[:, None] < xint)).sum(axis=1)
    return crossings % 2 == 1


def surface_polyline(domain) -> np.ndarray:
    """The gas/solid interface: one open curve spanning the window, left to right.

    The *last* level set is the union of every material, so its zero contour is
    the exposed surface -- the thing the etch actually moves.
    """
    import viennals as vls

    vls.Logger.setLogLevel(vls.LogLevel.ERROR)
    mesh = vls.Mesh()
    vls.d2.ToSurfaceMesh(domain.getLevelSets()[-1], mesh).apply()
    nodes = np.asarray(mesh.getNodes(), dtype=np.float64)
    lines = np.asarray(mesh.getLines(), dtype=np.int64)
    if lines.size == 0:
        raise RuntimeError("surface mesh carried no line elements")
    poly = _ordered_polyline(nodes, lines)
    if len(poly) < len(nodes):
        raise RuntimeError(
            f"polyline walk covered {len(poly)}/{len(nodes)} nodes -- surface is "
            "branching or disconnected, the SDF sign would be wrong"
        )
    return poly


def rasterise(poly: np.ndarray, n: int) -> np.ndarray:
    gx = np.linspace(X_MIN, X_MAX, n)
    gy = np.linspace(Y_MAX, Y_MIN, n)  # row 0 = top of the window
    return _segment_sdf(poly, gx, gy)


def grid_axes(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(X_MIN, X_MAX, n), np.linspace(Y_MAX, Y_MIN, n)


def build_domain(rec: Recipe, grid_delta: float):
    vps = _lazy_vps()
    d2 = vps.d2
    dom = d2.Domain(gridDelta=grid_delta, xExtent=X_MAX - X_MIN, yExtent=60.0)
    d2.MakeTrench(
        domain=dom,
        trenchWidth=rec.trench_width,
        trenchDepth=4.0,
        maskHeight=rec.mask_height,
    ).apply()
    return dom


def make_process(rec: Recipe, dom, duration: float):
    vps = _lazy_vps()
    d2 = vps.d2
    model = d2.SF6O2Etching(
        ionFlux=rec.ion_flux,
        etchantFlux=rec.etchant_flux,
        oxygenFlux=rec.oxygen_flux,
        meanIonEnergy=rec.ion_energy,
        sigmaIonEnergy=10.0,
        oxySputterYield=1.5,
    )
    p = d2.Process()
    p.setDomain(dom)
    p.setProcessModel(model)
    p.setProcessDuration(duration)
    return p


@dataclass
class Trajectory:
    recipe: Recipe
    sdf: np.ndarray  # (T+1, n, n)
    polylines: list = field(default_factory=list)
    seconds: float = 0.0
    steps_ok: bool = True


def simulate(
    rec: Recipe,
    n_steps: int = 8,
    dt: float = 0.35,
    grid_delta: float = 0.2,
    n: int = 128,
    keep_polylines: bool = False,
) -> Trajectory:
    """Roll the etch forward `n_steps` x `dt` minutes, rasterising after each.

    Timing excludes rasterisation: the speedup table compares the *solver* to the
    operator, and rasterisation is our own bookkeeping, charged to neither.
    """
    dom = build_domain(rec, grid_delta)
    frames = [rasterise(surface_polyline(dom), n)]
    polys = [surface_polyline(dom)] if keep_polylines else []
    seconds = 0.0
    ok = True
    for step in range(n_steps):
        proc = make_process(rec, dom, dt)
        t0 = time.perf_counter()
        proc.apply()
        seconds += time.perf_counter() - t0
        poly = surface_polyline(dom)
        left_window = poly[:, 1].min() < Y_MIN + 0.5
        if left_window:
            ok = False
        frames.append(rasterise(poly, n))
        if keep_polylines:
            polys.append(poly)
        # Stop once the surface has left the window. Nothing below Y_MIN is
        # representable on the fixed grid, so further steps add no information --
        # and they are not cheap: the level set keeps growing and the advection
        # takes more CFL substeps each time, so cost per step climbs without
        # bound. Adaptive dt kept every training trajectory inside the window, so
        # this never fired during generation. Inverse design proposes arbitrary
        # recipes, where it does: an unguarded verification run sat for >13
        # minutes on a single trajectory before being killed. Remaining frames
        # repeat the last and `steps_ok` stays False, so a truncated trajectory
        # can never be mistaken for a completed one.
        if left_window and step < n_steps - 1:
            last = frames[-1]
            frames.extend([last] * (n_steps - 1 - step))
            if keep_polylines:
                polys.extend([poly] * (n_steps - 1 - step))
            break
    return Trajectory(
        recipe=rec, sdf=np.stack(frames).astype(np.float32), polylines=polys,
        seconds=seconds, steps_ok=ok,
    )


def probe_rate(rec: Recipe, grid_delta: float = 0.2, t_probe: float = 0.1) -> float:
    """Vertical etch rate (um/min) at the *start* of the etch, by short probe.

    Used only to choose a timestep, never as a label. Measured on a fresh
    geometry and thrown away, so the trajectory the dataset stores is not
    contaminated by it.
    """
    dom = build_domain(rec, grid_delta)
    y0 = surface_polyline(dom)[:, 1].min()
    make_process(rec, dom, t_probe).apply()
    y1 = surface_polyline(dom)[:, 1].min()
    return max((y0 - y1) / t_probe, 1e-6)


# The etch rate across the recipe box spans ~20x: the fastest corner clears the
# window in 1.5 min while the slowest moves 0.5 um in the same time. A single
# global timestep therefore has to make most of the box nearly static, and a
# nearly static target is one a persistence predictor solves for free -- the
# rel-L2 clause would be met by a model that does nothing. So each trajectory
# gets its own dt, chosen from a cheap probe so every recipe evolves by a
# comparable depth, and dt is handed to the operator as a conditioning input.
TARGET_DEPTH_RANGE = (4.0, 10.0)
DT_RANGE = (0.01, 1.0)


def choose_dt(rec: Recipe, n_steps: int, target_depth: float, grid_delta: float = 0.2) -> float:
    rate = probe_rate(rec, grid_delta)
    dt = target_depth / (rate * n_steps)
    return float(np.clip(dt, *DT_RANGE))


def simulate_adaptive(
    rec: Recipe,
    rng: np.random.Generator,
    n_steps: int = 10,
    grid_delta: float = 0.2,
    n: int = 128,
    keep_polylines: bool = False,
) -> tuple[Trajectory, float, float]:
    """Pick a per-recipe dt, then simulate. Returns (trajectory, dt, target_depth)."""
    target = float(rng.uniform(*TARGET_DEPTH_RANGE))
    dt = choose_dt(rec, n_steps, target, grid_delta)
    tr = simulate(rec, n_steps=n_steps, dt=dt, grid_delta=grid_delta, n=n,
                  keep_polylines=keep_polylines)
    return tr, dt, target
