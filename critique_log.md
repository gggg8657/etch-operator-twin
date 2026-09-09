# critique log — E4, etch-operator-twin

Append-only. Each entry: what was measured, what it is being compared against and
where that comparison number comes from, why the gap exists, and what would
distinguish the stated explanation from the obvious alternative.

---

## 2026-09-09 — turn 1: standing up the ground truth

### What was measured

`scripts/verify_solver.py` → `runs/verify_solver.json`, on ViennaPS 4.6.2 /
ViennaLS 5.8.5, one OpenMP thread:

| check | result | tolerance |
|---|---|---|
| SDF rasteriser vs analytic plane | max abs err 2.0e-13 µm | < 1e-3 µm |
| `IsotropicProcess` recession vs rate×time | rel err ≤ 1.2e-8 over t ∈ {0.5, 1, 2} min | < 2e-2 |
| grid convergence, Δ 0.4→0.2 | mean surface shift 0.051 µm | — |
| grid convergence, Δ 0.2→0.1 | mean surface shift 0.018 µm | — |

The dataset is generated at Δ = 0.2, so its discretisation error against a
Δ = 0.1 reference is **0.018 µm mean / 0.089 µm max** in the band. Everything
the operator is later asked to do is measured against a target that is itself
only this accurate — an operator error below 0.018 µm would be measuring the
solver's grid, not the operator.

### Three things that were wrong and are now fixed

1. **`pip install viennaps` produces a broken install.** It resolves ViennaLS to
   5.9.0; `import viennaps` then dies with `arg(): could not convert default
   argument 'boundary: viennahrle::BoundaryType'`. Pinning viennals==5.8.5 fixes
   it. The brief said "4.6.2 verified from this host" — 4.6.2 is indeed
   available, but the default resolution of its dependency is not usable, and a
   turn was spent finding that out. Recorded in `requirements.txt` and asserted
   in `tests/test_solver.py::test_import_and_version_pin`.

2. **`Domain.getSurfaceMesh()` is the wrong extraction point.** It overlays the
   mask outline on the substrate outline, giving one connected component with
   four loose ends and degree-3 junction vertices. A walk over it returned 80 of
   229 nodes — one sub-chain covering x ∈ [−10, −3] — and the resulting "surface"
   did not move as the etch progressed, so the first trajectories looked
   perfectly static. Meshing the *top level set* instead
   (`ToSurfaceMesh(getLevelSets()[-1])`) gives a clean non-branching chain, 2
   endpoints, no junctions. The walk now raises if it fails to cover every node,
   so this cannot fail silently again.

3. **The SDF mis-signed an entire grid column.** The surface's endpoints sit
   exactly on x = ±10, so closing the polygon at the window edge put a vertical
   edge through the last grid column and the crossing test tied there: 115 of
   16384 points took the wrong sign, max error 36 µm. The closing ring is now
   padded 1 µm outside the window. `tests/test_sdf.py::test_right_edge_column_signed`
   is the regression.

### A protocol decision, made before any training

The etch rate across the recipe box spans about **20×**: the fastest corner
(ion 30, etchant 5e3) clears the 20 µm window in 1.5 min, while the slowest
(ion 5, etchant 5e2) moves the surface 0.5 µm in the same time. Under a single
global timestep, containing the fast corner forces most of the box to be nearly
static — and *a nearly static target is one that a do-nothing predictor solves*.
The rel-L2 ≤ 0.05 clause would then be met by a model that outputs its input.

So each trajectory gets its own dt, chosen from a cheap discarded probe so every
recipe covers a comparable etch depth (4–10 µm), and dt is a conditioning input.
Consequences, stated now rather than discovered later:

- The operator is solving a *harder* problem than a fixed-dt operator would, not
  an easier one: it must learn the rate law, not just apply it.
- Every reported number will carry a **persistence column** (predict no change).
  Any operator number that does not beat persistence is not a result, whatever
  the KPI threshold says. `scripts/train.py::evaluate` computes it on the same
  batch, so it cannot be quietly dropped.

### Two measurement-resolution problems found before they could contaminate a result

- **Hard-threshold area cannot resolve the 5% shape-error clause.** Counting
  `sdf < 0` cells quantises every area to a whole cell; on a plane-vs-plane case
  with a hand-computable answer of 40.0 µm², it returns 41.27 µm² — 3.2% high,
  the same order as the 5% the KPI is judged against. A metric whose
  quantisation floor sits at the decision threshold cannot decide the KPI.
  Replaced with a sub-cell occupancy `clip(0.5 − φ/h, 0, 1)`, exact for a
  straight contour, which returns 40.0 ± 0.5. Both readings are reported.
- **The `solver_s` recorded during dataset generation is unusable for the
  speedup clause.** It was measured with 90 workers on the box (load 105, plus
  two other tracks' jobs), so it is a contention number, not a solver number.
  The ≥1000× clause will be measured by a separate benchmark on a quiet box with
  the thread count stated, and the generation-time figure will not appear in
  that table.

### What is not yet known

No operator has been trained. There is no rel-L2, no speedup and no shape error
in this repo yet, and none will be written down until a run produces them.
