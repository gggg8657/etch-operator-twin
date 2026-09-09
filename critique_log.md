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

---

## 2026-09-09 — turn 1b: the dataset generator was 11× oversubscribed

### What was measured

`scripts/bench_workers.py` → `runs/worker_scaling.json`. Identical trajectories,
identical code, only the worker count changes:

| workers | traj/s | wall per trajectory (s) | speedup vs 1 worker |
|---|---|---|---|
| 1 | 0.306 | 3.2 | 1.00× |
| 8 | 0.703 | 10.6 | 2.30× |
| 24 | 0.677 | 32.4 | 2.21× |
| 48 | 0.637 | 68.5 | 2.08× |
| 90 | 0.626 | 131.5 | 2.05× |

### The critique

The first generation run used **90 workers and was slower than 8**. It completed
about 175 trajectories in 10 minutes — roughly 300 core-seconds each against 4.7 s
measured serially — and was killed.

Parallel efficiency at 8 workers is already only 29% (2.30× on 8 cores), and the
curve is *falling* after that. This is the signature of a bandwidth-bound
workload, not a compute-bound one: ViennaPS's flux calculation is a Monte Carlo
ray trace over a level-set surface with scattered memory access, so the cores
were queuing on memory, not arithmetic. Eighty-two of the ninety cores were
producing negative value, and they were taken from two other tracks sharing this
box.

**What would distinguish this explanation from the obvious alternative.** The
obvious alternative is OpenMP oversubscription — each worker silently spawning
its own thread pool, 90 × N threads on 192 cores. That would show a load average
far above the worker count; the measured load was ~105 with 91 processes, i.e.
about one runnable thread per process, so the threads are not there. The
remaining candidate is memory. The falling-throughput shape (0.703 → 0.626 while
cores go 8 → 90) is what a saturated memory system looks like and is not what
scheduler contention looks like — scheduler contention plateaus, it does not
decline.

**Consequence for the KPI.** The ≥1000× clause compares "seconds per simulated
wafer" for solver and operator. The solver's cost per wafer moves by a factor of
40 between 1 worker and 90 depending only on what else is running. So a speedup
number is meaningless without the thread count and the machine state attached,
and `scripts/bench_speed.py` records the load average at measurement time for
exactly this reason. The `solver_s` column stored inside the dataset was measured
under contention and is excluded from that table.

Regenerated at 16 workers.

---

## 2026-09-09 — turn 2: a measured negative result about my own orchestration

### What was measured

`scripts/bench_workers.py` → `runs/worker_scaling.json`. Generation throughput
against worker count, 10-step trajectories, `OMP_NUM_THREADS=1` per worker, on
the shared 192-core box:

| workers | trajectories/s | wall per trajectory (s) | parallel efficiency vs 1 worker |
|---|---|---|---|
| 1 | 0.306 | 3.2 | 1.00 |
| 8 | 0.703 | 10.6 | 0.29 |
| 24 | 0.677 | 32.4 | 0.09 |
| 48 | 0.637 | 68.5 | 0.04 |
| 90 | 0.626 | 131.5 | 0.02 |

**Throughput peaks at 8 workers and then falls.** Ninety workers occupy eleven
times the machine to deliver 11% *less* throughput than eight.

### Why the gap exists

ViennaPS's plasma-etch flux calculation is a Monte Carlo ray trace over the
surface disk mesh. Ray tracing has scattered, cache-hostile memory access and
almost no arithmetic intensity, so it saturates memory bandwidth long before it
saturates cores. Past that point, extra workers only add contention for the same
bandwidth and last-level cache.

### What would distinguish this from the obvious alternative

The obvious alternative explanation is CPU oversubscription — that
`OMP_NUM_THREADS=1` was not taking effect and each worker was spawning its own
thread pool. Two observations rule that out. First, load average sat at ~105
with 91 processes; genuine oversubscription at 192 threads per worker would have
put load in the thousands. Second, per-worker CPU utilisation was 91–93%, i.e.
each worker held roughly one core, which is what the variable is supposed to
produce. A bandwidth ceiling reproduces the observed shape — sublinear gain to a
plateau, then slow decay — while oversubscription would show a sharp collapse and
much lower per-worker CPU%.

A cleaner discriminator, not yet run and worth an hour if this ever matters
again: pin workers to cores on a single NUMA node with `numactl` and re-measure.
Bandwidth-bound work improves markedly when it stops crossing sockets;
oversubscribed work does not care.

### What it cost, and the honest accounting

The first generation run used 90 workers on the assumption that a 192-core box
means 90-way parallelism is free. It completed roughly 175 of 1800 trajectories
in 10 minutes — about 300 core-seconds each against 3.2 s measured serially —
and was killed and discarded. Two costs, both mine: ~15 minutes of wall clock,
and 90 cores taken from the two other tracks sharing this machine for that
period. The rerun uses 16 workers.

This is the same error the seed-count lesson warns about in a different
costume: an effect (here, "more workers is faster") assumed rather than measured,
in a regime where the assumption happens to be false. The five-point sweep cost
under ten minutes and would have cost nothing had it been run first.

### Consequence for the KPI

None directly — but it removes a trap from the speedup clause. The `solver_s`
field recorded during dataset generation is a *contention* number: at 90 workers
a trajectory takes 131.5 s of wall clock against 3.2 s serial, a factor of 41.
Quoting that as "the simulator takes 131 s" would have inflated the reported
speedup by 41× for free, and it is exactly the sort of number that looks
defensible because a script did produce it. `scripts/bench_speed.py` therefore
re-times the solver in a clean subprocess with the thread count pinned and
records the box load average alongside, and `scripts/report.py` never reads
`solver_s` from the generation report.

### Still not known

No operator trained yet. `RESULTS.md` currently reads `[not measured]` for all
three KPI clauses, which is the correct state.
