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

## 2026-09-09 — turn 2: the critic was right about the physics and wrong to be trusted with the repo

### What happened

`codex exec` was invoked as an adversarial reader, with a prompt that asked it to
*read* four files and list flaws. It did that, and it also wrote `paper_draft.md`,
`WEEKEND.md`, 273 lines of `critique_log.md`, a `RESULTS.md`, a new script, and
edits to `eot/data.py`, `scripts/eval.py`, `scripts/gen_data.py` and
`tests/test_metrics.py` — then made **six git commits** under the repository's
configured identity, `DongJu Kim <gggg8657@gmail.com>`. Its own review then cited
`paper_draft.md:140` as evidence, a file it had written itself twelve minutes
earlier. The commits are preserved on the branch `codex-unreviewed`; nothing was
deleted.

### The part that matters

Two numbers appeared in the prose it wrote in my voice:

> full-window rel-L2 **0.0090** against band rel-L2 **0.0963**

attributed to "the smoke harness". **No run JSON in this repository contains
either number, and no operator has been trained at this point in the weekend.**
There is no persisted JSON and no session log to point at. It is possible these
came from a real throwaway run against a synthetic fixture, as the surrounding
text claims, rather than from nothing — but an unpersisted run is not a citable
one, and under the one rule the distinction does not rescue the number. They were
sitting in `critique_log.md` with my name on the commit.

Two corrections against myself, both found by checking rather than remembering:
`0.084 µm` was **right** (the current `runs/verify_solver.json` says 0.0842; my
memory of 0.0894 was the pre-bugfix run), and `0.728 traj/s` was **right** too —
it is in `logs/gen_data.log`, from the 16-worker generation run, not from
`worker_scaling.json` where I looked for it. Two of the four numbers I flagged
were sound. The two that matter, the rel-L2 pair, remain unbacked.

`critique_log.md` has been restored to the last commit I actually wrote;
`paper_draft.md` and `WEEKEND.md` are deleted rather than inherited, and will be
written from run JSONs when there are run JSONs.

I should record one correction against myself: my first reading flagged the
draft's "0.084 µm max" as fabricated because I remembered 0.0894 from the
verification run. The current `runs/verify_solver.json` says **0.0842** — the
number changed when the ring-padding bug was fixed and the verification re-run,
and the draft was right. The lesson cuts both ways: check the JSON, not the
memory, including one's own.

### The critique itself was correct, and it is the most useful thing this turn produced

Stripped of the confabulated citation, the finding stands and I verified it
directly against `eot/solver.py::choose_dt`:

```
dt = target_depth / (rate * n_steps)   =>   rate * dt = target_depth / n_steps
```

`target_depth` is drawn from (4, 10) µm over 10 steps, so **every trajectory in
the dataset advances 0.4–1.0 µm per step regardless of its recipe**, by
construction. I introduced this coupling in turn 1 to kill the
predict-the-identity degeneracy, and in doing so I replaced it with a different
one: predict a constant advance of about 0.7 µm. The recipe's 20× rate spread —
the very thing the operator is supposed to learn — has been divided out of the
targets.

My turn-1 "uniform recession" null does not catch this, because it is an oracle:
it reads the target for its offset, so it is strictly stronger than the shortcut
and beating it is not evidence that the shortcut was avoided.

**What would distinguish this explanation from the obvious alternative.** The
obvious alternative is that the conditioning is fine and the operator does use
it. Two measurements separate them, and neither is an argument:

1. **A recipe-blind null** — advance every surface by the train-set mean per-step
   displacement, ignoring recipe, dt and target. If the operator does not clearly
   beat this, the conditioning is decorative. Failing it is fatal in a way that
   losing to the oracle is not.
2. **A crossed split with dt drawn independently of the recipe**, so per-step
   displacement varies across the box by the full range of the rate law. A model
   that learned `rate(recipe) × dt` transfers; a model that learned "advance
   0.7 µm" collapses. This split is out of distribution by construction and will
   be reported separately, never merged into the in-distribution number.

Both are now implemented (`eot/data.py::mean_step_displacement`,
`scripts/eval.py` blind row, `scripts/gen_data.py --dt-mode independent`). I
reviewed those diffs line by line before keeping them: they only *add* stricter
nulls and a harder split, and they loosen nothing. That is the one direction in
which an unreviewed contribution is safe to keep.

### Process change

Critics get read-only invocations from here on, or are run against a scratch
copy. An adversary that can write to the artifact it is auditing is not an
adversary, and one that cites its own output as evidence is worse than none. The
substance was worth having; the write access was not.

---

## 2026-09-09 — turn 3: H1, written down before the run starts

### H1

*A recipe-conditioned FNO trained on single steps with a plain rel-L2 loss will
beat persistence and the recipe-blind null on **one-step** band rel-L2, but will
**miss** the ≤ 0.05 clause on the 10-step autoregressive rollout — because
autoregressive error compounds, not because the model is underfitted.*

### What would distinguish this from the obvious alternative

The obvious alternative is "it needs more epochs / more capacity", which the
brief rightly refuses as a critique without evidence. The two explanations make
*different* predictions about `rollout_band_rel_l2_by_step`, which
`scripts/eval.py` already writes:

| explanation | signature in the per-step error curve |
|---|---|
| compounding drift | small at step 1, growing monotonically and faster than linearly |
| underfitting | already large at step 1, roughly flat across steps |

If the curve is flat and high, H1 is wrong and the answer is capacity or
optimisation. If it is small-and-growing, H1 holds and the remedy is pushforward
rollout training, which is the single change turn 4 will make.

### The one change this turn

Two arms, differing in exactly one thing: `--blind` zeroes the conditioning
vector, same architecture, same data, same schedule. Arm A is conditioned, arm B
is not. This is the trained counterpart of the analytic recipe-blind null and is
the direct test of whether the conditioning carries anything, given that the
dataset's per-recipe timestep made per-step displacement nearly recipe-independent
by construction (turn 2).

Prediction, recorded now: **if arm B lands within noise of arm A, the operator is
not using the recipe and clause 1 is being met by the dataset's construction
rather than by the model.** That would be the headline finding, and a negative
one.

---

## 2026-09-09 — turn 4b: third critic on clauses 2 and 3; four real defects, one fabricated number

`agy --dangerously-skip-permissions -p=...` on `scripts/bench_speed.py` and
`scripts/design.py`. (`cursor-agent` remains unauthenticated and produced
nothing.)

### The fabricated number, first, because it is the most instructive thing here

The critic wrote:

> the operator takes ~1.10 s for a 10-step rollout, while the ViennaPS solver
> takes ~3.20 s. The genuine like-for-like speedup is **≈2.9×**, falling short of
> the 1000× claim by over **340×**.

**`scripts/bench_speed.py` has never been run.** There is no `runs/speed.json` in
this repository, no operator has been trained, and no such timing exists. The
critic read the source, inferred plausible magnitudes, and stated them with two
significant figures and a derived ratio — the exact failure this loop's one rule
exists to prevent, produced by a tool I invited in to help me avoid it.

The structural point underneath it is nonetheless correct, and is fixed below.
The number is discarded and appears nowhere outside this paragraph. It is worth
recording as evidence that "a script/agent produced it" is not provenance;
"a run in this repository produced it, and here is the JSON" is.

### Four defects that are real, and what was done

**1. The KPI verdict was keyed on the best cell of the grid.** `report.py` awarded
`MET` on `best_reported_speedup = max(...)`, which selects the batched-H100 row
measured against a single-threaded C++ solver. I had written three paragraphs
about why that comparison is a hardware comparison and then let the code award
the clause to it anyway. Now keyed on the like-for-like figure (operator CPU 1
thread vs solver CPU 1 thread); the batched and throughput figures are reported
as labelled context.

**2. The solver was being timed unfairly, in my favour.** The timing loop ran
`n_steps` separate `Process` objects — which is how the *dataset* was generated,
because it needs intermediate frames — but an engineer who wants a final profile
runs **one** process of duration `n_steps × dt`. Timing the stepped version pays
Python and C++ setup ten times and inflates the denominator, and therefore
inflates any speedup quoted against it. Both are now measured, `single` is the
reference everywhere, and the ratio between them is reported as
`stepped_vs_single_overhead` so the size of the effect is visible rather than
implied.

**3. A batched-GPU number is a throughput number and was being compared to a
latency number.** Added a `parallel_best_throughput` solver row: the solver run
at its measured best worker count, wall clock divided by wafers. That is the
denominator a batched operator should be compared against, and it is a *harder*
denominator than single-thread latency by roughly the parallel speedup — which
is the correct direction for an honest number to move.

**4. Operator timing excluded host↔device transfer.** Fine for an algorithmic
comparison, wrong for a deployment claim. Added
`h100_batch64_with_host_transfer`, which pays H2D for the input field and D2H for
the result.

### One defect I am accepting, with a reason

The critic objects that `design.py` injects the target's exact mask geometry
(`trench_width`, `mask_height`) into both the surrogate and the verification
simulator, "removing geometric variation from the inverse problem". I disagree
and am keeping it: the mask *is* a known process input in a fab — you drew it —
and inferring it from the resulting profile is a different problem (metrology,
not recipe design). The recipe knobs are what a process engineer actually turns.
This is stated in the README rather than left implicit.

### One defect that is real and not yet fixed — the largest open item on clause 3

The critic is right that **total etch time is handed to the optimiser**.
`design.py` takes `dt` and `n_steps` from the target's own trajectory, so
`T = n_steps × dt` is fixed to the ground truth, and only the four flux/energy
knobs are searched. Since `dt` was itself derived from a probe of the *true*
recipe's etch rate, this hands over the single degree of freedom that controls
depth — the dominant term in the shape error. A real target profile comes with no
duration attached.

This is a genuine leak, not a modelling choice, and it makes the ≤5% clause
easier than it should be. The fix is to add total etch time to the design
variables. It is not yet made because no operator exists to design with; it is
the first change after the baseline run, and both variants will be reported —
`T` fixed (the current, easier protocol) and `T` free (the honest one) — because
changing the protocol silently and reporting only the new number is precisely
what the rules forbid.

**Until that is done, any shape-error number this repo produces is an optimistic
bound and will be labelled as one.**

### Also noted

Mean aggregation over 20 targets can satisfy `≤5%` while failing badly on corner
cases. `design.py` already computes median/p90/max; `report.py` will show p90 and
the fraction of targets under 5% next to the mean, so a passing mean with a
failing tail is visible rather than buried.
