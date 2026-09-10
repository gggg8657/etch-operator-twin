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

---

## 2026-09-09 — turn 4: H1 is falsified, and clause 1 depends on which reading you take

### The numbers

Test split, 250 trajectories × 10 steps, from `runs/base/test_eval.json` and
`runs/blind/test_eval.json`. Both arms: identical architecture (26,248,025
parameters), identical data, identical schedule, 80 epochs, seed 0. They differ
in exactly one thing — arm B has its conditioning vector zeroed.

| predictor | rollout band rel-L2 (mean over steps) | terminal step | one-step |
|---|---|---|---|
| **conditioned operator** | **0.0400** | **0.0668** | 0.0085 |
| recipe-blind operator (trained, cond zeroed) | 0.4054 | 0.6679 | 0.0647 |
| recipe-blind null (constant advance) | 1.7234 | — | — |
| uniform recession null (oracle) | 1.6679 | — | — |
| persistence null | 2.1439 | — | — |

### H1 was wrong, and the discriminator I committed to says why

H1 predicted the conditioned operator would **miss** ≤ 0.05 on the 10-step
rollout through compounding drift. It came in at 0.0400. The prediction is
falsified.

The *mechanism* half of H1 survives, and the pre-committed discriminator is what
establishes it rather than hindsight. The per-step curve is

```
0.0107 0.0178 0.0247 0.0312 0.0378 0.0443 0.0502 0.0558 0.0612 0.0668
```

— small at step 1 and growing 6.25× by step 10, with near-constant increments of
about 0.0065. That is the "compounding drift" signature I wrote down in turn 3,
not the "already large at step 1, roughly flat" signature of underfitting. So the
error is accumulating as predicted; I simply mis-estimated its scale by enough to
change the verdict. Underfitting is ruled out by the same curve, which is what
the discriminator was for.

### The reading that matters, and the one I nearly let stand

**Mean-over-steps is the permissive reading.** It averages the cheap first step
in with the expensive last one. The number a process engineer holds at the end of
an etch is the *terminal* step, and that is **0.0668 — above 0.05**.

So clause 1 is met on the mean and missed on the terminal step. Reporting the
0.0400 alone would have been exactly the move the rules call the worst one
available: passing a clause by choosing the looser of two readings of the same
measurement, without saying that a stricter one exists and fails. The clause is
now scored on **both readings, and is only "MET" if both pass**. It currently
does not.

For completeness the loosest reading of all, whole-window rel-L2, is 0.0084 —
five times inside the threshold. The far field of an SDF is a smooth ramp with a
large norm and the model gets it nearly free; that number is in `RESULTS.md`
labelled as the loose reading and is not the headline.

### The turn-2 confound is answered by measurement

The recipe-blind *trained* arm reaches 0.4054 against the conditioned arm's
0.0400 — a factor of 10 on identical capacity. The analytic recipe-blind null
(constant advance by the train-set mean displacement) reaches 1.7234. So the
operator is not scoring by predicting a constant ~0.7 µm advance, and the
conditioning carries most of the signal. The confound was real, the guard was
worth building, and the guard says the model is clean on this split.

Two caveats I am not entitled to drop:

1. This is **in-distribution**, where dt still comes from the per-recipe probe.
   The crossed split (dt drawn independently of the recipe) is generating and is
   the harder test; it is out of distribution by construction and will be
   reported separately.
2. This is **one seed**. The gap between arms (10×) is far outside any plausible
   seed noise, so that comparison is safe. But the clause-1 verdict sits 20%
   from its threshold, and this workspace has already paid three days to learn
   that two identical invocations can differ by more than a reported effect. An
   8-seed sweep is running; until it lands, the clause-1 number is a screen, not
   a verdict.

### Where the accuracy actually sits, physically

Mean surface distance between predicted and true final profile is **0.048 µm**,
Hausdorff **0.134 µm**. The solver's own grid-convergence uncertainty at Δ = 0.2
is 0.018 µm mean (`runs/verify_solver.json`). The operator is therefore within
about 2.7× of the accuracy of the ground truth it was trained against — close
enough that further gains would start to be measured against the solver's
discretisation rather than against the physics. That, not the rel-L2, is the
number that says what this surrogate is worth.

---

## 2026-09-09 — turn 5: clause 1 measured. H1 falsified. The crossed split is the real story.

### The numbers

`runs/t4_base_s0/test_eval.json` and `runs/t4_base_s0/test_crossed_eval.json`.
Model: width 64, modes 16, 4 layers, 16,810,841 parameters, 60 epochs one-step
training, seed 0, 900 train trajectories.

| predictor | in-distribution test | crossed-dt test |
|---|---|---|
| **operator, rollout band rel-L2** | **0.0130** | **0.1524** |
| operator, terminal step | 0.0195 | 0.2972 |
| persistence null | 2.1439 | 1.5829 |
| recipe-blind null | 1.7234 | 1.4825 |
| uniform-recession null (oracle) | 1.6679 | 1.2338 |
| operator, one-step band | 0.0038 | — |
| loose full-window reading | 0.0027 | — |

**Clause 1 is MET in-distribution (0.0130 ≤ 0.05) and MISSED on the crossed
split (0.1524).**

### H1 was wrong

Turn 4 predicted: *the operator will beat all three nulls at one step but miss
≤0.05 on the 10-step rollout, because error compounds.* It beat the nulls by two
orders of magnitude and it did **not** miss — 0.0130 against a 0.05 target, a
factor of 3.8 of headroom. The hypothesis is falsified and I am recording it as
such rather than reframing it.

The discriminator I committed to in advance still did its job. The per-step curve
is `0.0055, 0.0076, 0.0094, 0.0110, 0.0125, 0.0140, 0.0155, 0.0168, 0.0181,
0.0195` — monotonic, roughly linear, 3.5× growth over ten steps. So the
mechanism I named (compounding autoregressive drift) is real and visible; what I
got wrong was its magnitude relative to the threshold. Drift exists, and starts
from a base low enough that it does not matter over this horizon. Had I not
written the discriminator down first I would now be free to claim I had
predicted this, which is exactly why it was written down first.

### The result that actually matters

The adversarial critic's confound, measured end to end:

| split | displacement spread (max/min) | CV | operator band rel-L2 |
|---|---|---|---|
| adaptive dt (training protocol) | 5.1× | 0.311 | 0.0130 |
| crossed dt (independent) | 34.2× | 0.702 | 0.1524 |

The adaptive timestep compresses the rate law's natural spread by about **6.7×**,
and removing that compression costs the operator a factor of **11.7** in band
rel-L2. So:

- The protocol materially helps the headline number. Clause 1 cannot honestly be
  quoted without this attached, and `RESULTS.md` now carries both readings in
  adjacent rows.
- The operator has nonetheless learned real dynamics rather than a constant. It
  beats the recipe-blind null by 114× in-distribution and still by 9.7× on the
  crossed split, where a model that had learned "advance ~0.17 µm" would have
  collapsed to roughly the null. It did not.
- Conditioning explains R² ≈ 0.88 of displacement variance in-distribution and
  0.87 on the crossed split, so the recipe channels carry rate information, not
  only shape.

Both things are true at once and neither cancels the other. The honest summary is
that the operator works, and that the in-distribution figure overstates how well
by roughly an order of magnitude on the distribution a real user would present.

### A ceiling worth naming before anyone tries to improve on it

Mean surface distance between operator and ground truth on the test split is
**0.0123 µm**. The solver's own grid-convergence error between Δ=0.2 (what the
dataset uses) and Δ=0.1 is **0.018 µm mean**. The operator is therefore already
closer to its training target than that target is to a converged solution.
Further reduction of the in-distribution number is fitting the Δ=0.2
discretisation, not the physics. This was flagged as the accuracy floor in turn 1
before any model existed; it is now binding, and it is the reason the next
experiment should be the crossed split rather than a bigger model.

### 16.4% of the crossed split had to be dropped

41 of 250 crossed trajectories etched out of the window and were rejected. That
is a property of drawing dt independently — some recipe/dt pairs simply clear the
20 µm window — and it means the crossed split is not a uniform sample of the
recipe box: it is biased against fast recipes at long dt. The 0.1524 is therefore
measured on the 209 that stayed, and is if anything optimistic. Reported here
rather than buried, and the rejection count is in
`data/gen_report_crossed.json`.

### Two operational failures this turn, both mine

1. **A ViennaPS call in-process permanently breaks cuFFT.** Reproduced cleanly: a
   backward pass through the spectral layers succeeds before a single
   `S.simulate()` call and fails after it with `CUFFT_EXEC_FAILED`, nothing else
   changed. ViennaPS 4.6.2 ships its own GPU path and evidently disturbs the CUDA
   context. All simulation in `design.py` now goes to a **spawn** subprocess pool
   created before CUDA initialises — spawn and not fork, because a forked child
   inherits the context being protected.

2. **`simulate()` had no guard against a runaway etch.** Adaptive dt kept every
   training trajectory inside the window, so this never fired during generation.
   Inverse design proposes arbitrary recipes, where it does: a verification run
   sat for over 13 minutes on one trajectory before I killed it, because the
   level set keeps growing past the window and the advection takes more CFL
   substeps every step. Now stops at the window, pads the remaining frames, and
   leaves `steps_ok` False so a truncated trajectory can never read as a
   completed one. `tests/test_solver.py::test_runaway_etch_stops_at_the_window_and_is_flagged`
   is the regression and asserts a wall-clock bound.

### Provenance, which nearly went wrong

A concurrent turn of this loop trained into `runs/base` while my run was using
it, and overwrote `logs/train_base.log` and `runs/base/last.pt`. The two models
are distinguishable only by parameter count — mine is modes 16 (16,810,841),
theirs is modes 20 (26,248,025) — and I initially read the sizes backwards and
concluded my checkpoint had been destroyed. It had not. Every number above comes
from `runs/t4_base_s0/`, a private copy whose checkpoint was **verified by
loading it and counting parameters** against its own `args.json` before anything
was evaluated. That check is now the thing I trust, not the filename.

---

## 2026-09-09 — turn 6: clause 2 is UNREACHABLE, and the measurement fix decided it

### The grid

`runs/speed.json`. Intel Xeon Platinum 8558 / one H100 NVL, load average 11.6 at
measurement. One "wafer" = a 10-timestep trajectory from initial trench to final
profile. Rasterisation excluded from both sides.

| configuration | s / wafer | vs solver 1-thread single |
|---|---|---|
| ViennaPS, 1 thread, **single process** | 1.538 | 1.00× |
| ViennaPS, 1 thread, stepped (10 processes) | 4.457 | 0.35× |
| ViennaPS, 8 threads, single | 1.254 | 1.23× |
| ViennaPS, 16 threads, single | 1.336 | 1.15× |
| ViennaPS, 8 procs × 1 thread (best throughput) | 0.883 | 1.74× |
| operator, CPU 1 thread, batch 1 | 1.044 | **1.47×** |
| operator, H100 batch 1 | 0.0236 | 65× |
| operator, H100 batch 64, incl. host transfer | 0.00471 | 327× |
| operator, H100 batch 256 | 0.00443 | 347× |

**Every reading falls short of 1000×.** The clause is `UNREACHABLE`.

- Like-for-like, same device and thread count: **1.47×**.
- Throughput against throughput (solver at its best parallel setting vs batched
  H100): **199×**.
- The most flattering cell in the grid, batched H100 against a single-threaded
  C++ solver: **347×**.

### The thing worth stopping on

Timing the solver as ten separate `Process` objects instead of one process of the
full duration inflates it by **2.90×**. Against that inflated denominator the
best GPU cell reads **1006×**.

**1006 ≥ 1000.** Had the adversarial review not caught the stepped-versus-single
distinction, this repository would have recorded clause 2 as `MET`, by 0.6%,
entirely on an artefact of how the reference was timed — with a JSON behind it, a
script that produced it, and a plausible-sounding protocol paragraph. It would
have been the single worst outcome available this weekend, and nothing in my own
process caught it; a subprocess critic reading the file did.

The lesson generalises past this clause: for any ratio, the denominator deserves
at least as much adversarial attention as the numerator, because the denominator
is the part nobody is excited about and therefore the part nobody checks.

### Why the honest number is so small

The operator is 16.8M parameters doing ten autoregressive steps, each with eight
128×128 FFTs. That is not cheap on one CPU core — 1.044 s/wafer, against 1.538 s
for ViennaPS to advect a level set through the same 2 minutes of process time.
The surrogate's advantage is not that its arithmetic is less; it is that its
arithmetic is dense, regular and batchable, which is worth 65× on a GPU at batch
1 and 347× at batch 256. The simulator's is a scattered Monte Carlo ray trace,
which is why it also refuses to parallelise (8 threads buys 1.23×, 16 threads
buys 1.15× — the same bandwidth ceiling as turn 2's worker sweep, now measured a
second way and agreeing).

So the honest claim is *"~350× on a GPU against a single CPU core, ~200×
throughput against the solver's own best parallel configuration, and ~1.5× if you
hold the hardware fixed"*. None of those is 1000×, and the first is the one a
paper would print without the other two.

### What would change it

Not more training. The operator would have to get roughly 3× cheaper at equal
accuracy for the batched figure to reach 1000×: fewer modes, fewer layers, a
smaller width, or fewer autoregressive steps per wafer. Since the accuracy is
already below the solver's own grid error (turn 5), there is real headroom to
spend — a deliberately undersized operator is the experiment that could move this
clause, and it is a cheaper experiment than anything else outstanding. It is
recorded as an option for Monday rather than run now, because it changes the
model and clause 1 would have to be re-measured under it.

### The conditioning ablation, which is the strongest evidence in the repo

`runs/seed_spread.json`:

| arm | seeds | band rel-L2 |
|---|---|---|
| conditioned (modes 20, 80 ep) | 1, 5 | 0.01292, 0.01306 |
| **recipe-blind, same architecture** | 0 | **0.40540** |

Removing the recipe conditioning — same capacity, same data, same schedule, the
model simply cannot see the recipe — costs a factor of **31.2**. Measured
seed-to-seed range within the conditioned arm is **0.000144**, so the effect is
about **2700× the noise floor**.

This is a much stronger statement than "beats the recipe-blind null", because the
blind *model* is free to fit everything except the recipe, whereas the null
predicts a constant. It settles the question codex raised in turn 3b: the
conditioning is not decorative.

Honesty about its status: this is 2 seeds against 1, so by the standing 8-seed
rule it is a **screen, not a verdict**. I am recording it as a screen. An effect
2700× the measured spread is not going to reverse, but the rule exists precisely
so that judgement is not mine to make, and more seeds are training.

### A bug in my own analysis, caught by the number looking wrong

`scripts/seed_spread.py` grouped runs by architecture and epochs but **not** by
whether conditioning was ablated, so the blind run was pooled with the
conditioned ones and reported as a 31× *seed outlier* — an ablation result read
as pipeline instability. Had I taken that at face value I would have written
"run-to-run spread is 0.39, so nothing in this repo is distinguishable from
anything else", which is false and would have invalidated every comparison. Fixed
by putting `blind` in the grouping key; the ablation is now computed explicitly
as a matched-config contrast.

---

## Turn 7 — 2026-09-09 04:2x UTC

### Pre-registration, written before the runs below were launched

**Headline run.** Clauses 1 and 3 need one designated model, and there are now
several finished ones. Picking the best of them after seeing their scores is
cherry-picking, so the rule is fixed here in advance and does not change again:

> The headline run is the **lowest seed index in the converged conditioned arm**
> (modes 20, width 64, layers 4, 80 epochs, `blind: false`, `log.jsonl` complete
> at 80 epochs). That is **`runs/seed1`**, band rel-L2 0.012919.

`runs/base` (seed 0) would have been the natural headline and is excluded on a
rule that predates its score: it is still training as of this turn, at epoch 23
of 80. It becomes the headline only if it finishes *and* the rule above still
selects it, which it will, and at that point clause 1 gets re-reported under it.

**H2, the hypothesis under test this turn.** Not an architecture change — the
question this turn resolves is whether clause 3 is measurable at all:

> The operator's gradients are informative enough that gradient descent on the
> recipe beats a matched-budget random search *when both are scored in ViennaPS*,
> not on the surrogate.

The discriminator is already built into `design.py`: `operator_gd` vs
`random_search`, both re-simulated. If GD does not beat random search, the
finding is that the operator is a usable **ranker** whose **gradients are not
informative**, and that is the result — it does not get reframed as needing more
iterations. The `surrogate_opinion` row measures the surrogate-reality gap
separately, and a small simulator error with a large gap would mean the KPI is
met for the wrong reason.

### Why clause 3 read `[not measured]` for three turns, which was my fault

`logs/design.log` shows both inverse-design variants dying identically:

    FileNotFoundError: 'runs/t4_base_s0/args.json'

When turn 6 caught the two-trainers-one-directory race it renamed the affected
run directories to `UNTRACED_t4_base_s0` and `base_RACED_do_not_use`. That was
the right call for the data. But `scripts/design.py`, `scripts/report.py` and
`scripts/weekend.py` were all still being invoked with `--run runs/t4_base_s0`,
so every one of them failed or silently emitted `[not measured]`, and **three
KPI cells have been blank since for a path reason, not a scientific one.**

Worth naming the shape of this: the repo's honesty machinery worked exactly as
designed — a missing JSON printed `[not measured]` instead of a guess — and that
correct behaviour *masked a broken pipeline* for three turns, because a blank
cell looks the same whether the measurement is impossible or the script simply
crashed. `[not measured]` needs to be distinguishable from `[not run]`. Logged
as a defect against `report.py`, fixed below.

### `runs/seed_spread.json` on disk was stale and contaminated

The file dated 04:14 pools two runs that were **still training when they were
scored**:

| run | epochs logged | epochs requested | band rel-L2 as scored |
|---|---|---|---|
| `runs/seed2` | 68 | 80 | 0.012838 |
| `runs/base` | ~23 | 80 | **0.049133** |

`runs/seed2/test_eval.json` is timestamped 04:19:01 against a `log.jsonl` still
being appended at 04:19:29 — an eval of a mid-flight checkpoint, filed next to
converged ones. The 0.0491 is not a seed outlier, it is a partly-trained model,
and pooled with the four converged seeds it inflated the arm's reported spread to
**range 0.0365, cv 0.81**. Had I quoted that as the noise floor I would have
declared every effect in this repo indistinguishable from zero, including the
31× conditioning ablation — the exact inversion of turn 6's error, where an
ablation was misread as noise.

`scripts/seed_spread.py` *already* carries the completeness guard that catches
this; it was added in turn 6 and its comment describes this very case. The script
was fixed and **the JSON was never regenerated**. So the defect this turn is not
a missing guard, it is that a stale artefact outlived the fix that invalidated
it. Regenerated below.

### GPU lease health, measured rather than assumed

Track β recorded GPU 2 as unusable — 91 leaked CUDA contexts, 2.88 TFLOP/s
against 44.07 on a clean device. `nvidia-smi` shows my leased GPU 1 carrying
three foreign contexts (PIDs 342714, 350120, 350220, ~10 GB) at 95% reported
utilisation, which is the same picture. Clause 2 is a timing claim, so I measured
the device instead of inheriting β's conclusion (`runs/gpu_health.json`):

| device | median bf16 TFLOP/s | best | worst | run-to-run spread |
|---|---|---|---|---|
| cuda:0 (my two trainers) | 531.7 | 540.1 | 525.8 | 2.7% |
| cuda:1 (foreign contexts) | 530.6 | 534.0 | 528.1 | 1.1% |

Ratio 1.002. **The leaked-context pathology does not reproduce on my lease** —
GPU 1 delivers full throughput despite the foreign contexts and the 95% reading,
so the utilisation figure is not tracking work that competes with mine. β's
finding was real on GPU 2 and does not generalise to GPU 1; inheriting it would
have cost me a device for the weekend. Recorded as a negative result, and
`scripts/gpu_health.py` is now in the repo so clause 2's timings can state the
health of the hardware they were taken on.

Incidental, and not mine to fix: `/tmp/struct.py` on this box shadows the stdlib
`struct` module and makes *any* `python /tmp/foo.py` fail at `import torch`. It
references a `scripts/hypothesis_ledger.py` that does not exist in this repo, so
it belongs to another loop. Routed around by keeping scratch scripts in the repo.

---

## 2026-09-09 — turn 7: I nearly reported a 3.8× seed effect that was an unfinished training run

### The sequence, including the part where I was wrong

1. Two converged seeds (1, 5) of the modes-20 configuration gave 0.01292 and
   0.01306. I wrote that the seed-to-seed range was **0.000144** and used it as
   the noise floor against which the conditioning ablation was "2700× the noise".
2. `runs/base` — same configuration, seed 0 — evaluated to **0.0491**, 3.8×
   worse, with a terminal-step reading of 0.0888 that misses the clause outright.
   That looked like a genuine and important training-stability finding: one seed
   in three landing in a much worse basin, with direct consequences for whether
   clause 1 can be claimed at all.
3. Before writing it up I checked the training curves. `runs/base` had a final
   train loss of 0.00213 against ~0.00064 for the others — consistent with a
   worse basin, and I nearly stopped there.
4. `runs/base/log.jsonl` had **26 lines against seed1's 80**. It was not a worse
   basin. It was a job that was **still running**, 26 epochs into an 80-epoch
   budget, whose `best.pt` I had evaluated mid-flight. `runs/seed2` was likewise
   at 69/80.

### What the numbers actually are

Restricted to runs that completed their epoch budget:

| arm | seeds | band rel-L2 | range |
|---|---|---|---|
| conditioned, modes 20, 80 ep | 1, 5, 6 | 0.0129, 0.0131, 0.0126 | **0.00041** |
| recipe-blind, same architecture | 0 | 0.4054 | — |

So the converged seed range is **0.00041**, and the conditioning ablation is
**31.5×**, about 950× that range. Both my earlier figures were wrong in opposite
directions: 0.000144 was too small (two seeds), and the 0.0365 I briefly believed
was an artefact of an unfinished run.

### The correction to the previous entry

Turn 6 said the ablation was "~2700× the noise floor". The honest multiple
against three converged seeds is **~950×**. The conclusion — that the recipe
conditioning is doing real work and is not decorative — is unchanged, and the
effect remains far outside any plausible seed variation. The number was wrong and
is corrected here rather than quietly restated.

### Why this was so easy to get wrong, and what now prevents it

A mid-training checkpoint is a perfectly valid model file. It loads, its
parameter count matches its `args.json`, it evaluates, and it produces a number
with a JSON behind it. Every provenance check I had built — and I had built
several this weekend, after the `runs/base` collision — passes on it. The only
thing that distinguishes it is that the run had not finished, and nothing was
looking at that.

`scripts/seed_spread.py` now refuses any run whose `log.jsonl` is shorter than
its requested `epochs`, and lists what it excluded and why in
`excluded_incomplete`. That is a cheap check and it should have existed from the
first seed comparison.

The general form, which is the third instance of the same shape this weekend:
**a number having a JSON behind it is necessary and not sufficient.** Turn 2's
`solver_s` was real and contention-contaminated. Turn 6's 1006× was real and
measured against the wrong denominator. This one is real and measured on a model
that was still moving. In all three the artefact was upstream of the file, where
the provenance check was not looking.

### Status of the seed claim

Three converged seeds. By the standing rule this is a **screen, not a verdict**;
seeds 0 and 2 are still training and will be folded in when they finish. What can
be said now is that among converged runs the spread is 0.00041 — two orders of
magnitude below the 0.05 threshold — so clause 1's in-distribution verdict is not
seed-sensitive, whereas the crossed-split miss (0.15–0.16 across the two models
measured) is far outside it and is not a seed artefact either.

---

## 2026-09-09 — turn 8: H2, written before the run

### The finding it responds to

Clause 1 is met in-distribution (0.0129) and missed on the crossed-dt split
(0.1524). The training protocol picks dt per recipe so every trajectory covers a
comparable etch depth, which compresses per-step displacement spread from 34.2×
to 5.1×. The operator has learned the rate law — the recipe-blind *model* ablation
costs 32.1× — but it has only been asked to apply it over a narrow range of
per-step advances, and it degrades by ~12× when that range widens.

### H2

*Training on a mixture of adaptive-dt and independent-dt trajectories will reduce
crossed-split band rel-L2 by at least 3× without pushing in-distribution band
rel-L2 above 0.05.*

The claim behind it: the crossed-split failure is a **coverage** failure, not a
capacity or architecture failure. The operator can represent the rate law over a
wide range of dt; it has simply never been shown one.

### What would distinguish this from the obvious alternatives

| explanation | prediction |
|---|---|
| **coverage (H2)** | mixed training largely closes the crossed gap; in-distribution error rises slightly or not at all, because the model has spare capacity (it is already below the solver's grid error) |
| **capacity** | mixed training trades one split against the other — crossed improves and in-distribution degrades by a comparable factor — because the model is at its representational limit |
| **the crossed split is intrinsically harder** (long-dt trajectories have genuinely more surface change per step, so the same relative accuracy is a larger absolute error) | mixed training improves crossed error but plateaus well above the in-distribution value, and the residual gap tracks per-step displacement rather than dt itself |

The third is worth taking seriously and is partly testable already: crossed-split
error can be regressed on per-step displacement using data I have. If the gap is
fully explained by displacement magnitude, "coverage" is the wrong word for it.

### The protocol change, and how it will be reported

This changes the training distribution, so **numbers before and after are not
comparable and both will be reported**. The existing adaptive-only model stays as
the baseline with its own numbers intact; the mixed model is a second arm, scored
on the *same* two test splits, neither of which it was trained on. The KPI cell
for clause 1 will continue to name which arm and which split produced it.

### Cost, and what is running

900 independent-dt training trajectories at the measured 0.74 traj/s ≈ 20 min of
CPU, launched now so it overlaps the inverse-design job on GPU 1 rather than
queueing behind it. Generation uses 12 workers, below the measured throughput
peak, because a design run is sharing the box.

---

## 2026-09-09 — turn 9: the crossed-split failure localised, and a suspicion of mine refuted

### What was measured

`runs/crossed_analysis.json`, model `runs/seed1`, per-trajectory band rel-L2 and
a matched **absolute** band error in microns.

| group | n | band rel-L2 | absolute band error (µm) |
|---|---|---|---|
| adaptive test split | 250 | 0.0129 | 0.0077 |
| crossed, per-step displacement **inside** the trained range | 137 | 0.0465 | 0.0284 |
| crossed, displacement **below** the trained range | 63 | 0.6398 | 0.4903 |
| crossed, displacement **above** the trained range | 9 | 0.6117 | 0.3911 |
| crossed, all | 209 | 0.2497 (median 0.0236) | 0.1833 (median 0.0151) |

Trained per-step displacement range: 0.0617–0.3118 µm.
**`frac_dt_outside_trained_range` = 0.000** — every crossed dt is inside the range
the model was trained on.

### The suspicion I had, and why it was wrong

Seeing a mean of 0.2497 against a median of 0.0236, and a *negative* rank
correlation between displacement and error (Spearman −0.449), I expected the
crossed-split failure to be substantially an artefact of a **relative** metric:
rel-L2 divides by the norm of the target field in the band, so a trajectory that
barely moves has a small denominator and reports a large relative error for a
small absolute one. The crossed split contains such trajectories by construction
— drawing dt independently of the recipe pairs slow recipes with short timesteps —
and the adaptive split does not.

That is a clean story and it is **wrong**. Adding an absolute metric refutes it:
the below-range group is at **0.4903 µm** of absolute band error against
**0.0077 µm** in-distribution, a factor of 64. The model is not being penalised
by arithmetic for barely moving; it is actively predicting the wrong thing — motion
where there is little, on trajectories whose per-step advance is smaller than
anything it trained on. Had I reported the metric-artefact explanation without
building the absolute check, it would have excused a real failure.

### What the failure actually is

Not dt coverage — every crossed dt was in range. **Displacement coverage.** The
operator degrades sharply once the per-step advance leaves the interval the
adaptive protocol confined training to, in *both* directions, and the degradation
is real in microns.

Two components, separable:

1. **Out-of-range displacement**, 72 of 209 trajectories, absolute error ~50–64×
   in-distribution. This is the bulk of the mean.
2. **A residual at matched displacement**: 137 trajectories inside the trained
   displacement range are still 3.6× worse in rel-L2 and 3.7× worse in absolute
   terms (0.0284 µm vs 0.0077 µm). So unfamiliar recipe×dt *combinations* cost
   something even when their product is familiar — smaller than component 1, and
   not nothing.

### Consequences

- **H2 is supported so far, and sharpened.** The right axis for the mixed
  training set is per-step *displacement* coverage, not dt coverage — which is
  what generating independent-dt trajectories happens to produce, but the
  hypothesis should be stated in terms of displacement or the experiment will be
  described wrongly. The third alternative from turn 8 ("the crossed split is
  intrinsically harder because it moves further per step") is ruled out in the
  direction it predicted: the worst group moves *less*, not more.
- **The median crossed trajectory is fine.** 0.0151 µm absolute, still below the
  solver's own Δ0.2→Δ0.1 grid error of 0.018 µm. The crossed-split miss is a tail
  phenomenon concentrated in 72 of 209 trajectories, not a uniform collapse, and
  `RESULTS.md` should say so rather than leaving 0.1524 to imply the latter.
- **The clause-1 headline is unchanged.** In-distribution 0.0129, crossed 0.1524,
  both reported. What changes is the *explanation* attached to the second number,
  which is now measured rather than assumed.

### Method note

`scripts/analyse_crossed.py` now reports an absolute band error in microns
alongside every relative one. Any relative metric on a corpus with heterogeneous
target magnitudes needs this; without it there is no way to tell an accuracy
collapse from a small denominator, and the two call for opposite responses.

### Two loop instances are running this repo at once

`ps` on the full command lines:

    PID 2419080  claude -p "# Rules for every autonomous loop this weekend..."  04:18:40
    PID 2419219  claude -p "# Rules for every autonomous loop this weekend..."  04:18:41

Two agents, same brief, started one second apart. The consequences were already
visible in the working tree before I understood the cause: a commit (`02c1def`)
appeared on `master` between two of my own reads; `runs/seed3` and `runs/seed7`
trainers I did not launch were running on my GPU; and a second `design.py --run
runs/seed1` had started at 04:20, two minutes before mine.

So turn 6's "two trainers wrote to `runs/base`" was **not** a slip by one careless
process. It is structural, it will recur every turn, and being careful cannot fix
it. That reframes the withdrawn 0.0400 number: it was a symptom, not the disease.

What I did and did not do:

* **Did:** `eot/runlock.py`, an `fcntl.flock` on a sidecar `.lock`, acquired by
  `train.py` (on the run dir), `eval.py` and `design.py` (on the output file).
  A second writer to the same output now exits 3 with the holder's pid and argv
  instead of silently interleaving. The lock dies with the process, so a killed
  job does not wedge a directory. Six tests in `tests/test_runlock.py`.
* **Did:** strengthened the completeness guard. Line-counting `log.jsonl` was
  never sufficient — `train.py` opens it in **append** mode, so two writers reach
  the epoch budget with neither model converged. It now demands `done.json`, or
  failing that an exact, duplicate-free `0..epochs-1` sequence. Run against the
  existing tree it immediately and independently re-derived turn 6's incident:

      runs/base_RACED_do_not_use -- duplicate epoch records [0,1,2,3,4]
                                    -- two writers shared this log

  That is forensic evidence rather than inference, produced by the guard on its
  first execution against data it was not written for.
* **Did not:** kill the other instance's jobs. The boundary rule says never touch
  another loop's processes, and it holds even when the other loop is me. Its
  `design.py` writes `runs/design_Tfixed.json` and mine writes
  `runs/seed1/design.json`, so they do not collide, and running the identical
  computation twice under different names turns the waste into a free
  determinism check — I will compare them when both land.

**This needs a human and is now D4 in `WEEKEND.md`.** No amount of in-repo
locking fixes a supervisor that starts two α loops; the lock only converts silent
corruption into a loud refusal.

### The crossed-dt number does not survive its own seed spread — H3 killed on arrival

I had a tidy story ready. Holding the crossed-dt data fixed at the original
209-trajectory set: the old modes-16/60-epoch model scored **0.1524** (committed
in `17115c5`) and the new modes-20/80-epoch `seed1` scores **0.2497**. In
distribution the two are indistinguishable — 0.0130 vs 0.0129, against a measured
noise floor of 0.00114. The story writes itself: *the extra capacity and the
extra 20 epochs bought nothing in distribution and cost 64% out of it, so the
model is overfitting to the adaptive-dt protocol.*

Before writing that down I ran the other three converged seeds on the same 209
trajectories. All four models, one architecture, one dataset, seed only:

| run | in-distribution | crossed-dt | crossed terminal | beats recipe-blind null |
|---|---|---|---|---|
| seed1 | 0.0129 | 0.2497 | 0.4962 | yes |
| seed2 | 0.0119 | **0.0641** | 0.1107 | yes |
| seed5 | 0.0131 | **0.2909** | 0.7716 | yes |
| seed6 | 0.0126 | 0.1061 | 0.1793 | yes |
| **mean** | **0.0126** | **0.1777** | — | 4/4 |
| **range** | **0.00114** | **0.2268** | — | — |

**The seed range on the crossed split is 0.2268 — larger than the mean itself,
and 200× the in-distribution range.** The 0.097 gap I was about to attribute to
architecture sits entirely inside it; seed2 at 0.0641 is *better* than the old
model's 0.1524 and seed5 at 0.2909 is worse than seed1, at identical
configuration. H3 is dead, and it was never alive: the effect is smaller than the
noise on the axis it was measured on.

The brief's seed-count lesson cost another track three days. It nearly cost me
this turn, and the thing that saved it was running three more evals — about
ninety seconds of GPU — *before* writing the paragraph rather than after.

Two consequences that outlive the dead hypothesis:

1. **The noise floor is protocol-dependent, and nobody measures it twice.** This
   repo had one noise floor, 0.00114, measured in distribution, and I had been
   treating it as *the* noise floor. On the OOD probe the same pipeline is 200×
   noisier. A tolerance established on the test split licenses nothing on the
   crossed split. Any future crossed-dt comparison needs 8 seeds per arm and an
   exact test; below that it is not reportable at all, not even as a screen.
2. **The previously committed 0.1524 should be read as one draw, not a value.**
   It is not withdrawn — it was honestly measured — but `17115c5`'s framing
   ("removing the adaptive timestep costs a factor of 11.7") rests on a single
   seed of a quantity whose seed range is 4.5×. The factor is somewhere between
   about 5× and 23× depending on which model you happened to train. That is the
   honest statement and it is the one that goes in `RESULTS.md`.

What survives, and it is the substantive physics result: **all four seeds beat
the recipe-blind null on the crossed split.** The operator did learn
recipe-dependent dynamics rather than a protocol-shaped constant — that
conclusion is 4/4 and does not depend on the unstable magnitude. What is unstable
is *how much* accuracy the adaptive-dt protocol was buying, not *whether* the
model learned anything.

### Where clause 1 stands

`[not measured]` is gone from two cells. In distribution, headline run `seed1`:
**0.0129 mean / 0.0194 terminal — MET**, and the arm mean is 0.0126 over 4 seeds
with range 0.00114. Out of distribution on the crossed-dt probe: **0.1777 mean
over 4 seeds, range 0.2268 — MISSED**, on every seed individually. The clause as
written does not name a protocol, so both readings are reported and neither is
allowed to stand alone.

### The other instance's H2, tested against my seed spread — and it survives

While I was measuring the crossed split's seed spread, the other α instance was
attacking the same failure from the other side, and committed
`scripts/analyse_crossed.py` and `runs/crossed_analysis.json` under H2:
*the crossed-split failure is displacement coverage, not capacity, not dt, not
the metric.* Its evidence was `runs/seed1` alone: crossed error 0.0465 on the 137
trajectories whose mean per-step displacement lies inside the training range,
against 0.6363 on the 72 outside it.

That is exactly the kind of claim my own result this turn should have made
suspect. A conclusion drawn from one seed of a quantity whose seed range is
0.2268 is not a conclusion, and 0.0465 sits close enough to the 0.05 clause that
a different seed could plausibly have put it on the other side. So rather than
accept it or duplicate it, I ran its analysis on the other three converged seeds:

| run | in-distribution | crossed (all) | crossed, displacement IN range | crossed, OUT of range |
|---|---|---|---|---|
| seed1 | 0.0129 | 0.2497 | 0.0465 | 0.6363 |
| seed2 | 0.0119 | 0.0641 | 0.0294 | 0.1299 |
| seed5 | 0.0131 | 0.2909 | 0.0340 | 0.7796 |
| seed6 | 0.0126 | 0.1061 | 0.0389 | 0.2341 |
| **mean** | 0.0126 | 0.1777 | **0.0372** | 0.4450 |
| **range** | 0.00114 | 0.2268 | **0.0171** | 0.6496 |

**H2 holds, and it holds on the axis that could have killed it.** All four seeds
meet the clause in range — worst 0.0465, and seed1 that the other instance
happened to pick is the *worst* of the four, so its claim was if anything
conservative. None of the four meets it out of range.

The decisive detail is one I would not have looked for: `frac_dt_outside_trained_
range` is **0.0** on both splits. Not one crossed trajectory has a dt the model
never saw. The crossed split was built to test dt decoupling and it does not test
dt decoupling at all — dt stays in range and *displacement* leaves it. So the
variable named in the probe's design is not the variable that breaks it.

That retires my own earlier framing, and turn 5's. "Removing the adaptive
timestep costs a factor of 11.7" attributes the loss to dt. It is not dt.
The adaptive protocol helped only because choosing dt per recipe kept per-step
displacement in a narrow band; decouple dt but keep displacement covered and the
clause is met (0.0372 mean, 4/4 seeds). The honest statement is **coverage, not
protocol**, and the two were perfectly confounded until this decomposition split
them.

It also explains my seed-spread result rather than merely coexisting with it. The
seed range is 0.0171 in range against 0.6496 out of it — a factor of 38. The
pipeline is not "unstable"; it is stable where the data covers it and
seed-dominated where it extrapolates, which is what extrapolation looks like and
is not a defect to be trained away.

**Clause 1, stated honestly and completely:**

* in-distribution (adaptive dt): **0.0126** mean over 4 seeds, range 0.00114 — **MET**
* crossed dt, displacement inside training coverage: **0.0372**, worst seed 0.0465 — **MET**
* crossed dt, displacement outside coverage: **0.4450**, seed range 0.6496 — **NOT MET**
* aggregate over the whole crossed split: **0.1777** — **NOT MET**, and this is
  the number a paper would be tempted to omit

The clause is met wherever the operator is used inside its training coverage,
including under a protocol it was not trained on. Whether that counts as passing
is a judgement about scope, not about measurement, and it is stated as a decision
in `WEEKEND.md` rather than resolved by me in the direction I would prefer.

A note on the collaboration, since it was forced rather than chosen: two
instances sharing one working tree produced, this turn, one genuinely better
result than either was going to get alone — its decomposition, my seed spread,
and neither is sufficient without the other. That is not an argument for running
two loops on one repo. The same collision killed both inverse-design jobs after
eight minutes of duplicated compute and left clause 3 unmeasured for a fourth
turn, which is the larger cost.

### codex took the coverage claim apart, and four of its objections were right

I put the clause-1c claim in front of `codex exec` — specifically the claim that
displacement coverage, not dt, explains the crossed failure, and that all four
seeds meet ≤0.05 in coverage at 0.0372 mean / 0.0465 worst. Quoting it:

> **The strongest objection is that "training coverage" is actually a
> ground-truth, test-defined selection that can preferentially recover
> trajectories resembling the adaptive protocol. It does not isolate displacement
> as the cause.**
>
> 1. *"The bounds come from the adaptive test set, not training.*
>    `analyse_crossed.py:126` … sets `lo, hi = disp_t.min(), disp_t.max()` … Yet
>    `RESULTS.md:151` describes **training 1st–99th percentiles, 0.073–0.330 µm,
>    76/225 trajectories**. … a total of **209**, not 225. Thus the passing
>    scores do not establish performance on the advertised training-coverage
>    subset."
> 2. *"Conditioning on a narrow displacement interval therefore selects
>    compatible recipe–dt combinations, potentially bringing them closer to the
>    adaptive training relationship. … the comment 'so only dt differs' … is
>    unjustified"*, and *"dt inside its marginal minimum/maximum does not imply
>    coverage of joint (recipe, geometry, dt) inputs"*.
> 3. *"0.0465 is a passing point estimate, not a robust pass. Its margin is only
>    **0.00347**, compared with an observed in-subset seed range of **0.01711**,
>    about five times larger. … There is also a metric mismatch: … `RESULTS.md:11`
>    says **both mean and terminal-step readings must pass**. Clause 1c provides
>    no subset terminal-step check."*
> 4. *"The report's verdict is hard-coded. `report.py:440` … prints 'Every seed
>    meets' … without testing the threshold."*

All four are real. Taking them in the order that mattered:

**(1) is a straight contradiction and it was mine.** Two definitions of "in
range" were printed in adjacent sections of one document: a table selecting on
the test split's displacement min/max over 209 trajectories, and a paragraph
describing the train split's 1st–99th percentile over 225. I generated the second
from `confound.json` without checking it described the same rule as the first. A
reader comparing the two would have found 137 and 76 for the same concept.

**(4) is the same sin I had spent this turn fixing elsewhere.** I removed a
hard-coded "34.2×" from `report.py` in the morning and then wrote "Every seed
meets the clause" as literal prose in the afternoon. An assertion in a generator
is not a computed result, whether the number beside it is or not.

**(3) is the one that changed the answer.** I wrote `scripts/coverage_verdict.py`
to do what the critic asked: per-trajectory errors, both readings, three coverage
rules instead of one, and a 10,000-resample bootstrap **over trajectories, paired
across seeds** — the four seeds share the same trajectories, so resampling seeds
would be resampling the wrong thing.

| coverage rule | n in | reading | in-range [95% CI] | worst seed | ≤0.05? |
|---|---|---|---|---|---|
| A test min/max | 137/209 | mean-over-steps | 0.0372 [0.0241, 0.0535] | 0.0465 | no |
| A test min/max | 137/209 | **terminal-step** | 0.0600 [0.0384, 0.0865] | 0.0798 | **no** |
| B train p1–p99 | 121/209 | mean-over-steps | 0.0330 [0.0214, 0.0474] | 0.0411 | yes |
| B train p1–p99 | 121/209 | **terminal-step** | 0.0531 [0.0340, 0.0767] | 0.0703 | **no** |
| C train min/max | 139/209 | mean-over-steps | 0.0410 [0.0269, 0.0581] | 0.0515 | no |
| C train min/max | 139/209 | **terminal-step** | 0.0665 [0.0432, 0.0947] | 0.0889 | **no** |

**Verdict in coverage: NOT MET, under every rule.** The mean-over-steps reading
survives on the strictest rule; the terminal-step reading fails everywhere. Clause
1 has required both readings since turn 5, and the subset analysis had only ever
computed the mean. So **I withdraw the claim I committed earlier this same turn**
— "clause 1 MET within displacement coverage, 4/4 seeds" is wrong. It was wrong
by omission of the stricter reading, and it took an adversary to notice, which is
the argument for running one.

**(2) survives as a limitation rather than a refutation, and it got worse when
measured.** The critic is right that selecting on true displacement is
target-informed and cannot certify anything in deployment. So I computed the
deployable version: the same bounds applied to the displacement the *model
predicts*, which needs no oracle. It agrees with the oracle selector on 87.1% of
trajectories — and gives **0.1402** where the oracle gives 0.0330, 4.3× worse.

The mechanism is worth stating because it is not obvious: a trajectory the model
gets badly wrong also mis-predicts its own displacement, and therefore **admits
itself into the "covered" set by virtue of its own error**. The selector is
anti-correlated with what you want exactly where it matters. So the coverage
finding explains the failure but **does not license a domain of validity you
could ship**, and I have stopped writing it as though it did.

(A first version of that selector disagreed with the oracle 56% of the time. That
was my bug, not a finding: I computed predicted displacement as a whole-field
mean absolute difference while `per_step_displacement` uses a band-weighted
signed normal advance. Matching the definitions moved agreement from 0.435 to
0.871. Two quantities with the same name and different definitions, again.)

One objection I do not accept, and why: the critic implies the 209-vs-225
discrepancy might indicate the scores come from the wrong population. It does
not — 209 is the crossed set under `data/`, 225 is the one under `data_crossed/`,
built later with a different window-rejection rule. Both are real and the
analysis is internally consistent on 209 throughout. The defect was describing
one while measuring the other, which is a reporting error, not a scoring error.

**Where this leaves clause 1**, with everything now computed rather than asserted:

* in-distribution, adaptive dt: **0.0126** over 4 seeds, range 0.00114 — **MET**
* crossed dt, inside displacement coverage: 0.0330 mean-over-steps (upper bound
  0.0474, passes) but 0.0531 terminal-step (upper bound 0.0767, fails) — **NOT MET**
* crossed dt, outside coverage: 0.3767 — **NOT MET**
* using a selector that needs no oracle: 0.1402 — **NOT MET**

The clause holds in distribution and nowhere else. That is a narrower and less
satisfying result than the one I had at midday, and it is the one the
measurements support.

---

## Turn 8 — clause 3 has a number at last, and three things about it need saying

Both inverse-design runs landed while the loop was down (`runs/design_Tfixed.json`
04:53, `runs/design_Tfree.json` 05:13, both from `runs/seed1`, invocations
recorded in the `.lock` files and per-target progress in `logs/design.log`).
`n_failed_simulation` is 0/20 in both, `n_pinned_at_wall` is 0, and the minimum
box margin is 0.061 (T-pinned) / 0.083 (T-free), so no proposed recipe sits on a
wall of the region the simulator was validated in. The tripwire row — the true
recipe re-simulated — reads 0.0023, which is the pipeline's own floor.

| protocol | area error vs removed, in ViennaPS | max over 20 | targets < 5% | random search |
|---|---|---|---|---|
| T pinned to target | 0.0098 | 0.0587 | 19/20 | 0.0232 |
| T searched (honest) | **0.0061** | 0.0098 | 20/20 | 0.0233 |

**Clause 3's threshold is 5% and the honest protocol reads 0.61%.** That is a
factor of 8 of margin, and unlike clause 1 it is measured in the simulator on the
recipe the operator proposed, not on the surrogate's opinion of it.

### (a) The metric is the harder of the two on offer, which is worth stating

`shape_error` computes two normalisations and the report uses `area_error_vs_removed`
(mismatch over the area the target etch removed). The other reading,
`area_error_vs_solid`, gives **0.0009** — seven times smaller. Had I quoted that,
the clause would pass by 55× instead of 8×. It would also have been meaningless:
dividing by the whole solid area makes the score depend on how much of the window
happens to be filled, not on the etch. And the hard-threshold reading (0.0070,
i.e. counting cells by sign rather than by smoothed occupancy) sits *above* the
soft one, so the occupancy smoothing is not hiding a quantisation gap either. For
once the number I had already committed to was the conservative one; recording
this because the reverse is the failure mode the rules warn about, and the only
way to know which way it went is to compute both.

### (b) The recipe it finds is not the recipe that made the target, and by a lot

Measured from the same JSONs (`scripts/analyse_design.py`), distance between the
proposed and the true recipe as a fraction of the recipe box, per axis and RMS:

| protocol | ion_flux | etchant_flux | oxygen_flux | ion_energy | RMS | > 10% of box |
|---|---|---|---|---|---|---|
| T pinned | 0.169 | 0.041 | 0.122 | 0.218 | 0.167 | 13/20 |
| T searched | 0.197 | 0.097 | 0.135 | 0.184 | **0.177** | 18/20 |

And in the T-free arm the etch time it recovers is wrong by **23% on average and
55% at worst** — while the *shape* it produces is better than the arm that was
handed the true time. Correlation between recipe distance and shape error is 0.22.

So the inverse problem is **degenerate over (rate × time)**: the operator finds a
different process that lands on the same profile, and gets a better profile
precisely because searching T gives it a fourth-plus-one dimensional family to
slide along instead of a fixed-depth constraint. That also explains the
counter-intuitive ordering in the table above — I had labelled T-pinned "the
optimistic bound" in `run_design.sh` and in `inverse.py`'s docstring, expecting it
to win. It loses. The label is wrong and I am changing it: pinning T is not
optimistic, it is *constrained*, and the constraint costs more than the
information it hands over.

The consequence for the claim is sharp and I want it in the headline rather than a
caveat: **clause 3 as written asks for 형상오차 — shape error — and shape error is
met. "The twin recovers the process recipe" is a different claim and it is NOT
supported: 18 of 20 recovered recipes sit more than 10% of the box away from the
truth.** Anyone using this to *set a tool* rather than to *hit a profile* is using
a result that was not measured.

### (c) The random-search baseline was not budget-matched, and I wrote that it was

`design.py`'s own docstring says random search gets "a comparable number of
operator evaluations". The invocation that produced these numbers was
`--iters 200 --restarts 3 --random-budget 256`. Gradient descent therefore spent
3 × 200 = **600 rollouts forward and 600 backward**; random search spent **256
rollouts forward and none backward**. Taking a backward pass at roughly twice a
forward, GD had on the order of **7× the operator compute**, not a comparable
amount. The 3.8× quality gap (0.0061 against 0.0233) is therefore, as measured, a
gap between two methods at unequal budget — which is precisely the shape of
comparison the portfolio rules single out ("the baseline decides whether the
number means anything… report the objective-versus-compute curve rather than one
number, so the comparison cannot hide in the budget").

### H3, written before the run

> **Gradient descent's advantage over random search is not a budget artefact.**
> Random search ranked by the same operator, given 64× its original budget
> (16,384 candidates, ≈27× GD's total operator compute), will still not reach
> GD's simulated area error of 0.0061 on the T-free protocol.
>
> Falsified if the curve reaches ≤0.0061 at any budget ≤16,384. Given 4 search
> dimensions, best-of-N error should fall roughly as N^(-1/4), so 256 → 16,384 (a
> 64× budget) predicts about a 2.8× improvement: 0.0233 → ~0.0083. That lands
> *just above* GD, which is why the experiment is worth running rather than
> arguing — the predicted margin is thin enough that I cannot call it from the
> exponent.
>
> The alternative explanation this distinguishes against: the operator is a
> usable *ranker* but its gradients carry no extra information, in which case
> enough random samples buy the same answer and the word "design" is not earned.

Running it costs one simulator call per budget per target, because only the
best-of-budget candidate is verified; the ranking itself is batched and forward-only.

### Also this turn, and separately: clause 3 rests on one seed

`runs/seed1`. The seed-count lesson in the brief is explicit that one is not a
verdict, and this repo has already withdrawn a crossed-split claim whose seed
range exceeded its own mean. Clause 1 is reported over 4 converged seeds; clause 3
must be too before it goes on the board as met. Replication on seeds 2, 5 and 6 is
launched alongside H3. The margin is 8×, so a spread would have to be enormous to
flip the verdict — but "would have to be enormous" is an argument, and the board
takes measurements.

## Turn 8, continued — `codex` on clause 3: six objections, and the first one is a leak

Run with stdin closed (the earlier invocation hung reading stdin and returned an
empty file, which is why the first adversary attempt produced nothing). Ranked by
the critic itself. Verdicts are mine, and two of the six change code.

**1. "The supposedly unknown duration leaks through initialization." REAL, and it
is the most damaging thing found today.** `eot/inverse.py:167` computed the
starting point of the duration search as

```python
u0 = (np.log(dt) - np.log(lo)) / (np.log(hi) - np.log(lo))
```

where `dt` is *the target's own timestep*, which `gen_data` obtained from a
simulator probe of the true recipe's etch rate. Restarts randomise the four
recipe knobs (`design.py`, `init=`) and never touched `z_dt`. So every restart of
the arm labelled **"total etch time searched (T unknown, honest)"** began at the
ground-truth duration. The label is not supported by the code, and I wrote both.

It is a *warm start*, not a pin — the optimiser leaves it, and by a lot: the
recovered dt is 23% from the true value on average and 55% at worst. So the leak
does not trivially explain the result. But "it moves away afterwards" is an
argument, and the rules here say the baseline gets re-measured rather than
argued. `--dt-init {target,mid,random}` now exists; `target` remains the default
so the published numbers stay reproducible, and the JSON records
`dt_init_was_the_target` so no reader has to reconstruct the invocation.

> **H4, written before the run.** The T-searched result does not depend on being
> warm-started at the true duration. Re-running with the duration initialised
> independently of the target — log-uniform in the trained dt range, a fresh draw
> per restart, and separately at the geometric centre of that range — keeps the
> mean area error under the 0.05 clause.
>
> Falsified if either honest initialisation exceeds 0.05. I expect degradation:
> the surrogate loss over (recipe, dt) is what the optimiser sees, and a 4+1
> dimensional non-convex landscape entered from a random point should find worse
> minima than one entered at the true depth. My guess is it lands between 0.006
> and 0.02 — worse, still passing. If it lands above 0.05 then clause 3 as
> published was carried by the warm start and must be withdrawn.
>
> Distinguishes against: "the operator's gradients are strong enough that the
> starting point does not matter", which is what a null result here would mean,
> and which the 3-restart random-init behaviour of the recipe knobs already
> weakly suggests.

**2. "`_occupancy` is not the symmetric-difference area it claims." Real as a
statement about the metric, but already bracketed by measurements I had.** The
critic is right that `clip(0.5 - sdf/h, 0, 1)` is exact only for contours
parallel to a cell edge, and right that equal cell fractions can hide sub-cell
disagreement. What decides whether it matters is the size of the effect, and
`shape_error` already computes the hard-threshold reading (count cells by the
sign of the SDF, no smoothing) beside the soft one:

| reading | T-searched | T-pinned |
|---|---|---|
| smoothed occupancy (published) | 0.0061 | 0.0098 |
| hard sign threshold | 0.0070 | 0.0089 |
| true recipe re-simulated (floor) | 0.0024 | 0.0023 |

The two readings differ by 0.0009, and the *published* one is the smaller in the
T-searched arm — so quantisation contributes about a thousandth, not the ~3% the
critic feared, and the honest floor of the whole pipeline is 0.0024. The
operator's error is 2.5× that floor and 8× under the clause. Not accepted as a
threat to the verdict; accepted as a reason the report must print both readings,
which it now does.

**3. "Targets are a curated, reachable, simulator-generated family." Real, and it
was stated in `design.py`'s docstring and nowhere a reader would look.** Every
target is a profile ViennaPS produced from a recipe inside the training box, with
the true initial geometry supplied. That is a deliberate choice — it means a
solution provably exists, so a failure is the method's — and it is also a hard
scope limit: nothing here measures inversion of an independently specified
manufacturing target, a different depth regime, or a geometry outside the box.
Promoted from a docstring to the RESULTS.md scope paragraph.

**4. "Global area error can conceal local failures; Hausdorff never gates `met`."
Real, and the answer is a number I had not put beside it.** The grid is
Δ = 0.2 µm. Worst-case Hausdorff distance over 20 targets is **0.0733 µm** for
the operator's proposal and **0.0348 µm** for the true recipe re-simulated —
**0.37 and 0.17 of one grid cell**. The largest *local* deviation anywhere in the
worst target is a third of the discretisation. That is the defensible reply to
"the global metric could hide a notch", and it is stronger than the area number
it supports. It goes in the report next to the area figure.

**5. "`met` checks only the mean, and the T-pinned arm has a failing target."
Real.** 19/20 with one at 0.0587. `met_mean` and `met_every_target` are now
separate flags, both printed. Under the T-searched protocol the distinction is
moot (max 0.0098); under T-pinned it decides the verdict, which is exactly why it
cannot be left implicit.

**6. "`steps_ok=False` is recorded and not enforced." Real, and it never bit.**
`solver.simulate()` stops at the window and repeats its final frame with
`steps_ok` False; `simulate_recipe` returns `failed=None` regardless, and `agg`
counted only `failed`. So `n_failed_simulation: 0` did not mean verification
completed. Measured across all three arms of both published runs: **0 of 20
truncated, everywhere**. The published numbers are unaffected. `agg` now reports
`n_truncated_simulation` and `n_verified_complete`, because a hole that has not
bitten yet is still a hole, and this repo has already lost a number to one.

Nothing in the six was wrong. That is a worse result for me than the last review,
where I could dismiss one of four.
