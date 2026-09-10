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

---

## 2026-09-10, turn 8 — H3 lands, and E4 is declared

### H3: the random-search baseline, resolved against a paired exact test

**The number.** Gradient descent through the operator: **0.00607** mean
normalised area error, measured in ViennaPS, 20 targets (`runs/design_Tfree.json`).
Random search over the same operator, same 20 targets, nested prefixes so the
curve is free of independent-draw noise, and keeping the target's *true* etch
time throughout — which makes the baseline strictly stronger than the searched-T
method it is compared against (`runs/random_curve.json`):

| candidates | budget vs GD | random | GD wins | p (exact sign-flip, 2^20) |
|---|---|---|---|---|
| 64 | 0.0× | 0.0467 | 20/20 | 0.0000 |
| 256 | 0.1× | 0.0233 | 19/20 | 0.0000 |
| 1024 | 0.6× | 0.0130 | 18/20 | 0.0001 |
| 1800 | 1.0× | 0.0119 | 18/20 | 0.0002 |
| 4096 | 2.3× | 0.0099 | 18/20 | 0.0014 |
| 16384 | 9.1× | 0.0064 | 13/20 | **0.4842** |

**The hypothesis I wrote before the run was that random search would not reach
GD at any budget up to 16,384. On the means it did not — 0.0064 against 0.0061 —
and reporting that as "H3 survives" would have been exactly the mistake this
repo keeps making.** A 5% gap on 20 paired targets is inside the range where
this workspace has already been wrong twice. The paired test settles it: at
16,384 candidates GD wins 13 of 20 targets, p = 0.484 on an exact sign-flip over
all 2^20 assignments, p = 0.263 on an exact sign test. Not separated.

**So the claim changes shape.** What differentiability buys on this problem is
**~9× less search compute for the same profile error**, not a better optimum.
GD beats random search decisively at every budget up to 2.3× its own cost
(p ≤ 0.0014) and loses its significance somewhere between 2.3× and 9.1×. The
framing "the gradients find minima sampling cannot" is not supported by any
measurement here and is withdrawn from the report; `scripts/random_curve_test.py`
now generates the paragraph that replaced it.

**What would distinguish this from the obvious alternative.** The obvious
alternative is that the recipe box is only 4-dimensional and small, so random
search is competitive for reasons that have nothing to do with the operator
being good or bad. That is very likely part of it, and it is a limit on the
generality of the clause-3 result rather than a defect in it: on a 4-D box with
a smooth forward map, 16k samples is dense. It predicts that the gradient
advantage grows with recipe dimension, which this repo has not tested and which
is now written into `paper_draft.md` as the experiment that would decide it. It
does *not* rescue the withdrawn framing, because the measurement stands
whatever the reason for it.

### Declaration

E4 is **UNREACHABLE**, 2 of 3 clauses met, declared under the 10-turn rule:

- **Clause 1, rel-L2 ≤ 0.05 — MET in distribution and nowhere else.** 0.01256
  band rel-L2 over 7 converged seeds, range 0.00114 (`runs/seed_spread.json`).
  Crossed-dt fails under every coverage rule; the deployable, oracle-free
  selector scores 0.1402. Two of this repo's own claims about this clause were
  withdrawn getting here, and the narrow version is what survived.
- **Clause 2, ≥1000× — UNREACHABLE.** 1.47× like-for-like, short by 679×. Not a
  margin an implementation closes. The denominator that would have passed it
  (solver timed as 10 concurrent processes: 1006×) is in the grid and rejected
  in the text.
- **Clause 3, shape error ≤5% — MET.** 0.0061 and 0.0064 on two independently
  trained operators, 20/20 targets each, measured in ViennaPS on the recipe the
  operator proposed, 0 pinned at a box wall, 0 truncated simulations, floor
  0.0024. Worst-case Hausdorff 0.0733 µm = 0.37 of one grid cell. It is profile
  targeting: the recovered recipe sits 0.177 of the box from the truth.

**One check is still running and can withdraw clause 3** — H4, the duration
search initialised independently of the target rather than at the target's own
dt (`logs/dt_honest.log`, GPU 1, two arms). If either honest initialisation
exceeds 0.05 the clause comes down and this entry is corrected the same turn.
That is the one thing that outranks the turn budget: a wrong number in a
document is always worth a turn. Clause 3 is published with the warm start
disclosed in the protocol string and recorded in the JSON as
`dt_init_was_the_target`, so no reader is misled while the check runs.

---

## Turn 2026-09-10 08:2x — REOPENED by harness addendum 2026-09-10b. Rung 2, and a contaminated measurement found while preparing for it.

The declaration above stands as a record but is no longer the verdict: the
addendum reopens clause 2 on the ground that `UNREACHABLE` was reported without
the architecture ever being attacked. It is right, and the route it names
(predict K steps per application) is the route I would pick. Before running it I
went to re-read `runs/speed.json` and found something worse than the clause.

### Finding 1 (defect, mine, in the flattering direction): the speed table times a model that no surviving run produced

`runs/speed.json` records `"params": 16810841`. No run on disk has that
parameter count. The nine converged accuracy runs (`runs/base`, `runs/blind`,
`runs/seed1..8`) are all `width=64, modes=20, layers=4` = **26,248,025**
parameters. The only two directories that carry 16,810,841 are `modes=16`:

- `runs/base_RACED_do_not_use` — the raced directory withdrawn in 7a9a66e, and
- `runs/UNTRACED_t4_base_s0`.

`bench_speed.py` reads `width/modes/layers` out of `<run>/args.json` and defaults
`--run runs/base`, so the table was produced when `runs/base` still held the
`modes=16` config — i.e. **the entire clause-2 speed grid, including the
headline 1.47× like-for-like figure, times a network 1.56× smaller in parameters
than the network whose 0.01256 accuracy clause 1 claims.**

Why this matters and which way it cuts: a smaller operator is *faster*, so
1.47× is an **over**-statement of the real like-for-like speedup, and every
derived figure (199× throughput, 347× batched) is over-stated by the same ratio.
The error therefore runs in the direction that flatters the clause, which is the
direction this repo has already had to correct twice (the 40.8× clock-ramp
artefact in A4, the stale `test_eval.json` here). It also means clause 1's
accuracy and clause 2's cost were never measured on the same object, which is
exactly the pairing the KPI is about — an accuracy number and a speed number
belong to one model or they belong to nothing.

This is rung 3 of the ladder (fix your own setup) and it is not optional:
re-bench on `runs/seed1`, the model that carries a reported accuracy, before any
K-step number is compared against it. Both the old and the new row go in the
table; the old one is labelled with the run it actually timed.

I do not yet know the size of the correction. `modes=20` vs `modes=16` changes
the spectral einsum by (20/16)^2 = 1.56× in the multiply and nothing in the
FFTs, so the wall-clock ratio will be smaller than 1.56× and larger than 1. It
will be measured, not estimated, and this paragraph will be replaced by the
measurement.

### Finding 2: the step count is not the only thing between 1.47× and 1000×, and arithmetic says so before any run

The addendum's reasoning is that cost is linear in the number of applications,
so K applications collapsed into one divides operator cost by K. That is
correct, and with T = 10 steps per wafer the largest possible K is 10. So the
ceiling of the strict like-for-like reading under this change is

    (measured 1-application CPU cost) -> speedup_K=10 = 10 x speedup_K=1

which, taking the (contaminated, about to be corrected) 1.47× at face value, is
**~14.7×, against a clause of 1000×.** The addendum's "199× becomes ~199K×"
lands at ~1990× only on the *throughput-vs-throughput* row, and that row is the
one this repo already refuses to quote as the verdict because it compares a
batched H100 against parallel CPU processes — different hardware. Collapsing
steps does not convert a hardware comparison into a like-for-like one.

So the honest statement of the gap, before the run: horizon collapse can buy at
most one order of magnitude of the two-and-a-half orders that are missing. The
remaining factor has to come from the *per-application* cost — a 26.2M-parameter
FNO at 128×128 costs ~0.10 s on one CPU thread, and 1000× against a 1.65 s
solver requires 1.65 ms. That is a 60× cut in per-application cost, and it is a
statement about network size, not about horizon.

**This does not make the K-step experiment pointless — it makes it the first of
two axes**, and it is the axis that also attacks clause 1, which is why it goes
first exactly as the addendum says. What I am refusing to do is run K and then
report "199K×" as though the clause had moved, when the reading that the verdict
rests on would read 14.7×.

### H5 (one hypothesis, one change): horizon collapse

**Change:** the operator is trained on pairs (phi_t, recipe, K·dt) -> phi_{t+K}
instead of (phi_t, recipe, dt) -> phi_{t+1}. Architecture, width, modes, layers,
epochs, optimiser, normalisation constants: all unchanged. K in {1, 2, 5, 10},
which are the divisors of T = 10, so every arm ends at the same physical time
and the terminal states are comparable field-for-field.

The conditioning already carries `log_dt`, so no architectural change is needed
to express a longer step; the K arms differ only in which pairs they see and in
the constant `log K` added to that input. `data/norm.json` is **not** refitted
per arm — all arms are standardised by the same train-split constants, so the
comparison is not confounded by a change of normalisation. The consequence is
that the K=10 arm sees standardised `log_dt` up to ~+5.2 where K=1 saw at most
+1.8; that is stated because it is a real difference between the arms and it is
the price of keeping the constants fixed.

**Prediction, written before the run, so it can be wrong:**

1. *Terminal-step* band rel-L2 falls from K=1 to some intermediate K, because
   compounding over 10 applications is replaced by compounding over 10/K, and
   then rises again as the single-application map gets harder. If it is
   monotonically *increasing* in K, horizon collapse buys nothing for clause 1
   and the accuracy-vs-K curve is a negative result.
2. The crossing point is where the two error sources balance. I do not have a
   prediction for where, and refuse to guess one.
3. Clause 2's strict reading multiplies by K, to a hard ceiling of 10× the
   corrected K=1 figure. Measured, not assumed: per-application CPU cost is
   re-timed for each arm, because a K arm is not guaranteed to cost the same per
   application as a K=1 arm even at identical architecture.

**What would distinguish this from the obvious alternative.** The obvious
alternative explanation for any accuracy gain at K>1 is not horizon collapse but
*fewer optimisation targets*: a K=10 arm has 900 training pairs where K=1 has
9000, so if K=10 wins it might be winning on an easier fit, and if it loses it
might be losing on 10× less data rather than on map difficulty. Two things
separate them, both of which I will run: (a) the pair count per arm is recorded
in the run JSON so the comparison is never quoted without it, and (b) a
**data-matched control** at each K trains on all T-K+1 overlapping start
offsets rather than the K non-overlapping ones, which restores the pair count
to ~9000 at every K. If the K-curve has the same shape in both, the effect is
horizon; if it flattens in the data-matched arms, the effect was data volume and
the horizon story is withdrawn.

**Seeds.** The seed spread at K=1 is 0.00114 over 7 seeds on the in-distribution
band reading. Any K effect smaller than that is not an effect. Verdict arms get
8 seeds and an exact paired sign-flip test across the shared trajectory set, per
the seed-count lesson; the first pass is 3 seeds and is labelled **screen, not
verdict** until the eighth lands.

---

## Turn 2 (reopened) — H4 landed and it withdraws clause 3's headline

`runs/design_Tfree_dtinit_random.json`, 20 targets, `runs/seed1`, `--restarts 3
--iters 200`, duration initialised log-uniformly in the trained `dt` range,
independently of the target (`dt_init_independent_of_target: true`).

| reading | leaky init (`dt_init=target`, the declared number) | honest init (`dt_init=random`) |
|---|---|---|
| mean normalised area error over 20 targets | **0.0061** | **0.0613** |
| median | 0.0060 | 0.0069 |
| p90 | 0.0087 (`design_Tfree.json`) | 0.0125 |
| max | 0.0087 | **1.0799** |
| targets under the 5% clause | 20/20 | **19/20** |
| `dt_rel_err` median / max | 0.201 / 0.548 | 0.458 / 1.678 |
| best surrogate loss, median / max | 0.0144 / 0.0197 | 0.0159 / **1.9199** |

**What the number was and what it is.** Clause 3 was declared MET at 0.0061 on
the arm labelled "T unknown, honest". That arm was warm-started at the target's
own duration (`eot/inverse.py`, found by `codex`), and the fix was implemented
and the honest arm queued before the declaration, with the declaration saying in
writing that it could withdraw clause 3. **It has, on the mean reading:
0.0613 > 0.05.** The mean is not a near miss; it is 10× the leaky figure.

**Why.** Not a uniform degradation — 19 of 20 targets are essentially unchanged
(median 0.0069 against 0.0060) and one target, idx 237, goes to 1.0799. Its
duration search converged to `dt = 0.7906` against a true 0.5249 (`dt_rel_err`
0.506), which over-etches by 50% and truncates the simulation
(`sim_status.steps_ok: false`). So the honest protocol does not make the
inverse design worse; it makes the **duration search's non-convexity visible**,
and with 3 restarts drawn log-uniformly across the whole trained `dt` range, 1
in 20 targets keeps a bad basin as its best-of-3.

**The binding constraint is the multi-start budget I chose, not the operator.**
Two measurements say so rather than one intuition:

1. The failure is **self-announcing without any ground truth**. Best surrogate
   loss is ≤ 0.0327 for all 19 successes and **1.9199** for the failure — a 59×
   separation, and rank correlation between surrogate loss and simulated area
   error rises from −0.117 (leaky) to **+0.630** (honest). Any threshold in
   (0.033, 1.92) rejects exactly the one bad design and keeps all 19. Restart
   selection is *already* by surrogate loss, so this is the same oracle-free
   quantity the optimiser has in hand — no protocol change, no label.
2. Random search with 256 candidates scored **0.0260** on idx 237 — the target
   is not intrinsically hard, and GD's failure there is a basin, not a limit of
   the operator's accuracy on that profile.

**What would distinguish this from the obvious alternative.** The obvious
alternative is that the honest protocol is simply harder and the operator cannot
hit 5% without being told the duration — in which case error would rise across
*all* targets and more restarts would not help. What I measure instead is a
19-vs-1 split with the 19 unmoved, which predicts that a larger multi-start
budget removes the failure and that the accepted-design error stays at ~0.007.
If a restart curve flattens with the failure still present, the alternative is
right and clause 3 is genuinely lost under the honest protocol.

**Also true and reported next to it:** `dt_rel_err` median 0.458 under the
honest init against 0.201 under the leaky one. The (rate × time) degeneracy this
repo already recorded is *worse* than it looked: with no hint about duration, the
recovered process is off by ~46% in etch time while the profile matches to
0.7%. Profile targeting, not recipe identification — unchanged conclusion,
larger effect.

### H6 (one hypothesis, one change): the multi-start budget, and a curve not a point

**Change:** `--restarts 3` becomes a restart *curve* R ∈ {1, 2, 3, 6, 12} under
the honest `dt_init=random`, with the design at each R chosen by the same
surrogate-loss argmin over the first R restarts and simulated in ViennaPS.
Everything else — operator, targets, iters, box, metric — is held fixed.

**Prediction, written before the run:**

1. Mean area error falls below 0.05 by some R ≤ 12 because idx 237's bad basin
   is escaped, and the median stays ~0.007 throughout (the 19 are already
   converged, so extra restarts must not move them).
2. The surrogate-loss screen at any τ ∈ (0.05, 1.0) accepts 19/20 at R = 3 and
   20/20 at the R where the mean passes; if it ever *rejects* a design whose
   simulated error is < 5%, the screen is not free and that is reported.
3. Forward-equivalents scale linearly in R, so R = 12 costs 4× R = 3. Clause 3
   has no compute clause, but the GD-vs-random comparison does, and the matched
   budget for random search moves with R — the R = 12 row must be compared
   against a random-search budget of 12/3 × the R = 3 one or the comparison is
   the budget, not the method. `runs/random_curve_test.json` already carries the
   64→16,384 curve to read that off.

**What this is not.** It is not a protocol loosening: the initialisation stays
independent of the target, the selector stays oracle-free, and the number
reported for a given R is the mean over all 20 targets with no target dropped.
If the mean only passes because a target was screened out, the screened figure
is reported as a *screened* figure with its rejection rate next to it, and the
all-targets mean stays in the same table.

---

## Turn 2 (reopened) — H5's first pass falsifies its own prediction 1

`runs/kcurve_screen.json`, one seed per K arm against the 8-seed K=1 anchor,
terminal-step band rel-L2 (the only reading comparable across K), coverage rule
B (train p1–p99 displacement, 121 of 209 crossed trajectories inside).

| arm | pairs | applications/wafer | in-dist terminal | crossed **in coverage** terminal |
|---|---|---|---|---|
| K=1 (anchor, 8 seeds) | 9000 | 10 | **0.01890** | **0.05433** |
| K=2 nv | 4500 | 5 | 0.02803 | 0.06832 |
| K=5 nv | 1800 | 2 | 0.03460 | 0.08482 |
| K=10 nv | 900 | 1 | 0.04868 | 0.11363 |
| K=2 ov (data-matched) | 8100 | 5 | 0.10859 | 0.14481 |
| K=5 ov (data-matched) | 5400 | 2 | 0.08057 | 0.12956 |

**Prediction 1 was that the terminal error would fall from K=1 to some
intermediate K and rise after. It does not fall anywhere.** The curve is
monotonically increasing in K on both splits. Every arm loses to the anchor on
a paired per-trajectory comparison — the pairing unit is the test trajectory,
which every arm sees — with mean differences +0.0091 to +0.0897 in distribution
and +0.0140 to +0.0905 in coverage, at sign-test p < 1e-4 over 121–250 pairs.
The smallest of those effects is **5× the anchor's own 8-seed range** (0.00172
on the terminal reading), so this is not seed noise.

**What that does to the two clauses the addendum coupled.**

* Clause 1 in coverage moves the **wrong way**: 0.0543 → 0.1136 at K=10. The
  route that was supposed to rescue the reading that failed makes it fail worse.
  The compounding saved (10 applications → 1) is smaller than the accuracy lost
  in making one application cover ten timesteps.
* Clause 2's K multiplier is real but it buys a number nobody can use: the
  arm with 1 application per wafer is the arm whose in-coverage terminal error
  is 0.1136, i.e. 2.3× the clause-1 threshold. **The K knob trades clause 1 for
  clause 2 rather than moving both**, and the accuracy-versus-K curve above is
  the measurement of that trade. This is the result the addendum asked for; it
  is a negative one.

**Screen, not verdict** — one seed per arm. The 8-seed grid is running
(`scripts/kcurve.sh`, `runs/kcurve/`), and the sign of every effect here is far
outside the anchor's seed range, so I expect the sign to hold and the sizes to
move. Nothing from this table goes into RESULTS.md until the arms are at 8 seeds.

**The data-matched control refutes the obvious alternative, and in the
surprising direction.** The alternative to "a K-step map is intrinsically
harder" was "a K arm trained on K× fewer pairs". The `ov` arms restore the pair
count (8100 and 5400 against the anchor's 9000) and they are **worse than the
`nv` arms at the same K**, by 0.081 at K=2. So more training pairs at the same
horizon *hurt*, which no data-volume story predicts. The explanation I can
defend is **input-distribution mismatch**: with non-overlapping starts the
training inputs are exactly the states the rollout visits (t = 0, K, 2K, …),
while overlapping starts spend capacity on start times the rollout never sees.
That is a testable claim and the test is cheap — score the `ov` arm's
single-application error on rollout-visited starts against all starts — so it
is queued rather than asserted.

**One alternative is not yet excluded and is being run.** An `nv` arm at stride
K trains on T/K pairs for the same 80 epochs, so it takes K× fewer gradient
steps. `scripts/kcurve_sm.sh` adds the step-matched control: identical pairs and
identical input distribution to `nv`, epochs = 80K, so the gradient-step count
matches the anchor's ~22.5k. If the `sm` arms collapse onto the `nv` arms, the
K-curve is about the map and the horizon story is settled negative; if `sm`
recovers the anchor, my arms were undertrained and the curve above is an
artefact of my epoch budget. I would rather find the second, and the first is
what the `ov` result already points at.

---

## Turn 2 (reopened) — a defect in my own speed table, found by arithmetic

`runs/speed.json` records `params: 16810841`. The architecture every accuracy
number in this repo comes from — width 64, modes 20, layers 4 — has
**26,248,025** parameters, and `param_count()` on width 64 / **modes 16** /
layers 4 is exactly 16,810,841. The committed speed table was therefore measured
on a network **1.56× smaller in parameters** than the one it was quoted beside,
because it was run against an early `runs/base` whose default was `modes=16`
(the same `runs/base` later withdrawn as raced, commit 8cca0d1, which is why the
file has no `--run` field to catch it with).

**Direction: it flattered the clause.** A smaller network is faster, so the
published **1.47×** like-for-like is an upper bound on the real figure for the
reported model, and clause 2's shortfall is larger than 679×, not smaller. The
verdict does not change — the clause was already `UNREACHABLE` by two orders of
magnitude — but the number in the headline was not a measurement of the model in
the headline, which is exactly the rule this workspace exists to enforce.

Fixed: `scripts/bench_speed.py` now records `run`, `arch` and
`applications_per_wafer` in its output, times `n_steps/stride` applications so a
K-step arm is priced correctly, and reports seconds per application beside
seconds per wafer. Re-measuring on `runs/seed1` (the modes=20 anchor) and on
`runs/kcurve/K10_nv_s1`; the old file is kept for the record and every document
that quotes 1.47× is regenerated from the new one.

---

## Turn 3 (reopened) — the K-curve at 5 seeds, and every speedup number in this repo withdrawn

### 1. The addendum's route is measured, and it is a trade, not a joint win

`runs/kcurve.json`, terminal-step rel-L2 (the only reading comparable across K,
since every arm ends at the same physical time while the mean-over-emitted-states
reading averages T/K states):

| arm | seeds | apps/wafer | in-dist | seed range | crossed-in-coverage | seed range |
|---|---|---|---|---|---|---|
| K1_nv (anchor) | 8 | 10 | **0.01890** | 0.00172 | **0.05433** | 0.03511 |
| K2_nv | 5 | 5 | 0.03075 | 0.04662 | 0.06710 | 0.07651 |
| K5_nv | 5 | 2 | 0.04429 | 0.05316 | 0.08608 | 0.06443 |
| K10_nv | 5 | 1 | 0.04847 | 0.01069 | 0.10057 | 0.01850 |
| K2_ov | 5 | 5 | 0.07130 | 0.20476 | 0.10643 | 0.24262 |
| K5_ov | 4 | 2 | 0.04090 | 0.06899 | 0.09658 | 0.10925 |
| K2_sm | 2 | 5 | 0.09503 | 0.15001 | 0.12728 | 0.14877 |
| K5_sm | 2 | 2 | 0.06842 | 0.08888 | 0.12129 | 0.06852 |

**The addendum predicted the wrong sign.** Its argument was that error compounds
over K× fewer applications, so clause 1's terminal reading "gets easier". It does
not: terminal error rises monotonically in K on *both* splits — 0.0189 → 0.0308 →
0.0443 → 0.0485 in distribution, 0.0543 → 0.0671 → 0.0861 → 0.1006 on the
crossed split that clause 1 actually fails. Paired per trajectory against the
anchor, every K>1 arm is worse with sign-test p ≤ 1e-20 and sign-flip p ≈ 1e-5
(100k permutations); at K=2 the anchor wins 243 of 250 in-distribution
trajectories. The reduced-compounding benefit is real but is dominated by the
K-step map being intrinsically harder to fit. **The K knob buys clause 2 with
clause 1, which is what the previous turn's one-seed screen said and this now
says at 5 seeds with an exact paired test.**

The one place it nearly holds: K10_nv in distribution is 0.04847, under the 0.05
threshold on the **point estimate only** — its upper bootstrap CI is 0.05193 and
it fails `met_every_seed`. So K=10 converts a clause-1 pass that was robust under
all three coverage rules (0.01890, every seed, upper CI 0.02044) into one that
survives on the loosest rule alone. Reporting that as "clause 1 still met at
K=10" would be exactly the silent test-loosening the rules forbid.

**The undertraining alternative is refuted, not assumed.** An `nv` arm at stride
K takes K× fewer gradient steps, so "the arms are undertrained" was the
explanation to kill. `sm` matches the anchor's ~22.5k gradient steps at identical
pairs and identical input distribution (epochs = 80K). It is **worse**: K2_sm
0.09503 against K2_nv 0.03075, K5_sm 0.06842 against K5_nv 0.04429. More gradient
steps on the same pairs hurt, so the curve is not an epoch-budget artefact — it
is overfitting a smaller pair set. Two K's, same direction. **Screen, not
verdict**: 2 seeds each, and K2_sm's seed range is 0.15001, larger than its own
point estimate. The 8-seed grid and the `sm` seeds are running.

`ov` (the data-matched control, pair count restored) remains worse than `nv` at
the same K, and K2_ov's seed range of 0.20476 exceeds its point estimate of
0.07130 — this arm is unstable, and nothing about it is a result yet.

### 2. Every like-for-like speedup number this repo has published is withdrawn

This is the turn's real finding and it is a defect in my own measurement, found
by arithmetic on a quantity that should have been constant. The like-for-like
denominator is ViennaPS at one OpenMP thread on one wafer — identical code,
identical grid (0.2 µm, 128), identical 10 timesteps in every measurement — and
it has been reported as:

| file | solver s/wafer | load avg | like-for-like speedup |
|---|---|---|---|
| `runs/speed.json` | 1.538 | 11.56 | 1.47× |
| `runs/speed_seed1_cpu.json` | 8.457 | 40.65 | 6.57× |
| `runs/speed_K10_cpu.json` | 8.390 | 22.32 | 119.12× |

A 5.5× spread on a fixed computation. **My first explanation was box load and it
is wrong.** `runs/solver_drift.json` times the same wafer at 1.11 CPU-s under
load 11.3 and 1.26 CPU-s under load 41.0 — 1.15× across a 3.6× load range. Load
moves this by ~15%, not 5.5×.

**My second explanation was process reuse and it is also wrong, but it found the
real defect.** Recording each wafer separately instead of a median:

```
reuse_single_only       1.257, 0.191, 0.396, 0.297, 0.302, 0.104, 0.256, 0.352
fresh_process_per_wafer 1.253, 0.888, 1.070, 0.988, 1.011, 0.790, 0.945, 1.046
```

The first wafer in a process costs 1.257 s and every later one 0.10–0.40 s.
**ViennaPS charges a large one-time initialisation inside the timed `.apply()`
region.** So reuse makes the solver look *cheaper*, not dearer, and the committed
8.46 s is still unexplained by either hypothesis — see §3.

The consequence is mine. `scripts/bench_paired.py`, which I wrote this turn,
timed the solver in a **fresh process per wafer** (paying init every time) while
giving the operator **3 untimed warm-up rollouts** (paying its FFT-plan creation
on none). Solver cold, operator warm, ratio labelled "like-for-like". Under
`scripts/bench_symmetric.py`, which pays the one-time cost on both sides or
neither:

| arm | marginal_warm | cold_single_wafer |
|---|---|---|
| K=1 (`runs/seed1`) | **0.41×** | **0.73×** |
| K=10 (`runs/kcurve/K10_nv_s1`) | **2.37×** | **7.15×** |

solver cold/warm = 3.15×. **At K=1 the operator is slower than the simulator it
replaces**, like-for-like, on the same hardware at the same verified thread
count. Withdrawn: 1.47×, 6.57×, 119.12×, and this turn's own 1.53×/13.84× from
`bench_paired.py`. The inflation factor on the headline was 16–50×.

Clause 2's shortfall is therefore **140× at the most favourable of the four
honest rows**, not the 679× recorded at declaration — the declaration's figure
was itself computed from a contaminated denominator, and it was too kind in the
sense that mattered less and too harsh in the sense that mattered more: the
*route* is worse than believed because K=1 has no speedup at all.

Two directions this closes off and one it does not. It closes off "the operator
is intrinsically faster and the reporting is just conservative" — on CPU at one
thread, for a 128×128 2-D level set over 10 steps, ViennaPS costs 0.10–0.40
CPU-s warm and a 26.2 M-parameter FNO forward costs ~0.084 CPU-s per
application, so the two are the same order and the operator loses at K=1 on
application count alone. It does not close off the GPU throughput reading, which
is a different claim about different hardware and is labelled as such.

### 3. Recorded as unexplained rather than explained away

Neither load (1.15× across 3.6×) nor process reuse (which lowers the number)
accounts for 8.457 and 8.390 s/wafer, against 1.00 s cold and 0.296 s warm
measured under three conditions at comparable load in one invocation. I have no
account of it. It is not being reinterpreted into a story: the two files are
marked contaminated, every number derived from them is withdrawn, and the
reproduction is the open item. The `stepped_vs_single_overhead` field (2.52×) is
in the right ballpark to matter but does not close a 28× gap, and both files
already used `single` as the reference.

### 4. What the verdict looks like now

Clause 2 is *more* firmly `UNREACHABLE` at 1000×, on a protocol that no longer
flatters it, and clause 1's K-route is a measured negative. What changed is that
both are now wrong for stated reasons rather than for a reason that turned out to
be a measurement artefact. The board entry's "short by 679×" is withdrawn along
with the number it came from.

Infrastructure, same turn: four `runs/kcurve` arms had been killed mid-training
when a parent process exited and took its process group with it — their logs end
mid-write at `Traceback (most recent call last):\n  File "/h`. `train.py`'s
`log.jsonl` guard then refused to restart them, because it could not tell an
orphan from a race. It can: `runlock.acquire` holds an exclusive `flock` that
dies with its holder, so reaching that guard proves no live writer. Added
`runlock.reclaim_orphan`, which archives such a directory to `runs/_orphaned/`
with an `orphaned.json` recording the epoch count and why — archive rather than
append, since appending is what produced the interleaved log the guard exists
for. All long jobs this turn went to their own tmux sessions.

### 5. The measurement's own run-to-run spread, measured before quoting it

Two invocations of `bench_symmetric.py` with identical arguments disagreed by up
to 1.8× (K=10 `marginal_warm` read 3.83× then 2.14×; `runs/seed1`'s operator cost
0.9506 then 1.2585 CPU-s per wafer), while *within* either invocation the spread
over 8 rounds was 1.03–1.17×. So the noise is **per-process**, not per-round —
core assignment, frequency, NUMA placement, cache pressure from the other
projects on this box — and adding rounds cannot see it. It is the CPU analogue of
the H100 clock-ramp artefact that made `uq-surrogate-kit` report 40.8× for a
23.5× quantity, and the seed-count lesson applies to a timing measurement
unchanged.

`scripts/speed_spread.py` repeats the whole invocation 8×. Best arm/reading
(`K10_nv_s1`, `cold_single_wafer`): per-invocation 11.76, 12.16, 9.73, 9.70,
8.86, 8.89, 11.38, 12.04 → **median 10.55×, range 8.86–12.16×, max/min 1.37×**
(`runs/speed_spread.json`). Short of 1000× by **95× at the median and 82× at the
fastest invocation**, so no choice among the eight passes the clause and the
verdict is robust to the harness's own noise. The solver side contributes almost
none of that spread (0.2758 and 0.2767 CPU-s warm across two invocations, 0.3%);
the instability is the operator's, so the ratio inherits it.

`RESULTS.md` now carries the interval rather than one draw, and one thing this
retires is my own §2 table above: the single-invocation numbers there (K=1
0.29×/0.85×, K=10 3.83×/6.95×) are one draw each, and the K=10 `marginal_warm`
figure in particular moved 3.83× → 2.14× on a re-run. The *ordering* across K is
stable — every invocation puts K=10 fastest and K=1 slowest, and K=1 below 1.0 —
because that ordering is set by application count, which noise does not touch.
The individual figures are not, and are labelled accordingly.

### 6. Rung 4, asked the way the addendum says to ask it

`codex`, given "how would you make this clause pass?" rather than "what is wrong
with this", priced the gap instead of finding defects: to reach 1000× the
operator would need **≤276 µs per wafer warm against its current 72 ms**, a
further **261×**. Its four routes and my reading of each:

- **Distil into a compact endpoint model** (narrow CNN, or predict a
  low-dimensional surface basis and decode). Legitimate and untried. This is the
  only route that attacks the clause without changing what the claim covers, and
  a 261× parameter/FLOP reduction from a 26.2 M-parameter FNO is a large ask but
  not an obviously impossible one. **This is the route to take next.**
- **GPU batch throughput.** Legitimate as a deployment claim, and this repo
  already reports 199×/347× as hardware comparisons and refuses them as the KPI.
  It notes correctly that a GPU row must be timed in synchronised wall clock, not
  `process_time()`, so my CPU-seconds estimator does not carry over.
- **A genuinely more expensive simulator workload** (finer grid, 3-D, more
  physics). Honest only with a convergence study establishing that the finer
  setting is *required* for the target accuracy; otherwise it is denominator
  inflation. The clause would then cover a different problem than the one this
  repo trained on, and would have to say so.
- **Full recipe-to-deliverable pipeline timing.** Legitimate if that output is
  actually required, and it would move the denominator by including rasterisation
  — which both sides currently exclude. Not obviously worth 261×.

It also listed what would only manufacture a pass, and every item is something
this repo has either already rejected or has now withdrawn: cold-solver /
warm-operator (worth only ~10× even if taken, which matches §2), resurrecting the
8.4 s denominator, charging the solver for intermediate frames the operator does
not produce, and slowing the solver through contention. That the adversary's list
of tricks matches the list of this repo's own corrected mistakes is the useful
part of the exercise.

**Its three defects in my new script were real and two changed a number.** The
docstring claimed per-round interleaving that `main` does not do (blocks, which
CPU-seconds makes acceptable — the docstring now says so instead of claiming
otherwise). `marginal_warm` priced wafers 1–8 while `cold_single_wafer` priced
0–7, so the cold/warm factor mixed initialisation with geometry across recipes
that vary ~4× in cost; fixed, and re-measured at 2.67× against 2.64×, so the
confound was small but was not zero and was not known to be small before the fix.
Operator inputs are random fields; FNO cost is set by tensor shape rather than
values so the timing stands, but it is now stated in the JSON that no accuracy
claim may be read off that file.

### 7. Where clause 2 stands

`UNREACHABLE` at 1000× is now supported by a protocol that does not flatter it,
an interval rather than a point, and a shortfall of 82× at the most favourable of
eight invocations of the most favourable of four honest rows. What changed this
turn is not the verdict but its basis: the previously recorded "short by 679×"
came from a contaminated denominator and is withdrawn. The clause is not being
declared again yet, because rung 2 has one route left that has not been tried —
codex's distillation route — and a clause that is still moving has no budget.

---

## Turn 3, H6 — written before the run: is 1000× reachable by ANY model in this output representation?

codex's distillation route requires the operator to reach **≤276 µs per wafer
warm** (solver warm 0.2758 CPU-s / 1000). Before training anything smaller, there
is a cheaper question that bounds the whole route, and an arithmetic sketch says
it may already be decided.

The deployed operator emits a 128×128 signed-distance field per application:
16,384 float32 outputs, 65 KB. On one CPU thread, *writing* that array is already
tens of microseconds, and a single 3×3 convolution at 8 channels is
16384·8·8·9·2 ≈ 19 MFLOP, which at a few GFLOP/s per core is **milliseconds** —
an order of magnitude past the 276 µs budget before any useful capacity exists.

**H6: the 1000× clause is unreachable on this hardware for *any* model whose
output is a 128×128 field, independent of architecture, because the cost floor of
producing that output exceeds the budget.** If true, this converts "our operator
is 95× short" into a statement about the problem formulation rather than about
our network, and it identifies the only escape: change the output
representation.

The prediction that distinguishes H6 from "we did not shrink hard enough": the
measured cost of a *deliberately useless* model — an identity map, a 1×1
convolution — will already exceed 276 µs. A distillation story predicts the
opposite, that the floor is well under budget and the gap is capacity we chose to
buy.

**The escape H6 implies, and the reason it is worth measuring rather than
asserting.** The etch front is a curve, not a field: the 128×128 SDF is a
*rasterisation* of ~128 surface heights. A model that emits the surface directly
(N heights, or a low-dimensional basis) has an output 100–1000× smaller and a
correspondingly lower floor. This is codex's "low-dimensional surface basis" and
it is the only one of its four routes that is both legitimate and not yet tried.
Clause 3 already measures shape error from contours, so the metric survives the
change of representation — but clause 1 is a *field* rel-L2, so a surface-output
model would have to be rasterised back to be scored, and that rasterisation cost
belongs on the operator's side of the ratio. Both sides of that trade get
measured, not argued.

`scripts/cost_floor.py` measures, at one application per wafer, CPU-seconds for:
identity, 1×1 conv, single 3×3 conv, tiny FNO (width 8 / modes 4), the deployed
FNO (width 64 / modes 20), and a surface-output MLP (recipe → 128 heights) both
with and without rasterisation to a 128×128 field. **No accuracy is claimed by
any row**: these are cost floors, and a row that is fast and useless is the point
of the exercise, not a result.

### H6 FALSIFIED, and my arithmetic was wrong by an order of magnitude

`runs/cost_floor.json`, CPU-seconds per wafer at one application, budget = the
measured solver cost / 1000 (warm 277 µs, cold 739 µs):

| model | output | warm | cold | warm speedup | cold speedup |
|---|---|---|---|---|---|
| identity | field 128×128 | 1 µs | 9 µs | 244662× | 84081× |
| 1×1 conv | field 128×128 | 10 µs | 288 µs | 26667× | 2568× |
| **3×3 conv, width 8** | field 128×128 | **370 µs** | 2905 µs | **748×** | 254× |
| FNO width 8 / modes 4 / 2 layers | field 128×128 | 1782 µs | 6807 µs | 155× | 109× |
| **FNO width 64 / modes 20 / 4 layers (deployed)** | field 128×128 | **87835 µs** | 80836 µs | **3.2×** | 9.1× |
| MLP → 128 surface heights | surface | 47 µs | 744 µs | **5830×** | 993× |
| the same + rasterise to a field | surface → field | 33 µs | 887 µs | 8375× | 833× |

**H6 is falsified and withdrawn.** I predicted that producing a 128×128 field
would itself exceed the budget; a 3×3 convolution at width 8 produces one in
370 µs, which is **748×** — within 1.34× of the clause — and a 1×1 convolution
manages 26,667×. My sketch put a two-layer 8-channel stack in the milliseconds
by assuming a few GFLOP/s per core; the measured rate is ~13 GFLOP/s
(4.7 MFLOP in 370 µs), because this is an AVX-512 Xeon and MKL vectorises it. An
order of magnitude, in the direction that would have let me stop.

So the finding is the opposite of the one I wrote down, and it is much better:
**clause 2 is not architecture-bound. It is bound by the architecture we chose.**
The deployed operator is **317× over the warm budget** and its 26,248,025
parameters are the entire reason the clause fails. The distillation route codex
named is not merely open, it is nearly closed already on cost: something between
a 1×1 and a 3×3 convolution at width 8 sits on the 1000× line, and a
surface-output model clears it by 5.8× *including* the rasterisation needed to
score it against clause 1's field metric.

The honest statement of clause 2's position has therefore changed from "short by
95× and that is not a gap an implementation closes" to **"short by 95× because we
spent 317× of the budget on capacity, and whether that capacity is necessary for
rel-L2 ≤ 0.05 is not measured."** The earlier framing asserted the answer to a
question nobody had asked. That framing is withdrawn too.

Note also that the cold reading is *unfavourable* to cheap models and favourable
to the deployed one (identity 1 µs warm against 9 µs cold; the deployed FNO 3.2×
warm against 9.1× cold), because a cheap model's cost is dominated by the
one-time initialisation the cold reading charges it. Both readings stay in the
table; neither is dropped.

---

## Turn 3, H7 — written before the run: what does shrinking cost in accuracy?

Cost is now known at every scale; accuracy at those scales is not. The frontier
is the deliverable, and one branch of it needs no new code, since `train.py`
already takes `--width`, `--modes` and `--layers`.

**H7: the capacity the deployed operator spends is mostly unnecessary, so a
shrunken FNO at stride 10 holds terminal-step rel-L2 within the anchor's seed
range while costing 1–2 orders of magnitude less.**

The decision rule, fixed now so it cannot be chosen after seeing the numbers:

* The cheapest FNO measured, width 8 / modes 4 / 2 layers, is **155×** warm — so
  **no FNO on this frontier reaches 1000×**. This sweep therefore cannot pass
  clause 2 by itself, and is not being run in the hope that it will.
* It is run because it is the cheap half of the decision about whether to write
  the code for a convolutional or surface-output model. **If width 8 / modes 4
  already fails rel-L2 ≤ 0.05 badly**, then a model 6× cheaper again will fail
  worse, the frontier is closed below 1000×, and clause 2 is `UNREACHABLE` for a
  reason about the problem rather than about our taste in architectures. **If it
  holds accuracy**, capacity is cheap here, and the convolutional and
  surface-output models — which do reach 1000× on cost — are worth building.
* Screen at 3 seeds per config, verdict at 8 only for whichever config the
  decision turns on. K=1's in-distribution terminal reading is 0.01890 with a
  seed range of 0.00172, and the K=10 anchor is 0.04847 with a range of 0.01069;
  a shrunken arm's gap must exceed those to mean anything.

Confound to keep in view: every arm here is at **stride 10**, so it inherits
K=10's accuracy penalty (terminal 0.04847 in distribution against K=1's 0.01890)
*before* any shrinking. An arm that fails might be failing at the horizon rather
than at the width, and the K=1 shrink arms are the control that separates those.
Both are in the sweep.

### Infrastructure finding: four tests had never executed once

`scripts/weekend.py`'s "still running" section had been hand-written prose for
three turns and was stale every turn — it described clause-3 jobs that had long
landed and claimed a test count of 41, a figure wrong since the count passed 41.
Deriving it instead of typing it produced a disagreement: the derived count said
**72 defined** where the runners printed **68 run**.

The cause is a class of bug rather than an instance. Each test file executes
itself by iterating `globals()` from inside `if __name__ == "__main__"`, so a
test defined *below* that block is never reached — the runner finishes before the
interpreter defines it. Four files had grown that way as tests were appended over
the weekend: `test_inverse_protocol.py` hid 3 tests, `test_solver.py` 1, and
`test_runlock.py` and `test_metrics.py` would have hidden this turn's own
additions had I not moved their runners while adding to them.

**All four previously-unexecuted tests pass**, so nothing was broken. But three
of them guard `design.py`'s inverse-design protocol — including the fix for the
asymmetric baseline, where the random-search arm used to be handed the target's
own etch duration for free while the gradient arm searched it. Clause 3 is a MET
clause resting on that protocol, and the test that pins it had never run. That is
the kind of gap that is only visible from a count nobody was checking, which is
the argument for deriving counts rather than typing them, and it is the third
time this weekend that generating a document from state has caught something
prose was hiding (the others: `report.py` printing "26 claims" while claiming 40
in `claim-auditor`, and `runs/base` re-admitted to a seed group with a stale
eval here).

`test_metrics.py::test_no_test_is_defined_after_its_files_main_block` now fails
if any file regrows the defect. 73 tests, all passing.

---

## Turn 4 — two defects in my own cost-floor table, and H8 on the escape route

`runs/cost_floor.json` falsified H6 and opened the surface-output route by
measuring it at **5830×** warm (8375× "including rasterisation"). Two things
about that table are wrong, and both are mistakes I had just finished correcting
elsewhere in this repo.

**1. Its small rows are point estimates inside my own measured noise band.** Each
model is timed in its own subprocess, and I measured the between-invocation
spread of exactly that at **1.37×** (`runs/speed_spread.json`) — which is why
`RESULTS.md` now carries an interval for clause 2. The surface rows read 47 µs
and 33 µs, a ratio of **1.42×**, sitting squarely inside that band. So the
"+raster" row came out *cheaper than the model it adds work to*, which is not a
measurement of rasterisation; it is a draw from the harness. I corrected this
mistake for the speed clause on the previous turn and then made it again in the
new experiment on the same turn.

**2. The surface-output row is not a model of this task.** It is
`MLP(recipe) → 128 heights`: it never sees the input surface. An operator maps
(current surface, recipe) → next surface, and a recipe-only map solves a
different and easier problem — final profile from recipe, with no state. Its cost
is therefore not the cost of the escape route, and quoting 5830× for it
overstates what the route buys. The rows were all labelled
`is_a_real_surrogate: false`, so nothing false was written down, but the
*conclusion* I drew leaned on that row and the label does not carry the
conclusion.

**3. And the rasterisation it prices is not the quantity clause 1 scores.** The
row computes `ys - h`, a signed *vertical* distance. Clause 1's headline is band
rel-L2 against ViennaPS's **signed distance function**, and vertical distance
equals Euclidean distance only where the surface is horizontal. So that 33 µs
buys a field the metric would compare against a different quantity.

### H8, written before the run: does the surface representation admit clause 1 at all?

This is the question that decides the escape route, and it can be answered
**without training anything**, which is why it comes before any model code.

**H8: the etch fronts in this dataset are single-valued in x, and reconstructing
an SDF from 128 surface heights loses less than clause 1's 0.05 threshold — so
the surface representation is admissible and the route is open.**

The measurement: take the true final SDF of every test trajectory, extract the
zero crossing per column to get 128 heights, then rebuild a field two ways and
score each against the true SDF under the *same* band mask the KPI uses —

* `vertical` — `ys - h`, the cheap broadcast my cost row actually timed;
* `edt` — an exact Euclidean distance transform of the sign mask, the honest
  reconstruction.

This is an **information floor**: it is the best band rel-L2 any surface-output
model could reach *even with perfect heights*, because it measures only what the
representation throws away. The distinguishing predictions:

* If the `edt` floor is **above 0.05**, the surface route cannot satisfy clause 1
  however well it is trained, and the route is closed for a reason about the
  output representation rather than about optimisation. That would be a decisive
  negative obtained for the price of a distance transform.
* If the `edt` floor is far below 0.05 but `vertical` is not, then the route is
  open but its cost must include the distance transform, and my 33 µs row is
  replaced by whatever the EDT costs — which is the number that decides whether
  1000× survives the change of representation.

Also recorded, because it is a hard limit rather than an error: the fraction of
test surfaces that are **not single-valued** in x. A re-entrant or undercut
profile has no representation as 128 heights at all, and any such trajectory
bounds the route independently of the floor above.

The cost side is re-measured under the interval protocol at the same time, so no
row in the new table is a single draw.

---

## Turn 4 (reopened) — H8: the cheap output representation cannot express the data

H6's falsification put a surface-output model at **5830×** warm (8375× including
rasterisation) against the deployed field FNO's 3.2× and a 1000× clause, making a
change of output representation the cheapest route to clause 2. Before building
it, H8 asked whether a height field can represent these fronts at all. **It
cannot, and the route is closed.**

`runs/surface_representable.json`, all 250 test trajectories × 11 emitted
timesteps = 2750 frames, no model involved:

| quantity | value |
|---|---|
| columns carrying a trapped void (solid above *and* below) | **7,338 of 223,636** = 3.28% |
| frames with at least one | **2,291 of 2,750** = 83.3% |
| trajectories with at least one | **249 of 250** = 99.6% |
| worst trapped void a height field would have to fill in | **16.8 µm = 84 grid cells** |
| median thickness of the real ones | 20 cells = 4.0 µm |
| growth with etch time | **t=0: 1.2% → t=10: 99.6%** |

The mechanism is unambiguous. `traj 17, t=10, col 44` reads, top to bottom,
void(4) → **solid(8)** → void(73) → solid(43): a mask slab with 14.6 µm of
cavity beneath it, at symmetric column pairs 43/44 and 83/84 — the two mask
edges. **The etch undercuts the mask**, and the overhang deepens monotonically
with etch time, from 1.2% of frames before any etching to 99.6% at the final
step. That is exactly the signature I pre-registered as refuting H8, at the
location I pre-registered.

**So the 5830× is a speedup at predicting something this dataset does not
contain.** A height output would silently fill in every cavity — a median of 4 µm
and a worst case of 16.8 µm of invented solid, against a clause-3 threshold of 5%
shape error and a grid cell of 0.2 µm. The route is dead on representability,
before any question of what a network could learn, and the negative result is
worth more than the training run it saves: it also retro-justifies the level-set
field this repo already chose, since a level set represents overhangs natively
and that is *why* it costs 128×128.

### Two errors of mine on the way, both caught by running the thing

**First measurement, discarded.** I binned zero-contour points by column and
flagged any column whose points spanned more than a grid cell vertically. It
flagged **100% of frames including t=0**, before any etch physics, with a worst
span of 17.0 µm. Cause: the initial trench's *vertical sidewall* puts a whole
column of contour points at one x, but a vertical wall is a transition *between*
adjacent columns and each of those columns is still a single solid interval —
perfectly height-representable. The metric conflated the wall with the undercut
it existed to detect. Numbers discarded, not reinterpreted.

**Second criterion, corrected.** I then counted sign changes down each column and
called ≥3 re-entrant. That misses the dominant case: a column whose mask reaches
the top of the domain reads solid-void-solid, which is **two** changes, not
three. The contradiction surfaced in the JSON itself — `max_sign_changes: 2`
printed beside an 84-cell trapped void in the same column — and it is only
visible because the file records the worst case's column and its sign count
rather than a summary statistic. The verdict now rests on trapped-void runs,
which do not depend on whether a column starts solid or void; sign changes are
kept and labelled diagnostic. The verdict was `false` under both criteria, so
nothing published changes, but its stated basis was wrong and a reader checking
the criterion would have found it did not mean what it said.

**One exclusion, declared.** Trapped voids of ≤2 cells are a level-set
discretisation artefact at the trench corner, not undercuts: they sit at the
symmetric sidewall pairs (49/50, 77/78) as a 2-row sliver of the wrong sign where
the vertical wall meets the floor, and they are present at t=0 before any etch
runs. Excluding them takes t=0 from 27.5% of frames to 1.2%, which is what makes
the growth curve legible. They are **26.4% of all 10,441 trapped-void runs** and
the count is in the JSON, so the exclusion is auditable rather than asserted. It
is identified by mechanism and by its t=0 presence, not chosen because it
improved a number — and it makes the verdict *harder* to reach, not easier.

### What this leaves for clause 2

The frontier for an output the physics needs — a field — is bounded by
`runs/cost_floor.json`: a 1×1 convolution reaches 26,667× with no capacity, a
two-layer width-8 3×3 stack reaches **748×**, just under the clause. So a
field-output model *can* in principle clear 1000×, but with less capacity than
that stack. Whether anything that small reaches rel-L2 ≤ 0.05 is H7's question,
running now. H8 has removed the one route that looked cheap, and what remains is
a genuine accuracy-versus-cost frontier rather than a representational trick.

---

## Turn 4, part 2 — WITHDRAWING two claims I made last turn. The seed count did it again.

The K-grid reached 7–8 seeds on the `nv`/`ov` arms and 3 on `sm`, and both of
last turn's headline readings do not survive.

| arm | seeds | apps | in-dist terminal | seed range | crossed terminal | seed range |
|---|---|---|---|---|---|---|
| K1_nv (anchor) | 8 | 10 | 0.01890 | 0.00172 | 0.05433 | 0.03511 |
| K2_nv | 8 | 5 | 0.03958 | **0.09859** | 0.07577 | 0.12007 |
| K5_nv | 7 | 2 | 0.03367 | 0.00553 | 0.07743 | 0.02302 |
| K10_nv | 7 | 1 | 0.04866 | 0.01069 | 0.10040 | 0.01850 |
| **K2_sm** | 3 | 5 | **0.01942** | 0.00115 | **0.05170** | 0.00487 |
| K5_sm | 3 | 2 | 0.02482 | 0.00523 | 0.08225 | 0.01127 |
| K10_sm | 3 | 1 | 0.03927 | 0.03501 | 0.11848 | 0.03485 |

**Withdrawal 1: "terminal-step rel-L2 rises monotonically in K on both splits."**
It does not. At 7–8 seeds, K5_nv (0.03367) is *better* than K2_nv (0.03958)
in distribution. The monotonicity was a 5-seed artefact, and K2_nv is the arm
that moved — 0.03075 at 5 seeds to 0.03958 at 8, with a **seed range of 0.09859,
2.5× its own point estimate.** I put that monotonic sequence in a commit message,
a board row and the portfolio log as the headline of the turn. It was four
numbers, three of which are noise-dominated, arranged into a trend.

**Withdrawal 2: "the undertraining alternative is refuted, not assumed."** The
opposite, at K=2. `K2_sm` — identical pairs, identical input distribution,
epochs = 80K so the gradient-step count matches the anchor's ~22.5k — is
**statistically indistinguishable from the K=1 anchor**: in-distribution
mean_diff **+0.00052**, 132 of 250 trajectories better, exact sign test
**p = 0.411**, 100k sign-flip **p = 0.1032**, and the gap is *smaller than the
anchor's own seed range*. On the crossed split it is **better** than the anchor
(mean_diff −0.00263, flip p = 0.00858). So at K=2 the entire horizon penalty was
my epoch budget, which is precisely the alternative `kcurve_sm.sh` was written to
kill and which last turn's 2-seed screen appeared to confirm — on a point
estimate of 0.09503 with a seed range of 0.15001, larger than the estimate
itself. I did label the `sm` arms "screen, not verdict"; I then wrote the
conclusion into three documents anyway. The label is worthless if the claim
travels regardless.

**What survives.** The horizon penalty is real at K=5 and K=10 *after* matching
gradient steps — K5_sm +0.00592 (sign p = 1.4e-25), K10_sm +0.02036
(p = 1.8e-67) — but it is smaller than the `nv` arms implied (K10_sm 0.03927
against K10_nv 0.04866). So the honest statement is: **step-matching removes the
K=2 penalty entirely and roughly a third of the K=10 penalty; what remains at
K≥5 is a horizon effect.** Every `nv` arm remains worse than the anchor at
p ≤ 1e-25, so "a K-step map trained on K× fewer steps is worse" holds; "a K-step
map is intrinsically worse" is only established for K ≥ 5.

### This is now the live route to clause 2, and it points the compute

`K10_sm` sits at **0.03927 in-distribution terminal, under the 0.05 threshold**,
at one application per wafer — the configuration whose speedup is 10.55×. Its
seed range is 0.03501 on 3 seeds, so the margin (0.011) is inside its own noise
and this is a screen. But if it holds at 8 seeds, clause 1 is met
in-distribution at K=10, and clause 2's remaining gap is capacity alone, which is
H7's question. That makes the `sm` arms the highest-value compute in this repo
right now, and seeds 4–8 are queued behind the running `sm` queue rather than
alongside it.

Note what this does to last turn's framing of the addendum's route. I wrote that
the K knob "trades clause 1 for clause 2". At K=2 it does not trade at all once
the training budget is honest. Whether it trades at K=10 is not yet decided, and
I should not have decided it on three seeds twice in a row.

---

## Turn 4, part 3 — CORRECTION to my own reasoning, and a concurrency incident

### I concluded "there is no defect" from a tree that had already been repaired

Earlier this turn I measured the test suite per file and got `test_solver.py`
**6 passed** against 7 defined and `test_inverse_protocol.py` **5 passed**
against 8. That was the real defect: tests defined below their file's
`__main__` runner never execute. I then re-measured, got 7 and 8, saw every test
function sitting above the runner, and concluded my first count had been stale
and **"no tests are silently skipped, there's no defect."**

That conclusion is wrong. Between my two measurements, commit `e1ebc29` moved the
`__main__` block in `test_solver.py` from line 97 to the end of the file, and did
the same in `test_inverse_protocol.py`. **My first measurement was correct and my
second was of a fixed tree.** I attributed a real change in the code to staleness
in my own measurement, which is the same class of error as attributing a real
change in a number to noise — and I did it while the evidence (a diff, a commit
30 seconds old) was one `git log` away.

The finding stands as `e1ebc29` states it: 4 tests had never executed, 3 of them
guarding `design.py`'s inverse-design protocol, including the asymmetric-baseline
fix where the random-search arm was handed the target's own etch duration while
the gradient arm searched it. Clause 3 is a MET clause resting on that protocol
and the test pinning it had never run. All 4 pass.

### The concurrency incident: two loop instances in one repository

`e1ebc29` and `91a19da` were pushed to `origin/main` at 13:23 by a process I am
not. Checking the tree:

```
 693697  ppid 2749455  started 13:19:18   claude -p --continue Continue the aut...
 811827  ppid 3201067  started 13:24:33   claude -p --continue Continue the aut...
```

811827 is me; my parent 3201067 is the pid in `.lock_trkA`. 693697's parent
2749455 is the pid in **`.lock_trkC`** — a different track's wrapper — yet the
commits it produced are in this repository, on this track's project.

The visible cost is small but it is exactly the shape of the D4 incident this
repo already paid for on 2026-09-10, when two α instances killed each other's
inverse-design jobs after 8 minutes of duplicated compute:

- **duplicated work.** We each independently found the same test defect and each
  wrote a meta-test for it: `test_metrics.py::test_no_test_is_defined_after_its
  _files_main_block` (theirs, landed first) and
  `test_runlock.py::test_no_test_file_defines_a_test_the_runner_cannot_reach`
  (mine). I have removed mine; two tests asserting one invariant is worse than
  one, and theirs is the one a pushed commit message references.
- **a false correction.** I nearly committed a retraction of a *true* finding,
  and would have, had I not checked the commit's diff before writing it.
- **the run directories are shared.** `91a19da` swept up `log.jsonl` files from
  five `runs/kcurve` arms that my tmux sessions are writing. `runlock`'s flock
  protects a run directory from two *trainers*, and `train.py`'s guard now
  reclaims orphans — but nothing stops two loops from `git add -A` on the same
  half-written run.

**NEEDS HUMAN, stated as a decision.** The `.lock_trkA` / `.lock_trkC` scheme
records a pid per track but nothing enforces that a `--continue` session resumes
under the wrapper that owns its project. Either (a) the wrapper should refuse to
`--continue` a session whose project is not its track's, or (b) each repository
should carry a lock naming the track that owns it, checked before any commit. (b)
is enforceable inside a repository and is the one I would take; (a) needs the
harness. I have not implemented either, because a repo-level commit lock that a
concurrent instance does not know about would simply lock that instance out
mid-turn, and I should not do that to a process I cannot see the state of.

What I have done instead is stop assuming I am the only writer: this entry, the
duplicate removal, and a note in `WEEKEND.md`.

---

## Turn 5 — a fourth reconstruction bug, and the surface route's cost measured for the first time

### The 0.0726 floor from last turn was inflated by a half-cell misalignment

`rebuild_multi_fine` builds the phase on an `u`x grid, distance-transforms it,
and samples back to the coarse grid. It sampled at fine index `u//2, 3u/2, ...`
— cell *centres* — but every caller and the dataset itself define the coarse
grid point `(j, x)` at `y = j*delta` (`ys = arange(H) * delta`), which is fine
index `j*u`. So every sampled value was offset by half a coarse cell, 0.1 µm.

Caught by a test written to assert something I believed anyway: a **flat**
interface is represented *exactly* by a piecewise-constant-in-x reconstruction,
so it must round-trip to ~0. It scored **0.131**. Now `<0.02`
(`tests/test_repr.py::test_a_flat_front_reconstructs_almost_exactly`).

This is the **fourth** bug in this one reconstruction, after the inverted sign
convention, the hard-coded void top phase, and the ignored non-unit gradient.
Every one of them produced a plausible-looking floor that was really my own
error, which is the argument for the round-trip test rather than for more care.

**The corrected floor is worse, and it now diverges under refinement.**
`runs/repr_floor_aligned.json`, 25 trajectories whose window contains all the
geometry: median **0.0761** (was 0.0726), and the convergence check reads
u=4 → 0.0736, u=8 → **0.0785**, u=16 → **0.0831**, rising rather than settling.
Before the fix it appeared converged at 0.0728/0.0729 — because the coarse EDT's
own discretisation error was partly *cancelling* the horizontal quantisation
error, and removing the misalignment removed the cancellation. So the previous
"converged in the reconstruction grid" claim is withdrawn: the quantity was
converging on the wrong value for the wrong reason, and the true
piecewise-constant-in-x limit is above 0.083 and not yet bracketed.

### The other loop closed the cheap variant; this is about the one it did not

A concurrent instance measured `runs/surface_representable.json` and closed the
**one-height-per-column** representation properly: a column carrying a trapped
void (solid above *and* below) appears in **249 of 250** trajectories and
**83.3%** of frames, the etch undercuts the mask, and the overhang grows
monotonically with etch time (1.2% of frames at t=0 to 99.6% at t=10). That is
correct and it kills `cost_floor.json`'s 5830× row, which was an
`MLP(recipe) → 128 heights` — exactly the representation that cannot express an
undercut. I accept it and am not re-deriving it.

It does **not** touch the representation this script measures. `crossings_from_sdf`
keeps *every* crossing per column, so a trapped void is exactly what its multiple
crossings encode; `n_crossings_per_column` measures up to **6** on real data and
`test_repr.py::test_multi_interface_columns_are_counted_not_collapsed` pins it.
The multi-crossing form is still ~63 numbers per wafer against the field's
16,384, a **260×** reduction, so it remains the live version of the route and the
one worth deciding.

### And its cost, measured here for the first time, is the binding constraint

The floor argument has been the wrong one to spend turns on. `runs/repr_floor_aligned.json`'s
cost block, CPU-seconds per call, 8 separate processes each (the interval
protocol, because this box's between-process spread is 1.37×):

| reconstruction | cost | vs the 277 µs budget | admissible? |
|---|---|---|---|
| `rebuild_vertical` (broadcast) | **36 µs** | 0.13× | **No** — signed *vertical* distance, not the Euclidean quantity clause 1 scores; and single-height, so undercuts kill it anyway |
| `rebuild_edt` (coarse, single height) | **1,626 µs** | **5.9×** | **No** — single-height |
| `rebuild_multi_fine` u=8 (expressive) | **155,130 µs** | **560×** | expressive, and 560× over |

So the only reconstruction that can express the data costs **155 ms, which is
56% of the solver's entire 277 ms warm cost per wafer.** The cheap one is
inadmissible and the admissible one is 560× over budget. Clause 2's cheapest
route is squeezed from both ends: the other loop's finding closes the
representation that is cheap, and this closes the one that is expressive.

**But 155 ms is my implementation, not the route.** It upsamples to 1024×1024,
pads by 120 fine cells a side for the reflective boundary, and runs two Euclidean
distance transforms over ~1.6 M cells. Declaring the route dead on that number
would repeat precisely the error this repo has been correcting all weekend —
attributing an implementation's limit to the task. It is not evidence about the
representation until a reasonable implementation has been tried.

### H9, written before the run

**H9: an analytic distance-to-polyline reconstruction is both exact in x and
cheap, so it simultaneously (a) removes the horizontal quantisation that makes
the floor diverge and drops it below clause 1's 0.05, and (b) costs little
enough that the surface route survives on cost.**

One change answers both open questions, which is why it is preferred to the
interpolation experiment I had queued. Distance from a grid point to a
line-segment set is exact — no grid, no upsampling, no padding — so the
reconstruction stops quantising the interface horizontally *and* stops paying for
1.6 M cells.

Connectivity is the one judgement call and it is stated rather than hidden: the
i-th crossing of column *x* is joined to the i-th of column *x+1* **only when the
two columns carry the same number of crossings**, and the polyline is broken
otherwise. That is a heuristic, it is wrong at exactly the columns where an
overhang begins or ends, and the count of broken joins is recorded in the output
so the reader can see how often it fires.

Distinguishing predictions:

* If the floor drops below 0.05 **and** the cost lands under 277 µs, the surface
  route is open on both counts and the previous three turns' "the route is
  closed / squeezed" readings were all measurements of my rasteriser. I would
  rather find this.
* If the floor drops but the cost does not, the route dies on **cost**, and the
  statement is about converting a compact representation back to a field metric
  rather than about surfaces.
* If the floor stays above 0.05 with an exact-in-x reconstruction, the loss is
  genuine information loss at 0.2 µm column spacing, and the route dies on
  **accuracy** — which would also mean the KPI's field metric is what closes it,
  since the same surfaces satisfy clause 3's contour metric at 0.0061.

---

## Turn 4, second half — WITHDRAWN: "the undertraining alternative is refuted"

The 8-seed grid advanced and the step-matched arms reached 3 seeds. The sign
reversed.

| arm | seeds | apps/wafer | in-dist terminal | range | crossed terminal | range |
|---|---|---|---|---|---|---|
| K1_nv (anchor) | 8 | 10 | 0.01890 | 0.00172 | 0.05433 | 0.03511 |
| K2_nv | 8 | 5 | 0.03958 | 0.09859 | 0.07577 | 0.12007 |
| **K2_sm** | 3 | 5 | **0.01942** | 0.00115 | **0.05170** | 0.00487 |
| K5_nv | 7 | 2 | 0.03367 | 0.00553 | 0.07743 | 0.02302 |
| **K5_sm** | 3 | 2 | **0.02482** | 0.00523 | 0.08225 | 0.01127 |
| K10_nv | 7 | 1 | 0.04866 | 0.01069 | 0.10040 | 0.01850 |
| **K10_sm** | 3 | 1 | **0.03927** | 0.03501 | 0.11848 | 0.03485 |

**What I wrote last turn is wrong.** I reported that the step-matched control was
*worse* than `nv` at K=2 and K=5 (0.09503 against 0.03075, 0.06842 against
0.04429) and concluded "the undertraining alternative is refuted, not assumed".
At three seeds the `sm` arms are **better** than `nv` at every K — 0.01942
against 0.03958, 0.02482 against 0.03367, 0.03927 against 0.04866. The sign
flipped. I labelled that table "screen, not verdict" at two seeds, which was
right, and then drew a verdict-shaped conclusion from it in the commit message
and the board entry, which was not. Both are withdrawn here and corrected there.

This is the seed-count lesson doing exactly what `.overnight/RULES.md` says it
does: two seeds got the *sign* wrong, not merely the size. It is the second time
this weekend a two-or-three-seed screen in this repo has pointed the opposite way
from its own eight-seed run.

**And the correction changes the headline, not just a footnote.** Last turn's
conclusion was "the K knob trades clause 1 for clause 2". At *matched gradient
steps* it barely trades at all: K2_sm reaches 0.01942 in distribution against the
anchor's 0.01890 — a gap of 0.0005, well inside the anchor's own 0.00172 seed
range — while using **5 applications per wafer instead of 10**. So half the
inference cost appears to be free, and the trade I reported was an artefact of
giving the K arms K times fewer gradient steps.

More striking: **K2_sm's crossed-in-coverage terminal reading is 0.05170**,
against the anchor's 0.05433 and a clause threshold of 0.05. That is the closest
anything in this repo has come to the crossed split, it is *better* than K=1, and
its seed range is 0.00487 rather than the anchor's 0.03511. The crossed split is
the reading clause 1 has failed under every coverage rule since it was first
measured, and a K=2 arm trained to the anchor's step count is within 0.0017 of
it.

**Three seeds is a screen and this is not a verdict.** The 8-seed extension is
queued (`e4-sm8`, seeds 4-8 behind the `sm` pane, GPU 1). Two things to watch
when it lands, both of which could kill this:

* K2_nv's seed range is **0.09859 against a point of 0.03958** — the range
  exceeds twice the point estimate. If `sm`'s tight 0.00115 range is a
  three-seed accident rather than a property, the comparison dissolves. The
  ranges here are not yet trustworthy at either arm.
* K2_sm at 8 seeds could land above 0.05 on the crossed split by more than
  0.0017, which is a distance smaller than most of this repo's seed ranges.

**H9, written before that run lands: at matched gradient steps the K-step
horizon costs little or no accuracy, so clause 2's application-count win is
nearly free, and last turn's trade was an artefact of the epoch budget.** The
distinguishing prediction against "sm is a three-seed accident": at 8 seeds the
`sm` arms stay below their `nv` counterparts at every K, and K2_sm's
in-distribution gap to the anchor stays inside the anchor's seed range. If
instead `sm` regresses toward `nv` as seeds accumulate, the horizon penalty is
real and it was the `sm` screen that was noise.

Nothing from this table has gone into `RESULTS.md`; the report reads
`runs/kcurve.json` for the arms that are at eight seeds and labels the rest.

### H9 answered: the cost half decisively, the accuracy half not at all

`runs/repr_floor_aligned.json`, cost block, CPU-seconds per call as the median of
8 separate processes. The last column is the one that matters: it is
`solver_warm / reconstruction_cost`, i.e. **the largest speedup the surface route
could reach even if the model itself were free**.

| reconstruction | cost | vs 277 µs budget | speedup ceiling | admissible? |
|---|---|---|---|---|
| `rebuild_vertical` | **65.5 µs** | 0.2× | 4224× | **No**, twice over: signed *vertical* distance is not the Euclidean quantity clause 1 scores, and one height per column cannot express the undercut the other loop measured in 249/250 trajectories |
| `rebuild_edt` (coarse, single height) | **1,170 µs** | 4.2× | **236×** | **No** — single height |
| `rebuild_polyline` (exact in x, expressive) | **34,567 µs** | 125× | **8.01×** | yes |
| `rebuild_multi_fine` u=8 (grid, expressive) | **133,364 µs** | 482× | **2.07×** | yes |

**H9's cost prediction is falsified and the route dies on cost.** I expected an
analytic reconstruction to be cheap; it is 4× cheaper than the grid one and still
**125× over budget**. Two implementations that share no code — an upsampled
distance transform over 1.6 M cells, and exact point-to-segment distance with no
grid at all — land at ceilings of 2.07× and 8.01×. That the two agree within 4×
while differing completely in method is what makes this a statement about the
problem rather than about my code, which the single 155 ms figure was not
entitled to be.

And the ceiling holds even for the inadmissible cheap-ish row: `rebuild_edt` caps
at **236×**, still short of 1000×. **The only reconstruction that clears 1000× on
cost is `rebuild_vertical`, and it is the one that computes the wrong quantity
for the metric *and* cannot represent an undercut.**

So: **converting a compact interface representation back into a field that
clause 1 can score costs more than the entire 1000× budget, by 125–482×.** The
route is closed on cost, independently of the accuracy floor — which is
fortunate, because the accuracy half is unresolved.

**H9's accuracy prediction is also unsupported, and by my own implementation
rather than by the representation.** The analytic reconstruction is exact in x —
it round-trips a flat front to machine zero, where the grid version managed
0.131 before the alignment fix — and it is nonetheless **worse** on real data:
median **0.3197** against the grid version's 0.0761, with individual
trajectories at 2.35 and 2.70. The cause is the connectivity heuristic I
pre-registered as the weak point: joins break where crossing counts differ, 4–6
times per trajectory, and each break leaves a gap with no nearby segment. So the
floor is still bracketed only from above and I am not reporting a value for it.

What this leaves genuinely open, and it is now a question about clause 1's metric
rather than about clause 2: the same surfaces that reconstruct to 0.08–0.32 as a
*field* satisfy clause 3's contour-based shape error at **0.0061**. The field
metric is what makes the compact representation expensive, since it forces a
rasterisation the contour metric never asks for. I am not touching the metric —
clause 1 says rel-L2 and this repo scores it as band rel-L2 on the field, and
rewriting that after seeing which representation it excludes is precisely the one
thing the rules forbid. But it is the honest description of where the constraint
lives, and it belongs in the paper's discussion rather than in a clause verdict.

**Net effect on clause 2.** Rung 2 of the ladder is now exhausted for the output
representation: the cheap representation cannot express the data (other loop),
and the expressive one cannot be scored within budget (here). The surviving
routes from `cost_floor.json` are the ones that keep the field output and shrink
the *network* — a 3×3 convolution at width 8 emits a full field in 370 µs, which
is **748×**, and H7's shrink sweep is queued to say what that costs in accuracy.
That is where the next turn goes.

### Infrastructure: a job of this repo's was seen outside the GPU lease

`scripts/train.py` from this repository was running with
`CUDA_VISIBLE_DEVICES=2,3` at 13:55 (pid 1456532). The α lease is devices 0 and
1; 2 and 3 belong to other tracks. None of this track's tmux sessions can produce
that — `kcurve.sh` and `shrink.sh` launch on GPU 0, `kcurve_sm.sh` on GPU 1, and
all four launchers set the variable inline — so it came from the other loop
sharing this working tree (D4, above).

**What is and is not known.** The process had exited by the time I looked, and
every compute application on the box was on GPU 0 or GPU 1 throughout
(`nvidia-smi --query-compute-apps`, cross-referenced against the device UUIDs).
So no allocation was ever observed on 2 or 3 and nothing is known to have been
disturbed on another track. I am not claiming it consumed another track's GPU; I
am recording that it could have.

I did not kill it and would not have: it is not mine to kill, and it is in this
repository, so the fix belongs in the code rather than in a launcher — whichever
loop starts a trainer, the check fires. `runlock.assert_gpu_lease` refuses any
`CUDA_VISIBLE_DEVICES` containing a device outside `EOT_GPU_LEASE` (default
`0,1`, from the brief), and refuses an *unset* variable too, since unset means
every GPU on the box is reachable and torch's choice is then a gamble. It checks
`CUDA_VISIBLE_DEVICES` rather than `--device`, because `--device` is an index
*into* that set and is `cuda:0` in every script here regardless of which physical
GPU that is. Called before `acquire`, so a lease violation cannot even take a run
directory. Four tests, and all four existing launchers verified to pass it.

### Cross-check with the concurrent instance: its estimate, my measurement

`37b9d44` reached the same structural conclusion independently and by a
different route, which is worth more than either of us reaching it alone. Where
we overlap we agree inside this box's known 1.37× between-process spread:

| reconstruction | theirs (`runs/repr_floor.json`) | mine (`runs/repr_floor_aligned.json`) |
|---|---|---|
| coarse EDT | 1,756 µs | 1,170 µs |
| sub-cell EDT, u=8 | 116,371 µs | 133,364 µs |

**Where we differ, its figure was an estimate and mine is a measurement.** That
commit says a point-to-segment reconstruction "would land around 1–3 ms, the
same order as the coarse EDT, so the structural conclusion does not turn on the
constant." Measured, `rebuild_polyline` costs **34,567 µs [25,643–38,977]** — 10
to 30× above the estimate. The structural conclusion does indeed survive (both
are over budget, which is what that sentence was defending), but the constant it
guessed is wrong, and the guess was load-bearing for the sentence "a
reconstruction two orders of magnitude cheaper than a distance transform" would
be needed to overturn it: from 34.6 ms the requirement is **125×**, not the ~4×
that 1–3 ms against 277 µs would have implied.

Checked and clean: the 1–3 ms figure never left that commit message for any
document, so nothing needed correcting under the no-unmeasured-numbers rule.
This entry records it because the estimate is the kind of thing that gets quoted
later, and because a peer instance's guess deserves the same treatment as
`codex`'s.

---

## Turn 5 — every K-curve number was scored on partial checkpoints, and correcting it reverses my own conclusion

### The defect

`scripts/kcurve_report.py` evaluates each arm's `best.pt` itself rather than
reading a committed `test_eval.json`. Its `arms()` collector required only
`args.json` and `best.pt` — **not completion**. So an arm that was still
training, or that had been killed mid-training, had a checkpoint on disk and was
scored at whatever epoch it had reached, then joined the seed group as if it had
finished.

Eight arms were being scored that way at the moment the previous turn's table
was generated (`excluded_incomplete_arms` in `runs/kcurve.json` now lists them
with the guard's own reason):

```
K10_sm_s1   384/800    K2_nv_s5   26/80    K2_ov_s4  26/80    K2_ov_s8  38/80
K2_sm_s4     90/160    K5_ov_s4   26/80    K5_ov_s8  44/80    K5_sm_s4 214/400
```

The two clearest: `K2_nv_s5`, killed by the process-group kill at epoch 26, was
scored **0.06705**; `K2_nv_s8`, four minutes into an 80-epoch run, was scored
**0.11902**. The six completed K2_nv seeds all sit between 0.02043 and 0.02291.
Those two partial checkpoints moved that arm's mean from **0.02169 to 0.03958**
and its seed range from **0.00248 to 0.09859** — and I was one step from writing
the resulting spread up as a *bimodal training-instability finding*. It was two
undertrained checkpoints.

The bias does not cancel. A partial checkpoint always scores worse, and which
arms get caught depends on queue order, so an arm with more seeds queued later is
penalised more than one with fewer. That is precisely the shape that manufactures
a monotone trend.

The guard is now `scripts/seed_spread.completed`, this repo's canonical answer,
rather than a fourth notion of completeness invented here — it already encodes
that `test_eval.json` must not predate `best.pt` (the `runs/base` staleness worth
33× on a seed range) and that a full-length log is not proof because `train.py`
appends. **My first attempt at the guard required `done.json` alone and was too
strict in the other direction**: it silently cut the K=1 anchor from 8 seeds to 2,
because `runs/seed1..8` all logged 80/80 epochs with an eval but only `seed4` and
`seed8` carry a `done.json` — `runlock.mark_done` was added after the other six
ran. A guard that drops the anchor changes the seed group as surely as one that
admits a partial checkpoint. Both directions are now pinned by tests, and the
restored anchor reproduces its historical 0.01890 / range 0.00172 exactly, which
is the check that the guard is right.

### The corrected table

Terminal-step rel-L2, completed arms only:

| arm | seeds | apps/wafer | grad steps | in-dist | seed range | crossed-in-coverage |
|---|---|---|---|---|---|---|
| K1_nv (anchor) | 8 | 10 | 22560 | **0.01890** | 0.00172 | 0.05433 |
| K2_nv | 7 | 5 | 11280 | 0.02169 | 0.00248 | 0.05640 |
| K2_ov | 6 | 5 | 20320 | 0.02006 | 0.00179 | 0.05174 |
| K2_sm | 3 | 5 | 22560 | **0.01942** | 0.00115 | 0.05170 |
| K5_nv | 8 | 2 | 4560 | 0.03336 | 0.00553 | 0.07680 |
| K5_ov | 6 | 2 | 13520 | 0.02338 | 0.00165 | 0.07119 |
| K5_sm | 3 | 2 | 22800 | 0.02330 | 0.00136 | 0.08226 |
| K10_nv | 8 | 1 | 2320 | 0.04781 | 0.01399 | 0.09898 |
| K10_sm | 2 | 1 | 23200 | **0.02610** | 0.00059 | 0.10440 |

### WITHDRAWN: "the step-matched control refutes undertraining"

The previous turn wrote: *"The undertraining alternative is refuted, not assumed
… `sm` is **worse**: K2_sm 0.09503 against K2_nv 0.03075"*, and concluded that
more gradient steps on the same pairs hurt. **That is backwards and it is
withdrawn.** Both `sm` figures came from arms scored mid-training (`K2_sm_s4` at
90 of 160 epochs, and a 2-seed group of which one was partial). On completed
arms the step-matched control is *better* than `nv` at every K, and gradient
steps are the controlling variable rather than the horizon:

* **K=2 at matched steps is statistically indistinguishable from K=1
  in-distribution** — mean paired difference **+0.00052** over 250 shared
  trajectories, anchor better on 118/250, **exact sign test p = 0.411**,
  sign-flip p = 0.1032, and the gap is *smaller than the anchor's own seed
  range*. It uses **half** the applications per wafer.
* On the crossed split K=2 at matched steps is **better** than the anchor:
  mean difference **−0.00263**, sign-flip p = **0.00858**.
* **K=10 at matched steps meets clause 1 in-distribution at 0.02610** (2 seeds,
  range 0.00059, upper CI under 0.05) with **one** application per wafer, against
  0.04781 for the same horizon at 2320 steps. Step-matching removes about
  three-quarters of K=10's penalty: the paired gap to the anchor falls from
  **+0.02890** (`K10_nv`, anchor better on 247/250) to **+0.00719** (`K10_sm`,
  anchor better on 211/250).

So the previous turn's headline — *"the K knob trades clause 1 for clause 2"* —
is **half wrong and now split**: in-distribution the horizon is nearly free once
the gradient-step budget is matched, and at K=2 it is free outright. **On the
crossed split it is not free and gets monotonically worse with K** (0.05433 →
0.05170 → 0.08226 → 0.10440 at matched steps), which is where clause 1 already
failed and still fails. The trade is real in exactly one of the two readings, and
saying so required separating the horizon from the optimisation budget — which
the contaminated table could not do.

Also corrected, same cause: the in-distribution `nv` figures were 0.0308 /
0.0443 / 0.0485 for K = 2 / 5 / 10 and are **0.02169 / 0.03336 / 0.04781**. The
monotone rise in the `nv` arms survives; its size does not, and its explanation
has changed from the horizon to the gradient-step budget that `nv` varies along
with it.

### What this does to the clauses

Clause 1 in-distribution is met by every arm on the point estimate, and `K10_sm`
meets it on the upper-CI rule too — so the 10× reduction in applications per
wafer that clause 2's best row already uses does **not** cost clause 1
in-distribution, which is what the previous turn claimed it did. Clause 2 remains
short by 95× at the median of eight invocations (`runs/speed_spread.json`,
unaffected by this defect — it times checkpoints rather than scoring them), so
the horizon route still needs the compact-architecture route to reach 1000×.

`K10_sm` at 2 completed seeds is a **screen, not a verdict**. The 8-seed
extension is running.

---

## Turn 6 — H10: make the reconstruction as fast as the peer estimated, and see if the clause cares

The peer instance and I disagreed about the cost of rebuilding a field from a
compact interface representation: its `37b9d44` estimated "around 1–3 ms", my
`5bd6a5b` measured **34,567 µs [25,643–38,977]**. It has already recorded the
resolution correctly — the figure was an estimate, mine was a measurement, and
the structural conclusion survives because both are over budget. I accept that
and am not re-litigating it.

What the exchange leaves open is different and is the reason for this turn.
**Neither of us tried to make the reconstruction fast.** My `rebuild_polyline`
computes exhaustive point-to-segment distance — every one of 16,384 grid points
against every one of ~700 segments, chunked but complete. That is the honest cost
*of that algorithm*, and I was careful to say the 155 ms grid version was "my
implementation, not the route". The same caution applies to 34.6 ms, and I did
not apply it. An objection of the form "you never optimised it" is currently
open, and it should not be, because the clause does not depend on the answer.

**H10, in two parts, written before the run.**

1. A nearest-neighbour reconstruction — sample the interface segments densely,
   build a k-d tree, query every grid point — reaches the peer's 1–3 ms, because
   it replaces an O(points × segments) scan with O(points log samples).
2. **And the clause fails anyway.** The ceiling on the whole route is
   `solver_warm / reconstruction_cost`, the best it could do with a free model.
   The solver's warm cost is **0.2767 CPU-s** (`runs/speed_symmetric.json`), so
   even a **1 ms** reconstruction gives a ceiling of **277×** and a 3 ms one
   gives **92×**, both short of 1000×. The reconstruction would have to come in
   under **277 µs** — the entire budget — leaving nothing for the model.

Part 2 is arithmetic on an already-measured denominator, so I can state it now;
part 1 is the measurement. If part 1 holds, the negative result stops depending
on my implementation quality at all, which is worth more than the 30× it costs
me to concede on the constant.

**The approximation has to be bounded, not just fast.** Sampling segments at
spacing `s` and taking the distance to the nearest *sample* rather than to the
segment overestimates by at most `s/2`. At `s = delta/4 = 0.05 µm` that is
0.025 µm against a band of 1.5 µm, so it cannot manufacture accuracy. The sign
still comes from per-column phase parity, which is exact. And the fast version is
checked **against the exhaustive one on the same trajectories** rather than
against the truth, so the comparison isolates the approximation from every other
error in the pipeline — of which this reconstruction has already had five.

---

## Turn 6 — the paper draft predated the entire reopening

Jobs were running (43 of 51 kcurve arms done, seed 8 in flight, H7's shrink queue
correctly waiting on the kcurve driver rather than starving), so this turn went to
the draft, as the brief directs. Auditing it found one absence and three defects,
all of them mine.

**The absence.** `paper_draft.md` had sections 4.1–4.4, all of which predate the
2026-09-10 reopening. Every finding since — the cost floor, H6's falsification,
the representability result, H9's reconstruction ceiling — existed only in this
log. The long-form record was missing the weekend's load-bearing negative result.

Added `### 4.5 Why the speedup clause is not an implementation problem`,
generated by `scripts/paper.py` from `runs/cost_floor.json`,
`runs/repr_floor_aligned.json`, `runs/surface_representable.json` and
`runs/speed_spread.json`. Nothing in it is typed; the two tables and every figure
are read from those files, and the section reads `[not measured]` if they are
absent.

**Three defects in what I had written, each caught by reading my own section
against my own log.**

1. *The cost table presented rows I had already shown to be noise.*
   `mlp_surface_128` and `mlp_surface_128_plus_raster` appear at 5,830× and
   8,374× — the row that *adds* a rasterisation is the cheaper of the two, which
   is impossible and is exactly the between-process noise I measured at 1.37×
   (the two differ by 1.44×, i.e. by nothing). Presenting them unqualified in a
   paper would have repeated, in the document, the error I spent a turn removing
   from the clause. The table now carries the spread and the fact that its
   rasterisation is a signed *vertical* distance, which is not the quantity
   clause 1 compares against.

2. *The abstract contradicted the new section.* It said the speedup clause is
   decided by "the *denominator*, not the model". Both halves were once true —
   the denominator decided every figure we first reported — but §4.5 shows the
   *model* decides reachability, since a small convolution emits the same field
   inside the budget. Rewritten to carry both, in that order.

3. *The abstract asserted a magnitude the runs do not support.* "Missed by
   roughly an order of magnitude on a crossed split" conflates two readings that
   differ by more than 10×: `runs/coverage_verdict.json` puts the crossed
   terminal-step error inside the trained displacement range at an upper CI of
   0.0767–0.0947 — 1.5–1.9× the 0.05 threshold — and outside it at 1.118–1.351,
   which is 22–27×. The blanket phrase was wrong for the in-coverage reading and
   understated for the out-of-coverage one. It now says which is which and
   points at §4.1 for the figures, with no multiplier in the abstract.

Also rewrote the abstract's closing claim. It said "three of the four numbers
this project nearly reported were wrong" — a count I cannot substantiate now that
the number of withdrawals is well past four, and a tidier claim than the truth.
It now describes *where* the errors were (denominators, warm-up asymmetries, seed
counts under the noise floor, and five consecutive bugs in one reconstruction)
and how each was caught, which is checkable against this log.

**One process note.** My previous turn's fix to `tests/test_kcurve.py` — the
fixture that asserted against an empty dict after `arms()` began filtering by
`is_complete` — did land, but inside the concurrent instance's commit `8bc0b35`,
which swept my working tree with `git add -A`. The work is preserved and the
attribution is not; that is the same shared-tree hazard as `91a19da` in the other
direction, and it is a further argument for the repo-level ownership lock still
recorded as needing a human.

### H10 answered: part 1 falsified, part 2 confirmed, and the result is now implementation-independent

`runs/repr_floor_kd.json`, CPU-seconds per call, median of 8 separate processes.
The ceiling is `solver_warm / reconstruction_cost` — the best the whole route
could do **with a free model**:

| reconstruction | cost | × the 277 µs budget | ceiling | Euclidean? | multi-interface? |
|---|---|---|---|---|---|
| `rebuild_vertical` | **43 µs** | 0.2× | 6411× | **no** | **no** |
| `rebuild_edt` | **988 µs** | 3.6× | **280×** | yes | **no** |
| `rebuild_kdtree` | **13,561 µs** | 49.0× | **20.4×** | yes | yes |
| `rebuild_polyline` | 27,068 µs | 97.8× | 10.2× | yes | yes |
| `rebuild_multi_fine` u=8 | 105,385 µs | 380.8× | 2.6× | yes | yes |

**Part 1 is falsified. The k-d tree is faster and nowhere near 1–3 ms.** It
halves the exhaustive scan (13.6 ms against 27.1 ms) and matches it to four
decimal places in band rel-L2, but the peer's estimate sits **4.5–13.6× below
what the algorithm it described actually delivers**. So the estimate was
optimistic not only relative to my implementation but relative to the method
itself. I am recording this as a falsified prediction of *mine* as much as of the
peer's: I wrote H10 expecting the tree to reach 1–3 ms.

The approximation is verified rather than argued: `max |kdtree − exhaustive|` in
the band is **0.0096–0.0200 µm** against the `sample_um/2 = 0.0250 µm` bound, and
`test_repr.py::test_the_kdtree_reconstruction_stays_inside_its_error_bound` pins
both the bound and its one-sidedness (a nearest-sample distance may only ever
overestimate).

**Part 2 is confirmed, and this is what makes the negative result stop depending
on anyone's implementation.** Five reconstruction algorithms now span **2,400×**
in cost, from a 43 µs broadcast to a 105 ms upsampled distance transform, and
**every one that computes a Euclidean signed distance caps the route below
1000×**. The best fully admissible one — Euclidean *and* able to express the
undercut the peer measured in 249/250 trajectories — caps at **20.4×**.

The arithmetic behind it is not about algorithms at all. To clear 1000× the
*entire* per-wafer cost must fall under 277 µs, so a reconstruction costing
anything comparable to that leaves nothing for the model. `rebuild_edt` is the
cheapest Euclidean reconstruction measured, at 988 µs, and it is **already 3.6×
over the whole budget while still being unable to represent an undercut.** The
only row under budget computes signed *vertical* distance, which is not the
quantity clause 1 compares against, and is single-height besides.

So the "you never optimised it" objection is now closed by measurement rather
than by assertion, which is what it cost a turn to buy. **Clause 2's
output-representation route is dead on cost, and no longer for a reason anyone
can attribute to how I wrote the code.**

What remains genuinely open is unchanged and is not about clause 2: the same
surfaces reconstruct to 0.08–0.32 as a *field* while satisfying clause 3's
contour metric at 0.0061, so it is clause 1's choice of a field metric that makes
compact representations expensive. I am not touching that metric — rewriting it
after seeing which representation it excludes is the one thing the rules forbid —
but it is the honest location of the constraint and belongs in the discussion.

### H11 — the corrected K-curve invalidates my own queued experiment, before it ran

*(Renumbered from H10 after the fact: the concurrent instance had already used
H10, both in this log above and in `scripts/train.py`'s `--strides` flag, for a
different hypothesis about reconstruction speed. Two hypotheses under one label
in a shared document is how a citation goes wrong later, and their claim was
written first.)*

Commit `ec16fcd` excluded eight partial checkpoints from `runs/kcurve.json`, and
the clean table says something none of the three readings this repo has published
said. In-distribution terminal rel-L2, seeds and seed range now that only
finished arms are scored:

| arm | seeds | apps/wafer | in-dist terminal | range | crossed |
|---|---|---|---|---|---|
| K1_nv (anchor) | 8 | 10 | 0.01890 | 0.00172 | 0.05433 |
| K2_nv | 7 | 5 | 0.02169 | 0.00248 | 0.05640 |
| K5_nv | 8 | 2 | 0.03336 | 0.00553 | 0.07680 |
| K10_nv | 8 | 1 | 0.04781 | 0.01399 | 0.09898 |
| K2_sm | 3 | 5 | 0.01942 | 0.00115 | 0.05170 |
| K5_sm | 3 | 2 | **0.02330** | 0.00136 | 0.08226 |
| K10_sm | 2 | 1 | **0.02610** | 0.00059 | 0.10440 |

Two things follow, and the second is the one that matters this turn.

**Undertraining was the explanation after all.** The step-matched arms — same
pairs, same input distribution, `epochs = 80*K` so the gradient-step count
matches the anchor's ~22.5k — beat the fixed-epoch arms at the same horizon by
**1.43× at K=5** (0.02330 vs 0.03336) and **1.83× at K=10** (0.02610 vs
0.04781), with seed ranges of 0.00136 and 0.00059 against gaps of 0.010 and
0.022. That is the opposite of what I wrote two turns ago ("the undertraining
alternative is refuted"), which I drew from two seeds of contaminated data and
labelled a screen; and it is also not the concurrent instance's withdrawal, which
was drawn from the same contamination. The clean answer is that most of the
K-curve at the deployed architecture was a training budget, not a horizon
penalty. K10_sm is at 2 seeds and is a screen; the 8-seed extension is running on
GPU 1 (`e4-sm8`, seeds 4–8), so I have not touched it.

**And `scripts/shrink.sh` — H7, mine, queued for three turns and not yet
started — was about to make exactly the mistake the correction just exposed.** It
trained every arm at `--epochs 80`, including its five stride-10 configurations.
That is the `nv` condition, the one now measured at 0.04781 where the
step-matched condition reaches 0.02610. So H7 would have measured shrunken
networks under a training budget known to be short by 1.83× at that horizon —
larger than the accuracy differences the sweep exists to resolve — and its
pre-registered decision rule ("if width 8 / modes 4 already fails rel-L2 ≤ 0.05
badly, the frontier is closed below 1000×") would have fired on a training
artefact and closed the last surviving route to clause 2 for the wrong reason.

**H11: the step-matched correction transfers to the shrunken architectures, so a
compact network at stride 10 trained to matched gradient steps holds clause 1
in-distribution where the same network at 80 epochs does not.** Fixed
`shrink.sh` to `epochs = 80*stride` and recorded why in the script itself. The
stride-1 control arms are unaffected (80×1 = 80) so they stay comparable to
`runs/seed1..8` element-for-element. `queue_after.sh` re-reads the script when
the kcurve driver exits and `runs/shrink` is still empty, so the correction lands
before the first arm rather than after the sweep.

The distinguishing prediction, so this is falsifiable rather than a hedge: if H11
holds, the step-matched shrink arms will beat their 80-epoch equivalents by
roughly the 1.83× seen at the deployed width, and the frontier decision will be
made on capacity. If the shrunken arms show *no* step-matching benefit, then the
benefit at width 64 was about that architecture's capacity to exploit extra steps
rather than about the horizon, which would be a finding about the anchor and not
about H7 — and I would then have to re-read the K-curve a fourth time.

**What this costs.** Fifteen stride-10 arms at 800 epochs instead of 80. The
networks are 8–32 wide against the anchor's 64, so per-epoch cost is far lower,
but the sweep is now the long pole on GPU 0 rather than a quick screen. That is
the right trade: a fast answer to the wrong question is what the last three turns
kept producing.

### H10, written before the run: is the crossed-split penalty at high K about input-state diversity?

The corrected table splits cleanly by reading. In-distribution, the horizon is
nearly free once the gradient-step budget is matched (0.01890 → 0.01942 →
0.02330 → 0.02610 for K = 1/2/5/10 at ~22.5k steps). On the crossed split it is
not, and worsens monotonically (0.05433 → 0.05170 → 0.08226 → 0.10440). So
something the crossed split needs is *not* supplied by more gradient steps.

**The coverage explanation fails before it is run, so it is not the hypothesis.**
A K-step arm advances K× the per-step displacement per application, but its
trained range in that quantity scales by K too, so the selection
`lo ≤ disp ≤ hi` is scale-invariant and the crossed subset stays the same
comparable set for every arm. Recording that here because it was my first guess
and it is wrong for a reason worth keeping.

What the arms already say points elsewhere. Distinct start offsets per
trajectory are `T−K+1` for `ov` and `T/K` for `nv`/`sm`:

| K | arm | starts | pairs | grad steps | crossed |
|---|---|---|---|---|---|
| 5 | ov | 6 | 5400 | 13520 | **0.07119** |
| 5 | sm | 2 | 1800 | 22800 | 0.08226 |
| 2 | ov | 9 | 8100 | 20320 | 0.05174 |
| 2 | sm | 5 | 4500 | 22560 | 0.05170 |

At K=5, `ov` beats `sm` on the crossed split by 0.011 **with 40% fewer gradient
steps**; at K=2 they tie, where `sm` already has 5 starts. That is the signature
of input-state diversity rather than optimisation budget, and it is a screen (6
and 3 seeds), not a verdict.

**And K=10 cannot be given start diversity at all**: on a T=10 trajectory,
`T−K+1 = T/K = 1`, so `ov` and `nv` coincide by construction and every K=10 arm
in this repo has seen exactly one input state per trajectory — the initial
trench. A model that only ever sees initial trenches as inputs has memorised
"trench + recipe → final profile" over the training recipes, which is precisely
the thing that would generalise worst to unseen recipe/dt combinations.

**H10: the crossed-split penalty at K=10 is caused by input-state narrowness, not
by horizon length. Training ONE operator jointly on strides {1, 2, 5, 10} — the
conditioning already carries log(K·dt), so a single network can serve every
horizon — will improve the K=10 crossed reading over the stride-10-only arm at
matched gradient steps.**

Distinguishing prediction. If long horizons are intrinsically harder, the mixed
arm will not improve K=10's crossed reading, and may worsen it by spending
capacity on strides nobody asked for. If input narrowness is the cause, the mixed
arm improves crossed at K=10 while keeping in-distribution comparable. The mixed
dataset carries **16,200 pairs and all ten start offsets** against the
stride-10 arm's 900 pairs and one, at the same architecture and a matched step
budget (45 epochs × 506 steps ≈ 22.8k, against `K10_sm`'s 23,200).

Two things this is not allowed to be read as. The mixed arm is evaluated at
stride 10 — `args.json` records `eval_stride` and `eval.py` now prefers it, so
the arm is scored at its deployment horizon and not at the `--stride` default it
never used. And it is *not* a K-curve arm: it lives in `runs/mixed/` so
`kcurve_report`'s glob cannot pick it up and misgroup it by `cfg["stride"]`,
which would silently pollute the very table this turn just cleaned.
