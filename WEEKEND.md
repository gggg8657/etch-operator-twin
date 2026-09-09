# WEEKEND — E4, Etch Neural Operator Twin

*Living document, rewritten each loop turn. Last updated 2026-09-09 03:27 UTC.*
*Authority for numbers is `RESULTS.md` (machine-generated). Long form is
`critique_log.md` and `paper_draft.md`.*

> **KPI:** 2D 표면진화 rel-L2 ≤0.05 · 시뮬 대비 ≥1000× 가속 · 역설계 형상오차 ≤5%

## Where this stands right now

**Status: RUNNING. No KPI clause has a number yet, and that is the accurate
state — not a delay.** The repository was created from nothing this session. All
three cells in `RESULTS.md` read `[not measured]`, which is what they must read
until a run produces them.

| clause | Friday | now |
|---|---|---|
| rel-L2 ≤ 0.05 | repo did not exist | `[not measured]` — dataset still generating |
| ≥1000× speedup | repo did not exist | `[not measured]` — benchmark written, needs a trained model |
| shape error ≤ 5% | repo did not exist | `[not measured]` — needs a trained model |

What *is* measured is the ground truth the operator will be scored against, and
it is exact where a closed form exists:

| check | result | tolerance |
|---|---|---|
| SDF rasteriser vs analytic plane | 2.0 × 10⁻¹³ µm | < 10⁻³ µm |
| isotropic recession vs rate × time | 1.2 × 10⁻⁸ relative | < 2 × 10⁻² |
| grid convergence Δ0.2 → Δ0.1 | 0.018 µm mean surface shift | — (this is the accuracy floor) |

## What has been tried that did not work, and what it rules out

1. **90-way parallel dataset generation.** Throughput *peaks at 8 workers
   (0.703 traj/s) and falls to 0.626 at 90* — eleven times the machine for 11%
   less work. Rules out the assumption that a 192-core box makes ViennaPS
   embarrassingly parallel; its flux solver is a bandwidth-bound Monte Carlo ray
   trace. Cost: ~15 min and 90 cores taken from two other tracks. Consequence
   for the KPI: solver wall-clock varies ~40× with machine load, so the speedup
   clause is only meaningful with thread count pinned and load recorded.

2. **A single global timestep.** The etch rate spans 20× across the recipe box,
   so one dt forces most of the box to be nearly static — and a static target is
   one that a do-nothing predictor solves. Rules out the naive fixed-dt dataset.

3. **My fix for (2), which introduced its own confound.** Choosing dt per recipe
   as `target_depth / (rate · n_steps)` makes per-step displacement identically
   `U[0.4, 1.0]` µm *for every recipe*. An adversarial `codex` review caught it.
   Rules out reporting rel-L2 against persistence alone: a constant-advance
   predictor that ignores the recipe entirely is now strong. Two measurements
   were added to catch it (recipe-blind null, crossed-dt test split) rather than
   quietly re-tuning the dataset.

4. **`Domain.getSurfaceMesh()` as the extraction point.** Branches at
   mask/substrate junctions; a naive walk covered 80 of 229 nodes and the
   extracted surface did not move at all as the etch progressed. Fixed by meshing
   the top level set; the walk now raises rather than silently truncating.

## Decisions that need a human, stated as decisions

**D1 — If the speedup clause lands between 100× and 1000× like-for-like, which
number goes on the board?**
The grid will contain both a batched-H100-vs-1-thread-CPU figure (large) and a
CPU-1-thread-vs-CPU-1-thread figure (smaller, honest). The catalog says "시뮬
대비 ≥1000× 가속" without naming hardware.
- *Option A (recommended):* report the like-for-like figure as the KPI number and
  mark the clause `UNREACHABLE` if it falls short, with the full grid as
  evidence. Consistent with how the other tracks' withdrawals were handled.
- *Option B:* report the batched-GPU figure as the KPI number, with the
  like-for-like figure in the same table. Defensible only if the deployment story
  is explicitly "batched inference on an H100".

**D2 — Is the crossed-dt split a reported result or a training target?**
The crossed split (dt drawn independently of the recipe) is out-of-distribution
by construction.
- *Option A (recommended):* keep the training set as-is, report crossed-split
  performance separately as a generalisation probe. Cheap, and an honest negative
  there is a real finding.
- *Option B:* retrain on a mixture of adaptive and independent dt. Costs another
  ~20 min of generation plus a training run, and makes the headline number
  in-distribution for a broader distribution — arguably the better model, but it
  changes the protocol mid-flight, so the before/after numbers would not be
  comparable and both would have to be reported.

**D3 — Dataset size.** Currently 900 train trajectories (9000 transitions). If
the operator is data-limited rather than capacity-limited, more data is ~25 min
per 1000 trajectories at the measured 0.74 traj/s. A data-scaling ablation would
answer it but costs a training run per point.

## Still running, and how to check it

```bash
cd ~/Documents/workspace/etch-operator-twin
tail -f logs/gen_data.log          # dataset generation, ~0.74 traj/s
ls -la data/                       # train/val/test .npz + gen_report.json when done
python scripts/report.py           # regenerate RESULTS.md from whatever JSONs exist
for t in tests/test_*.py; do python $t; done   # 24 tests, all passing
```

Dataset generation (`scripts/gen_data.py --workers 16`) was at 600/900 train
trajectories at 03:27 UTC, then builds val (150) and test (250). Nothing else is
running. GPUs 0 and 1 (the track's lease) are idle and untouched.

## The next three things, in order

1. Train the baseline operator; evaluate on the test split against all three
   nulls; that produces clause 1.
2. Run `scripts/bench_speed.py` on a quiet box — clause 2.
3. Run `scripts/design.py` (gradient inverse design, scored in ViennaPS) and
   `scripts/inverse_baseline.py` (what the surrogate is worth in simulator
   calls) — clause 3.
