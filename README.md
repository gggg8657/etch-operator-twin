# etch-operator-twin — E4, Etch Neural Operator Twin

> **KPI (verbatim from the portfolio board):**
> **2D 표면진화 rel-L2 ≤0.05 · 시뮬 대비 ≥1000× 가속 · 역설계 형상오차 ≤5%**

Plasma etch surface evolution in 2-D, learned as a neural operator against a
ViennaPS ground truth, then inverted: descend on the recipe to hit a target
profile, and check the recipe it finds back in the simulator.

Status: **REOPENED 2026-09-10** (attack ladder, rung 2: a K-step horizon
operator, which attacks the accuracy and speedup clauses together). Two of three
clauses met, one short by a factor of ~95.

| clause | measured | verdict |
|---|---|---|
| 2D 표면진화 rel-L2 ≤0.05 | **0.0126** band rel-L2, 7 seeds, range 0.0011 | **MET, in distribution and nowhere else** — the crossed-dt probe misses under every coverage rule (terminal step 0.0531–0.0665, upper CI to 0.0947) |
| 시뮬 대비 ≥1000× 가속 | **10.55×** like-for-like, CPU-seconds, one-time cost paid on both sides or neither — median over **8 whole invocations**, range **8.86–12.16×** (`K10_nv_s1`, `cold_single_wafer`) | **UNREACHABLE** — short by **95×** at the median and **82×** at the fastest of 8 invocations, so no choice among them passes. At one step per application the operator is *slower* than ViennaPS (0.22–0.59×); only a 10-step horizon makes it faster at all, and that costs clause 1. The harness's own run-to-run spread is 1.37×, measured, and the batched-H100 readings are hardware comparisons rather than the KPI number |
| 역설계 형상오차 ≤5% | **0.0061** normalised area error, 20/20 targets, measured in ViennaPS | **MET** — profile targeting only; the inverse problem is degenerate over (rate × time), so the recovered recipe is not the one that made the target |

The verdict rows in `RESULTS.md` are derived from the run JSONs by
`scripts/report.py`, not typed. The clause that fails is the one the brief warned
would be cheated by accident, and it was — by us, three times. **Withdrawn on
2026-09-10: 1.47×, 6.57× and 119.12×.** All three timed a *cold* reference,
which pays a one-time ViennaPS initialisation inside `apply()`, against a *warm*
operator given untimed warm-up rollouts, and quoted the ratio as like-for-like.
`scripts/bench_symmetric.py` pays that cost on both sides or neither and reports
both readings; `runs/solver_drift.json` is the diagnosis. Two of the three also
rest on a reference denominator (8.46 s/wafer) that no condition reproduces, and
that discrepancy is recorded as unexplained rather than explained away.

`RESULTS.md` is regenerated from run JSONs by
`scripts/report.py`; no number in this repo is hand-typed, and none of them come
from a paper. Published baselines, when cited, live in their own column with a
source.

## The three clauses, and how each is measured

| clause | metric this repo reports | why it is the honest reading |
|---|---|---|
| rel-L2 ≤ 0.05 | rel-L2 of the SDF **in a narrow band** around the interface, headline; full-window rel-L2 also reported | The far field of a signed distance field is a smooth ramp that any model gets nearly free, so full-window rel-L2 flatters the operator. The band is where the physics is. Both numbers appear side by side. |
| ≥1000× speedup | seconds per simulated wafer for solver and operator, with hardware and thread count stated for each, **per-sample and batched** | A batched H100 forward against a single-threaded C++ solver is a hardware comparison, not a speedup. If the honest figure is short of 1000×, the clause is reported `UNREACHABLE` with the table. |
| 역설계 형상오차 ≤5% | normalised area difference between achieved and target profile (primary), Hausdorff distance in µm (secondary), **measured in the simulator** on the recipe the operator proposed | A recipe that only works on the surrogate is the failure mode this clause exists to catch. Recipes found outside the training box are reported as extrapolated, not as solutions. |

## The nulls, which are not optional

Every accuracy number in `RESULTS.md` is reported beside three predictors that
know progressively more, because a KPI threshold on its own cannot tell a working
operator from a flattering metric:

| null | what it knows | why it is there |
|---|---|---|
| persistence | nothing; predicts no change | an SDF over one step is mostly unchanged, so this scores far better than it deserves to |
| **recipe-blind** | the train-set mean per-step displacement | the timestep is chosen per recipe so every trajectory covers a comparable depth, which makes displacement nearly recipe-independent *by construction*. A model could score well while ignoring the recipe entirely. **Failing this one is fatal.** |
| uniform recession | the *ground truth's* mean displacement (oracle) | handed the correct amount of etch, lacking only the shape. Losing to it is informative, not damning; beating it is strong. |

A **crossed test split** (`scripts/gen_data.py --dt-mode independent`) draws dt
without reference to the recipe, so per-step displacement varies across the box
by the full range of the rate law. A model that learned "advance by the
middle of the 0.4-1.0 µm/step design range"
collapses there; one that learned `rate(recipe) × dt` does not. It is
out-of-distribution by construction and is reported separately, never merged into
the in-distribution number.

## Ground truth

ViennaPS 4.6.2, `SF6O2Etching` on a 2-D trench, units µm/min.

**Version pin:** viennaps 4.6.2 requires **viennals 5.8.5**. `pip install viennaps`
resolves ViennaLS to 5.9.0, which changes the registered
`viennahrle::BoundaryType` and makes `import viennaps` fail at load with
`could not convert default argument`. `requirements.txt` pins it.

State representation is *not* ViennaPS's own level set: its narrow-band grid
grows its bounding box as the trench deepens, so frames would not share a frame.
Instead each snapshot is an exact geometric SDF resampled on a fixed window
(x ∈ [−10, 10] µm, y ∈ [−18, 2] µm), computed from the explicit zero contour of
the top level set. Negative inside solid.

## Layout

```
eot/         package: solver wrapper, dataset, operator, inverse design
scripts/     one script per stage; report.py regenerates RESULTS.md
tests/       run by CI, no GPU needed
runs/        run JSONs -- the only source of numbers
```
