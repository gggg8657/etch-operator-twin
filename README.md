# etch-operator-twin — E4, Etch Neural Operator Twin

> **KPI (verbatim from the portfolio board):**
> **2D 표면진화 rel-L2 ≤0.05 · 시뮬 대비 ≥1000× 가속 · 역설계 형상오차 ≤5%**

Plasma etch surface evolution in 2-D, learned as a neural operator against a
ViennaPS ground truth, then inverted: descend on the recipe to hit a target
profile, and check the recipe it finds back in the simulator.

Status: **RUNNING**. `RESULTS.md` is regenerated from run JSONs by
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
by the full range of the rate law. A model that learned "advance ~0.7 µm"
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
