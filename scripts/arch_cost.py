"""Price named architectures per wafer, same estimator as cost_floor.py.

    python scripts/arch_cost.py --out runs/arch_cost.json

`cost_floor.py` prices *useless* models to establish floors; `cnn_cost.py`
prices the CompactCNN ladder. This file prices architectures that are candidate
surrogates -- models this repo intends to train -- so the decision to spend GPU
hours on one is made against its measured cost rather than an estimate. Every
row here is a real surrogate by construction, which is the opposite of
`cost_floor.py`'s convention, so the flag is spelled out per row.

**Cost is per wafer**: `per_apply * applications_per_wafer`. A stride-10 model
applies once to advance a wafer through all 10 dataset timesteps; a stride-1
model applies ten times. Per-application cost would flatter the former by 10x.

Reads the budget from `runs/speed_symmetric.json`, the symmetric benchmark that
pays ViennaPS's one-time init on both sides or neither -- the file that replaced
every earlier speed number in this repo after they were found to have timed a
cold solver against a warm operator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.bench_symmetric import solver_warm  # noqa: E402
from scripts.bench_workload import (solver_allframes,  # noqa: E402
                                    solver_fixed_duration, solver_terminal)
from scripts.cost_floor import time_model  # noqa: E402

# n_apply is the stride the row is meant to be deployed at: 10 dataset
# timesteps / stride.
SPECS = {
    "fno_w8m4L2_K10": dict(
        build="from eot.operator import EtchOperator\n"
              "M = EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2)",
        call="M(phi, cond)", n_apply=1,
        note="the accuracy-admissible FNO from runs/shrink.json (0.04717 "
             "in-distribution terminal at 3 seeds). Reference row: this is the "
             "cost the multiscale architecture has to beat by 6.4x."),
    "multiscale_s4_w8m4L2_wf8_K10": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, "
              "width_full=8, scale=4)",
        call="M(phi, cond)", n_apply=1,
        note="H15's candidate: the same body at 32x32 plus a full-resolution "
             "FiLM-conditioned 1x1 path."),
    "multiscale_s2_w8m4L2_wf8_K10": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, "
              "width_full=8, scale=2)",
        call="M(phi, cond)", n_apply=1,
        note="half the downsample, 4x the body cost: the control that says "
             "whether scale=4 was necessary or merely sufficient."),
    "multiscale_s8_w16m4L4_wf8_K10": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=16, modes=4, n_layers=4, "
              "width_full=8, scale=8)",
        call="M(phi, cond)", n_apply=1,
        note="if s=4 fits, s=8 buys back capacity inside the same budget: "
             "double the width and double the depth on a 16x16 body."),
    "multiscale_s4_w8m4L2_wf8_K1": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, "
              "width_full=8, scale=4)",
        call="M(phi, cond)", n_apply=10,
        note="the same model deployed at stride 1, ten applications per wafer. "
             "Included because the stride is the difference between clearing "
             "the clause and missing it by 10x, and that must be visible."),
}


# Components, priced to locate the cost rather than infer it. H15 predicted the
# multiscale body would be 16x cheaper than the full-resolution one (s^2), and
# the measured whole-model saving was 2.26x. Either the body is not where the
# cost is, or the parts that stayed at full resolution are. These rows answer
# which, and NONE of them is a surrogate -- each computes a fragment.
COMPONENTS = {
    "coords_cat_only": dict(
        build="from eot.operator import EtchOperator\n"
              "class M_(torch.nn.Module):\n"
              "    def forward(self, phi, cond):\n"
              "        B, _, H, W = phi.shape\n"
              "        return torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], 1)\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="building the two coordinate channels at 128x128 and "
             "concatenating them to phi. No parameters, no arithmetic on the "
             "data. Every architecture in this repo pays this."),
    "fullres_local_path_only": dict(
        build="import torch.nn.functional as F\n"
              "from eot.operator import MultiScaleOperator\n"
              "class M_(torch.nn.Module):\n"
              "    def __init__(s):\n"
              "        super().__init__(); s.m = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, width_full=8, scale=4)\n"
              "    def forward(s, phi, cond):\n"
              "        B, _, H, W = phi.shape\n"
              "        from eot.operator import EtchOperator\n"
              "        emb = s.m.cond(cond)\n"
              "        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], 1)\n"
              "        h = s.m.local_in(x); g, b = s.m.film(emb).chunk(2, 1)\n"
              "        h = F.gelu(h * (1 + g[:, :, None, None]) + b[:, :, None, None])\n"
              "        return s.m.local_out(h)\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="the FiLM-conditioned full-resolution 1x1 path alone, coords and "
             "cat included. What the multiscale model pays to keep any "
             "full-resolution processing at all."),
    "coarse_body_only": dict(
        build="import torch.nn.functional as F\n"
              "from eot.operator import MultiScaleOperator, EtchOperator\n"
              "class M_(torch.nn.Module):\n"
              "    def __init__(s):\n"
              "        super().__init__(); s.m = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, width_full=8, scale=4)\n"
              "    def forward(s, phi, cond):\n"
              "        B, _, H, W = phi.shape\n"
              "        emb = s.m.cond(cond)\n"
              "        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], 1)\n"
              "        xc = F.avg_pool2d(x, 4)\n"
              "        c = emb[:, :, None, None].expand(-1, -1, H // 4, W // 4)\n"
              "        z = s.m.lift(torch.cat([xc, c], 1))\n"
              "        for blk in s.m.blocks:\n"
              "            z = blk(z)\n"
              "        return F.interpolate(s.m.proj(z), size=(H, W), mode='bilinear', align_corners=False)\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="the coarse spectral body alone, including the avg_pool down and "
             "the bilinear up. What H15's argument said should be ~1/16 of the "
             "full-resolution body."),
    "fullres_fno_body_only": dict(
        build="from eot.operator import EtchOperator\n"
              "class M_(torch.nn.Module):\n"
              "    def __init__(s):\n"
              "        super().__init__(); s.m = EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2)\n"
              "    def forward(s, phi, cond):\n"
              "        B, _, H, W = phi.shape\n"
              "        c = s.m.cond(cond)[:, :, None, None].expand(-1, -1, H, W)\n"
              "        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype), c], 1)\n"
              "        x = s.m.lift(x)\n"
              "        for blk in s.m.blocks:\n"
              "            x = blk(x)\n"
              "        return s.m.proj(x)\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="the same body at 128x128: the thing the coarse body replaces. "
             "The ratio of this row to coarse_body_only is the saving H15 "
             "claimed, isolated from everything that did not change."),
}


# Located, not inferred. `fullres_local_path_only` measured 1193 us for two 1x1
# convolutions, a FiLM affine and one GELU at 128x128 -- about 1 MFLOP of
# arithmetic, which at this box's measured ~13 GFLOP/s single-thread should be
# under 100 us. Something other than arithmetic is being paid, and these rows
# take the path apart one operation at a time so the answer is measured.
COMPONENTS_FULLRES = {
    "gelu_only_w8_fullres": dict(
        build="class M_(torch.nn.Module):\n"
              "    def forward(self, phi, cond):\n"
              "        return torch.nn.functional.gelu(phi.expand(-1, 8, -1, -1))\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="one GELU over 8x128x128 = 131,072 elements and nothing else. "
             "torch's default GELU is erf-based, so this is the price of one "
             "transcendental pointwise nonlinearity at full resolution."),
    "relu_only_w8_fullres": dict(
        build="class M_(torch.nn.Module):\n"
              "    def forward(self, phi, cond):\n"
              "        return torch.nn.functional.relu(phi.expand(-1, 8, -1, -1))\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="the same tensor through ReLU. The difference from the GELU row "
             "is what the choice of nonlinearity costs, holding the memory "
             "traffic fixed."),
    "fullres_1x1_linear_no_act": dict(
        build="class M_(torch.nn.Module):\n"
              "    def __init__(s):\n"
              "        super().__init__(); s.a = torch.nn.Conv2d(3, 8, 1); s.b = torch.nn.Conv2d(8, 1, 1)\n"
              "    def forward(s, phi, cond):\n"
              "        from eot.operator import EtchOperator\n"
              "        B, _, H, W = phi.shape\n"
              "        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], 1)\n"
              "        return s.b(s.a(x))\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="the same two 1x1 convolutions with NO nonlinearity and NO FiLM. "
             "Subtracting this from fullres_local_path_only isolates what the "
             "conditioning and the activation add."),
    "fullres_1x1_plus_gelu": dict(
        build="class M_(torch.nn.Module):\n"
              "    def __init__(s):\n"
              "        super().__init__(); s.a = torch.nn.Conv2d(3, 8, 1); s.b = torch.nn.Conv2d(8, 1, 1)\n"
              "    def forward(s, phi, cond):\n"
              "        from eot.operator import EtchOperator\n"
              "        B, _, H, W = phi.shape\n"
              "        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], 1)\n"
              "        return s.b(torch.nn.functional.gelu(s.a(x)))\n"
              "M = M_()",
        call="M(phi, cond)", n_apply=1,
        note="linear path plus the GELU, still no FiLM. The step from "
             "fullres_1x1_linear_no_act to here is the activation's price in "
             "situ; the step from here to fullres_local_path_only is FiLM's."),
}
COMPONENTS.update(COMPONENTS_FULLRES)


# LADDER RUNG 3: fix your own setup. Every cost this repo has published was
# measured in eager PyTorch on one thread. The component rows above show why
# that matters: `coarse_body_only` costs 612 us at 32x32 with width 8, about
# 0.1 MFLOP of arithmetic, and `relu_only_w8_fullres` costs 13.2 us to touch
# 131,072 floats. Neither number is arithmetic. At these tensor sizes the price
# is per-operation dispatch, allocation and (for the spectral blocks) FFT plan
# lookup -- roughly 20 operations at ~30 us each. That is a property of the
# inference stack, not of the operator or of the etch problem, and measuring the
# clause against it makes the verdict a statement about eager mode.
#
# `torch.compile` fuses the elementwise chains and removes most of the dispatch,
# and it is what anyone deploying this would do. It changes no weight, no
# output, no metric -- so it is rung 3 (fix the setup), not protocol-loosening.
#
# **The compile cost is real and is reported, not hidden.** Compilation happens
# on the first call. The `warm` reading excludes it, which is the same treatment
# `bench_symmetric.py` gives ViennaPS's one-time init after that asymmetry was
# found to have inflated every earlier speedup in this repo by up to 119x. The
# `cold` reading here therefore includes a full compilation and is dominated by
# it; it is a per-process figure and it is labelled, not quoted as a speedup.
COMPONENTS_COMPILED = {}
MODELS_COMPILED = {
    "fno_w8m4L2_K10_compiled": dict(
        build="from eot.operator import EtchOperator\n"
              "M = EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2)\n"
              "M.eval()\n"
              "M = torch.compile(M, dynamic=False)",
        call="M(phi, cond)", n_apply=1,
        note="the accuracy-admissible FNO (0.04717 in-distribution terminal, "
             "runs/shrink.json), compiled. Same weights, same output."),
    "multiscale_s4_w8m4L2_wf8_K10_compiled": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=8, modes=4, n_layers=2, "
              "width_full=8, scale=4)\n"
              "M.eval()\n"
              "M = torch.compile(M, dynamic=False)",
        call="M(phi, cond)", n_apply=1,
        note="H15's architecture, compiled. Accuracy UNMEASURED as of this "
             "row: no accuracy is claimed for it anywhere in this file."),
    "multiscale_s8_w16m4L4_wf8_K10_compiled": dict(
        build="from eot.operator import MultiScaleOperator\n"
              "M = MultiScaleOperator(cond_dim=7, width=16, modes=4, n_layers=4, "
              "width_full=8, scale=8)\n"
              "M.eval()\n"
              "M = torch.compile(M, dynamic=False)",
        call="M(phi, cond)", n_apply=1,
        note="the capacity-restored variant, compiled. Accuracy UNMEASURED."),
    "fno_w64m20L4_DEPLOYED_compiled": dict(
        build="from eot.operator import EtchOperator\n"
              "M = EtchOperator(cond_dim=7, width=64, modes=20, n_layers=4)\n"
              "M.eval()\n"
              "M = torch.compile(M, dynamic=False)",
        call="M(phi, cond)", n_apply=1,
        note="the deployed 26.2M-parameter operator, compiled. Its eager cost "
             "is 87835 us (runs/cost_floor.json); this says how much of that "
             "was dispatch overhead rather than arithmetic. At 26.2M params "
             "this row should be arithmetic-bound, so it is the control that "
             "says compilation is not a measurement trick."),
}
SPECS.update(MODELS_COMPILED)


# THE LADDER THAT DECIDES CLAUSE 2. Every row is a candidate surrogate, trained
# at stride 10 (one application per wafer), and the rows straddle the 1000x
# threshold. The point of measuring cost first is that the accuracy experiment
# is only worth running on rows whose cost is known, and known BEFORE the
# accuracy is, so the frontier cannot be assembled after the fact from whichever
# rows happen to look good.
#
# Pointwise rows have NO spatial mixing at any resolution: the residual at a
# pixel is a per-pixel MLP in (phi, x, y, recipe). That is a real restriction and
# it is stated where the row is defined -- the mask geometry in this dataset is a
# two-parameter family (`trench_width`, `mask_height` are conditioning inputs),
# so (x, y, trench_width, mask_height) determines the layout and a pointwise
# model can in principle infer it. On an arbitrary mask layout it could not, and
# a pointwise result therefore does not transfer to one.
LADDER = {
    f"pw_wf{wf}_n{nl}_{act}_K10": dict(
        build="from eot.operator import MultiScaleOperator\n"
              f"M = MultiScaleOperator(cond_dim=7, scale=0, width_full={wf}, "
              f"n_local={nl}, act='{act}')",
        call="M(phi, cond)", n_apply=1,
        note=f"pointwise per-pixel MLP, width {wf}, {nl} hidden layer(s), "
             f"{act}. NO spatial mixing at any resolution.")
    for wf, nl, act in [(8, 1, "relu"), (16, 2, "relu"), (32, 3, "relu"),
                        (64, 4, "relu"), (32, 3, "gelu")]
}
LADDER.update({
    f"ms_s{sc}_w{w}m4L{nl}_wf16_relu_K10": dict(
        build="from eot.operator import MultiScaleOperator\n"
              f"M = MultiScaleOperator(cond_dim=7, scale={sc}, width={w}, modes=4, "
              f"n_layers={nl}, width_full=16, n_local=2, act='relu')",
        call="M(phi, cond)", n_apply=1,
        note=f"pointwise path plus a spectral body on a {128 // sc}x{128 // sc} "
             f"grid, width {w}, {nl} layers. The cheapest rows that mix "
             "spatially at all.")
    for sc, w, nl in [(8, 8, 2), (8, 16, 4), (16, 16, 4), (4, 8, 2)]
})
SPECS.update(LADDER)


# H18: attack the OPERATION COUNT, the one axis the three failed attacks on
# clause 2 never touched. See eot.operator.SpectralPropagator for why this form
# is the leading-order physics and not merely a cheap shape.
SPECS.update({
    f"specprop_m{m}_ma{ma}_K10": dict(
        build="from eot.operator import SpectralPropagator\n"
              f"M = SpectralPropagator(cond_dim=7, modes={m}, modes_a={ma})",
        call="M(phi, cond)", n_apply=1,
        note=f"conditioned linear Fourier propagator, modes={m}, "
             f"modes_a={ma}. Four full-resolution operations (rfft2, masked "
             "multiply-add, irfft2, add) against the FNO's ~20, and the "
             "transforms act on one channel not eight. Accuracy UNMEASURED at "
             "the time this row was priced.")
    for m, ma in [(4, 4), (4, 16), (8, 32)]
})


# H20: runs/spectral_floor_fine.json makes every modes_a below 63 provably
# unable to meet clause 1 -- the oracle projection floor is 0.879 at ma=4, 0.441
# at ma=16, 0.285 at ma=32, and crosses 0.05 only between ma=62 and ma=63. The
# residual of an SDF over a 10-step etch is BROADBAND, because the surface moves
# several micron and the change is spatially localised at the interface. So the
# three configs launched for H18 are all doomed, and the fix is the thing the
# class was designed around: the ADDITIVE term's mode count is free in
# operations, because irfft2 costs the same whatever fraction of the spectrum is
# non-zero. These rows check that it is also affordable in TIME -- the a_head
# MLP emits 2*(2*ma)*ma numbers, so its parameter count grows as ma^2 even
# though it runs on a (1, 7) tensor and never touches the grid.
SPECS.update({
    f"specprop_m{m}_ma{ma}_K10": dict(
        build="from eot.operator import SpectralPropagator\n"
              f"M = SpectralPropagator(cond_dim=7, modes={m}, modes_a={ma})",
        call="M(phi, cond)", n_apply=1,
        note=f"modes={m}, modes_a={ma}. Oracle floor at this modes_a is in "
             "runs/spectral_floor_fine.json; ma>=63 is the only band that "
             "admits clause 1 at all.")
    for m, ma in [(4, 63), (4, 64), (8, 64), (16, 64)]
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--cold-rounds", type=int, default=None,
                    help="rounds for the cold reading (each is a fresh "
                         "process, so this dominates wall-clock). Defaults to "
                         "--rounds; 0 skips the cold reading entirely.")
    ap.add_argument("--sym", default="runs/speed_symmetric.json")
    ap.add_argument("--out", default="runs/arch_cost.json")
    ap.add_argument("--target", type=float, default=1000.0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test",
                    help="the split whose recipes and dt values the solver is "
                         "timed on. Defaults to the split accuracy is scored on.")
    ap.add_argument("--solver-wafers", type=int, default=16,
                    help="wafers per solver denominator. Solver cost varies ~4x "
                         "across recipes, so a small count gives an unstable "
                         "median: the fixed-duration column read a 12x spread "
                         "at 32 wafers.")
    ap.add_argument("--stored-budget", action="store_true",
                    help="use the solver cost stored in --sym instead of "
                         "re-timing it here. Leaves every ratio unpaired; only "
                         "for reproducing an older row.")
    a = ap.parse_args()

    runlock.acquire(a.out, what="arch_cost")

    # THE DENOMINATOR IS MEASURED IN THIS INVOCATION, and that is a correction.
    #
    # Every "speedup_vs_solver" in runs/cost_floor.json and runs/cnn_cost.json
    # divides a model cost timed *now* by a solver cost read out of
    # runs/speed_symmetric.json, timed hours or days earlier. That is the same
    # shape of error as the one that inflated this repo's headline speedup up to
    # 119x -- two sides of a ratio measured under conditions that were not the
    # same -- with load standing in for warm-versus-cold.
    #
    # It is not hypothetical. Nine ladder rows were timed here at load average
    # 26.4 (another job on this box holding ~20 cores) and read 2.5x the cost
    # the same architectures read at load ~11 minutes earlier: the multiscale
    # s=4 model went 1405 -> 3569 us/wafer without a line changing. Divided by a
    # stored denominator that is a factor of 2.5 too small for present
    # conditions, its speedup would have been understated by the same factor --
    # and had the load fallen instead of risen, overstated.
    #
    # A ratio of two costs measured in the same invocation under the same load
    # is what the clause is about, so the solver is re-timed here by
    # bench_symmetric.solver_warm -- the same function, same recipe stream, same
    # grid and step count. `--stored-budget` keeps the old behaviour available
    # and says in the JSON that it was used.
    sym = json.loads(Path(a.sym).read_text())
    # These live under "protocol" in speed_symmetric.json; read them rather
    # than restating them, so the re-timed solver is the SAME workload the
    # stored budget was measured on. A different dt or grid_delta here would
    # silently change the denominator.
    cfg = sym["protocol"]
    n_steps = int(cfg["n_steps_per_wafer"])
    fixed_total = n_steps * float(cfg["dt"])
    grid_delta = float(cfg["grid_delta"])
    seed = int(cfg.get("recipe_seed", sym.get("recipe_seed", 0)))

    # THREE DENOMINATORS, ALL MEASURED HERE, because there is not one honest
    # denominator -- there is one per output. See runs/bench_workload.json for
    # the derivation; the short version:
    #
    #   terminal_one_apply    one apply of 10*dt per wafer, terminal state only.
    #                         The MATCHED denominator for a stride-10 operator,
    #                         which emits exactly that in one application.
    #   all_frames_ten_applies  ten applies of dt, all ten states. The matched
    #                         denominator for a stride-1 operator. Rasterisation
    #                         is EXCLUDED: it is 76% of that column and it is our
    #                         own unoptimised numpy, so charging the solver for
    #                         it would inflate the denominator with our slow code.
    #   fixed_duration        one apply at 10*0.2 = 2.0 minutes, the duration
    #                         speed_symmetric.json chose. Kept because it is what
    #                         every published number in this repo divided by, so
    #                         the correction stays visible as a difference.
    #
    # The dataset's own dt has median 0.344, i.e. a 3.44-minute etch, so the
    # fixed 2.0 understates the median wafer by 2.10x. Adopting the matched
    # denominator SILENTLY would be the protocol-loosening the rules forbid;
    # every row below therefore carries its speedup against all three.
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    dset = np.load(Path(a.data) / f"{a.split}.npz")
    n_take = a.solver_wafers + 1                    # one discarded as warm-up
    rows, dts = dset["recipe"][:n_take], dset["dt"][:n_take]

    loadavg = float(Path("/proc/loadavg").read_text().split()[0])
    if a.stored_budget:
        dens = {"fixed_duration": sym["solver"]["marginal_warm"]["median_cpu"]}
        budget_provenance = {"paired": False, "source": a.sym,
                             "why": "--stored-budget was passed; the ratios in "
                                    "this file are NOT paired and are only "
                                    "comparable to the load at which --sym ran"}
        matched_key = "fixed_duration"
    else:
        term = solver_terminal(rows, dts, gen["steps"], gen["grid_delta"])
        allf = solver_allframes(rows, dts, gen["steps"], gen["grid_delta"],
                                gen["grid_n"])
        fixd = solver_fixed_duration(rows, fixed_total, gen["grid_delta"])
        allf_no_raster = [c - r for c, r in zip(allf["cpu"], allf["cpu_raster"])]
        dens = {
            "terminal_one_apply": float(np.median(term["cpu"])),
            "all_frames_ten_applies_excl_raster": float(np.median(allf_no_raster)),
            "fixed_duration": float(np.median(fixd["cpu"])),
        }
        matched_key = "terminal_one_apply"
        budget_provenance = {
            "paired": True,
            "measured_in_this_invocation": True,
            "solver_wafers": a.solver_wafers,
            "split": a.split,
            "same_wafers_as_accuracy": "the recipes and dt values of the split "
                                       "the operator's accuracy is scored on",
            "denominators_cpu_s": dens,
            "all_frames_raster_excluded_cpu_s": float(np.median(allf["cpu_raster"])),
            "stored_warm_median_cpu_s": sym["solver"]["marginal_warm"]["median_cpu"],
            "drift_vs_stored": dens["fixed_duration"]
            / sym["solver"]["marginal_warm"]["median_cpu"],
            "loadavg_1min_at_start": loadavg,
            "matched_denominator_for_one_application": matched_key,
            "why": "a stride-10 operator emits the terminal state in one "
                   "application, so the terminal one-apply column is the "
                   "like-for-like denominator for it. All three are reported "
                   "per row so the choice is visible rather than adopted.",
        }
    solver_w = dens[matched_key]
    solver_c = sym["solver"]["cold_single_wafer"]["median_cpu"]
    budget_w, budget_c = solver_w / a.target, solver_c / a.target

    res = {
        "question": "which candidate surrogate architectures fit the clause-2 "
                    "budget, measured before any of them is trained",
        "protocol": {
            "estimator": "CPU-seconds per wafer on one verified thread, fresh "
                         "subprocess per config",
            "rounds": a.rounds, "warm_rollouts_untimed": 3,
            "budget_warm_s": budget_w, "budget_cold_s": budget_c,
            "solver_warm_cpu_s": solver_w, "solver_cold_cpu_s": solver_c,
            "speedup_target": a.target,
            "budget_provenance": budget_provenance,
            "no_accuracy_claimed": "cost only. Accuracy for these rows is in "
                                   "runs/shrink.json and runs/multiscale.json.",
        },
        "models": {},
        "components": {},
    }
    all_specs = [(k, v, True) for k, v in SPECS.items()] + \
                [(k, v, False) for k, v in COMPONENTS.items()]
    for name, spec, is_surrogate in all_specs:
        if a.only and name not in a.only:
            continue
        # Load is recorded PER ROW, not once per file. The ratio this file
        # reports is not load-invariant even in CPU-seconds: solver_drift.json
        # measures the solver at 1.15x across a 3.6x load range, while the
        # ladder rows here moved 2.5x between load ~11 and load 26.4. A
        # ray-tracing solver is compute-bound and a 9,914-parameter network at
        # 128x128 is bound by memory traffic and per-op dispatch, so contention
        # does not scale the two sides together and pairing the denominator
        # fixes the drift in the denominator without making the ratio a
        # constant. Under load the operator is penalised more, so a speedup
        # measured under load is a LOWER BOUND on the quiet-box figure.
        load_before = float(Path("/proc/loadavg").read_text().split()[0])
        warm = time_model(spec, n_rep=a.rounds, n_warm=3)
        n_cold = a.rounds if a.cold_rounds is None else a.cold_rounds
        colds = [time_model(spec, n_rep=1, n_warm=0) for _ in range(n_cold)]
        cold_cpu = [c for r in colds for c in r["cpu"]]
        load_after = float(Path("/proc/loadavg").read_text().split()[0])
        row = {"note": spec["note"], "params": warm["params"],
               "applications_per_wafer": spec["n_apply"],
               "is_a_real_surrogate": is_surrogate,
               "loadavg_1min": {"before": load_before, "after": load_after}}
        for tag, cpu, budget, solver in (("warm", warm["cpu"], budget_w, solver_w),
                                         ("cold", cold_cpu, budget_c, solver_c)):
            if not cpu:
                row[tag] = {"skipped": "--cold-rounds 0"}
                continue
            pa = float(np.median(cpu))
            pw = pa * spec["n_apply"]
            row[tag] = {
                "per_apply_cpu_s": pa, "per_wafer_cpu_s": pw,
                "min_cpu_s": float(np.min(cpu)), "max_cpu_s": float(np.max(cpu)),
                "spread_factor": float(np.max(cpu) / np.min(cpu)),
                "over_budget_factor": pw / budget,
                "under_budget": bool(pw <= budget),
                "speedup_vs_solver": solver / pw,
                "speedup_vs_each_denominator": {
                    k: v / pw for k, v in dens.items()
                } if tag == "warm" else None,
                "matched_denominator": matched_key if spec["n_apply"] == 1
                else "all_frames_ten_applies_excl_raster",
                # The between-invocation spread on this harness reaches 2x
                # (runs/cnn_cost.json), so the interval is what a reader needs.
                "speedup_range": [solver / (float(np.max(cpu)) * spec["n_apply"]),
                                  solver / (float(np.min(cpu)) * spec["n_apply"])],
                "n": len(cpu),
            }
        res["models" if is_surrogate else "components"][name] = row
        w = row["warm"]
        print(f"{name:34s} p={row['params']:7d} n_app={spec['n_apply']:2d} "
              f"{w['per_wafer_cpu_s']*1e6:9.1f} us/wafer  {w['speedup_vs_solver']:8.1f}x "
              f"[{w['speedup_range'][0]:.0f}-{w['speedup_range'][1]:.0f}] "
              f"{'UNDER' if w['under_budget'] else 'over'} budget")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
