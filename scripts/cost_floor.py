"""H6: the inference-cost floor of the output representation, not of our network.

    python scripts/cost_floor.py

`runs/speed_spread.json` puts the like-for-like speedup at 10.55× (8.86-12.16×
over 8 invocations) against a 1000× clause, so the operator would have to reach
**276 µs per wafer** to pass -- the solver's warm cost of 0.2758 CPU-s divided by
1000. codex's route to that is distillation into a compact model.

**This script asks whether that route is open at all**, by timing models chosen
to be *useless*. The deployed operator emits a 128x128 signed-distance field per
application: 16,384 float32 values. If an identity map or a 1x1 convolution --
which compute nothing of value -- already cost more than 276 µs on one CPU
thread, then no model with that output can pass, whatever its architecture, and
the clause is a statement about the problem formulation rather than about our
choice of width and modes.

The escape that would imply is a change of output representation. An etch front
is a curve: the field is a rasterisation of ~128 surface heights, so a model
emitting the surface directly has an output 128x smaller. Both the raw
surface-output cost and the cost *including* rasterising back to a field are
measured, because clause 1 is scored as a field rel-L2 and that rasterisation
would belong on the operator's side of the ratio.

**No row here claims any accuracy.** These are cost floors. A row that is fast
and predicts nothing is the point.

Same estimator as `scripts/bench_symmetric.py`: CPU-seconds on one verified
thread, in a fresh subprocess per model, warm (untimed warm-ups) and cold.

Writes `runs/cost_floor.json`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402

# Each entry builds a torch.nn.Module in the child and is called once per wafer.
# `note` says what the row is for; `useful` records whether the row could
# possibly be a surrogate, so no reader mistakes a floor for a result.
MODELS = {
    "identity_field": dict(
        build="M = torch.nn.Identity()",
        call="M(phi)",
        note="returns its input unchanged. The absolute floor for a 128x128 "
             "field output: no arithmetic, only the tensor handling.",
        useful=False, output="field 128x128"),
    "conv1x1_field": dict(
        build="M = torch.nn.Conv2d(1, 1, 1)",
        call="M(phi)",
        note="one multiply-add per pixel. The floor for any convolutional model "
             "with this output.",
        useful=False, output="field 128x128"),
    "conv3x3_w8_field": dict(
        build="M = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), "
              "torch.nn.GELU(), torch.nn.Conv2d(8, 1, 3, padding=1))",
        call="M(phi)",
        note="the smallest convolutional stack with any spatial capacity at all.",
        useful=False, output="field 128x128"),
    "fno_w8_m4_field": dict(
        build="from eot.operator import EtchOperator\n"
              "M = EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2)",
        call="M.rollout(phi, cond, 1)",
        note="a deliberately tiny spectral operator: 1/8 the width, 1/5 the "
             "modes and half the depth of the deployed one.",
        useful=False, output="field 128x128"),
    "fno_w64_m20_field_DEPLOYED": dict(
        build="from eot.operator import EtchOperator\n"
              "M = EtchOperator(cond_dim=7, width=64, modes=20, n_layers=4)",
        call="M.rollout(phi, cond, 1)",
        note="the architecture every accuracy number in this repo comes from, "
             "26,248,025 parameters. The only row here that is a real surrogate.",
        useful=True, output="field 128x128"),
    "mlp_surface_128": dict(
        build="M = torch.nn.Sequential(torch.nn.Linear(7, 128), torch.nn.GELU(), "
              "torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 128))",
        call="M(cond)",
        note="recipe -> 128 surface heights, no field at all. The alternative "
             "output representation, at a capacity that is at least arguable.",
        useful=False, output="surface, 128 heights"),
    "mlp_surface_128_plus_raster": dict(
        build="M = torch.nn.Sequential(torch.nn.Linear(7, 128), torch.nn.GELU(), "
              "torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 128))\n"
              "ys = torch.arange(128, dtype=torch.float32).view(1, 128, 1)",
        call="h = M(cond); _ = ys - h.view(1, 1, 128)",
        note="the same surface model, plus the broadcast that turns 128 heights "
             "into a 128x128 signed-height field -- the cost of scoring it "
             "against clause 1's field metric, which belongs on the operator's "
             "side of the ratio.",
        useful=False, output="surface -> field 128x128"),
}

_CHILD = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, sys
sys.path.insert(0, {root!r})
import torch
torch.set_num_threads(1)
{build}
if hasattr(M, "eval"):
    M.eval()
torch.manual_seed(0)
phi = torch.randn(1, 1, {H}, {H}); cond = torch.randn(1, 7)
wall, cpu = [], []
with torch.no_grad():
    for _ in range({n_warm}):
        {call}
    for _ in range({n_rep}):
        w0 = time.perf_counter(); c0 = time.process_time()
        {call}
        wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
n_par = sum(p.numel() for p in M.parameters()) if hasattr(M, "parameters") else 0
print(json.dumps({{"wall": wall, "cpu": cpu, "params": n_par}}))
'''


def time_model(spec, n_rep, n_warm, H=128):
    body = _CHILD.format(root=str(ROOT), build=spec["build"], call=spec["call"],
                         H=H, n_rep=n_rep, n_warm=n_warm)
    out = subprocess.run([sys.executable, "-c", body], capture_output=True,
                         text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", default="runs/cost_floor.json")
    ap.add_argument("--speed", default="runs/speed_spread.json")
    ap.add_argument("--sym", default="runs/speed_symmetric.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="cost_floor")

    # The budget comes from the measured solver cost, not from a round number.
    sym = json.loads(Path(a.sym).read_text())
    solver_warm = sym["solver"]["marginal_warm"]["median_cpu"]
    solver_cold = sym["solver"]["cold_single_wafer"]["median_cpu"]
    budget_warm = solver_warm / 1000.0
    budget_cold = solver_cold / 1000.0

    res = {
        "hypothesis": "H6: the 1000x clause is unreachable on this hardware for "
                      "ANY model whose output is a 128x128 field, because the "
                      "cost floor of producing that output exceeds the budget. "
                      "Falsified if a useless model comes in under budget.",
        "protocol": {
            "estimator": "CPU-seconds per wafer, one verified thread, fresh "
                         "subprocess per model, one application per wafer",
            "rounds": a.rounds, "warm_rollouts_untimed": 3,
            "budget_source": f"{a.sym}: solver marginal_warm {solver_warm:.4f} "
                             f"CPU-s and cold_single_wafer {solver_cold:.4f} "
                             f"CPU-s, divided by the clause's 1000x",
            "budget_warm_s": budget_warm, "budget_cold_s": budget_cold,
            "no_accuracy_claimed": "every row but the deployed one is chosen to "
                                   "be useless. These are cost floors; a fast "
                                   "row here is not a surrogate.",
        },
        "models": {},
    }

    for name, spec in MODELS.items():
        row = {"note": spec["note"], "output": spec["output"],
               "is_a_real_surrogate": spec["useful"]}
        for reading, n_warm in (("warm", 3), ("cold", 0)):
            n_rep = a.rounds if reading == "warm" else 1
            cpus, walls, par = [], [], 0
            reps = 1 if reading == "warm" else a.rounds
            for _ in range(reps):
                r = time_model(spec, n_rep, n_warm)
                cpus += r["cpu"]
                walls += r["wall"]
                par = r["params"]
            cw = [c / w for c, w in zip(cpus, walls) if w > 0]
            row["params"] = par
            row[reading] = {
                "median_cpu_s": float(np.median(cpus)),
                "min_cpu_s": float(np.min(cpus)),
                "cpu_over_wall_median": float(np.median(cw)) if cw else None,
                "n": len(cpus),
            }
            budget = budget_warm if reading == "warm" else budget_cold
            row[reading]["over_budget_factor"] = float(np.median(cpus)) / budget
            row[reading]["under_budget"] = bool(float(np.median(cpus)) <= budget)
            row[reading]["speedup_vs_solver"] = (
                (solver_warm if reading == "warm" else solver_cold)
                / float(np.median(cpus)))
        res["models"][name] = row
        print(f"{name:36} warm {row['warm']['median_cpu_s']*1e6:9.0f} us "
              f"({row['warm']['over_budget_factor']:7.1f}x budget)  "
              f"speedup {row['warm']['speedup_vs_solver']:8.1f}x", flush=True)

    floor = min((v for k, v in res["models"].items()
                 if v["output"] == "field 128x128"),
                key=lambda v: v["warm"]["median_cpu_s"])
    surf = res["models"]["mlp_surface_128"]
    surf_r = res["models"]["mlp_surface_128_plus_raster"]
    res["verdict"] = {
        "cheapest_field_model_cpu_s": floor["warm"]["median_cpu_s"],
        "cheapest_field_model_is": floor["note"],
        "budget_warm_s": budget_warm,
        "field_floor_over_budget": floor["warm"]["over_budget_factor"],
        "H6_supported": bool(not floor["warm"]["under_budget"]),
        "H6_reading": (
            "SUPPORTED: even a model that computes nothing exceeds the 1000x "
            "budget for a 128x128 field output, so the clause is unreachable for "
            "any architecture in this output representation on this hardware -- "
            "not merely for ours."
            if not floor["warm"]["under_budget"] else
            "FALSIFIED: a useless model comes in under budget, so the field "
            "output does not bound the clause and the gap is capacity we chose. "
            "The distillation route is open and H6 is withdrawn."),
        "surface_output_cpu_s": surf["warm"]["median_cpu_s"],
        "surface_output_over_budget": surf["warm"]["over_budget_factor"],
        "surface_plus_raster_cpu_s": surf_r["warm"]["median_cpu_s"],
        "surface_plus_raster_over_budget": surf_r["warm"]["over_budget_factor"],
        "escape_note": "the surface rows are the alternative representation. If "
                       "they are under budget and the field floor is not, then "
                       "the clause turns on the output representation, which is a "
                       "design decision this repo made and can revisit -- and the "
                       "'+raster' row prices scoring it against clause 1's field "
                       "metric. Neither surface row has been trained or scored "
                       "for accuracy; this file bounds cost only.",
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdict"], indent=2))


if __name__ == "__main__":
    main()
