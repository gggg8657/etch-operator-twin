"""What a *conditioned* compact operator costs, against clause 2's budget.

    python scripts/cnn_cost.py

**The number this corrects is mine.** `runs/cost_floor.json` priced a two-layer
3x3 convolution at **370 us, 748x** — the only architecture this repo has
measured within reach of the 1000x clause, and the row that turned clause 2 from
"unreachable" into "bound by the architecture we chose". That row is

    Conv2d(1, 8, 3) -> GELU -> Conv2d(8, 1, 3)

on a bare field: **no recipe embedding, no broadcast, no coordinate channels.**
It cannot advance a surface *given a recipe*, which is the task, so its cost is
a floor for the output shape and not a price for a surrogate. `cost_floor.json`
labels every such row `is_a_real_surrogate: false`, but the conclusion I drew
leaned on it anyway, and the label does not carry the conclusion.

`eot.operator.CompactCNN` is the honest version: it keeps the recipe MLP, the
broadcast to H x W, the two coordinate channels and the residual form — every
structural element the conditioning needs — and replaces only the spectral
blocks with 3x3 convolutions. This script prices a ladder of its sizes against
the same budget, so the question "does a compact model clear 1000x once it
actually takes a recipe?" gets an answer rather than an extrapolation.

Protocol, matching `scripts/bench_symmetric.py` and `scripts/speed_spread.py`:
CPU-seconds on one verified thread, fresh subprocess per configuration,
**repeated across processes** because this box's between-process spread is 1.37x
and a single draw is not a measurement. Budget is the measured solver cost
divided by 1000, both readings.

**No accuracy is claimed by any row here.** These are prices. Whether a model
this small can reach rel-L2 <= 0.05 is what `scripts/shrink.sh` is measuring for
the spectral family and what a trained CNN arm would have to measure for this
one.

Writes `runs/cnn_cost.json`.
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

# (label, kwargs). The ladder brackets the budget rather than hunting for a
# passing cell: the smallest entry is barely a model and the largest is the
# smallest thing with any depth and channel mixing.
LADDER = [
    ("cnn_w4_L1_c4", dict(width=4, n_layers=1, cond_ch=4)),
    ("cnn_w8_L1_c8", dict(width=8, n_layers=1, cond_ch=8)),
    ("cnn_w8_L2_c8", dict(width=8, n_layers=2, cond_ch=8)),
    ("cnn_w16_L2_c8", dict(width=16, n_layers=2, cond_ch=8)),
    ("cnn_w16_L4_c16", dict(width=16, n_layers=4, cond_ch=16)),
    ("cnn_w32_L4_c24", dict(width=32, n_layers=4, cond_ch=24)),
]

_CHILD = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, sys
sys.path.insert(0, {root!r})
import torch
torch.set_num_threads(1)
from eot.operator import CompactCNN
m = CompactCNN(cond_dim=7, **{kw!r})
m.eval()
torch.manual_seed(0)
phi = torch.randn(1, 1, {H}, {H}); cond = torch.randn(1, 7)
wall, cpu = [], []
with torch.no_grad():
    for _ in range({n_warm}):
        m.rollout(phi, cond, 1)
    for _ in range({n_rep}):
        w0 = time.perf_counter(); c0 = time.process_time()
        m.rollout(phi, cond, 1)
        wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
print(json.dumps({{"wall": wall, "cpu": cpu,
                   "params": sum(p.numel() for p in m.parameters())}}))
'''


def time_cfg(kw, n_rep, n_warm, H=128):
    body = _CHILD.format(root=str(ROOT), kw=kw, H=H, n_rep=n_rep, n_warm=n_warm)
    r = subprocess.run([sys.executable, "-c", body], capture_output=True,
                       text=True, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def summ(xs):
    xs = np.asarray(xs, float)
    return {"median": float(np.median(xs)), "min": float(xs.min()),
            "max": float(xs.max()),
            "max_over_min": float(xs.max() / max(xs.min(), 1e-12)),
            "n": int(xs.size), "per_process": [float(x) for x in xs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6, help="separate processes per config")
    ap.add_argument("--calls", type=int, default=20, help="timed calls inside each")
    ap.add_argument("--sym", default="runs/speed_symmetric.json")
    ap.add_argument("--out", default="runs/cnn_cost.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="cnn_cost")
    sym = json.loads(Path(a.sym).read_text())
    sw = sym["solver"]["marginal_warm"]["median_cpu"]
    sc = sym["solver"]["cold_single_wafer"]["median_cpu"]
    budget = {"marginal_warm": sw / 1000.0, "cold_single_wafer": sc / 1000.0}

    rows = {}
    for label, kw in LADDER:
        row = {"config": kw}
        for reading, n_warm in (("marginal_warm", 3), ("cold_single_wafer", 0)):
            meds, params = [], 0
            for _ in range(a.reps):
                r = time_cfg(kw, a.calls if reading == "marginal_warm" else 1, n_warm)
                meds.append(float(np.median(r["cpu"])))
                params = r["params"]
            st = summ(meds)
            row["params"] = params
            row[reading] = {
                "cpu_s_per_application": st,
                "budget_s": budget[reading],
                "over_budget_factor": st["median"] / budget[reading],
                "under_budget": bool(st["median"] <= budget[reading]),
                "speedup_vs_solver": (sw if reading == "marginal_warm" else sc)
                / st["median"],
            }
        rows[label] = row
        print(f"{label:16} params {row['params']:7d}  warm "
              f"{row['marginal_warm']['cpu_s_per_application']['median']*1e6:8.0f} us "
              f"({row['marginal_warm']['speedup_vs_solver']:7.1f}x)  cold "
              f"{row['cold_single_wafer']['cpu_s_per_application']['median']*1e6:8.0f} us "
              f"({row['cold_single_wafer']['speedup_vs_solver']:7.1f}x)", flush=True)

    passing = {r: [k for k, v in rows.items() if v[r]["under_budget"]]
               for r in ("marginal_warm", "cold_single_wafer")}
    best = {r: max(rows.items(), key=lambda kv: kv[1][r]["speedup_vs_solver"])
            for r in ("marginal_warm", "cold_single_wafer")}
    res = {
        "question": "does a compact operator clear the 1000x clause once it "
                    "actually takes a recipe? cost_floor.json's 748x row does "
                    "not condition on one.",
        "protocol": {
            "estimator": "median CPU-seconds per application over timed calls "
                         "inside a process, then the spread of that median "
                         "across separate processes",
            "reps_processes": a.reps, "calls_per_process": a.calls,
            "why_repeats": "this box's between-process spread is 1.37x "
                           "(runs/speed_spread.json), so one draw is not a "
                           "measurement -- the same mistake cost_floor.json's "
                           "47us/33us surface rows made",
            "applications_per_wafer": 1,
            "budget_source": f"{a.sym}, solver CPU-seconds / 1000",
            "budget": budget,
            "no_accuracy_claimed": "every row is a price. Whether a model this "
                                   "small reaches rel-L2 <= 0.05 is not measured "
                                   "here and may not be inferred from here.",
            "what_differs_from_cost_floor": "cost_floor.json's conv rows are "
                                            "Conv2d(1,8,3)->Conv2d(8,1,3) on a "
                                            "bare field: no recipe embedding, no "
                                            "broadcast, no coordinate channels. "
                                            "CompactCNN keeps all three and the "
                                            "residual form, so it is priceable as "
                                            "a surrogate.",
        },
        "models": rows,
        "verdict": {
            "configs_under_budget_warm": passing["marginal_warm"],
            "configs_under_budget_cold": passing["cold_single_wafer"],
            "best_warm": {"config": best["marginal_warm"][0],
                          "speedup": best["marginal_warm"][1]["marginal_warm"]["speedup_vs_solver"]},
            "best_cold": {"config": best["cold_single_wafer"][0],
                          "speedup": best["cold_single_wafer"][1]["cold_single_wafer"]["speedup_vs_solver"]},
            "clause_reachable_on_cost": bool(passing["marginal_warm"]
                                             or passing["cold_single_wafer"]),
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdict"], indent=2))


if __name__ == "__main__":
    main()
