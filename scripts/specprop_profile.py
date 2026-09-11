"""Where do the ~350-600 us of a SpectralPropagator forward actually go?

    python scripts/specprop_profile.py

**Why this exists.** `codex`, asked the rung-4 question, proposed two routes to
the ~1.9x margin clause 2 needs. One of them is already dead and this script
exists because the other one needs a target:

* **"Replace any remaining full-grid erf-GELU with ReLU": worth exactly zero
  here.** `SpectralPropagator` has no full-grid activation at all. Instrumenting
  every activation module during one 128x128 forward finds two GELUs, both on
  **64-element** vectors inside `h_head` and `a_head`. The 152.7 us erf-GELU
  figure it reasoned from is real, but it was measured on the multiscale/FNO
  family, and I handed it to codex in a list of "prior measured facts" without
  saying which architecture each fact belonged to. Its arithmetic was right and
  its premise was mine and wrong. `tests/test_no_full_grid_activation.py` pins
  the fact so the proposal cannot be re-made.
* **"An exact, specialized native CPU forward pass [...] actual fused loops and
  reusable workspaces": still live**, and it needs to know which stage to fuse.

So this measures the stages rather than reasoning about them, which is the
mistake this repo has now made four times (torch.compile predicted to help and
2-3.3x slower; a coarse spectral body predicted 16x and worth 2.26x; codex's
state-head predicted at 25-55 us and costing 87-135; the erf-GELU above).

Each stage is timed in isolation on the same tensors the real forward uses, in
a fresh subprocess on one thread. Stage times do NOT sum to the whole -- they
exclude dispatch between stages and each pays its own timing overhead -- so the
whole is measured too and the shortfall is reported rather than hidden.

Writes `runs/specprop_profile.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402

_CHILD = '''
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time, json, sys
sys.path.insert(0, %r)
import torch
torch.set_num_threads(1)
from eot.operator import SpectralPropagator

H = W = %d
N = %d
M = SpectralPropagator(cond_dim=7, modes=4, modes_a=64).eval()
torch.manual_seed(0)
phi = torch.randn(1, 1, H, W); cond = torch.randn(1, 7)

def bench(fn, n=N, warm=20):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        cpu = []
        for _ in range(n):
            c0 = time.process_time(); fn(); cpu.append(time.process_time() - c0)
    return float(np.median(cpu)) if False else sorted(cpu)[len(cpu)//2]

import numpy as np
res = {}
with torch.no_grad():
    f = torch.fft.rfft2(phi.squeeze(1).float())
    out = torch.zeros_like(f)
    ca = M._cond_a(cond.float(), f)
    ac = M._coeffs(ca, M.modes_a, M.a_head)
    hc = M._coeffs(cond.float(), M.modes, M.h_head)

res["01_rfft2"]        = bench(lambda: torch.fft.rfft2(phi.squeeze(1).float()))
res["02_zeros_like"]   = bench(lambda: torch.zeros_like(f))
res["03_cond_a"]       = bench(lambda: M._cond_a(cond.float(), f))
res["04_coeffs_a"]     = bench(lambda: M._coeffs(ca, M.modes_a, M.a_head))
res["05_coeffs_h"]     = bench(lambda: M._coeffs(cond.float(), M.modes, M.h_head))
res["06_hermitian"]    = bench(lambda: M._hermitian_self_conjugate_columns(out.clone(), H, W))
res["07_irfft2"]       = bench(lambda: torch.fft.irfft2(out, s=(H, W)))
res["08_add_residual"] = bench(lambda: phi + phi)
res["99_whole_forward"] = bench(lambda: M(phi, cond))
res["98_whole_residual"] = bench(lambda: M.residual(phi, cond))
print(json.dumps(res))
''' % (str(ROOT), 128, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="runs/specprop_profile.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="specprop_profile")

    runs = []
    for i in range(a.repeats):
        env = dict(os.environ, OMP_NUM_THREADS="1")
        o = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True,
                           text=True, env=env, cwd=ROOT)
        if o.returncode != 0:
            raise RuntimeError(o.stderr[-3000:])
        runs.append(json.loads(o.stdout.strip().splitlines()[-1]))
        print(f"  repeat {i+1}/{a.repeats}", flush=True)

    keys = sorted(runs[0])
    stages = {k: {"median_us": float(np.median([r[k] for r in runs]) * 1e6),
                  "min_us": float(np.min([r[k] for r in runs]) * 1e6),
                  "max_us": float(np.max([r[k] for r in runs]) * 1e6)}
              for k in keys}
    whole = stages["99_whole_forward"]["median_us"]
    parts = {k: v for k, v in stages.items() if not k.startswith(("99", "98"))}
    summed = sum(v["median_us"] for v in parts.values())
    for k, v in stages.items():
        v["pct_of_forward"] = 100.0 * v["median_us"] / whole if whole else None

    res = {
        "question": "which stage of a SpectralPropagator forward carries the "
                    "cost that clause 2 needs 1.9x of?",
        "dead_route_recorded_here":
            "codex's 'replace the full-grid erf-GELU with ReLU' is worth ZERO "
            "for this architecture: instrumenting every activation module "
            "during one 128x128 forward finds two GELUs, both on 64-element "
            "vectors. The 152.7 us erf-GELU figure belongs to the "
            "multiscale/FNO family. I gave codex that fact without saying "
            "which architecture it came from, so the bad premise is mine.",
        "protocol": {
            "grid": 128, "inner_reps": 400, "warmup": 20,
            "repeats": a.repeats,
            "estimator": "CPU-seconds, one thread, fresh subprocess per repeat",
            "stages_do_not_sum":
                "stage timings exclude inter-stage dispatch and each pays its "
                "own timing overhead, so they are not expected to sum to the "
                "whole. The shortfall is reported, not hidden.",
        },
        "stages": stages,
        "accounting": {
            "whole_forward_us": whole,
            "whole_residual_us": stages["98_whole_residual"]["median_us"],
            "sum_of_stages_us": summed,
            "unaccounted_us": whole - summed,
            "unaccounted_pct": 100.0 * (whole - summed) / whole if whole else None,
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'stage':24s} {'median us':>10s} {'% of fwd':>9s}")
    for k in keys:
        v = stages[k]
        print(f"{k:24s} {v['median_us']:10.1f} {v['pct_of_forward']:8.1f}%")
    print(f"\nsum of stages {summed:.1f} us vs whole forward {whole:.1f} us "
          f"-> unaccounted {whole - summed:.1f} us "
          f"({100.0 * (whole - summed) / whole:.1f}%)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
