"""H25: this operator's cost is weight BYTES STREAMED, not arithmetic.

    python scripts/bandwidth_bound.py

**What forced this.** `runs/specprop_profile.json` times the stages of a
`specprop_m4_ma64` forward and finds one stage carrying **46.3%** of it:
`_coeffs(cond, modes_a, a_head)` at **243.1 us** of a 525.5 us forward. That
stage is dominated by one `nn.Linear(64 -> 16384)`, which holds **1,048,576 of
the model's 1,070,144 parameters** -- 4.19 MB of the model's 4.28 MB.

At batch 1 that layer is a matvec: every weight is read exactly once and used
exactly once, an arithmetic intensity of 2 FLOP per 4 bytes = **0.5 FLOP/byte**.
The two ways of reading the same 243.1 us are:

* arithmetic-bound: **8.6 GFLOP/s** -- implausibly low for one AVX-512 core;
* bandwidth-bound: **17.3 GB/s** -- an entirely ordinary single-core streaming
  rate on a Xeon 8558.

**Why this matters beyond one stage.** If it holds, it is the cost model this
repo has been missing for four turns, and it retroactively explains every cost
surprise in the log: `torch.compile` was 2.0-3.3x SLOWER because fusing
arithmetic does not reduce weight traffic; the coarse-grid spectral body bought
2.26x of a predicted 16x because shrinking the GRID does not shrink the
WEIGHTS; and "per-field-op cost is not constant, it spans 56-118 us" is exactly
what you see when the binding resource is bytes rather than operations.

It also decides which clause-2 margin routes are worth building. `codex`
proposed a native fused CPU forward with reusable workspaces; fusion cannot
beat a bandwidth bound, so under H25 that route is capped at the 14.2%
unaccounted dispatch plus the small stages, not the 46.3%.

**THE DISCRIMINATING TEST, and it is the reason this script is not just a
re-timing.** Vary the layer's shape and dtype and compute BOTH implied rates:

* if **arithmetic-bound**, implied GFLOP/s is roughly CONSTANT across configs
  and implied GB/s varies;
* if **bandwidth-bound**, implied GB/s is roughly CONSTANT and implied GFLOP/s
  varies.

They cannot both be flat. Whichever is flat names the binding resource.

**Predictions, registered before the run:**

1. implied GB/s is flat to within ~1.5x across every config; implied GFLOP/s
   spans more than 3x.
2. `bfloat16` weights halve the bytes and cut the time by >= 1.6x.
3. the fitted line through (bytes, time) has a non-trivial slope and a small
   intercept -- i.e. bytes explain the time, not a fixed per-call overhead.
4. holding bytes fixed while changing the output shape (64->16384 vs
   128->8192, both 1,048,576 weights) changes the time by less than 1.25x.

Prediction 4 is the cleanest one: same bytes, same MACs, different shape. If
the time tracks bytes it should barely move.

**This measures COST ONLY.** `bfloat16` changes numerics and no accuracy is
claimed for it anywhere here; if it is ever adopted the accuracy must be
re-measured under the same protocol as every other arm.

Writes `runs/bandwidth_bound.json`.
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
import numpy as np
import torch
torch.set_num_threads(1)

CFGS = %s
N = %d
out = []
for cfg in CFGS:
    hid, nout, dt = cfg["hidden"], cfg["n_out"], cfg["dtype"]
    tdt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
           "float16": torch.float16}[dt]
    lin = torch.nn.Linear(hid, nout).to(tdt).eval()
    x = torch.randn(1, hid, dtype=tdt)
    with torch.no_grad():
        for _ in range(20):
            lin(x)
        cpu = []
        for _ in range(N):
            c0 = time.process_time(); lin(x); cpu.append(time.process_time() - c0)
    t = float(np.median(cpu))
    wbytes = lin.weight.numel() * lin.weight.element_size()
    macs = hid * nout
    out.append({**cfg, "weight_bytes": wbytes, "macs": macs,
                "median_s": t,
                "implied_gbps": wbytes / t / 1e9,
                "implied_gflops": 2 * macs / t / 1e9})
print(json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--inner", type=int, default=300)
    ap.add_argument("--out", default="runs/bandwidth_bound.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="bandwidth_bound")

    # The real layer, then the axes that separate bytes from arithmetic.
    cfgs = [
        # the actual a_head output projection
        dict(name="a_head_real_fp32", hidden=64, n_out=16384, dtype="float32"),
        # same layer, half the bytes, SAME MACs  -> prediction 2
        dict(name="a_head_bf16", hidden=64, n_out=16384, dtype="bfloat16"),
        dict(name="a_head_fp16", hidden=64, n_out=16384, dtype="float16"),
        # same bytes and MACs, different shape   -> prediction 4
        dict(name="same_bytes_shape_128x8192", hidden=128, n_out=8192,
             dtype="float32"),
        dict(name="same_bytes_shape_32x32768", hidden=32, n_out=32768,
             dtype="float32"),
        # the size sweep: bytes scale 1/8 .. 2x  -> predictions 1 and 3
        dict(name="hidden_8", hidden=8, n_out=16384, dtype="float32"),
        dict(name="hidden_16", hidden=16, n_out=16384, dtype="float32"),
        dict(name="hidden_32", hidden=32, n_out=16384, dtype="float32"),
        dict(name="hidden_128", hidden=128, n_out=16384, dtype="float32"),
        # what modes_a would do if it could be reduced (it cannot: accuracy)
        dict(name="ma32_equivalent", hidden=64, n_out=4096, dtype="float32"),
        dict(name="ma16_equivalent", hidden=64, n_out=1024, dtype="float32"),
    ]

    runs = []
    for i in range(a.repeats):
        body = _CHILD % (json.dumps(cfgs), a.inner)
        env = dict(os.environ, OMP_NUM_THREADS="1")
        o = subprocess.run([sys.executable, "-c", body], capture_output=True,
                           text=True, env=env, cwd=ROOT)
        if o.returncode != 0:
            raise RuntimeError(o.stderr[-3000:])
        runs.append(json.loads(o.stdout.strip().splitlines()[-1]))
        print(f"  repeat {i+1}/{a.repeats}", flush=True)

    rows = []
    for j, cfg in enumerate(cfgs):
        ts = np.array([r[j]["median_s"] for r in runs])
        r0 = runs[0][j]
        rows.append({
            **cfg,
            "weight_bytes": r0["weight_bytes"], "macs": r0["macs"],
            "median_us": float(np.median(ts) * 1e6),
            "min_us": float(ts.min() * 1e6), "max_us": float(ts.max() * 1e6),
            "implied_gbps": float(r0["weight_bytes"] / np.median(ts) / 1e9),
            "implied_gflops": float(2 * r0["macs"] / np.median(ts) / 1e9),
        })

    gb = np.array([r["implied_gbps"] for r in rows])
    gf = np.array([r["implied_gflops"] for r in rows])
    by = np.array([r["weight_bytes"] for r in rows], dtype=float)
    tt = np.array([r["median_us"] for r in rows]) * 1e-6
    slope, intercept = np.polyfit(by, tt, 1)
    pred = slope * by + intercept
    ss_res = float(((tt - pred) ** 2).sum())
    ss_tot = float(((tt - tt.mean()) ** 2).sum())

    def _spread(x):
        return float(x.max() / x.min())

    fp32 = next(r for r in rows if r["name"] == "a_head_real_fp32")
    bf16 = next(r for r in rows if r["name"] == "a_head_bf16")
    shape_rows = [r for r in rows if r["weight_bytes"] == fp32["weight_bytes"]
                  and r["dtype"] == "float32"]
    shape_spread = (max(r["median_us"] for r in shape_rows) /
                    min(r["median_us"] for r in shape_rows))

    verdict = ("BANDWIDTH-bound: implied GB/s is the flatter of the two"
               if _spread(gb) < _spread(gf) else
               "ARITHMETIC-bound: implied GFLOP/s is the flatter of the two")

    res = {
        "hypothesis": "H25: this operator's cost is weight BYTES STREAMED, not "
                      "arithmetic. At batch 1 the dominant layer is a matvec "
                      "at 0.5 FLOP/byte, so the binding resource should be "
                      "memory, and 243.1 us for 4.19 MB implies 17.3 GB/s "
                      "(ordinary) against 8.6 GFLOP/s (implausibly low).",
        "discriminating_test":
            "implied GB/s and implied GFLOP/s cannot both be flat across "
            "configs. Whichever is flat names the binding resource.",
        "predictions_registered_before_the_run": [
            "implied GB/s flat to within ~1.5x; implied GFLOP/s spans >3x",
            "bfloat16 halves the bytes and cuts the time by >=1.6x",
            "time is linear in bytes with a small intercept",
            "same bytes + same MACs, different shape: time moves <1.25x",
        ],
        "prediction_outcomes": {
            "1_gbps_flatter": {
                "gbps_spread": _spread(gb), "gflops_spread": _spread(gf),
                "outcome": ("CONFIRMED" if _spread(gb) < 1.5 and
                            _spread(gf) > 3.0 else "see numbers"),
            },
            "2_bf16_speedup": {
                "factor": fp32["median_us"] / bf16["median_us"],
                "outcome": ("CONFIRMED"
                            if fp32["median_us"] / bf16["median_us"] >= 1.6
                            else "FALSIFIED"),
            },
            "3_linear_in_bytes": {
                "slope_s_per_byte": float(slope),
                "implied_gbps_from_slope": float(1 / slope / 1e9),
                "intercept_us": float(intercept * 1e6),
                "r_squared": float(1 - ss_res / ss_tot) if ss_tot else None,
            },
            "4_shape_invariance_at_fixed_bytes": {
                "configs": [r["name"] for r in shape_rows],
                "spread": float(shape_spread),
                "outcome": "CONFIRMED" if shape_spread < 1.25 else "FALSIFIED",
            },
        },
        "protocol": {
            "estimator": "CPU-seconds, one thread, fresh subprocess per "
                         "repeat, batch 1, 20 untimed warmups",
            "inner_reps": a.inner, "repeats": a.repeats,
            "measures_cost_only": "bfloat16 changes numerics. No accuracy is "
                                  "claimed for any dtype here; adopting one "
                                  "would require re-measuring accuracy under "
                                  "the same protocol as every other arm.",
        },
        "rows": rows,
        "verdict": verdict,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'config':28s} {'bytes':>9s} {'us':>8s} {'GB/s':>7s} {'GFLOP/s':>8s}")
    for r in rows:
        print(f"{r['name']:28s} {r['weight_bytes']/1e6:8.2f}M "
              f"{r['median_us']:8.1f} {r['implied_gbps']:7.1f} "
              f"{r['implied_gflops']:8.1f}")
    print(f"\nimplied GB/s spread    {_spread(gb):.2f}x")
    print(f"implied GFLOP/s spread {_spread(gf):.2f}x")
    print(f"time ~ bytes: R^2 = {1 - ss_res/ss_tot:.4f}, "
          f"slope implies {1/slope/1e9:.1f} GB/s, "
          f"intercept {intercept*1e6:.1f} us")
    print(f"VERDICT: {verdict}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
