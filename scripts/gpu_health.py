"""Measure what the leased GPUs actually deliver before trusting any timing.

    python scripts/gpu_health.py --devices cuda:0 cuda:1

Track beta lost a headline to this: a device carrying leaked CUDA contexts
delivered 2.88 TFLOP/s where a clean one delivered 44.07, and a speedup measured
on it would have been wrong by 15x. Clause 2 of this KPI is a timing claim, so
the health of the device it was timed on is part of the measurement, not
background. Writes `runs/gpu_health.json`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch


def probe(dev, n=8192, iters=10, warmup=5):
    torch.cuda.set_device(dev)
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        a @ b
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    fl = 2 * n ** 3
    return {"device": dev, "matmul_n": n, "iters": iters,
            "median_tflops": fl / ts[len(ts) // 2] / 1e12,
            "best_tflops": fl / ts[0] / 1e12,
            "worst_tflops": fl / ts[-1] / 1e12,
            "spread_pct": 100.0 * (ts[-1] - ts[0]) / ts[len(ts) // 2]}


def foreign_contexts():
    """Compute apps nvidia-smi can see. Ours are usually invisible from here, so
    every row this returns is somebody else's context on a device we lease."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        uu = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # nvidia-smi absent or wedged
        return {"error": str(e)}
    idx = {}
    for line in uu.strip().splitlines():
        i, u, m, g = [x.strip() for x in line.split(",")]
        idx[u] = {"index": int(i), "memory_used": m, "utilization": g}
    apps = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        p, m, u = [x.strip() for x in line.split(",")]
        apps.append({"pid": int(p), "used_memory": m,
                     "gpu_index": idx.get(u, {}).get("index")})
    return {"devices": sorted(idx.values(), key=lambda d: d["index"]), "apps": apps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    ap.add_argument("--out", default="runs/gpu_health.json")
    a = ap.parse_args()
    out = {"lease": a.devices, "nvidia_smi": foreign_contexts(),
           "probes": [probe(d) for d in a.devices]}
    med = [p["median_tflops"] for p in out["probes"]]
    out["ratio_fastest_to_slowest"] = max(med) / min(med) if min(med) > 0 else None
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
