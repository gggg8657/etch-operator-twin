"""H26: is train.py's CPU burn real work, or an idle intra-op pool?

    python scripts/thread_cost.py --rounds 3

`train.py` never pins its intra-op pool, so torch sizes it to every core on the
box, while `--num-workers 0` keeps `PairDataset.__getitem__` in the main process
where each item's small CPU tensor ops wake that pool. This times one fixed
config at several thread caps and reports, per cell, the wall-clock per epoch
and the CPU seconds actually consumed.

The point of the CPU-seconds column is that it does not depend on box load: a
cell that uses 20 cores to do 1 core of work reads 20x here whatever else is
running.

Cells are interleaved within a round so a drift in the box hits all of them
equally. Writes `runs/thread_cost.json`; touches no existing run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import runlock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# The architecture the live arms train, so the reading transfers to them.
ARCH = ["--arch", "specprop", "--modes", "4", "--modes-a", "64",
        "--state-modes", "0", "--stride", "10"]


def run_cell(threads: int, epochs: int, run_dir: Path, device: str, seed: int):
    """One training subprocess at a fixed intra-op thread cap.

    Returns wall seconds, CPU seconds and the per-epoch times it logged. CPU
    seconds come from getrusage on children, which counts every thread's time,
    so a spinning pool shows up even though it computes nothing.
    """
    if run_dir.exists():
        shutil.rmtree(run_dir)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    cmd = [sys.executable, "scripts/train.py", "--run", str(run_dir),
           "--epochs", str(epochs), "--device", device, "--seed", str(seed),
           *ARCH]

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime))

    if p.returncode != 0:
        raise SystemExit(f"train.py failed at threads={threads}:\n{p.stderr[-3000:]}")

    log = run_dir / "log.jsonl"
    eps = [json.loads(l)["epoch_s"] for l in log.read_text().splitlines()
           if l.strip() and "epoch_s" in l]
    torch_threads = None
    args_p = run_dir / "args.json"
    if args_p.exists():
        torch_threads = json.loads(args_p.read_text()).get("torch_threads")
    return {"threads_env": threads, "wall_s": wall, "cpu_s": cpu,
            "cores_used": cpu / wall if wall > 0 else None,
            "epoch_s_median": float(np.median(eps)) if eps else None,
            "n_epochs_logged": len(eps), "torch_threads_reported": torch_threads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cells", type=int, nargs="+", default=[0, 16, 4, 1],
                    help="intra-op thread caps; 0 means leave unset (torch default)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/thread_cost.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="thread_cost")

    ncpu = os.cpu_count()
    cells = [ncpu if c == 0 else c for c in a.cells]
    scratch = ROOT / "runs" / "threadcost"
    scratch.mkdir(parents=True, exist_ok=True)

    load0 = os.getloadavg()[0]
    raw = []
    for r in range(a.rounds):
        for c in cells:  # interleaved: a drift in the box hits every cell
            rec = run_cell(c, a.epochs, scratch / f"t{c}_r{r}", a.device, seed=r)
            rec["round"] = r
            raw.append(rec)
            print(f"  round {r}  threads={c:>3}  epoch {rec['epoch_s_median']*1e3:7.1f} ms"
                  f"  cores {rec['cores_used']:6.1f}", flush=True)

    rows = []
    for c in cells:
        g = [x for x in raw if x["threads_env"] == c]
        rows.append({
            "threads": c,
            "epoch_s_median": float(np.median([x["epoch_s_median"] for x in g])),
            "cores_used_median": float(np.median([x["cores_used"] for x in g])),
            "cpu_s_median": float(np.median([x["cpu_s"] for x in g])),
            "wall_s_median": float(np.median([x["wall_s"] for x in g])),
            "n": len(g),
        })

    default = next(x for x in rows if x["threads"] == ncpu)
    one = next(x for x in rows if x["threads"] == 1)
    for x in rows:
        x["epoch_vs_default"] = x["epoch_s_median"] / default["epoch_s_median"]
        x["cores_vs_default"] = x["cores_used_median"] / default["cores_used_median"]

    slowdown = one["epoch_s_median"] / default["epoch_s_median"]
    res = {
        "hypothesis": "H26: train.py's CPU burn is an idle intra-op pool woken "
                      "by num_workers=0 collate, not work. Capping threads "
                      "leaves epoch wall-clock unchanged and cuts cores used.",
        "predictions_registered_before_run": {
            "1_epoch_within_15pct_at_1_thread": True,
            "2_cores_used_falls_with_cap": True,
            "3_falsifier": "median epoch_s rises >15% from default to 1 thread",
        },
        "protocol": {
            "arch": " ".join(ARCH), "epochs": a.epochs, "rounds": a.rounds,
            "device": a.device, "interleaved": True,
            "cpu_seconds_source": "getrusage(RUSAGE_CHILDREN) delta per subprocess",
            "ncpu": ncpu, "loadavg_at_start": load0,
            "loadavg_at_end": os.getloadavg()[0],
        },
        "cells": rows,
        "raw": raw,
        "one_thread_slowdown_vs_default": slowdown,
        "falsifier_fired": bool(slowdown > 1.15),
        "cores_saved_per_process": default["cores_used_median"] - one["cores_used_median"],
        "verdict": ("FALSIFIED: capping threads costs real throughput"
                    if slowdown > 1.15 else
                    "CONFIRMED: the pool was idle; capping it is free"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in
                      ["one_thread_slowdown_vs_default", "falsifier_fired",
                       "cores_saved_per_process", "verdict"]}, indent=2))


if __name__ == "__main__":
    main()
