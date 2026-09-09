"""How many generation workers actually maximise throughput on this box.

The first attempt used 90 workers and completed ~175 trajectories in 10 minutes
-- about 300 core-seconds each, against 4.7 s measured serially. ViennaPS's flux
solver is a Monte Carlo ray trace with scattered memory access, so it is
plausibly bandwidth-bound rather than compute-bound, in which case more workers
buy nothing and cost everyone else on the machine. This measures it instead of
assuming it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = """
import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, {root!r})
import numpy as np
from multiprocessing import Pool
from eot import solver as S

def one(seed):
    rng = np.random.default_rng(seed)
    rec = S.sample_recipe(rng)
    t0 = time.perf_counter()
    S.simulate_adaptive(rec, rng, n_steps=10)
    return time.perf_counter() - t0

if __name__ == "__main__":
    w = {w}
    n = {n}
    t0 = time.perf_counter()
    with Pool(w) as p:
        ts = p.map(one, range(50000, 50000 + n))
    wall = time.perf_counter() - t0
    print(json.dumps({{"workers": w, "n": n, "wall_s": wall,
                       "traj_per_s": n / wall,
                       "per_traj_wall_s_mean": float(np.mean(ts))}}))
"""


def main():
    out = []
    for w, n in [(1, 4), (8, 24), (24, 48), (48, 96), (90, 180)]:
        src = CODE.format(root=str(ROOT), w=w, n=n)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="1"), cwd=str(ROOT))
        if r.returncode != 0:
            print("FAILED", w, r.stderr[-1500:])
            continue
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        rec["load_before"] = os.getloadavg()[0]
        out.append(rec)
        print(json.dumps(rec), flush=True)
    Path(ROOT / "runs/worker_scaling.json").write_text(json.dumps(out, indent=2))
    best = max(out, key=lambda r: r["traj_per_s"])
    print("best:", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
