"""Count the ATen operations each architecture dispatches. Measured, not asserted.

    python scripts/op_count.py

**Why this exists.** H18's entire justification is that clause 2's cost is
per-operation rather than arithmetic -- the spectral body carries ~0.1 MFLOP on
8x32x32 tensors and costs 645 us, about 80x what this box's measured ~13
GFLOP/s implies -- and that `SpectralPropagator` therefore wins by doing "four
full-resolution operations instead of twenty". That count was an assertion. It
was written into a docstring, a commit message, `WEEKEND.md` and the board
before anybody counted.

So this counts them, with `TorchDispatchMode`, which sees every ATen call the
model actually makes rather than every line of Python that looks like one.

**The count is split by tensor SHAPE, and an earlier version split it by
element count, which was wrong.** An operation on a (1, 7) conditioning vector
costs nothing per pixel and must not be pooled with one on a 128x128 field.
Bucketing by `numel` seemed to do that, and `tests/test_specprop.py` caught it
failing: `SpectralPropagator(modes_a=32)` has a conditioning head whose final
weight is (4096, 64) = 262,144 elements, so the **transpose of that weight
matrix** was counted as a full-field operation. Two of the six "full" ops
attributed to `specprop_m8_ma32` were weight transposes inside a
grid-independent MLP, and that -- not any real extra field work -- is why it
appeared to dispatch more full ops than `modes_a=4` while measuring cheaper.

So the bucket is now decided by shape: an output is field-shaped if its last two
dimensions are (n_grid, n_grid) or (n_grid, n_grid // 2 + 1), the latter being
the half-spectrum an `rfft2` returns. Everything else is `coarse` if it still has
a spatial pair smaller than the grid, and `tiny` otherwise -- which is where a
weight matrix of any size now lands, because a weight is not a field.

`full_ops` is the number the H18 argument rests on, and it is the only one this
file lets a document quote.

Writes `runs/op_count.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from eot.operator import (CompactCNN, EtchOperator,  # noqa: E402
                          MultiScaleOperator, SpectralPropagator)


class Counted(TorchDispatchMode):
    """Records every ATen call and the element count of its largest output."""

    def __init__(self, n_grid=128):
        self.calls = []
        self.n_grid = n_grid

    def _class_of(self, t):
        """field / coarse / tiny, by SHAPE. A weight matrix is not a field."""
        n = self.n_grid
        if t.dim() >= 2:
            h, w = t.shape[-2], t.shape[-1]
            if h == n and w in (n, n // 2 + 1):
                return "full"
            # A spatial pair smaller than the grid: a downsampled body.
            if 1 < h < n and 1 < w <= n // 2 + 1 and h == w or \
               (1 < h < n and 1 < w < n and t.dim() >= 3):
                return "coarse"
        return "tiny"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        rank = {"tiny": 0, "coarse": 1, "full": 2}
        klass = "tiny"
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if isinstance(t, torch.Tensor):
                c = self._class_of(t)
                if rank[c] > rank[klass]:
                    klass = c
        self.calls.append((str(func), klass))
        return out


def bucket(calls, _unused=None):
    b = Counter()
    per_op = {"full": Counter(), "coarse": Counter(), "tiny": Counter()}
    for name, k in calls:
        b[k] += 1
        per_op[k][name.split(".")[-2] if "." in name else name] += 1
    return b, per_op


MODELS = {
    "fno_w64m20L4_DEPLOYED": lambda: EtchOperator(cond_dim=7, width=64, modes=20,
                                                  n_layers=4),
    "fno_w8m4L2": lambda: EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2),
    "compactcnn_w8L2": lambda: CompactCNN(cond_dim=7, width=8, n_layers=2),
    "multiscale_s4_wf8": lambda: MultiScaleOperator(cond_dim=7, width=8, modes=4,
                                                    n_layers=2, width_full=8,
                                                    scale=4),
    "pointwise_wf8_n1": lambda: MultiScaleOperator(cond_dim=7, width=8, modes=4,
                                                   n_layers=2, width_full=8,
                                                   scale=0, n_local=1,
                                                   act="relu"),
    "specprop_m4_ma4": lambda: SpectralPropagator(cond_dim=7, modes=4, modes_a=4),
    "specprop_m8_ma32": lambda: SpectralPropagator(cond_dim=7, modes=8,
                                                   modes_a=32),
}


def count(build, n_grid=128):
    m = build()
    m.eval()
    phi = torch.randn(1, 1, n_grid, n_grid)
    cond = torch.randn(1, 7)
    with torch.no_grad():
        with Counted(n_grid) as c:
            m(phi, cond)
    b, per_op = bucket(c.calls)
    return {
        "params": sum(p.numel() for p in m.parameters()),
        "total_aten_calls": len(c.calls),
        "full_ops": b["full"],
        "coarse_ops": b["coarse"],
        "tiny_ops": b["tiny"],
        "full_op_names": dict(per_op["full"]),
        "coarse_op_names": dict(per_op["coarse"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--out", default="runs/op_count.json")
    ap.add_argument("--cost", default="runs/arch_cost_h18.json",
                    help="cost file to join, so the MECHANISM can be tested: "
                         "if cost is op-count-bound, the cost ratio between two "
                         "models should track their full-op ratio")
    a = ap.parse_args()

    runlock.acquire(a.out, what="op_count")
    rows = {k: count(v, a.n_grid) for k, v in MODELS.items()}
    res = {
        "question": "how many ATen operations does each architecture dispatch, "
                    "and how many of them touch a full 128x128 field",
        "protocol": {
            "instrument": "torch.utils._python_dispatch.TorchDispatchMode, one "
                          "forward pass, batch 1, inference_mode off but "
                          "no_grad on",
            "n_grid": a.n_grid,
            "bucketing": f"by SHAPE: full if an output's last two dims are "
                         f"({a.n_grid},{a.n_grid}) or "
                         f"({a.n_grid},{a.n_grid // 2 + 1}) -- the rfft2 "
                         "half-spectrum -- coarse if it has a smaller spatial "
                         "pair, tiny otherwise",
            "why_not_numel": "bucketing by element count counted the TRANSPOSE "
                             "OF A WEIGHT MATRIX as a full-field op: "
                             "SpectralPropagator(modes_a=32) has a (4096, 64) "
                             "head weight, 262,144 elements. That inflated its "
                             "full_ops from 4 to 6 and manufactured an apparent "
                             "op-count-versus-cost inversion. Caught by "
                             "tests/test_specprop.py.",
            "why_bucketed": "an op on a (1,7) recipe vector has no per-pixel "
                            "cost and must not be pooled with one on a field. "
                            "full_ops is the quantity the H18 argument rests on.",
            "not_a_cost": "an operation count is not a cost. Costs are in "
                          "runs/arch_cost.json, measured in CPU-seconds. This "
                          "file exists because the count was being asserted.",
        },
        "models": rows,
        "specprop_vs_fno_full_ops": (
            rows["specprop_m4_ma4"]["full_ops"] / rows["fno_w8m4L2"]["full_ops"]
            if rows["fno_w8m4L2"]["full_ops"] else None),
    }

    # THE MECHANISM TEST. H18 claims the cost is bound by the operation count.
    # That claim predicts something checkable: between two models measured in
    # the same invocation, the cost ratio should track the full-op ratio. If it
    # does not, the count is a coincidence and the argument is decoration.
    cost_p = Path(a.cost)
    if cost_p.exists():
        cost = json.loads(cost_p.read_text())
        name_map = {"specprop_m4_ma4": "specprop_m4_ma4_K10",
                    "specprop_m8_ma32": "specprop_m8_ma32_K10",
                    "fno_w8m4L2": "fno_w8m4L2_K10",
                    "pointwise_wf8_n1": "pw_wf8_n1_relu_K10"}
        joined = {}
        for ok, ck in name_map.items():
            m = (cost.get("models") or {}).get(ck)
            if m and ok in rows:
                joined[ok] = {
                    "full_ops": rows[ok]["full_ops"],
                    "us_per_wafer": m["warm"]["per_wafer_cpu_s"] * 1e6,
                    "us_per_full_op": (m["warm"]["per_wafer_cpu_s"] * 1e6
                                       / max(rows[ok]["full_ops"], 1)),
                }
        ref = joined.get("fno_w8m4L2")
        res["mechanism_test"] = {
            "claim": "cost is bound by the count of full-resolution operations, "
                     "not by arithmetic. Predicts that between two models the "
                     "cost ratio tracks the full-op ratio.",
            "same_invocation": cost_p.name,
            "loadavg": cost["protocol"]["budget_provenance"].get(
                "loadavg_1min_at_start"),
            "per_model": joined,
            "vs_fno_w8m4L2": {
                k: {"op_ratio": ref["full_ops"] / v["full_ops"],
                    "cost_ratio": ref["us_per_wafer"] / v["us_per_wafer"],
                    "agreement": (ref["us_per_wafer"] / v["us_per_wafer"])
                    / (ref["full_ops"] / v["full_ops"])}
                for k, v in joined.items()
                if k != "fno_w8m4L2" and v["full_ops"]
            } if ref else None,
            "caveat": "an agreement near 1 supports the mechanism; it does not "
                      "prove per-op cost is constant across op TYPES, and an "
                      "rfft2 on one channel is not a 1x1 conv on eight. The "
                      "us_per_full_op column shows the spread.",
        }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"{'model':26s} {'params':>10s} {'total':>6s} {'full':>5s} {'coarse':>7s} {'tiny':>5s}")
    for k, v in rows.items():
        print(f"{k:26s} {v['params']:10d} {v['total_aten_calls']:6d} "
              f"{v['full_ops']:5d} {v['coarse_ops']:7d} {v['tiny_ops']:5d}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
