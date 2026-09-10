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


# ATen calls that return a VIEW or otherwise move no data. Counting these as
# operations is what produced two wrong versions of this file's headline number
# in one turn: a `slice` of a (1, 128, 65) spectrum has a field-shaped output and
# costs nothing, and `SpectralPropagator` issues eight of them.
VIEW_OPS = {
    "slice", "select", "view", "reshape", "expand", "squeeze", "unsqueeze",
    "permute", "transpose", "t", "detach", "alias", "as_strided", "narrow",
    "unbind", "split", "chunk", "contiguous",
}


class Counted(TorchDispatchMode):
    """Records every ATen call, its field-shape class, and whether it moves data.

    Two things this instrument got wrong before, both caught by
    `tests/test_specprop.py` and by tracing the calls by hand:

    1. **Bucketing by element count counted weight matrices as fields.**
       `SpectralPropagator(modes_a=32)` has a (4096, 64) head weight, 262,144
       elements, so the transpose of that weight was a "full-field op". That
       inflated its count and manufactured an apparent op-count-versus-cost
       inversion against `modes_a=4`. The bucket is now decided by SHAPE.
    2. **Views were counted as operations.** Under a shape-based bucket, a
       `slice` of the (128, 65) half-spectrum is field-shaped and free;
       `SpectralPropagator` issues eight and `EtchOperator` sixteen. Counting
       them put `fno_w8m4L2` at 45 "full ops" when the number of calls that
       actually materialise a field is 25.

    `full_materialising` is the only count any document may quote, and it is
    the one the cost argument concerns: calls that write O(H*W) memory.
    """

    def __init__(self, n_grid=128):
        self.calls = []
        self.n_grid = n_grid

    def _is_field(self, t):
        """Field-shaped: the last two dims are the grid, or the grid's rfft2
        half-spectrum (n, n//2 + 1). A 2-D weight matrix never qualifies."""
        n = self.n_grid
        return (t.dim() >= 3 and t.shape[-2] == n
                and t.shape[-1] in (n, n // 2 + 1))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        short = str(func).split(".")[-2] if "." in str(func) else str(func)
        is_field = any(self._is_field(t)
                       for t in (out if isinstance(out, (tuple, list)) else [out])
                       if isinstance(t, torch.Tensor))
        self.calls.append({"op": short, "field": is_field,
                           "view": short in VIEW_OPS})
        return out


def bucket(calls):
    b = Counter()
    names = {"full_materialising": Counter(), "full_view": Counter(),
             "other": Counter()}
    for c in calls:
        if c["field"] and not c["view"]:
            k = "full_materialising"
        elif c["field"]:
            k = "full_view"
        else:
            k = "other"
        b[k] += 1
        names[k][c["op"]] += 1
    return b, names


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
    # Inputs are built OUTSIDE the dispatch context. Building them inside made
    # the harness's own `randn` a counted field operation in every row.
    phi = torch.randn(1, 1, n_grid, n_grid)
    cond = torch.randn(1, 7)
    with torch.no_grad():
        with Counted(n_grid) as c:
            m(phi, cond)
    b, names = bucket(c.calls)
    return {
        "params": sum(p.numel() for p in m.parameters()),
        "total_aten_calls": len(c.calls),
        "full_materialising": b["full_materialising"],
        "full_view_only": b["full_view"],
        "other_calls": b["other"],
        "full_materialising_names": dict(names["full_materialising"]),
        "full_view_names": dict(names["full_view"]),
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
        "specprop_vs_fno_field_materialising": (
            rows["specprop_m4_ma4"]["full_materialising"] / rows["fno_w8m4L2"]["full_materialising"]
            if rows["fno_w8m4L2"]["full_materialising"] else None),
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
                    "full_materialising": rows[ok]["full_materialising"],
                    "us_per_wafer": m["warm"]["per_wafer_cpu_s"] * 1e6,
                    "us_per_field_op": (m["warm"]["per_wafer_cpu_s"] * 1e6
                                        / max(rows[ok]["full_materialising"], 1)),
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
                k: {"op_ratio": ref["full_materialising"] / v["full_materialising"],
                    "cost_ratio": ref["us_per_wafer"] / v["us_per_wafer"],
                    "agreement": (ref["us_per_wafer"] / v["us_per_wafer"])
                    / (ref["full_materialising"] / v["full_materialising"])}
                for k, v in joined.items()
                if k != "fno_w8m4L2" and v["full_materialising"]
            } if ref else None,
            "caveat": "an agreement near 1 supports the mechanism; it does not "
                      "prove per-op cost is constant across op TYPES, and an "
                      "rfft2 on one channel is not a 1x1 conv on eight. The "
                      "us_per_full_op column shows the spread.",
        }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"{'model':26s} {'params':>10s} {'total':>6s} {'FIELD-MAT':>9s} "
          f"{'field-view':>10s} {'other':>6s}")
    for k, v in rows.items():
        print(f"{k:26s} {v['params']:10d} {v['total_aten_calls']:6d} "
              f"{v['full_materialising']:9d} {v['full_view_only']:10d} "
              f"{v['other_calls']:6d}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
