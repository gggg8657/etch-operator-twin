"""H25: does factorising the dominant stage buy the margin clause 2 needs?

    python scripts/rank_cost.py

**The bottleneck, measured first.** `runs/specprop_profile.json` times the
stages of a 128x128 `SpectralPropagator` forward: `_coeffs_a` is **243.1 us,
46.3%** of the whole, against 15.4% for every FFT combined. That stage is one
`nn.Linear(64 -> 16384)` holding **98.0%** of the model's parameters, and at
2.10 MFLOP in 243.1 us it runs at 8.63 GFLOP/s -- **arithmetic-bound**, which
is what distinguishes it from every previous cost attack in this repo. Those
failed (torch.compile 2.0-3.3x SLOWER, coarse spectral body 2.26x of a
predicted 16x, state head 87-135 us against an estimated 25-55) because the
cost was dispatch and the reasoning was arithmetic.

**The registered prediction**, from the MAC counts alone: a_rank 4/8/16 give
6.4x/3.2x/1.6x fewer MACs in the stage, hence roughly 1.66x/1.48x/1.22x on the
whole forward. Said in advance: **that is not the 1.9x clause 2 needs**, so a
smaller measured gain must not be retold as a success.

**The registered falsifier**: if a_rank=4 does not beat a_rank=16 in the stage
by roughly their 4x MAC ratio, the stage is not purely arithmetic-bound and
the premise is wrong.

Every row is priced in ONE invocation, because this weekend established that
between-invocation cost comparisons on this box are worthless: one unchanged
architecture read 350-901 us across 56 invocations, driven by a CPU clock that
explains 78% of the variance.

The script also fits `stage = floor + MACs / rate` across the ranks, which
turns "how much is left in this route" from an argument into a number.

Writes `runs/rank_cost.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from scripts.arch_cost import SPECS  # noqa: E402
from scripts.cost_floor import time_model  # noqa: E402

# MACs in the additive-coefficient stage, from LowRankCoeffHead's structure at
# modes_a=64, hidden=64: dense is hidden*2*(2ma)*ma; rank-r is the projection
# hidden*2r*(3ma) plus the outer products 2*(2ma)*ma*r.
MA, HID = 64, 64
MACS = {0: HID * 2 * (2 * MA) * MA}
for _r in (4, 8, 16):
    MACS[_r] = HID * 2 * _r * (3 * MA) + 2 * (2 * MA) * MA * _r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--profile", default="runs/specprop_profile.json")
    ap.add_argument("--out", default="runs/rank_cost.json")
    a = ap.parse_args()
    runlock.acquire(a.out, what="rank_cost")

    prof = json.loads(Path(a.profile).read_text())
    stage_anchor = prof["stages"]["04_coeffs_a"]["median_us"]
    fwd_anchor_prof = prof["stages"]["99_whole_forward"]["median_us"]
    # "Everything but the stage", taken from the profile and held fixed, so the
    # per-rank stage cost is a subtraction rather than a second profile run.
    other_us = fwd_anchor_prof - stage_anchor

    order = [(0, "specprop_m4_ma64_K10"), (16, "specprop_m4_ma64_r16_K10"),
             (8, "specprop_m4_ma64_r8_K10"), (4, "specprop_m4_ma64_r4_K10")]
    load0 = os.getloadavg()[0]
    rows = []
    for rank, name in order:
        r = time_model(SPECS[name], n_rep=a.rounds, n_warm=a.warm)
        cpu = np.asarray(r["cpu"], dtype=float)
        rows.append({"a_rank": rank, "config": name, "params": r["params"],
                     "forward_us": float(np.median(cpu)) * 1e6,
                     "within_spread": float(cpu.max() / cpu.min()),
                     "stage_macs": MACS[rank]})
        print(f"  a_rank={rank:2d}  {rows[-1]['forward_us']:7.1f} us  "
              f"{r['params']:>9d} params", flush=True)

    dense = next(x for x in rows if x["a_rank"] == 0)
    for x in rows:
        x["stage_us"] = x["forward_us"] - other_us
        x["speedup_vs_dense"] = dense["forward_us"] / x["forward_us"]
        x["mac_ratio_vs_dense"] = dense["stage_macs"] / x["stage_macs"]
        x["stage_time_ratio_vs_dense"] = dense["stage_us"] / x["stage_us"]
        x["translation_efficiency"] = (
            (x["stage_time_ratio_vs_dense"] - 1) / (x["mac_ratio_vs_dense"] - 1)
            if x["mac_ratio_vs_dense"] > 1 else None)

    # stage = floor + macs / rate, least squares over the three rank rows.
    lr = [x for x in rows if x["a_rank"] > 0]
    A = np.array([[1.0, x["stage_macs"]] for x in lr])
    y = np.array([x["stage_us"] for x in lr])
    floor_us, per_mac = np.linalg.lstsq(A, y, rcond=None)[0]
    pred_dense = floor_us + per_mac * dense["stage_macs"]

    r4 = next(x for x in rows if x["a_rank"] == 4)
    r16 = next(x for x in rows if x["a_rank"] == 16)
    falsifier_ratio = r16["stage_us"] / r4["stage_us"]
    mac_ratio = r4["mac_ratio_vs_dense"] / r16["mac_ratio_vs_dense"]

    # The bound this route can ever reach: rank 1, i.e. essentially the floor.
    best_stage = floor_us + per_mac * (HID * 2 * 1 * (3 * MA) + 2 * (2 * MA) * MA)
    best_fwd = other_us + best_stage

    res = {
        "hypothesis": "H25: the additive coefficient head is arithmetic-bound "
                      "and 98% of the parameters, so factorising it buys real "
                      "time -- unlike the four previous cost attacks here, "
                      "which assumed arithmetic where the cost was dispatch.",
        "predictions_registered_before_the_run": {
            "a_rank=4": "~1.66x on the whole forward",
            "a_rank=8": "~1.48x",
            "a_rank=16": "~1.22x",
            "stated_in_advance": "none of these is the 1.9x clause 2 needs; "
                                 "this is a necessary part of the margin, not "
                                 "the whole of it",
        },
        "falsifier_registered_before_the_run":
            "if a_rank=4 does not beat a_rank=16 in the STAGE by roughly their "
            "4x MAC ratio, the stage is not purely arithmetic-bound",
        "protocol": {
            "all_rows_in_one_invocation": True,
            "why": "one unchanged architecture read 350-901 us across 56 "
                   "invocations this weekend, with the CPU clock explaining "
                   "78% of the variance, so only within-invocation deltas "
                   "mean anything on this box",
            "rounds": a.rounds, "warm": a.warm,
            "loadavg_at_start": load0,
            "stage_cost_method":
                f"forward minus a fixed 'everything else' of {other_us:.1f} us, "
                f"taken from {a.profile} (anchor forward "
                f"{fwd_anchor_prof:.1f} us minus anchor stage "
                f"{stage_anchor:.1f} us). Subtraction, not a second profile.",
            "no_accuracy_claimed": "these are cost rows only. A rank "
                                   "constraint encodes a separability prior on "
                                   "the additive spectral response that "
                                   "nothing has measured.",
        },
        "rows": rows,
        "floor_model": {
            "form": "stage_us = floor_us + macs / rate",
            "floor_us": float(floor_us),
            "us_per_mac": float(per_mac),
            "implied_gmac_per_s": float(1.0 / per_mac / 1e3),
            "fitted_on": "the three rank rows",
            "check_predicts_dense_stage_us": float(pred_dense),
            "measured_dense_stage_us": dense["stage_us"],
            "dense_prediction_error_pct":
                float(100 * (pred_dense - dense["stage_us"]) / dense["stage_us"]),
        },
        "falsifier_outcome": {
            "mac_ratio_r16_over_r4": float(mac_ratio),
            "measured_stage_ratio_r16_over_r4": float(falsifier_ratio),
            "verdict": ("FIRES: the stage is only partially arithmetic-bound"
                        if falsifier_ratio < 0.75 * mac_ratio else
                        "does not fire: the stage scales with its MACs"),
        },
        "bound_on_this_route": {
            "best_possible_stage_us_at_rank_1": float(best_stage),
            "best_possible_forward_us": float(best_fwd),
            "best_possible_speedup": float(dense["forward_us"] / best_fwd),
            "reading": "even at rank 1 this route cannot reach the 1.9x that "
                       "clause 2's worst draw needs, because a floor in the "
                       "stage does not shrink with the rank.",
        },
    }
    Path(a.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'a_rank':>7s} {'params':>9s} {'fwd us':>8s} {'stage us':>9s} "
          f"{'MAC red':>8s} {'time red':>9s} {'transl':>7s} {'vs dense':>9s}")
    for x in rows:
        te = x["translation_efficiency"]
        print(f"{x['a_rank']:7d} {x['params']:9d} {x['forward_us']:8.1f} "
              f"{x['stage_us']:9.1f} {x['mac_ratio_vs_dense']:7.2f}x "
              f"{x['stage_time_ratio_vs_dense']:8.2f}x "
              f"{(f'{100*te:.0f}%' if te else '—'):>7s} "
              f"{x['speedup_vs_dense']:8.2f}x")
    print(f"\nfloor model: stage = {floor_us:.1f} us + MACs / "
          f"{1/per_mac/1e3:.2f} GMAC/s   "
          f"(predicts dense {pred_dense:.1f} vs measured "
          f"{dense['stage_us']:.1f} us)")
    print(f"falsifier: {res['falsifier_outcome']['verdict']} "
          f"(MAC 4.00x -> time {falsifier_ratio:.2f}x)")
    print(f"BOUND: even at rank 1 this route gives at most "
          f"{res['bound_on_this_route']['best_possible_speedup']:.2f}x")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
