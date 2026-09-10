#!/usr/bin/env python
"""Is GD's win over random search bigger than the paired noise across targets?

`random_curve.py` compares two means. Two means differing by 5% over 20 targets
is not a result, so this script does the paired test the comparison needs:

* an **exact sign-flip (permutation) test** over all 2^20 sign assignments of
  the per-target differences (random@N - GD) -- exact, not sampled;
* an **exact binomial sign test** on how many targets GD wins.

Both are paired: every budget is evaluated on the same 20 targets with the same
model, so the target is the unit of analysis and the pairing is real.

Writes runs/random_curve_test.json. No number is typed anywhere else.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np


def exact_signflip_p(d: np.ndarray) -> float:
    """Two-sided p for H0: the paired differences are symmetric about 0.

    Enumerates all 2^n sign assignments in chunks. n<=22 keeps this exact and
    cheap; above that the caller should switch to sampling and say so.
    """
    n = len(d)
    if n > 22:
        raise ValueError(f"exact enumeration refuses n={n}; use a sampled test and label it")
    obs = abs(d.sum())
    total = 1 << n
    hits = 0
    step = 1 << 16
    bits = 1 << np.arange(n)
    for start in range(0, total, step):
        idx = np.arange(start, min(start + step, total), dtype=np.int64)
        signs = np.where((idx[:, None] & bits) > 0, 1.0, -1.0)
        hits += int((np.abs(signs @ d) >= obs - 1e-15).sum())
    return hits / total


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p at q=0.5 (ties excluded before calling)."""
    k = min(wins, n - wins)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (1 << n)
    return min(1.0, 2 * tail)


def main() -> None:
    curve = json.loads(Path("runs/random_curve.json").read_text())
    design = json.loads(Path("runs/design_Tfree.json").read_text())

    gd = {t["target_index"]: t["operator_gd"]["area_error_vs_removed"]
          for t in design["targets"]}
    budgets = sorted(curve["curve"], key=int)
    out = {
        "what": "paired exact tests, gradient descent vs random search, per target",
        "source": {"curve": "runs/random_curve.json", "design": "runs/design_Tfree.json"},
        "metric": "area_error_vs_removed, measured in ViennaPS",
        "gd_forward_equivalents": curve["compute"]["gd_forward_equivalents"],
        "budgets": {},
    }
    for b in budgets:
        pairs = [(gd[t["target_index"]], t["budgets"][b]["area_error_vs_removed"])
                 for t in curve["targets"] if t["target_index"] in gd]
        g = np.array([p[0] for p in pairs])
        r = np.array([p[1] for p in pairs])
        d = r - g                       # positive => GD better on that target
        nz = d[d != 0]
        wins = int((d > 0).sum())
        out["budgets"][b] = {
            "n_targets": len(d),
            "gd_mean": float(g.mean()),
            "random_mean": float(r.mean()),
            "mean_paired_diff": float(d.mean()),
            "median_paired_diff": float(np.median(d)),
            "gd_wins": wins,
            "random_wins": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
            "p_signflip_exact": exact_signflip_p(d),
            "p_sign_test_exact": sign_test_p(int((nz > 0).sum()), len(nz)),
            "budget_ratio_vs_gd": int(b) / curve["compute"]["gd_forward_equivalents"],
        }

    alpha = 0.05
    sig = [b for b in budgets if out["budgets"][b]["p_signflip_exact"] < alpha
           and out["budgets"][b]["mean_paired_diff"] > 0]
    tied = [b for b in budgets if out["budgets"][b]["p_signflip_exact"] >= alpha]
    out["verdict"] = {
        "alpha": alpha,
        "budgets_where_gd_wins_significantly": sig,
        "budgets_indistinguishable_from_gd": tied,
        "smallest_budget_indistinguishable": min(tied, key=int) if tied else None,
        "reading": (
            "GD is significantly better than random search up to the largest budget where "
            "it is listed under budgets_where_gd_wins_significantly; at the budgets listed "
            "under budgets_indistinguishable_from_gd the paired test does not separate them, "
            "so the honest statement is a compute ratio, not a quality win."),
    }
    Path("runs/random_curve_test.json").write_text(json.dumps(out, indent=2))
    for b in budgets:
        v = out["budgets"][b]
        print(f"N={b:>6} random {v['random_mean']:.5f} vs GD {v['gd_mean']:.5f}  "
              f"diff {v['mean_paired_diff']:+.5f}  GD wins {v['gd_wins']}/{v['n_targets']}  "
              f"p_flip={v['p_signflip_exact']:.5f} p_sign={v['p_sign_test_exact']:.5f}")
    print("verdict:", json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
