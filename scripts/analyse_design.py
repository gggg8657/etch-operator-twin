"""Does inverse design recover the *recipe*, or only the *profile*?

Reads the design run JSONs and answers a question the KPI does not ask but a
process engineer would: the clause is 형상오차 (shape error), and shape error is
what `design.py` scores. This script measures the other thing -- how far the
proposed recipe sits from the recipe that actually made the target -- so the two
claims cannot be conflated in the report.

Distances are taken in the unit recipe box, log-scaled on the axes the sampler
draws in log space, because a raw distance in cm^-2 s^-1 is not comparable to one
in eV.

    python scripts/analyse_design.py runs/design_Tfixed.json runs/design_Tfree.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import solver as S  # noqa: E402
from eot.inverse import DESIGN_KEYS  # noqa: E402


def to_unit(value: float, key: str) -> float:
    lo, hi, mode = S.RECIPE_BOX[key]
    if mode == "log":
        lo = max(lo, 1e-9)
        return (np.log(max(value, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (value - lo) / (hi - lo)


def analyse(path: Path) -> dict:
    doc = json.loads(path.read_text())
    rows = doc["targets"]
    per_axis = {k: [] for k in DESIGN_KEYS}
    rms, shape, dt_rel = [], [], []
    for r in rows:
        true_rec, prop = r["true_recipe"], r["operator_gd_surrogate"]["recipe"]
        d = [abs(to_unit(prop[k], k) - to_unit(true_rec[k], k)) for k in DESIGN_KEYS]
        for k, x in zip(DESIGN_KEYS, d):
            per_axis[k].append(x)
        rms.append(float(np.linalg.norm(d) / np.sqrt(len(d))))
        shape.append(float(r["operator_gd"]["area_error_vs_removed"]))
        dt_rel.append(float(r["operator_gd_surrogate"]["dt_rel_err"]))
    rms, shape, dt_rel = np.array(rms), np.array(shape), np.array(dt_rel)

    def stats(x):
        return {"mean": float(np.mean(x)), "median": float(np.median(x)),
                "max": float(np.max(x))}

    return {
        "source": str(path),
        "dt_optimised": doc["summary"]["dt_optimised"],
        "n_targets": len(rows),
        "recipe_distance_unit_box": {
            "per_axis": {k: stats(per_axis[k]) for k in DESIGN_KEYS},
            "rms": stats(rms),
            "frac_beyond_10pct_of_box": float((rms > 0.10).mean()),
        },
        "etch_time_rel_error": stats(dt_rel),
        "shape_error_area_vs_removed": stats(shape),
        # If the two were tightly coupled, a small shape error would certify the
        # recipe. It does not: this is the number that separates the claims.
        "corr_recipe_distance_vs_shape_error": (
            float(np.corrcoef(rms, shape)[0, 1]) if np.std(shape) > 0 else None),
        "interpretation": (
            "shape error is met while the recipe is not recovered -- the forward "
            "map is degenerate over (rate x time), so a matched profile does not "
            "identify the process that produced it"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="runs/design_degeneracy.json")
    a = ap.parse_args()
    out = {"arms": [analyse(Path(p)) for p in a.paths]}
    Path(a.out).write_text(json.dumps(out, indent=2))
    for arm in out["arms"]:
        rd = arm["recipe_distance_unit_box"]
        print(f"{arm['source']}  dt_opt={arm['dt_optimised']}  "
              f"shape {arm['shape_error_area_vs_removed']['mean']:.4f}  "
              f"recipe RMS {rd['rms']['mean']:.3f}  "
              f"beyond 10% of box {rd['frac_beyond_10pct_of_box']:.0%}  "
              f"dt rel err {arm['etch_time_rel_error']['mean']:.3f}  "
              f"corr {arm['corr_recipe_distance_vs_shape_error']:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
