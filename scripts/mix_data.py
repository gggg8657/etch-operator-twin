"""Build a mixed adaptive + independent-dt training set for H2.

    python scripts/mix_data.py --out data_mixed

H2 says the crossed-split failure is a *coverage* failure: the adaptive protocol
confines per-step displacement to 0.0617-0.3118 µm and the operator degrades
sharply outside that interval (turn 9). Mixing in trajectories whose timestep was
drawn without reference to the recipe widens that interval.

Test splits are **not** touched. Both models are scored on exactly the same
`test.npz` and `test_crossed.npz`, neither of which either was trained on, so
the arms remain comparable even though their training distributions differ.
Normalisation is refitted on the mixed training set and written to
`data_mixed/norm.json`; reusing the adaptive constants would standardise the new
dt range against statistics that never saw it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import fit_norm  # noqa: E402


def cat(paths, out):
    parts = [np.load(p) for p in paths if Path(p).exists()]
    if not parts:
        raise SystemExit(f"none of {paths} exist")
    keys = set(parts[0].files)
    for p in parts[1:]:
        keys &= set(p.files)
    np.savez_compressed(out, **{k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)})
    return [int(p["sdf"].shape[0]) for p in parts], sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_mixed")
    a = ap.parse_args()
    src, dst = Path(a.data), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)

    report = {"source": str(src), "parts": {}}
    for split in ("train", "val"):
        counts, keys = cat([src / f"{split}.npz", src / f"{split}_indep.npz"],
                           dst / f"{split}.npz")
        report["parts"][split] = {"adaptive": counts[0],
                                  "independent": counts[1] if len(counts) > 1 else 0,
                                  "total": sum(counts), "keys": keys}
    # test splits are copied unchanged: both arms must be scored on identical data
    for split in ("test", "test_crossed"):
        p = src / f"{split}.npz"
        if p.exists():
            shutil.copy2(p, dst / f"{split}.npz")
            report["parts"][split] = {"copied_unchanged": True}

    norm = fit_norm(dst / "train.npz")
    (dst / "norm.json").write_text(json.dumps(norm, indent=2))
    report["norm"] = {k: norm[k] for k in
                      ("mean_step_displacement_um", "dt_lo", "dt_hi", "n_train_trajectories")}
    # carry the generation metadata forward so downstream scripts still find it
    gen = json.loads((src / "gen_report.json").read_text())
    gen["mixed_from"] = ["adaptive", "independent"]
    gen["mixed_counts"] = report["parts"]
    (dst / "gen_report.json").write_text(json.dumps(gen, indent=2))
    (dst / "mix_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
