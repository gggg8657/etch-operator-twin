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


def cat(paths, out, match_n=None, seed=0):
    """Concatenate splits, optionally down to a fixed total trajectory count.

    `match_n` exists because gradient steps and data volume are both known to
    move these numbers: the K-curve's apparent horizon effect was largely its
    `nv` arms taking K times fewer gradient steps (critique_log, turn 5). A
    mixed set built by simple concatenation is 1.8x the adaptive set, so an arm
    trained on it at the same epoch count differs from the anchor in THREE ways
    at once -- dt coupling, data volume and step count -- and no reading of the
    result could attribute the effect. With `match_n` the mixed set holds the
    anchor's trajectory count, split evenly across sources, so pairs, epochs and
    steps all match the anchor and only the dt coupling of half the data differs.
    """
    parts = [np.load(p) for p in paths if Path(p).exists()]
    if not parts:
        raise SystemExit(f"none of {paths} exist")
    if match_n is not None:
        rng = np.random.default_rng(seed)
        per = match_n // len(parts)
        sel = []
        for q in parts:
            n = q["sdf"].shape[0]
            take = min(per, n)
            idx = np.sort(rng.choice(n, size=take, replace=False))
            sel.append({k: q[k][idx] for k in q.files})
        parts = sel
    keys = set(parts[0].files if hasattr(parts[0], "files") else parts[0].keys())
    for p in parts[1:]:
        keys &= set(p.files if hasattr(p, "files") else p.keys())
    np.savez_compressed(out, **{k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)})
    return [int(p["sdf"].shape[0]) for p in parts], sorted(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data_mixed")
    ap.add_argument("--match-n", type=int, default=None,
                    help="total trajectories in the mixed TRAIN split, split "
                         "evenly across sources. Set to the adaptive split's "
                         "count to hold data volume and gradient steps fixed so "
                         "only the dt coupling differs from the anchor.")
    ap.add_argument("--seed", type=int, default=0,
                    help="subsampling seed, recorded in mix_report.json")
    a = ap.parse_args()
    src, dst = Path(a.data), Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)

    report = {"source": str(src), "match_n": a.match_n, "subsample_seed": a.seed,
              "parts": {}}
    for split in ("train", "val"):
        # val is subsampled proportionally so the checkpoint-selection set is
        # drawn from the same distribution the arm is trained on
        mn = a.match_n if split == "train" else (
            None if a.match_n is None else max(a.match_n // 6, 2))
        counts, keys = cat([src / f"{split}.npz", src / f"{split}_indep.npz"],
                           dst / f"{split}.npz", match_n=mn, seed=a.seed)
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
                      ("mean_step_displacement_um", "dt_lo", "dt_hi", "n_train_trajectories")
                      if k in norm}
    # carry the generation metadata forward so downstream scripts still find it
    gen = json.loads((src / "gen_report.json").read_text())
    gen["mixed_from"] = ["adaptive", "independent"]
    gen["mixed_counts"] = report["parts"]
    (dst / "gen_report.json").write_text(json.dumps(gen, indent=2))
    (dst / "mix_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
