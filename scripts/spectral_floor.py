"""The error floor of a band-limited residual: SpectralPropagator's ceiling, no model.

    python scripts/spectral_floor.py

**Why this exists, and why it exists BEFORE the training finishes.** A
concurrent instance falsified H15's Nyquist argument with a control (commit
`ac880e0`): I had argued a coarse spectral body is representationally free
because the body truncates to `modes=4` and a 4x downsample leaves Nyquist at
16, so s=4 and s=8 should score the same at matched parameters. They differ by
1.31x with non-overlapping intervals. Its diagnosis is right and it is general:
**my bound was on what the body can OUTPUT, while the downsample acts on what it
can SEE**, and no mode-truncation bound on an output says anything about the
information available at the input.

**That diagnosis applies to `SpectralPropagator`, which I built two turns ago and
which is training right now.** Its update is

    phi_next = phi + irfft2( rfft2(phi) * H(recipe) + A(recipe) )

so the residual it emits is **band-limited to max(modes, modes_a) modes**, and
its dependence on `phi` runs through only the lowest `modes` x `modes` of the
input spectrum. Nobody had connected the falsification to it. This script does,
by computing the consequence that needs no training at all.

**The bound.** The model can only ever predict `phi_t + r` where `r` is
band-limited to `modes_a`. The true update is `phi_t + r_true`. The best such
`r` is the spectral projection `P(r_true)`, achieved by a perfect `H` and `A`.
So

    floor(modes_a) = band_rel_l2( phi_t + P_{modes_a}(r_true),  phi_{t+1} )

is a **lower bound on the error of any SpectralPropagator at that `modes_a`**,
attained by an oracle. It is computed from the stored trajectories with no
network, no checkpoint and no training, exactly like
`runs/surface_representable.json`.

Two readings are reported because they answer different questions:

* `oracle_projection` -- the bound above. If it exceeds 0.05, the architecture
  **cannot** meet clause 1 at that `modes_a`, and no amount of training changes
  that.
* `oracle_multiplicative_only` -- the same with `A` forced to zero, i.e. what a
  purely multiplicative propagator could reach. This isolates how much of the
  update is a recipe-dependent additive field rather than a transformation of
  `phi`, which is the design question the class docstring asserts an answer to.

The metric, the band and the scale are the repo's canonical ones
(`band_rel_l2`, `band_mask`, `norm.json`), and the reading is the terminal step
at stride 10 -- one application per wafer -- because that is the deployment the
cost measurement priced.

Writes `runs/spectral_floor.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot import runlock  # noqa: E402
from eot.data import band_mask  # noqa: E402
from eot.operator import band_rel_l2  # noqa: E402


def project(r: torch.Tensor, m: int) -> torch.Tensor:
    """Keep only the lowest m x m modes of a real field, as the class does.

    `rfft2` gives (H, W//2+1): the first axis is a full spectrum so it needs the
    positive AND negative m bands, the second is already half. Truncating the
    second axis symmetrically would discard half the retained content and make
    the floor look worse than the architecture really is -- a bound has to be
    tight to be worth reporting, and the direction of that error would have
    flattered the conclusion that the architecture is doomed.
    """
    f = torch.fft.rfft2(r)
    keep = torch.zeros_like(f)
    keep[..., :m, :m] = f[..., :m, :m]
    keep[..., -m:, :m] = f[..., -m:, :m]
    return torch.fft.irfft2(keep, s=r.shape[-2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--modes", type=int, nargs="+",
                    default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/spectral_floor.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="spectral_floor")
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]

    sdf = np.load(Path(a.data) / f"{a.split}.npz")["sdf"]     # (N, T+1, H, W)
    n_steps = sdf.shape[1] - 1
    if n_steps % a.stride:
        raise SystemExit(f"stride {a.stride} does not divide {n_steps} steps")
    dev = torch.device(a.device)
    # Normalised exactly as TrajDataset does, so the metric is the repo's.
    phi0 = torch.from_numpy(sdf[:, 0] / scale).to(dev).float()
    phiT = torch.from_numpy(sdf[:, a.stride] / scale).to(dev).float()
    r_true = phiT - phi0

    m_true = band_mask(phiT * scale, band_um)
    base = band_rel_l2(phi0.unsqueeze(1), phiT.unsqueeze(1),
                       m_true.unsqueeze(1), "none")

    rows = {}
    for m in a.modes:
        if m > sdf.shape[-1] // 2:
            continue
        pr = project(r_true, m)
        pred = phi0 + pr
        e = band_rel_l2(pred.unsqueeze(1), phiT.unsqueeze(1),
                        m_true.unsqueeze(1), "none")
        # A == 0: the residual must be a linear transform of phi's low modes,
        # so the best achievable is the projection of r_true onto the span of
        # phi's retained modes. Per-trajectory least squares in the frequency
        # domain, one complex coefficient per retained mode, which is exactly
        # what a perfect H(recipe) could do for that trajectory.
        f0 = torch.fft.rfft2(phi0)
        fr = torch.fft.rfft2(r_true)
        keep = torch.zeros_like(fr)
        for sl in ((slice(None), slice(0, m), slice(0, m)),
                   (slice(None), slice(-m, None), slice(0, m))):
            num, den = fr[sl], f0[sl]
            # Best per-mode complex scalar is fr/f0 where f0 is non-zero; where
            # phi has no content at a mode, no H can create any, so the target
            # content there is unreachable and stays zero.
            ok = den.abs() > 1e-8
            keep[sl] = torch.where(ok, num, torch.zeros_like(num))
        mult_pred = phi0 + torch.fft.irfft2(keep, s=phi0.shape[-2:])
        e_mult = band_rel_l2(mult_pred.unsqueeze(1), phiT.unsqueeze(1),
                             m_true.unsqueeze(1), "none")
        rows[str(m)] = {
            "modes": m,
            "oracle_projection": {
                "mean": float(e.mean()), "median": float(e.median()),
                "p90": float(torch.quantile(e, 0.9)), "max": float(e.max()),
                "frac_under_target": float((e <= a.target).float().mean()),
                "meets_target_on_mean": bool(e.mean() <= a.target),
            },
            "oracle_multiplicative_only": {
                "mean": float(e_mult.mean()), "median": float(e_mult.median()),
                "meets_target_on_mean": bool(e_mult.mean() <= a.target),
            },
        }

    trained = [m for m, r in rows.items()
               if not r["oracle_projection"]["meets_target_on_mean"]]
    res = {
        "question": "what is the lowest band rel-L2 any SpectralPropagator can "
                    "reach, given that its residual is band-limited to modes_a",
        "why": "a concurrent instance falsified H15's Nyquist argument (commit "
               "ac880e0) by showing a mode bound on an OUTPUT says nothing "
               "about information at the INPUT. SpectralPropagator, built and "
               "trained on that same reasoning, emits a band-limited residual, "
               "so the same objection gives it a computable ceiling.",
        "protocol": {
            "split": a.split, "stride": a.stride,
            "n_trajectories": int(sdf.shape[0]),
            "reading": "terminal step at stride 10, one application per wafer "
                       "-- the deployment the cost rows priced",
            "metric": "band_rel_l2 with the repo's band and scale from norm.json",
            "estimator": "ORACLE. r_true is projected onto the retained modes, "
                         "which is what a perfect H and A would achieve. No "
                         "network, no checkpoint, no training is involved, so "
                         "this is a bound and not a result about a model.",
            "target": a.target,
            "band_um": band_um, "sdf_scale_um": scale,
        },
        "do_nothing_baseline": {
            "mean": float(base.mean()), "median": float(base.median()),
            "what": "phi_next = phi_t, the persistence predictor. Any floor "
                    "must be far below this to mean anything.",
        },
        "floors": rows,
        "modes_that_cannot_meet_target": trained,
        "verdict": (
            "SpectralPropagator's band-limited residual is not what blocks it "
            "at the modes_a values being trained"
            if not trained or all(int(m) < 4 for m in trained) else
            f"modes_a in {trained} cannot meet {a.target} even with a perfect "
            "H and A"),
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"do-nothing baseline: {base.mean():.5f}")
    print(f"{'modes':>6s} {'oracle proj':>12s} {'p90':>8s} {'frac<=t':>8s} "
          f"{'mult-only':>10s}")
    for m, r in rows.items():
        o, mu = r["oracle_projection"], r["oracle_multiplicative_only"]
        print(f"{m:>6s} {o['mean']:12.5f} {o['p90']:8.5f} "
              f"{o['frac_under_target']:8.3f} {mu['mean']:10.5f}")
    print(f"\ncannot meet {a.target}: modes_a in {trained or 'none'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
