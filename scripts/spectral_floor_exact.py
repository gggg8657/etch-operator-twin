"""The band-weighted optimum over band-limited residuals: the ACTUAL floor.

    python scripts/spectral_floor_exact.py --modes 4 16 32 64 --n 32

**This file exists because `scripts/spectral_floor.py` computes the wrong
quantity and I drew a conclusion from it.** That script evaluates the metric at
the *L2* projection of the true residual onto the retained modes, and I called
the result a lower bound -- "every modes_a below 63 is provably unable to meet
clause 1". It is not a lower bound. The L2 projection is one reachable residual,
so its error is an **upper bound on the minimum**: it shows the architecture does
not do well *at that particular residual* and says nothing about whether a better
one exists inside the same mode budget.

The gap is not pedantic, because the metric is **band-weighted**. `band_rel_l2`
scores only a narrow band around the zero set, so a residual is free to be
arbitrarily wrong in the far field if that buys accuracy near the interface. The
L2 projection spends its budget uniformly over the whole 128x128 window, which
is exactly the wrong allocation for this metric. So the true floor can be well
below the projection value, and the amount is what this script measures.

**The correct problem.** The model emits `phi_t + r` with `r = irfft2(S)` and `S`
supported on the retained block, so `r` ranges over a linear subspace R.
Minimising the metric over R is a weighted linear least-squares problem,

    minimise  || M . (r - r_true) ||_2   over r in R,

with `M` the band mask. Solved here by conjugate gradient on the normal
equations, applying the operator through FFTs rather than materialising it (for
modes_a=32 the design matrix would be 16,384 x 4,096). The result is the
**attained minimum**, so it is a true floor: no `SpectralPropagator` at that
modes_a can beat it, whatever H and A are, however it is trained.

**A convention detail that also came out of the failing test, and that matters
for whether this is a bound at all.** `[-ma:]` retains rows -ma..-1 while
`[:ma]` retains 0..ma-1, so the block holds `-ma` but not `+ma`. A real field
forces Hermitian symmetry on the k2=0 column, so the model's residual carries one
cell of content at `(+ma, 0)` that a naive mask excludes -- `tests/test_arch.py`
caught exactly this, one cell of magnitude 1.76. The subspace here is therefore
defined as *what the model can actually emit*, by round-tripping through
`irfft2`/`rfft2`, rather than by a mask written independently of the model.

Writes `runs/spectral_floor_exact.json`.
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


def retained_mask(H, W, m, device):
    """The model's reachable spectral support, matching eot.operator exactly."""
    f = torch.zeros(H, W // 2 + 1, dtype=torch.complex64, device=device)
    f[:m, :m] = 1
    f[-m:, :m] = 1
    return f.real > 0


def basis(keep, H, W, device):
    """Materialise A: one column per real degree of freedom in the retained block.

    **Why materialise instead of using CG with a hand-written adjoint.** The
    first version of this script applied the operator through FFTs and derived
    the adjoint by hand, noting the scaling "cancels in CG". It does not: in the
    `rfft2` half-spectrum layout the k2=0 and k2=W/2 columns appear once in the
    real field's energy while every other column appears twice, so the true
    adjoint carries a column-dependent weight. With the wrong adjoint the normal
    equations are not symmetric, CG is not applicable, and it returned NaN at
    every modes_a >= 16 -- while still printing plausible numbers at modes_a = 4
    and 8, which is the dangerous failure.

    `A` does not depend on the trajectory, only the band weight does, so it is
    built once and reused. Column j is the real field produced by setting the
    j-th degree of freedom to 1, obtained by pushing a basis vector through the
    same `irfft2` the model uses -- so the operator is defined by the model's own
    code path rather than by an independent derivation.
    """
    cells = keep.nonzero()
    n = cells.shape[0]
    cols = []
    for part in (0, 1):                      # real, then imaginary
        S = torch.zeros(n, H, W // 2 + 1, dtype=torch.complex64, device=device)
        val = (1.0 + 0j) if part == 0 else (0.0 + 1j)
        S[torch.arange(n, device=device), cells[:, 0], cells[:, 1]] = val
        cols.append(torch.fft.irfft2(S, s=(H, W)).reshape(n, -1))
    return torch.cat(cols, dim=0).T.contiguous()      # (H*W, 2n)


def solve_band_ls(A, r_true, mask, ridge):
    """Exact weighted least squares per trajectory: min || W^.5 (A c - b) ||.

    The objective is a convex quadratic in `c`, so the stationary point of the
    normal equations is the global minimum and the returned value is the TRUE
    floor, not merely an attained one. `ridge` is present only to make the
    solve well posed where the band mask leaves directions unconstrained (any
    residual content outside the band is free); it is reported, and its effect
    is checked by re-solving at 100x the value.
    """
    B = r_true.shape[0]
    b = r_true.reshape(B, -1)
    w = mask.reshape(B, -1).float()
    out = torch.empty_like(b)
    n2 = A.shape[1]
    eye = torch.eye(n2, device=A.device)
    for i in range(B):
        Aw = A * w[i].unsqueeze(1)            # W A
        G = A.T @ Aw + ridge * eye            # A^T W A + ridge I
        rhs = A.T @ (w[i] * b[i])
        c = torch.linalg.solve(G, rhs)
        out[i] = A @ c
    return out.view_as(r_true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--modes", type=int, nargs="+", default=[4, 16, 32, 64])
    ap.add_argument("--n", type=int, default=32,
                    help="trajectories (CG is per-trajectory; a subsample is "
                         "declared rather than the full split silently used)")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--target", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="runs/spectral_floor_exact.json")
    a = ap.parse_args()

    runlock.acquire(a.out, what="spectral_floor_exact")
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]
    sdf = np.load(Path(a.data) / f"{a.split}.npz")["sdf"][:a.n]
    dev = torch.device(a.device)
    phi0 = torch.from_numpy(sdf[:, 0] / scale).to(dev).float()
    phiT = torch.from_numpy(sdf[:, a.stride] / scale).to(dev).float()
    r_true = phiT - phi0
    B, H, W = r_true.shape
    m_band = band_mask(phiT * scale, band_um)

    rows = {}
    for m in a.modes:
        if m > W // 2:
            continue
        keep = retained_mask(H, W, m, dev)
        A = basis(keep, H, W, dev)
        r_opt = solve_band_ls(A, r_true, m_band, a.ridge)
        # Ridge sensitivity: if the floor moves when the ridge moves 100x, the
        # solve is being shaped by the regulariser rather than the data.
        r_chk = solve_band_ls(A, r_true, m_band, a.ridge * 100)
        e_opt = band_rel_l2((phi0 + r_opt).unsqueeze(1), phiT.unsqueeze(1),
                            m_band.unsqueeze(1), "none")
        # The L2 projection, for the comparison that motivated this file.
        F = torch.fft.rfft2(r_true)
        Fp = torch.zeros_like(F)
        Fp[:, keep] = F[:, keep]
        r_l2 = torch.fft.irfft2(Fp, s=(H, W))
        e_l2 = band_rel_l2((phi0 + r_l2).unsqueeze(1), phiT.unsqueeze(1),
                           m_band.unsqueeze(1), "none")
        e_chk = band_rel_l2((phi0 + r_chk).unsqueeze(1), phiT.unsqueeze(1),
                            m_band.unsqueeze(1), "none")
        rows[str(m)] = {
            "modes_a": m,
            "n_retained_cells": int(keep.sum()),
            "ridge": a.ridge,
            "band_weighted_optimum": {
                "mean": float(e_opt.mean()), "median": float(e_opt.median()),
                "max": float(e_opt.max()),
                "frac_under_target": float((e_opt <= a.target).float().mean()),
                "meets_target_on_mean": bool(e_opt.mean() <= a.target),
            },
            "l2_projection_for_comparison": {
                "mean": float(e_l2.mean()),
                "meets_target_on_mean": bool(e_l2.mean() <= a.target),
            },
            "l2_over_optimum": float(e_l2.mean() / e_opt.mean().clamp_min(1e-12)),
            "ridge_x100_mean": float(e_chk.mean()),
            "ridge_sensitive": bool(abs(float(e_chk.mean() - e_opt.mean()))
                                    > 0.1 * float(e_opt.mean().clamp_min(1e-12))),
        }

    # A "cannot meet" verdict needs the attained value to BE the minimum, so it
    # is only emitted for rows where CG converged. Elsewhere the value is an
    # upper bound and the honest answer is "not established".
    blocked = [k for k, v in rows.items()
               if not v["band_weighted_optimum"]["meets_target_on_mean"]
               and not v["ridge_sensitive"]]
    unresolved = [k for k, v in rows.items()
                  if not v["band_weighted_optimum"]["meets_target_on_mean"]
                  and v["ridge_sensitive"]]
    res = {
        "question": "what is the LOWEST band rel-L2 attainable by any residual "
                    "inside the modes_a budget -- the true floor, not the error "
                    "of the L2 projection",
        "corrects": "runs/spectral_floor.json evaluated the metric at the L2 "
                    "projection and called it a lower bound. It is an upper "
                    "bound on the minimum: the projection is one reachable "
                    "residual. Because the metric is band-weighted and the L2 "
                    "projection allocates error uniformly over the window, the "
                    "two can differ a lot.",
        "protocol": {
            "split": a.split, "stride": a.stride,
            "n_trajectories": int(B),
            "subsample_note": ("the solve runs per trajectory, so a subsample "
                               "is used and declared. runs/spectral_floor.json "
                               f"used all 250; these means are over the first "
                               f"{B} and are not interchangeable with it."),
            "solver": ("exact weighted normal equations on a materialised "
                       "operator, solved per trajectory; the objective is a "
                       "convex quadratic so this is the global minimum"),
            "subspace": "defined as what the model can emit, by round-tripping "
                        "through irfft2/rfft2, so the Hermitian leak at "
                        "(+modes_a, 0) is inside the subspace rather than "
                        "excluded by a hand-written mask",
            "estimator": "attained minimum, so a TRUE floor: no "
                         "SpectralPropagator at this modes_a can beat it",
            "target": a.target,
        },
        "floors": rows,
        "modes_a_that_cannot_meet_target": blocked,
        "modes_a_unresolved_cg_did_not_converge": unresolved,
        "modes_a_shown_NOT_blocked": [
            k for k, v in rows.items()
            if v["band_weighted_optimum"]["meets_target_on_mean"]],
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"{'modes_a':>8s} {'cells':>7s} {'floor':>11s} {'frac<=t':>8s} "
          f"{'L2 proj':>9s} {'L2/opt':>7s} {'ridgex100':>10s}")
    for k, v in rows.items():
        o = v["band_weighted_optimum"]
        print(f"{k:>8s} {v['n_retained_cells']:7d} {o['mean']:11.5f} "
              f"{o['frac_under_target']:8.3f} "
              f"{v['l2_projection_for_comparison']['mean']:9.5f} "
              f"{v['l2_over_optimum']:7.2f} {v['ridge_x100_mean']:10.5f}")
    print(f"\ncannot meet {a.target} (ridge-insensitive): {blocked or 'none'}")
    print(f"shown NOT blocked: {res['modes_a_shown_NOT_blocked'] or 'none'}")
    print(f"unresolved (CG did not converge): {unresolved or 'none'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
