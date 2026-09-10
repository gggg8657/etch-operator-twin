"""SpectralPropagator: the properties its whole argument rests on.

Three claims are load-bearing and all three are testable without training:

1. **It dispatches four full-field operations.** This was asserted in a
   docstring, a commit and the board before it was counted, and the baseline it
   was compared against ("~20") turned out to be 45. So the count is pinned.
2. **`modes_a` is free.** The additive term may use far more modes than the
   multiplicative one at no extra operation cost, because `irfft2` costs the
   same whatever fraction of the spectrum is non-zero. If that stops being true
   the architecture's main design freedom is gone.
3. **It is linear in phi at fixed recipe.** That is what makes it a propagator
   and also what bounds it: a linear map cannot represent an undercut, whose
   advance depends nonlinearly on phi. The expected failure mode should be a
   property of the code, not a claim about it.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.operator import EtchOperator, SpectralPropagator  # noqa: E402
from scripts.op_count import count  # noqa: E402


def test_dispatches_four_full_field_ops():
    r = count(lambda: SpectralPropagator(cond_dim=7, modes=4, modes_a=4))
    assert r["full_materialising"] == 4, r
    fno = count(lambda: EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2))
    assert fno["full_materialising"] == 25, fno
    assert fno["full_materialising"] / r["full_materialising"] == 6.25


def test_modes_a_is_free_in_op_count():
    """The design freedom: a wider additive band must not cost operations."""
    small = count(lambda: SpectralPropagator(cond_dim=7, modes=4, modes_a=4))
    wide = count(lambda: SpectralPropagator(cond_dim=7, modes=4, modes_a=32))
    assert wide["full_materialising"] == small["full_materialising"], (small, wide)


def test_linear_in_phi_at_fixed_recipe():
    """f(a*x + b*y) - f(0) == a*(f(x) - f(0)) + b*(f(y) - f(0)).

    Affine in phi, so the check is on the map minus its value at zero. This is
    the architecture's defining property AND its stated limitation.
    """
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=8)
    # Random init is near-zero by construction, which would make any map look
    # linear; scale the heads up so the test has something to detect.
    with torch.no_grad():
        for h in (m.h_head, m.a_head):
            h[-1].weight.mul_(300.0)
    m.eval()
    cond = torch.randn(1, 7)
    x, y = torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128)
    z = torch.zeros(1, 1, 128, 128)
    a, b = 0.37, -1.9
    with torch.no_grad():
        f0 = m(z, cond)
        lhs = m(a * x + b * y, cond) - f0
        rhs = a * (m(x, cond) - f0) + b * (m(y, cond) - f0)
    assert torch.allclose(lhs, rhs, atol=1e-3), \
        f"not affine in phi: max dev {(lhs - rhs).abs().max():.3e}"


def test_an_fno_is_NOT_linear_in_phi():
    """The contrast that makes the previous test meaningful: if the check passed
    for a nonlinear model too, it would be testing nothing."""
    torch.manual_seed(0)
    m = EtchOperator(cond_dim=7, width=8, modes=4, n_layers=2)
    m.eval()
    cond = torch.randn(1, 7)
    x, y = torch.randn(1, 1, 128, 128), torch.randn(1, 1, 128, 128)
    z = torch.zeros(1, 1, 128, 128)
    a, b = 0.37, -1.9
    with torch.no_grad():
        f0 = m(z, cond)
        lhs = m(a * x + b * y, cond) - f0
        rhs = a * (m(x, cond) - f0) + b * (m(y, cond) - f0)
    assert not torch.allclose(lhs, rhs, atol=1e-3)


def test_recipe_actually_changes_the_output():
    """A propagator whose coefficients ignore the recipe would be a persistence
    predictor with extra steps, and this repo has already been caught once by a
    target that a do-nothing model solves."""
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=8)
    with torch.no_grad():
        for h in (m.h_head, m.a_head):
            h[-1].weight.mul_(300.0)
    m.eval()
    phi = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        a = m(phi, torch.randn(1, 7))
        b = m(phi, torch.randn(1, 7))
    assert not torch.allclose(a, b, atol=1e-4)


def test_starts_near_identity():
    """At init the residual must be small: the update is a residual and the
    rollout applies it repeatedly, so a large random init diverges."""
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=16)
    m.eval()
    phi = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        d = (m(phi, torch.randn(1, 7)) - phi).abs().max()
    assert d < 0.05, f"init is not near-identity: {d:.4f}"


def test_modes_above_nyquist_refused():
    for bad in (dict(modes=80, modes_a=4), dict(modes=4, modes_a=80)):
        try:
            SpectralPropagator(cond_dim=7, **bad)
        except AssertionError:
            continue
        raise AssertionError(f"{bad} accepted; rfft2 has {128 // 2 + 1} columns")


def test_output_is_finite_over_a_long_rollout():
    """A linear propagator can be unstable: |H| > 1 on any retained mode grows
    without bound under repeated application. Ten steps is what the KPI uses."""
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=16)
    m.eval()
    phi = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        out = m.rollout(phi, torch.randn(1, 7), 10)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
