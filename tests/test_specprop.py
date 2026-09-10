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


def test_state_conditioning_makes_the_residual_depend_on_phi_when_nothing_else_does():
    """The sharp property, and its exact complement.

    `modes=0` removes the multiplicative term, so the residual is
    `irfft2(A(recipe))` and is EXACTLY independent of phi -- the pre-registered
    null H20 used, which measured 0.32064 against the candidate's 0.08886 and
    established that this dataset can tell an operator from a recipe lookup.

    `state_modes > 0` feeds a spectral summary of the current field into that
    same head. So at `modes=0, state_modes>0` the multiplicative term is still
    gone and the residual must STILL depend on phi -- through a GELU, i.e.
    nonlinearly, which is what the multiplicative term can never be.

    Testing both directions in one function on purpose: an assertion that a
    difference is non-zero passes for a model that is subtly broken in some
    other way, and an assertion that it is zero passes for a model that ignores
    its input entirely. Together they pin that exactly one pathway is open.
    """
    torch.manual_seed(0)
    phi1 = torch.randn(2, 1, 128, 128)
    phi2 = torch.randn(2, 1, 128, 128)
    cond = torch.randn(2, 7)

    blind = SpectralPropagator(cond_dim=7, modes=0, modes_a=64, state_modes=0)
    blind.eval()
    with torch.no_grad():
        d = (blind.residual(phi1, cond) - blind.residual(phi2, cond)).abs().max()
    assert float(d) == 0.0, f"modes=0 state_modes=0 leaked phi: {float(d)}"

    stated = SpectralPropagator(cond_dim=7, modes=0, modes_a=64, state_modes=8)
    stated.eval()
    with torch.no_grad():
        d = (stated.residual(phi1, cond) - stated.residual(phi2, cond)).abs().max()
    assert float(d) > 0.0, (
        "state_modes=8 produced a phi-independent residual, so the state "
        "features are not reaching the additive head at all"
    )


def test_state_conditioning_is_bit_identical_to_the_old_class_when_off():
    """`state_modes=0` must not change a single parameter or output.

    Five specprop arms are already trained and scored. If adding this option
    perturbed the default path -- an extra LayerNorm, a changed input width, a
    different init draw order -- their checkpoints would still LOAD (shapes are
    the thing that must match) and would compute something else. That is the
    same hazard `hermitian_closed` was given a False default for.
    """
    torch.manual_seed(0)
    a = SpectralPropagator(cond_dim=7, modes=4, modes_a=64)
    torch.manual_seed(0)
    b = SpectralPropagator(cond_dim=7, modes=4, modes_a=64, state_modes=0)
    assert a.param_count() == b.param_count() == 1070144, (
        f"{a.param_count()} != 1070144: the default path changed size, and "
        f"every runs/specprop/m4_ma64_s* checkpoint was trained at 1070144"
    )
    assert b.state_norm is None
    phi, cond = torch.randn(1, 1, 128, 128), torch.randn(1, 7)
    with torch.no_grad():
        assert torch.equal(a(phi, cond), b(phi, cond))


def test_state_features_are_normalised_before_the_head_sees_them():
    """Un-normalised spectral coefficients would silently no-op.

    An SDF over this geometry has a DC coefficient orders of magnitude above its
    high modes, so the raw block spans many decades. Fed straight into a Linear
    followed by a GELU, the head saturates and learns nothing from the state --
    which looks exactly like 'state conditioning does not help' rather than like
    a bug. This pins that the features reaching the head are O(1).
    """
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=64, state_modes=8)
    m.eval()
    # A field with a huge DC term, which is what an SDF actually looks like.
    phi = torch.randn(4, 1, 128, 128) + 500.0
    f = torch.fft.rfft2(phi.squeeze(1).float())
    with torch.no_grad():
        ca = m._cond_a(torch.randn(4, 7), f)
    state = ca[:, 7:]
    assert state.shape[1] == m.state_dim == 2 * (2 * 8) * 8
    assert float(state.abs().max()) < 50.0, (
        f"state features reach {float(state.abs().max()):.1f}; the head will "
        f"saturate and the state pathway will be a silent no-op"
    )


def test_state_conditioning_BREAKS_linearity_in_phi():
    """The complement of `test_linear_in_phi_at_fixed_recipe`, and the whole
    point of the option.

    That test pins superposition as "the architecture's defining property AND
    its stated limitation". H22's entire claim is that `state_modes > 0` removes
    the limitation without removing the four-operation cost structure, so the
    same superposition identity must now FAIL. Asserting only that the outputs
    differ would be satisfied by any perturbation; asserting that superposition
    breaks is the property with a known answer.
    """
    torch.manual_seed(0)
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=8, state_modes=8)
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
    gap = float((lhs - rhs).abs().max())
    assert gap > 1e-3, (
        f"superposition still holds to {gap:.2e}: the state pathway is not "
        f"contributing any nonlinearity, so H22's mechanism is absent"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
