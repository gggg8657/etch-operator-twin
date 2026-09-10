"""The architecture factory, and the back-compatibility it has to preserve.

`build_from_cfg` was added when a second architecture appeared. The risk it
carries is not that it raises -- a raise is visible -- but that it silently
rebuilds a *different* model from the same checkpoint, because two
`MultiScaleOperator` configs differing only in `scale` have identical parameter
names and identical shapes. So the tests here pin, in order: every run written
before `--arch` existed still rebuilds as an `EtchOperator`; a multiscale config
round-trips through its own state dict; and a scale-mismatched load is caught by
something, since `load_state_dict` cannot catch it.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.operator import (EtchOperator, MultiScaleOperator,  # noqa: E402
                          build_from_cfg)


def test_legacy_cfg_without_arch_is_fno():
    """Every args.json in runs/ predating --arch has no `arch` key at all."""
    m = build_from_cfg({"width": 8, "modes": 4, "layers": 2}, 7)
    assert isinstance(m, EtchOperator), type(m)
    assert m.param_count() == EtchOperator(cond_dim=7, width=8, modes=4,
                                           n_layers=2).param_count()


def test_real_stored_run_still_loads():
    """Not a synthetic cfg: the actual args.json of a committed run, loaded into
    the model the factory picks, with its real checkpoint."""
    runs = sorted(Path("runs/shrink").glob("w8m4L2_K10_s*"))
    runs = [r for r in runs if (r / "args.json").exists() and (r / "best.pt").exists()]
    if not runs:
        return  # dataset-dependent; skip rather than fail a fresh clone
    cfg = json.loads((runs[0] / "args.json").read_text())
    assert "arch" not in cfg or cfg["arch"] == "fno"
    m = build_from_cfg(cfg, 7)
    m.load_state_dict(torch.load(runs[0] / "best.pt", map_location="cpu"))


def test_multiscale_roundtrip():
    cfg = {"arch": "multiscale", "width": 8, "modes": 4, "layers": 2,
           "width_full": 16, "scale": 4, "n_local": 2, "act": "relu"}
    a = build_from_cfg(cfg, 7)
    b = build_from_cfg(cfg, 7)
    b.load_state_dict(a.state_dict())
    phi, cond = torch.randn(2, 1, 128, 128), torch.randn(2, 7)
    a.eval(), b.eval()
    with torch.no_grad():
        assert torch.allclose(a(phi, cond), b(phi, cond))


def test_pointwise_has_no_body_and_no_spatial_mixing():
    """scale=0 must remove the body, and the result must be genuinely pointwise:
    changing one input pixel may only change that same output pixel. This is the
    property the whole clause-2 argument rests on, so it is tested rather than
    asserted in a docstring."""
    m = build_from_cfg({"arch": "multiscale", "width": 8, "modes": 4,
                        "layers": 2, "width_full": 8, "scale": 0,
                        "n_local": 2, "act": "relu"}, 7)
    assert not hasattr(m, "blocks") or len(getattr(m, "blocks", [])) == 0
    assert not m.coarse
    m.eval()
    phi = torch.zeros(1, 1, 128, 128)
    cond = torch.randn(1, 7)
    with torch.no_grad():
        base = m(phi, cond)
        phi2 = phi.clone()
        phi2[0, 0, 40, 70] = 3.0
        pert = m(phi2, cond)
    d = (pert - base).abs()
    assert d[0, 0, 40, 70] > 0, "the perturbed pixel did not change"
    d[0, 0, 40, 70] = 0
    assert d.max() < 1e-6, f"a pointwise model leaked to other pixels: {d.max()}"


def test_scale_mismatch_is_not_silent():
    """Two multiscale configs differing only in `scale` share every parameter
    name and shape, so load_state_dict CANNOT detect the mismatch -- it loads
    happily and computes something else. Pin that the state dicts really are
    interchangeable (the hazard is real) and that the outputs differ (the hazard
    matters), so nobody later assumes the loader protects them."""
    base = {"arch": "multiscale", "width": 8, "modes": 4, "layers": 2,
            "width_full": 8, "scale": 4, "n_local": 1, "act": "gelu"}
    a = build_from_cfg(base, 7)
    b = build_from_cfg({**base, "scale": 8}, 7)
    b.load_state_dict(a.state_dict())   # no error: that is the hazard
    a.eval(), b.eval()
    phi, cond = torch.randn(1, 1, 128, 128), torch.randn(1, 7)
    with torch.no_grad():
        assert not torch.allclose(a(phi, cond), b(phi, cond)), \
            "scale is not affecting the computation, so the ladder is a no-op"


def test_modes_above_coarse_nyquist_is_refused():
    """The architecture's whole justification is that the downsample discards no
    mode the body can represent. If modes exceed the coarse grid's Nyquist that
    justification is false, so construction must fail rather than quietly
    truncate."""
    try:
        MultiScaleOperator(cond_dim=7, width=8, modes=20, n_layers=2, scale=8)
    except AssertionError:
        return
    raise AssertionError("modes=20 at scale=8 (Nyquist 8) was accepted")


def test_relu_and_gelu_differ():
    """The act flag exists for a measured 11.6x cost reason; make sure it is
    wired to the computation and not just to args.json."""
    cfg = {"arch": "multiscale", "width": 8, "modes": 4, "layers": 2,
           "width_full": 8, "scale": 0, "n_local": 1, "act": "relu"}
    torch.manual_seed(0)
    a = build_from_cfg(cfg, 7)
    b = build_from_cfg({**cfg, "act": "gelu"}, 7)
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    phi, cond = torch.randn(1, 1, 128, 128), torch.randn(1, 7)
    with torch.no_grad():
        assert not torch.allclose(a(phi, cond), b(phi, cond))


def test_unknown_arch_raises():
    try:
        build_from_cfg({"arch": "transformer", "width": 8, "modes": 4,
                        "layers": 2}, 7)
    except ValueError:
        return
    raise AssertionError("an unknown arch was accepted")




def test_specprop_null_residual_is_exactly_phi_independent():
    """`modes=0` is H20's pre-registered null and its whole value is that the
    residual cannot depend on phi. Pinned as EXACT equality, not a tolerance:
    if a future edit lets any phi-dependent path leak into the modes=0 branch,
    the null stops being a null and the H20 decision rule silently breaks."""
    m = build_from_cfg({"arch": "specprop", "modes": 0, "modes_a": 64,
                        "width": 8, "layers": 2}, 7)
    m.eval()
    cond = torch.randn(1, 7)
    with torch.no_grad():
        r1 = m.residual(torch.randn(1, 1, 128, 128), cond)
        r2 = m.residual(torch.zeros(1, 1, 128, 128), cond)
        r3 = m.residual(torch.randn(1, 1, 128, 128) * 100, cond)
    assert torch.equal(r1, r2), float((r1 - r2).abs().max())
    assert torch.equal(r1, r3), float((r1 - r3).abs().max())
    assert not m.has_mult
    assert m.h_head is None


def test_specprop_with_modes_does_depend_on_phi():
    """The complement: the candidate arm must actually use phi, or the null
    comparison is vacuous."""
    m = build_from_cfg({"arch": "specprop", "modes": 4, "modes_a": 64,
                        "width": 8, "layers": 2}, 7)
    # At init both heads are ~zero, so give H real weights before testing.
    torch.nn.init.normal_(m.h_head[-1].weight, std=0.1)
    m.eval()
    cond = torch.randn(1, 7)
    with torch.no_grad():
        r1 = m.residual(torch.randn(1, 1, 128, 128), cond)
        r2 = m.residual(torch.zeros(1, 1, 128, 128), cond)
    assert not torch.allclose(r1, r2), "modes>0 residual ignored phi"


def _specprop_spectrum(ma, closed, std=0.5, seed=0):
    m = build_from_cfg({"arch": "specprop", "modes": 4, "modes_a": ma,
                        "width": 8, "layers": 2,
                        "hermitian_closed": closed}, 7)
    torch.nn.init.normal_(m.a_head[-1].weight, std=std)
    torch.nn.init.normal_(m.h_head[-1].weight, std=std)
    m.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        r = m.residual(torch.randn(1, 1, 128, 128), torch.randn(1, 7))
    return m, torch.fft.rfft2(r.squeeze(1).float())


def test_specprop_default_reaches_one_cell_outside_the_block():
    """The defect, pinned as it actually behaves under the default.

    Written asserting the support equals `[:ma] + [-ma:]`, it FAILED with one
    cell of magnitude ~1.2-1.8 at `(+ma, 0)`. That is real: `[-ma:]` retains
    frequencies -ma..-1 while `[:ma]` retains 0..ma-1, so the set holds `-ma`
    and not `+ma`, and a real field forces Hermitian symmetry on the k2=0
    column -- so content at `(-ma, 0)` obliges content at `(+ma, 0)`, which
    `irfft2` synthesises.

    Consequence: under the default the reachable subspace is slightly LARGER
    than a naive mask, so a floor computed from such a mask is not a bound.
    `scripts/spectral_floor_exact.py` therefore measures the subspace by pushing
    basis vectors through the model rather than trusting a mask.
    """
    ma = 8
    _m, f = _specprop_spectrum(ma, closed=False)
    allowed = torch.zeros_like(f, dtype=torch.bool)
    allowed[:, :ma, :ma] = True
    allowed[:, -ma:, :ma] = True
    allowed[:, ma, 0] = True          # the Hermitian partner of (-ma, 0)
    ib = float(f[allowed].abs().max())
    assert float(f[~allowed].abs().max()) / ib < 1e-6, \
        "support is wider than block+partner"
    # The partner cell really is populated, so the exception is necessary
    # rather than defensive.
    assert float(f[:, ma, 0].abs().max()) / ib > 1e-4


def test_specprop_hermitian_closed_is_exactly_band_limited():
    """`hermitian_closed=True` makes the retained set conjugate-closed --
    {0..ma-1} u {-(ma-1)..-1} -- and symmetrises the self-conjugate columns, so
    the residual's support IS the claimed block and a mask-derived floor is a
    genuine bound.

    The bound is RELATIVE, not absolute. The previous version of this assertion
    used an absolute 1e-4 and failed at 1.13e-4 on a correctly band-limited
    residual, because out-of-band content here is float32 round-trip noise that
    scales with the in-band magnitude -- which is set by an arbitrary weight-init
    scale, not by the architecture. Measured relative leakage is 1.7e-8 to
    7.7e-8 over modes_a 8..64 and a 10x change in that scale, i.e. float32 eps.
    """
    for ma, std in ((8, 0.5), (16, 0.5), (32, 0.5), (8, 5.0)):
        m, f = _specprop_spectrum(ma, closed=True, std=std)
        blk = torch.zeros_like(f, dtype=torch.bool)
        blk[:, :ma, :ma] = True
        blk[:, -(ma - 1):, :ma] = True
        rel = float(f[~blk].abs().max()) / float(f[blk].abs().max())
        assert rel < 1e-6, f"ma={ma} std={std}: relative leakage {rel:.2e}"
        assert m.retained_rows()["conjugate_closed"]
        assert m.retained_rows()["extra_reachable_cell_at_col0"] is None


def test_specprop_default_is_false_so_old_checkpoints_keep_their_meaning():
    """Five runs/specprop arms were mid-training when the defect was found.
    Their `args.json` has no `hermitian_closed` key and their checkpoints load
    under either setting, because the shapes are identical -- so if the default
    ever flips, those checkpoints silently start computing a model that was
    never trained. Pin the default.
    """
    m = build_from_cfg({"arch": "specprop", "modes": 4, "modes_a": 8}, 7)
    assert m.hermitian_closed is False, \
        "flipping this default re-scores every pre-fix specprop arm as a " \
        "different model; set hermitian_closed explicitly on new arms instead"


def test_specprop_the_two_modes_really_differ():
    """If the flag did not change the computation, the fix would be cosmetic and
    the two tests above would both be vacuous."""
    _a, fa = _specprop_spectrum(8, closed=False)
    _b, fb = _specprop_spectrum(8, closed=True)
    assert not torch.allclose(fa, fb), "hermitian_closed changed nothing"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
