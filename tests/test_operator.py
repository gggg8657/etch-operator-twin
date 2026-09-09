"""Operator shape, conditioning and metric behaviour. CPU only, no data needed."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.operator import EtchOperator, band_rel_l2, rel_l2  # noqa: E402
from eot.data import cond_vector  # noqa: E402


def _model(**kw):
    torch.manual_seed(0)
    return EtchOperator(cond_dim=7, width=16, modes=8, n_layers=2, cond_ch=8, **kw)


def test_shapes_and_residual_form():
    m = _model()
    phi = torch.randn(3, 1, 32, 32)
    c = torch.randn(3, 7)
    out = m(phi, c)
    assert out.shape == phi.shape
    # forward must be exactly phi + residual, or the identity-baseline argument
    # in the README stops holding
    assert torch.allclose(out, phi + m.residual(phi, c), atol=1e-6)


def test_rollout_shape_and_composition():
    m = _model()
    phi = torch.randn(2, 1, 32, 32)
    c = torch.randn(2, 7)
    r = m.rollout(phi, c, 4)
    assert r.shape == (2, 4, 1, 32, 32)
    assert torch.allclose(r[:, 0], m(phi, c), atol=1e-6)
    assert torch.allclose(r[:, 1], m(m(phi, c), c), atol=1e-5)


def test_conditioning_actually_conditions():
    """Two different recipes must give two different fields.

    A conditioning path that is silently dropped -- broadcast wrong, zeroed by a
    dead ReLU -- would leave the model predicting one average etch for every
    recipe, which would still train to a plausible-looking loss.
    """
    m = _model()
    phi = torch.randn(1, 1, 32, 32)
    a = m(phi, torch.zeros(1, 7))
    b = m(phi, torch.ones(1, 7) * 2.0)
    assert (a - b).abs().max() > 1e-4


def test_discretisation_invariance():
    """Same weights on a finer grid: the spectral layers must still run."""
    m = _model().eval()
    c = torch.randn(1, 7)
    for n in (32, 48, 64):
        with torch.no_grad():
            out = m(torch.randn(1, 1, n, n), c)
        assert out.shape == (1, 1, n, n)


def test_rel_l2_zero_on_exact():
    y = torch.randn(4, 1, 16, 16)
    assert rel_l2(y, y).item() < 1e-6


def test_band_rel_l2_ignores_outside_band():
    """Corrupting the field outside the mask must not move the band metric."""
    torch.manual_seed(1)
    y = torch.randn(2, 1, 16, 16)
    p = y.clone()
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, :, :8] = True
    p[:, :, 8:] += 100.0
    assert band_rel_l2(p, y, mask).item() < 1e-6
    assert rel_l2(p, y).item() > 1.0


def test_cond_vector_layout():
    rec = np.array([[10.0, 2000.0, 100.0, 120.0, 6.0, 2.0]], dtype=np.float32)
    dt = np.array([0.25], dtype=np.float32)
    c = cond_vector(rec, dt)
    assert c.shape == (1, 7)
    assert abs(c[0, 0] - np.log(10.0)) < 1e-5
    assert abs(c[0, 1] - np.log(2000.0)) < 1e-3
    assert abs(c[0, 6] - np.log(0.25)) < 1e-5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
