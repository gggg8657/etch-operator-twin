"""SpectralPropagator has no full-grid activation, and a proposal assumed it did.

`codex`, answering the rung-4 question "how would you make clause 2 pass",
offered two routes to the ~1.9x margin the clause needs. One was:

> **Replace any remaining full-grid erf-GELU with ReLU and retrain or distill.**
> Your isolated measurements imply a 139.5 µs saving per replacement. One
> replacement would ideally yield **210.5 µs, or 1.66× overall**.

It is worth **exactly zero** for this architecture. There is no full-grid
activation to replace: the only nonlinearities are two GELUs inside `h_head`
and `a_head`, both acting on 64-element vectors whose size does not depend on
the grid at all.

The 152.7 us erf-GELU figure it reasoned from is a real measurement, but of the
multiscale/FNO family. I handed codex a list of "prior measured facts" without
saying which architecture each one belonged to, so the arithmetic was its and
the false premise was mine. This test pins the fact so the proposal cannot be
made a second time, by a critic or by me.

What survives of that answer is the other route -- a native fused CPU forward --
and `runs/specprop_profile.json` gives it its target: `_coeffs_a` is 46.3% of
the forward and `a_head.2.weight` is 98.0% of the parameters.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot.operator import SpectralPropagator  # noqa: E402

ACTS = (nn.GELU, nn.ReLU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.ELU)
GRID = 128


def _activation_inputs(model, grid=GRID):
    seen = []
    handles = [m.register_forward_hook(
        lambda mod, inp, out, n=n: seen.append((n, inp[0].numel())))
        for n, m in model.named_modules() if isinstance(m, ACTS)]
    try:
        with torch.no_grad():
            model(torch.randn(1, 1, grid, grid), torch.randn(1, 7))
    finally:
        for h in handles:
            h.remove()
    return seen


def test_no_activation_sees_a_full_grid_tensor():
    for state_modes in (0, 8):
        m = SpectralPropagator(cond_dim=7, modes=4, modes_a=64,
                               state_modes=state_modes).eval()
        seen = _activation_inputs(m)
        assert seen, "no activations found at all; the hook did not fire"
        big = [(n, k) for n, k in seen if k >= GRID * GRID]
        assert not big, (
            f"state_modes={state_modes}: activation(s) on a full-grid tensor "
            f"{big}. codex's erf-GELU->ReLU route would then apply and this "
            f"test's premise would be wrong.")


def test_activation_cost_is_independent_of_grid():
    """The stronger statement: activation input size does not grow with H."""
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=64).eval()
    # 128 is the smallest grid modes_a=64 is valid on (rfft2 gives W//2+1
    # columns, so a 64-grid has only 33 and the additive band does not fit).
    small = sorted(k for _, k in _activation_inputs(m, grid=128))
    large = sorted(k for _, k in _activation_inputs(m, grid=256))
    assert small == large, (
        f"activation input sizes changed with the grid: {small} -> {large}")


def test_the_cost_is_in_the_additive_head_not_the_transforms():
    """Pins the profile's structural claim, which the next change targets."""
    m = SpectralPropagator(cond_dim=7, modes=4, modes_a=64)
    tot = sum(p.numel() for p in m.parameters())
    dom = dict(m.named_parameters())["a_head.2.weight"].numel()
    assert dom / tot > 0.95, (
        f"a_head.2.weight is {100 * dom / tot:.1f}% of parameters, not >95%. "
        f"runs/specprop_profile.json's attribution assumes it dominates.")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL {name}: {e}")
    print("FAILED" if fails else "all full-grid-activation tests pass")
    sys.exit(1 if fails else 0)
