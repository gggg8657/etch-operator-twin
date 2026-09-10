"""The inverse-design *protocol*, as opposed to its numbers.

Written after `codex` found that the arm labelled "total etch time searched
(T unknown, honest)" initialised its duration parameter at the target's own dt --
a value computed from a simulator probe of the true recipe's etch rate. Restarts
randomised the four recipe knobs and never touched it, so every restart began at
the answer. These tests pin down which invocation is honest and which is not, so
the label and the code cannot drift apart again.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eot.inverse import design  # noqa: E402
from eot.operator import EtchOperator  # noqa: E402

# A deliberately tiny operator: these tests are about the protocol, not accuracy.
# width must stay a multiple of 8 -- FNOBlock uses GroupNorm(8, width).
NORM = {
    "cond_mean": [0.0] * 7, "cond_std": [1.0] * 7,
    "sdf_scale_um": 5.0, "band_um": 1.5,
    "dt_lo": 1.0, "dt_hi": 100.0,
}
TRUE_DT = 40.0


def _fixture():
    torch.manual_seed(0)
    model = EtchOperator(cond_dim=7, width=8, modes=2, n_layers=1, cond_ch=8)
    for p in model.parameters():
        p.requires_grad_(False)
    n = 16
    phi0 = torch.zeros(1, 1, n, n)
    tgt = torch.ones(1, 1, n, n) * 0.1
    return model, phi0, tgt, {"trench_width": 6.0, "mask_height": 2.0}


def _design(**kw):
    model, phi0, tgt, geom = _fixture()
    return design(model, tgt, phi0, geom, TRUE_DT, NORM, n_steps=2,
                  iters=2, device="cpu", **kw)


def test_default_optimise_dt_starts_at_the_target_and_says_so():
    """The published protocol. It is a leak, it is kept for reproducibility, and
    it must announce itself in the output rather than hiding in the invocation."""
    d = _design(optimise_dt=True)
    assert d["dt_init_was_the_target"] is True
    assert d["dt_init"] == TRUE_DT == d["dt_given"]


def test_explicit_dt_init_does_not_start_at_the_target():
    d = _design(optimise_dt=True, dt_init=3.0)
    assert d["dt_init_was_the_target"] is False
    assert d["dt_init"] == 3.0
    assert d["dt_init"] != d["dt_given"]      # the target's dt is not the start
    assert d["dt_given"] == TRUE_DT           # but is still recorded, for scoring


def test_pinned_protocol_never_moves_the_duration():
    d = _design(optimise_dt=False, dt_init=3.0)
    assert d["dt_was_optimised"] is False
    assert d["dt"] == TRUE_DT, "with T pinned the returned dt must be the target's"


def test_searched_duration_stays_inside_the_trained_range():
    """dt is parameterised through a sigmoid onto [dt_lo, dt_hi], so a recipe the
    simulator was never validated at cannot be proposed however far the
    optimiser pushes."""
    for init in (NORM["dt_lo"], NORM["dt_hi"], 3.0, 90.0):
        d = _design(optimise_dt=True, dt_init=init)
        assert NORM["dt_lo"] <= d["dt"] <= NORM["dt_hi"], (init, d["dt"])


def test_an_out_of_range_dt_init_is_clipped_not_extrapolated():
    d = _design(optimise_dt=True, dt_init=1e6)
    assert d["dt"] <= NORM["dt_hi"]


if __name__ == "__main__":
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v()
            n += 1
            print(f"  ok  {k}")
    print(f"{n} passed")
