"""Every architecture flag must actually reach the model.

**The failure this exists to prevent, which already happened.** The H22 sweep
passed `--state-modes 4/8/16` to `scripts/train.py` for nine arms. The flag was
declared, parsed, and written into each run's `args.json` by `vars(a)` -- and
never passed to `SpectralPropagator`. All nine trained the `state_modes=0`
anchor. The proof is unambiguous: all nine carry `params = 1070144`, the
anchor's exact count, and their `best_val_roll_band` matches the corresponding
anchor seed to all seventeen digits.

It cost 2.4 GPU-hours and surfaced only because `eval.py` rebuilds the model
from `args.json` and the checkpoint would not load into it. Had eval rebuilt
from the checkpoint's own shapes instead, nine anchor runs would have been
scored and published as a state-head result -- the accuracy would have looked
flat, H22 would have been recorded as falsified, and the falsification would
have been of an architecture that was never trained.

`eot/` already had four property tests on `SpectralPropagator(state_modes=...)`
and every one passed, because the class was correct. Nothing tested the WIRING.
That is the gap this file closes: it drives `train.build_model`, the real
construction path, and asserts each flag changes the model it is supposed to
change.
"""
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import build_model  # noqa: E402

BASE = dict(arch="specprop", width=8, modes=4, layers=2, modes_a=64,
            state_modes=0, a_rank=0, hidden=0, cond_ch=0, no_norm=False,
            width_full=8, scale=2, n_local=1, act="gelu")


def _n(**over):
    return build_model(Namespace(**{**BASE, **over}), cond_dim=7).param_count()


def test_state_modes_reaches_the_model():
    """The exact bug: --state-modes was parsed and dropped."""
    base = _n(state_modes=0)
    for sm in (4, 8, 16):
        got = _n(state_modes=sm)
        assert got != base, (
            f"--state-modes {sm} did not change the model: {got} params, same "
            f"as state_modes=0. This is the H22 defect recurring.")
    # and it must grow monotonically with the summary width
    assert _n(state_modes=4) < _n(state_modes=8) < _n(state_modes=16)


def test_a_rank_reaches_the_model():
    """a_rank factorises 98% of the parameters; if the flag is dropped, the
    sweep silently retrains the dense anchor -- exactly the H22 failure."""
    base = _n(a_rank=0)
    for r in (4, 8, 16):
        got = _n(a_rank=r)
        assert got != base, (
            f"--a-rank {r} did not change the model: {got} params")
        assert got < base, (
            f"--a-rank {r} did not SHRINK the model: {got} vs {base}")
    assert _n(a_rank=4) < _n(a_rank=8) < _n(a_rank=16) < base


def test_a_rank_reaches_the_model():
    """`a_rank` was added to SpectralPropagator with a cost script and no way
    to train it: no CLI flag, absent from build_model. The sweep that would
    have measured its accuracy would have trained the dense anchor."""
    base = _n(a_rank=0)
    for r in (4, 8, 16):
        got = _n(a_rank=r)
        assert got != base, f"--a-rank {r} did not change the model: {got}"
    assert _n(a_rank=4) < _n(a_rank=8) < _n(a_rank=16) < base


def test_state_modes_zero_is_exactly_the_anchor():
    """1,070,144 is pinned by five trained checkpoints and must not drift."""
    assert _n(state_modes=0) == 1070144


def test_hidden_and_cond_ch_and_norm_reach_the_model():
    """Found by the reachability test, not by anyone remembering them."""
    assert _n(hidden=16) < _n(hidden=0) < _n(hidden=128)   # 0 = class default 64
    b = _n(arch="fno")
    assert _n(arch="fno", cond_ch=8) != b
    assert _n(arch="fno", no_norm=True) != b
    assert _n(arch="multiscale", cond_ch=16) != _n(arch="multiscale")


def test_specprop_modes_flags_reach_the_model():
    assert _n(modes=4) != _n(modes=16)
    assert _n(modes_a=16) != _n(modes_a=64)


def test_fno_flags_reach_the_model():
    b = _n(arch="fno")
    assert _n(arch="fno", width=16) != b
    assert _n(arch="fno", modes=8) != b
    assert _n(arch="fno", layers=4) != b


def test_multiscale_flags_reach_the_model():
    b = _n(arch="multiscale")
    assert _n(arch="multiscale", width=16) != b
    assert _n(arch="multiscale", width_full=16) != b
    assert _n(arch="multiscale", n_local=3) != b


def test_every_constructor_parameter_is_reachable_from_build_model():
    """The blind spot that let `a_rank` through, closed.

    The meta-test below asks "does every flag build_model READS have a wiring
    assertion?" -- and passed while `a_rank` existed on SpectralPropagator with
    no flag at all, because build_model did not mention it. That is the wrong
    direction of the invariant. A constructor parameter nothing can reach is
    exactly as useless as a flag that is silently dropped, and it fails the
    same way: the sweep trains the default and the hypothesis is recorded as
    falsified on an architecture that was never built.

    So this asks the other direction: for each model class, every constructor
    parameter that shapes the architecture must appear in build_model's source.
    Parameters that are genuinely not architectural are listed with a reason.
    """
    import inspect
    from eot.operator import EtchOperator, MultiScaleOperator, SpectralPropagator

    NOT_ARCHITECTURAL = {
        "self", "cond_dim",          # supplied by the caller from the norm file
        "n_grid",                    # fixed by the dataset, not swept
        "hermitian_closed",          # legacy-compat switch, pinned by its own test
    }
    src = inspect.getsource(build_model)
    missing = {}
    for cls in (EtchOperator, MultiScaleOperator, SpectralPropagator):
        for name in inspect.signature(cls.__init__).parameters:
            if name in NOT_ARCHITECTURAL:
                continue
            if name not in src:
                missing.setdefault(cls.__name__, []).append(name)
    assert not missing, (
        f"constructor parameters no CLI flag can reach: {missing}. Either wire "
        f"them into build_model or add them to NOT_ARCHITECTURAL with a reason.")


def test_every_arch_shaping_flag_is_covered_by_this_file():
    """A flag added to train.py without a wiring test here fails this test.

    Without this, the file protects exactly the flags someone remembered, which
    is the same failure mode one level up.
    """
    import inspect
    src = inspect.getsource(build_model)
    referenced = {n for n in BASE if f"a.{n}" in src}
    covered = set()
    for fn in (test_state_modes_reaches_the_model,
               test_a_rank_reaches_the_model,
               test_hidden_and_cond_ch_and_norm_reach_the_model,
               test_specprop_modes_flags_reach_the_model,
               test_fno_flags_reach_the_model,
               test_multiscale_flags_reach_the_model):
        covered |= {n for n in BASE if n in inspect.getsource(fn)}
    missing = referenced - covered - {"arch"}
    assert not missing, (
        f"build_model reads these flags but no test asserts they reach the "
        f"model: {sorted(missing)}")


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
    print("FAILED" if fails else "all flag-wiring tests pass")
    sys.exit(1 if fails else 0)
