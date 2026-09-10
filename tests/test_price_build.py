"""The cost path and the scoring path must build the SAME model.

`scripts/shrink_report.price` reconstructs an architecture from a source string
so it can be timed in a fresh single-threaded subprocess;
`eot.operator.build_from_cfg` reconstructs it in-process so a checkpoint can be
scored. One report row carries a cost from the first and an accuracy from the
second. If they diverge, the row is a cost for one model printed beside an
accuracy for another, and nothing about it announces the mismatch.

That is not hypothetical. `price` used to read

    if arch == "fno": EtchOperator(...) else: MultiScaleOperator(...)

so every `specprop` config was priced as a `MultiScaleOperator` -- silently,
because argparse writes `width`, `layers`, `scale` and `width_full` defaults
into every args.json whether the architecture consumes them or not, so the
wrong constructor found every key it needed. It raised only when `modes=0`
made the wrong build fail an einsum.

The check here is a PROPERTY WITH A KNOWN ANSWER -- two constructions of one
config must agree on parameter count -- rather than a second script that
recomputes a cost. This repo has now had four instrument bugs that a
recomputation could not catch and a property test caught immediately: an
op-count keyed on `numel` that could not tell a field from a weight matrix, a
`config_key` blind to `modes_a` that pooled two architectures into one seed
group, five reconstruction bugs in the surface metric, and this one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eot.operator import build_from_cfg  # noqa: E402
from scripts.shrink_report import ARCH_FIELDS, build_string  # noqa: E402

COND_DIM = 7

# One config per architecture, carrying the argparse defaults that the wrong
# constructor used to feed on. `width`/`layers`/`scale`/`width_full` are present
# in EVERY case on purpose: a specprop config that omitted them would have made
# the old bug raise a KeyError, and the whole point is that it did not.
CONFIGS = [
    dict(arch="fno", width=8, modes=4, layers=2, scale=4, width_full=8),
    dict(arch="multiscale", width=8, modes=4, layers=2, scale=4, width_full=8,
         n_local=1, act="relu"),
    dict(arch="specprop", modes=4, modes_a=64, width=64, layers=4, scale=4,
         width_full=8),
    dict(arch="specprop", modes=0, modes_a=64, width=64, layers=4, scale=4,
         width_full=8),
]


def _params_from_build_string(cfg):
    ns = {}
    exec(build_string(cfg, COND_DIM), ns)  # noqa: S102 - the string under test
    return sum(p.numel() for p in ns["M"].parameters())


def test_priced_model_matches_scored_model():
    """Same config, two constructions, same parameter count."""
    for cfg in CONFIGS:
        scored = sum(p.numel() for p in build_from_cfg(cfg, COND_DIM).parameters())
        priced = _params_from_build_string(cfg)
        assert priced == scored, (
            f"{cfg['arch']} modes={cfg['modes']}: priced model has {priced} "
            f"parameters, scored model has {scored}. The cost column and the "
            f"accuracy column are different models."
        )


def test_priced_model_runs_and_matches_scored_output_shape():
    """It must also be callable on the shape the timer uses (128x128)."""
    for cfg in CONFIGS:
        ns = {}
        exec(build_string(cfg, COND_DIM), ns)  # noqa: S102
        phi = torch.zeros(1, 1, 128, 128)
        cond = torch.zeros(1, COND_DIM)
        with torch.no_grad():
            priced_out = ns["M"](phi, cond)
            scored_out = build_from_cfg(cfg, COND_DIM)(phi, cond)
        assert priced_out.shape == scored_out.shape == phi.shape


def test_priced_model_is_the_same_CLASS_as_the_scored_model():
    """The check that does not depend on a coincidence, and the reason it exists.

    Parameter count nearly failed to catch the real bug. For the actual config
    on disk (`runs/specprop/m4_ma64_s1`) the wrongly-built `MultiScaleOperator`
    has 1,080,322 parameters against the `SpectralPropagator`'s 1,070,144 --
    within 1%. Exact equality still fails, so the count test does its job here,
    but it does it by a margin of 0.95% and a different width or layer count
    could have made the two coincide exactly while the COST differed by an
    order of magnitude. Class identity cannot coincide.
    """
    for cfg in CONFIGS:
        scored_cls = type(build_from_cfg(cfg, COND_DIM)).__name__
        ns = {}
        exec(build_string(cfg, COND_DIM), ns)  # noqa: S102
        priced_cls = type(ns["M"]).__name__
        assert priced_cls == scored_cls, (
            f"{cfg['arch']}: priced as {priced_cls}, scored as {scored_cls}"
        )


def test_every_known_arch_has_a_price_branch():
    """No architecture may fall through to a default constructor."""
    covered = {c["arch"] for c in CONFIGS}
    assert covered == set(ARCH_FIELDS), (
        f"architectures in ARCH_FIELDS but not priced here: "
        f"{set(ARCH_FIELDS) - covered}. Add a CONFIGS entry and a build_string "
        f"branch before scoring them, or the fall-through will price the wrong "
        f"model."
    )


def test_unknown_arch_raises_rather_than_defaulting():
    """The fall-through must be loud. This is the whole bug, in one line."""
    try:
        build_string(dict(arch="not_an_arch", width=8, modes=4, layers=2), COND_DIM)
    except ValueError as e:
        assert "unknown arch" in str(e), e
        return
    raise AssertionError("an unknown arch was priced instead of raising")


def test_real_specprop_run_prices_as_specprop():
    """Against a config on disk, not a synthetic one.

    The synthetic cases above could both be wrong in the same way if I wrote
    them from the same mental model. A real args.json cannot be.
    """
    run = ROOT / "runs" / "specprop" / "m4_ma64_s1"
    if not (run / "args.json").exists():
        print("   (skipped: specprop run not present in this tree)")
        return
    cfg = json.loads((run / "args.json").read_text())
    assert "SpectralPropagator" in build_string(cfg, COND_DIM)
    assert _params_from_build_string(cfg) == cfg["params"], (
        "priced parameter count disagrees with the count the TRAINING run "
        "recorded in args.json"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
