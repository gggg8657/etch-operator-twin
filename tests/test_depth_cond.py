"""Depth conditioning: back-compatibility, the guard, and the leakage label.

Depth conditioning exists to dissolve a ceiling: the operator needs its horizon
channel before it can be queried, and if the query is a target depth then
obtaining dt costs a solver probe, capping the speedup clause at 12.8x for every
architecture. Swapping the channel removes the probe.

The ways that change could go wrong quietly, which is what these pin:

1. **It changes dt-mode behaviour.** Every run in this repo was trained in dt
   mode. If `fit_norm`/`standardise` shift by even a float, every stored
   checkpoint is being scored under different constants than it was trained on.
2. **A depth-conditioned checkpoint gets scored under dt constants.** The
   channel counts are identical (7), so nothing would raise -- the model would
   be fed a standardised log(dt) in the channel it learned as micron of depth
   and would return plausible nonsense.
3. **The multi-application case is silently approximated.** The etch rate falls
   with depth, so no single depth describes a K-application rollout.
4. **The oracle reading is reported as deployable.** Conditioning an evaluation
   on the depth derived from the split's own frames uses the label.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import (COND_KEYS, COND_KEYS_DEPTH, PairDataset,  # noqa: E402
                      TrajDataset, fit_norm, load_depth, norm_path)

DATA = Path("data")
HAVE = (DATA / "train.npz").exists() and (DATA / "depth_train.npy").exists()


def test_dt_mode_is_byte_identical_to_the_stored_norm():
    """The regression that would invalidate every checkpoint in the repo."""
    if not HAVE:
        return
    stored = json.loads((DATA / "norm.json").read_text())
    refit = fit_norm(DATA / "train.npz", cond_mode="dt")
    assert refit["cond_keys"] == stored["cond_keys"] == COND_KEYS
    for k in ("cond_mean", "cond_std", "sdf_scale_um", "band_um"):
        a, b = stored[k], refit[k]
        if isinstance(a, list):
            assert all(abs(x - y) < 1e-12 for x, y in zip(a, b)), k
        else:
            assert abs(a - b) < 1e-12, k


def test_norm_path_keeps_dt_at_its_historical_location():
    assert norm_path(DATA, "dt").name == "norm.json"
    assert norm_path(DATA, "depth").name != "norm.json"


def test_depth_mode_has_its_own_channel_names():
    assert COND_KEYS_DEPTH[-1] == "depth_um"
    assert COND_KEYS[-1] == "log_dt"
    assert COND_KEYS_DEPTH[:-1] == COND_KEYS[:-1], "recipe channels must match"
    assert len(COND_KEYS_DEPTH) == len(COND_KEYS)


def test_depth_conditioning_refuses_multi_application_stride():
    """Hazard 3: approximating here would be wrong, not roughly right."""
    if not HAVE:
        return
    nd = fit_norm(DATA / "train.npz", cond_mode="depth")
    for bad in (1, 2, 5):
        try:
            PairDataset(DATA / "train.npz", nd, stride=bad)
        except ValueError as e:
            assert "one application per wafer" in str(e)
            continue
        raise AssertionError(f"stride {bad} was accepted in depth mode")
    PairDataset(DATA / "train.npz", nd, stride=10)  # the well-posed case


def test_requested_depth_is_refused_where_it_does_not_exist():
    """Hazard 4's other half: test_crossed stores NaN for all 209."""
    if not (DATA / "test_crossed.npz").exists():
        return
    try:
        load_depth(DATA / "test_crossed.npz", "requested")
    except ValueError as e:
        assert "non-finite" in str(e)
        return
    raise AssertionError("requested depth was returned for test_crossed")


def test_achieved_and_requested_differ_and_in_the_predicted_direction():
    """The leak is small but real, and its SIGN was predicted before measuring:
    choose_dt sizes dt from the rate on the initial flat geometry and the rate
    falls as the trench deepens, so the etch must under-deliver."""
    if not HAVE:
        return
    a = load_depth(DATA / "train.npz", "achieved")
    r = load_depth(DATA / "train.npz", "requested")
    assert a.shape == r.shape
    ratio = float(np.median(a / r))
    assert 0.90 < ratio < 1.0, f"achieved/requested = {ratio}"
    assert float(np.corrcoef(a, r)[0, 1]) > 0.8, "extraction looks wrong"


def test_the_two_sources_give_different_conditioning():
    """If they did not, the oracle/deployable distinction would be cosmetic."""
    if not HAVE:
        return
    nd = fit_norm(DATA / "train.npz", cond_mode="depth")
    ach = TrajDataset(DATA / "train.npz", nd, stride=10, depth_source="achieved")
    req = TrajDataset(DATA / "train.npz", nd, stride=10, depth_source="requested")
    assert not np.allclose(ach.cond, req.cond)
    # and only in the horizon channel
    assert np.allclose(ach.cond[:, :-1], req.cond[:, :-1])


def test_depth_norm_is_fitted_on_train_only():
    """val/test/crossed must be scored under constants they did not shape."""
    if not (DATA / "norm_depth.json").exists():
        return
    nd = json.loads((DATA / "norm_depth.json").read_text())
    assert nd["cond_mode"] == "depth"
    assert nd["n_train_trajectories"] == np.load(DATA / "train.npz")["sdf"].shape[0]
    refit = fit_norm(DATA / "train.npz", cond_mode="depth")
    assert all(abs(x - y) < 1e-9
               for x, y in zip(nd["cond_mean"], refit["cond_mean"]))


def test_derived_depth_is_monotone_and_finite():
    if not HAVE:
        return
    rep = json.loads(Path("runs/depth_derivation.json").read_text())
    for sp, row in rep["splits"].items():
        if row.get("absent"):
            continue
        assert row["n_nan"] == 0, sp
        assert row["monotone_fraction"] == 1.0, sp


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
