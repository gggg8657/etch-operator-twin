"""Dataset: ViennaPS trajectories -> (phi_t, recipe, dt) -> phi_{t+dt} pairs.

Normalisation constants come from the *train* split only and are written to
`data/norm.json`, so val/test are scored under constants they did not
contribute to. The SDF scale is a fixed 5 um rather than a fitted std: the
field's std is dominated by the far-field ramp and would drift with the window,
while 5 um is a stated length that stays comparable across any later change of
window or resolution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SDF_SCALE = 5.0  # micron
BAND_UM = 1.5  # half-width of the narrow band the headline metric uses

COND_KEYS = ["log_ion_flux", "log_etchant_flux", "oxygen_flux", "ion_energy",
             "trench_width", "mask_height", "log_dt"]

# The same six recipe channels, with the horizon expressed as the DEPTH the
# application must advance instead of the time it must advance.
#
# Why the alternative exists. runs/bench_workload.json measures a ceiling on the
# speedup clause that no architecture can beat: the operator needs `log_dt`
# before it can be queried, and if the caller's query is a target depth then
# obtaining dt costs eot.solver.probe_rate -- a real solver run, 39.9 ms --
# which bounds the speedup at solver/probe = 12.8x however fast the network
# gets. Conditioning on depth removes the probe by construction, because the
# depth IS the query.
#
# The depth channel is the *achieved* depth derived from the stored frames by
# scripts/derive_depth.py, not the `target_depth` the dataset stores: the stored
# field is the depth requested of choose_dt (which sizes dt from the rate on the
# initial flat geometry, so the etch under-delivers by a measured 2.3-2.6%), and
# it is NaN for all 209 test_crossed trajectories.
COND_KEYS_DEPTH = ["log_ion_flux", "log_etchant_flux", "oxygen_flux",
                   "ion_energy", "trench_width", "mask_height", "depth_um"]
COND_MODES = ("dt", "depth")


def cond_vector(recipe: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """(N,6) recipe + (N,) dt -> (N,7) raw conditioning, before standardising."""
    recipe = np.atleast_2d(recipe)
    dt = np.atleast_1d(dt)
    return np.stack(
        [
            np.log(recipe[:, 0]),
            np.log(recipe[:, 1]),
            recipe[:, 2],
            recipe[:, 3],
            recipe[:, 4],
            recipe[:, 5],
            np.log(dt),
        ],
        axis=1,
    ).astype(np.float32)


def cond_vector_depth(recipe: np.ndarray, depth_um: np.ndarray) -> np.ndarray:
    """(N,6) recipe + (N,) depth in micron -> (N,7) raw conditioning.

    Depth enters linearly, not as a log. dt is log-scaled because it spans
    0.060-1.000, a factor of 17, and because the rate law is multiplicative in
    the fluxes. Achieved depth spans 2.2-9.9 um on train -- a factor of 4.5,
    already on a scale the standardiser handles -- and 0 is a meaningful value
    for it (no etch) where log would send it to -inf.
    """
    recipe = np.atleast_2d(recipe)
    depth_um = np.atleast_1d(depth_um)
    return np.stack(
        [
            np.log(recipe[:, 0]),
            np.log(recipe[:, 1]),
            recipe[:, 2],
            recipe[:, 3],
            recipe[:, 4],
            recipe[:, 5],
            depth_um,
        ],
        axis=1,
    ).astype(np.float32)


def load_depth(npz: str | Path, source: str = "achieved") -> np.ndarray:
    """(N,) final depth for the split `npz` belongs to, by source.

    **The two sources are not interchangeable and the difference is a leakage
    question, not a detail.**

    * `achieved` is derived from the split's own stored frames by
      scripts/derive_depth.py. It is the right thing to TRAIN on -- it is what
      the model must actually reproduce -- and it exists for every split
      including test_crossed.
    * `requested` is the dataset's stored `target_depth`, the number
      `choose_dt` was asked for. It is what a CALLER would supply at deployment,
      because a caller does not know the achieved depth before running
      something.

    Conditioning an evaluation on `achieved` therefore hands the model a
    quantity derived from the label it is being scored against. The two differ
    by a measured 2.3-2.6% (achieved/requested median 0.974-0.977, Pearson r
    0.952-0.965, runs/depth_derivation.json), so the leak is small -- but a
    small leak reported as a deployable number is still the thing this repo
    forbids. `achieved` is labelled an ORACLE reading wherever it is scored, and
    `requested` is the deployable one.

    `requested` is unavailable on test_crossed: all 209 trajectories carry NaN,
    because that split was generated in independent-dt mode which never
    computes a target. Asking for it there raises.
    """
    if source == "requested":
        d = np.load(npz)
        if "target_depth" not in d.files:
            raise KeyError(f"{npz} stores no target_depth")
        t = d["target_depth"].astype(np.float64)
        if not np.isfinite(t).all():
            raise ValueError(
                f"{npz}: target_depth has {int((~np.isfinite(t)).sum())} "
                f"non-finite entries of {t.size}; the requested-depth reading "
                f"does not exist for this split (independent-dt generation). "
                f"Use source='achieved' and label it an oracle reading.")
        return t
    if source != "achieved":
        raise ValueError(f"unknown depth source {source!r}")
    return _load_achieved_depth(npz)


def _load_achieved_depth(npz: str | Path) -> np.ndarray:
    """(N, T+1) achieved depth for the split `npz` belongs to.

    Written by scripts/derive_depth.py next to the split it describes. Absent
    means the script has not been run, which must be an error rather than a
    silent fallback to dt conditioning -- a run whose args say `depth` and whose
    conditioning is dt would be unfalsifiable.
    """
    npz = Path(npz)
    p = npz.with_name(f"depth_{npz.stem}.npy")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing; run scripts/derive_depth.py before training with "
            f"--cond depth")
    return np.load(p)[:, -1]


def mean_step_displacement(sdf_um: np.ndarray, band_um: float = BAND_UM) -> float:
    """Mean per-step normal displacement of the interface, in um, over a split.

    Fitted on train only and used to build the *recipe-blind* null: advance every
    surface by this constant regardless of recipe, dt or target. Because the
    dataset's timestep is chosen per recipe so that each trajectory covers a
    comparable depth, per-step displacement is nearly constant across the recipe
    box by construction -- so this null is strong, and an operator that fails to
    beat it has learned nothing the conditioning was supposed to supply.
    """
    prev, nxt = sdf_um[:, :-1], sdf_um[:, 1:]
    band = (np.abs(nxt) < band_um) | (np.abs(prev) < band_um)
    w = band.reshape(band.shape[0], band.shape[1], -1).astype(np.float64)
    diff = (nxt - prev).reshape(w.shape)
    num = (diff * w).sum(axis=2)
    den = np.clip(w.sum(axis=2), 1.0, None)
    return float((num / den).mean())


def norm_path(data_dir: str | Path, cond_mode: str = "dt") -> Path:
    """Which norm file a given conditioning mode uses.

    Depth conditioning has different channels, different means and different
    stds, so it cannot share `norm.json`. Keeping the dt file at its historical
    path means every run and every script written before depth existed resolves
    to exactly the same file it always did.
    """
    data_dir = Path(data_dir)
    return data_dir / ("norm.json" if cond_mode == "dt"
                       else f"norm_{cond_mode}.json")


def fit_norm(train_npz: str | Path, cond_mode: str = "dt") -> dict:
    d = np.load(train_npz)
    if cond_mode not in COND_MODES:
        raise ValueError(f"cond_mode {cond_mode!r} not in {COND_MODES}")
    if cond_mode == "dt":
        c = cond_vector(d["recipe"], d["dt"])
    else:
        c = cond_vector_depth(d["recipe"], load_depth(train_npz))
    return {
        "cond_mode": cond_mode,
        "mean_step_displacement_um": mean_step_displacement(d["sdf"]),
        # dt range actually seen in training. Inverse design that treats total
        # etch time as unknown searches inside this, not outside it.
        "dt_lo": float(np.min(d["dt"])),
        "dt_hi": float(np.max(d["dt"])),
        "cond_mean": c.mean(axis=0).tolist(),
        "cond_std": (c.std(axis=0) + 1e-8).tolist(),
        "cond_keys": COND_KEYS if cond_mode == "dt" else COND_KEYS_DEPTH,
        "sdf_scale_um": SDF_SCALE,
        "band_um": BAND_UM,
        "n_train_trajectories": int(d["sdf"].shape[0]),
    }


def standardise_depth(recipe: np.ndarray, depth_um: np.ndarray,
                      norm: dict) -> np.ndarray:
    """Standardised depth-mode conditioning. No stride argument, deliberately.

    In dt mode a stride-K arm scales the horizon channel by K, because K
    applications of one dt-conditioned operator each advance the same dt. Depth
    does not work that way: the etch rate falls as the trench deepens, so the
    depth advanced by application 1 is larger than the depth advanced by
    application 2, and there is no single depth value that describes all K
    applications of a rollout. Multiplying or dividing by K would be silently
    wrong rather than approximately right.

    So depth conditioning is only well-posed where there is exactly ONE
    application per wafer, and the callers enforce that rather than papering
    over it. That is not a limitation of the experiment it was built for --
    stride 10, one application, the configuration that meets clause 1
    in-distribution at 8 seeds and is the cheap one -- but it does mean depth
    conditioning cannot be dropped into the K-curve.
    """
    c = cond_vector_depth(recipe, depth_um)
    return (c - np.array(norm["cond_mean"], np.float32)) / np.array(
        norm["cond_std"], np.float32)


def _cond_for(d, npz, norm, stride, depth_source="achieved"):
    """Whichever conditioning `norm` says this dataset was normalised for."""
    if norm.get("cond_mode", "dt") == "dt":
        return standardise(d["recipe"], d["dt"], norm, stride)
    T = d["sdf"].shape[1] - 1
    if stride != T:
        raise ValueError(
            f"depth conditioning needs one application per wafer (stride == "
            f"{T}), got stride={stride}. See standardise_depth: the depth "
            f"advanced differs between applications, so no single value "
            f"describes a {T // stride}-application rollout.")
    return standardise_depth(d["recipe"], load_depth(npz, depth_source), norm)


def standardise(recipe: np.ndarray, dt: np.ndarray, norm: dict,
                stride: int = 1) -> np.ndarray:
    """Standardised conditioning for an operator that advances `stride` steps.

    A K-step operator is applied once per K*dt of physical time, so the `log_dt`
    channel carries log(K*dt) -- the horizon the application actually covers, not
    the dataset's underlying timestep. Everything else is untouched.

    `norm` is NOT refitted per stride. All arms are standardised by the same
    train-split constants so a K-curve is not confounded by a change of
    normalisation; the price is that the K=10 arm sees standardised log_dt out
    where K=1 never went, which is a real difference between the arms and is
    recorded rather than hidden.
    """
    c = cond_vector(recipe, np.asarray(dt, dtype=np.float64) * stride)
    return (c - np.array(norm["cond_mean"], np.float32)) / np.array(
        norm["cond_std"], np.float32)


def pair_starts(T: int, stride: int, overlap: bool) -> list[int]:
    """Start offsets for (t -> t+stride) pairs within a trajectory of T steps.

    `overlap=False` takes the T//stride non-overlapping pairs, which is the
    partition the arm is actually rolled out over. `overlap=True` takes every
    one of the T-stride+1 legal starts, which restores the pair count to ~T at
    any stride -- the data-matched control that separates a horizon effect from
    a training-set-size effect.
    """
    if stride < 1 or stride > T:
        raise ValueError(f"stride {stride} outside 1..{T}")
    return list(range(0, T - stride + 1)) if overlap else list(
        range(0, T - T % stride, stride))


class PairDataset(Dataset):
    """Every (t -> t+stride) transition in the split, flattened.

    stride=1, overlap=False is the original one-step dataset, exactly.
    """

    def __init__(self, npz: str | Path, norm: dict, stride: int = 1,
                 overlap: bool = False, depth_source: str = "achieved"):
        d = np.load(npz)
        self.sdf = d["sdf"]  # (N, T+1, H, W), micron
        self.stride, self.overlap = int(stride), bool(overlap)
        self.cond = _cond_for(d, npz, norm, self.stride, depth_source)
        self.N, self.Tp1 = self.sdf.shape[0], self.sdf.shape[1]
        self.T = self.Tp1 - 1
        self.starts = pair_starts(self.T, self.stride, self.overlap)
        self.scale = norm["sdf_scale_um"]

    def __len__(self):
        return self.N * len(self.starts)

    def __getitem__(self, i):
        n, j = divmod(i, len(self.starts))
        t = self.starts[j]
        a = torch.from_numpy(self.sdf[n, t][None]).float() / self.scale
        b = torch.from_numpy(self.sdf[n, t + self.stride][None]).float() / self.scale
        return a, torch.from_numpy(self.cond[n]).float(), b


class TrajDataset(Dataset):
    """Whole trajectories, for rollout training and rollout evaluation."""

    def __init__(self, npz: str | Path, norm: dict, stride: int = 1,
                 depth_source: str = "achieved"):
        d = np.load(npz)
        self.sdf = d["sdf"]
        self.recipe = d["recipe"]
        self.dt = d["dt"]
        self.stride = int(stride)
        self.depth_source = depth_source
        self.solver_s = d["solver_s"] if "solver_s" in d else None
        self.cond = _cond_for(d, npz, norm, self.stride, depth_source)
        self.scale = norm["sdf_scale_um"]
        T = self.sdf.shape[1] - 1
        if T % self.stride:
            raise ValueError(
                f"stride {self.stride} does not divide the {T}-step trajectory, so "
                f"the arm would not end at the same physical time as the others")
        # the states a stride-K operator emits: t = K, 2K, ... T
        self.times = list(range(self.stride, T + 1, self.stride))

    def __len__(self):
        return self.sdf.shape[0]

    def __getitem__(self, i):
        traj = torch.from_numpy(self.sdf[i]).float() / self.scale  # (T+1,H,W)
        return (traj[0][None], torch.from_numpy(self.cond[i]).float(),
                traj[self.times][:, None])


def band_mask(target_um: torch.Tensor, band_um: float = BAND_UM) -> torch.Tensor:
    """True within `band_um` of the interface of the *ground truth* field.

    Keyed on the target, never the prediction, so a model cannot widen or narrow
    its own evaluation region.
    """
    return target_um.abs() < band_um
