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


def fit_norm(train_npz: str | Path) -> dict:
    d = np.load(train_npz)
    c = cond_vector(d["recipe"], d["dt"])
    return {
        "mean_step_displacement_um": mean_step_displacement(d["sdf"]),
        # dt range actually seen in training. Inverse design that treats total
        # etch time as unknown searches inside this, not outside it.
        "dt_lo": float(np.min(d["dt"])),
        "dt_hi": float(np.max(d["dt"])),
        "cond_mean": c.mean(axis=0).tolist(),
        "cond_std": (c.std(axis=0) + 1e-8).tolist(),
        "cond_keys": COND_KEYS,
        "sdf_scale_um": SDF_SCALE,
        "band_um": BAND_UM,
        "n_train_trajectories": int(d["sdf"].shape[0]),
    }


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
                 overlap: bool = False):
        d = np.load(npz)
        self.sdf = d["sdf"]  # (N, T+1, H, W), micron
        self.stride, self.overlap = int(stride), bool(overlap)
        self.cond = standardise(d["recipe"], d["dt"], norm, self.stride)
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

    def __init__(self, npz: str | Path, norm: dict, stride: int = 1):
        d = np.load(npz)
        self.sdf = d["sdf"]
        self.recipe = d["recipe"]
        self.dt = d["dt"]
        self.stride = int(stride)
        self.solver_s = d["solver_s"] if "solver_s" in d else None
        self.cond = standardise(d["recipe"], d["dt"], norm, self.stride)
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
