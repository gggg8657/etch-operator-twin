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


def fit_norm(train_npz: str | Path) -> dict:
    d = np.load(train_npz)
    c = cond_vector(d["recipe"], d["dt"])
    return {
        "cond_mean": c.mean(axis=0).tolist(),
        "cond_std": (c.std(axis=0) + 1e-8).tolist(),
        "cond_keys": COND_KEYS,
        "sdf_scale_um": SDF_SCALE,
        "band_um": BAND_UM,
        "n_train_trajectories": int(d["sdf"].shape[0]),
    }


class PairDataset(Dataset):
    """Every (t -> t+1) transition in the split, flattened."""

    def __init__(self, npz: str | Path, norm: dict):
        d = np.load(npz)
        self.sdf = d["sdf"]  # (N, T+1, H, W), micron
        self.cond = (cond_vector(d["recipe"], d["dt"]) - np.array(norm["cond_mean"], np.float32)) / np.array(
            norm["cond_std"], np.float32
        )
        self.N, self.Tp1 = self.sdf.shape[0], self.sdf.shape[1]
        self.T = self.Tp1 - 1
        self.scale = norm["sdf_scale_um"]

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, i):
        n, t = divmod(i, self.T)
        a = torch.from_numpy(self.sdf[n, t][None]).float() / self.scale
        b = torch.from_numpy(self.sdf[n, t + 1][None]).float() / self.scale
        return a, torch.from_numpy(self.cond[n]).float(), b


class TrajDataset(Dataset):
    """Whole trajectories, for rollout training and rollout evaluation."""

    def __init__(self, npz: str | Path, norm: dict):
        d = np.load(npz)
        self.sdf = d["sdf"]
        self.recipe = d["recipe"]
        self.dt = d["dt"]
        self.solver_s = d["solver_s"] if "solver_s" in d else None
        self.cond = (cond_vector(d["recipe"], d["dt"]) - np.array(norm["cond_mean"], np.float32)) / np.array(
            norm["cond_std"], np.float32
        )
        self.scale = norm["sdf_scale_um"]

    def __len__(self):
        return self.sdf.shape[0]

    def __getitem__(self, i):
        traj = torch.from_numpy(self.sdf[i]).float() / self.scale  # (T+1,H,W)
        return traj[0][None], torch.from_numpy(self.cond[i]).float(), traj[1:][:, None]


def band_mask(target_um: torch.Tensor, band_um: float = BAND_UM) -> torch.Tensor:
    """True within `band_um` of the interface of the *ground truth* field.

    Keyed on the target, never the prediction, so a model cannot widen or narrow
    its own evaluation region.
    """
    return target_um.abs() < band_um
