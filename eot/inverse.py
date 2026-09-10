"""Inverse design: descend on the recipe until the operator produces a target profile.

The operator is differentiable in its conditioning, so the recipe is just
another tensor to optimise. Two things make this an honest test rather than a
demonstration:

1. **The search is confined to the training box.** Recipes are parameterised as
   an unconstrained vector squashed through a sigmoid into the box the solver
   was sampled from, so the optimiser cannot walk into a region where the
   surrogate has never been checked. A recipe outside the box that "works" on
   the operator is the classic failure of this method; here it cannot be
   produced at all, and the distance to the box edge is reported so a solution
   pinned against a wall is visible.
2. **The answer is scored in the simulator, not on the surrogate.** The recipe
   the operator proposes is run through ViennaPS and the shape error is measured
   there. The surrogate's own opinion of its solution is reported too, and the
   gap between the two is the number that says whether the surrogate is usable
   for design.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import cond_vector
from .solver import GEOM_BOX, RECIPE_BOX, Recipe

# Order must match `gen_data.RECIPE_KEYS`.
DESIGN_KEYS = ["ion_flux", "etchant_flux", "oxygen_flux", "ion_energy"]


def _box(keys):
    los, his, modes = [], [], []
    allb = {**RECIPE_BOX, **GEOM_BOX}
    for k in keys:
        lo, hi, mode = allb[k]
        los.append(lo)
        his.append(hi)
        modes.append(mode)
    return np.array(los), np.array(his), modes


class RecipeParam(torch.nn.Module):
    """Unconstrained z -> recipe strictly inside the training box.

    Log-scaled knobs are interpolated in log space, matching how the dataset
    sampled them, so a uniform step in z is a uniform step in the sampling
    measure rather than one that crawls at the low end of a decade.
    """

    def __init__(self, keys=DESIGN_KEYS, init=None, device="cuda:0"):
        super().__init__()
        self.keys = list(keys)
        lo, hi, modes = _box(self.keys)
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32, device=device))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32, device=device))
        self.log_mask = torch.tensor([m == "log" for m in modes], device=device)
        if init is None:
            z0 = torch.zeros(len(self.keys))
        else:
            z0 = torch.logit(torch.clamp(self._to_unit(torch.as_tensor(init, dtype=torch.float32)), 1e-3, 1 - 1e-3))
        self.z = torch.nn.Parameter(z0.to(device))

    def _to_unit(self, v):
        """Inverse of `values`. Same clamp, same reason."""
        lo, hi = self.lo.cpu().clamp_min(1e-12), self.hi.cpu().clamp_min(1e-12)
        lm = self.log_mask.cpu()
        v_s = torch.as_tensor(v, dtype=torch.float32).clamp_min(1e-12)
        u = torch.where(lm,
                        (torch.log(v_s) - torch.log(lo)) / (torch.log(hi) - torch.log(lo)),
                        (v_s - self.lo.cpu()) / (self.hi.cpu() - self.lo.cpu()))
        return u

    def unit(self):
        return torch.sigmoid(self.z)

    def values(self):
        """Unit cube -> recipe box, log-interpolating the knobs sampled in log space.

        `lo` must be clamped away from zero before the log even though the
        zero-valued knob (oxygen flux, lo = 0) is *not* log-masked.
        `torch.where` computes both branches and differentiates both: with
        log(0) = -inf, the unused branch produces 0 * inf = NaN in the backward
        pass, `torch.where` propagates it through the mask, and the whole recipe
        goes NaN on the first step. Symptom was a silent one -- the optimiser
        reported nan loss, proposed a nan recipe, and the failure surfaced only
        as a crashed verification subprocess with an empty stderr.
        """
        u = self.unit()
        lin = self.lo + u * (self.hi - self.lo)
        lo_s = self.lo.clamp_min(1e-12)
        hi_s = self.hi.clamp_min(1e-12)
        lg = torch.exp(torch.log(lo_s) + u * (torch.log(hi_s) - torch.log(lo_s)))
        return torch.where(self.log_mask, lg, lin)

    def margin(self):
        """Distance to the nearest box wall, in units of the box width.

        0 means pinned against a wall -- the optimiser wanted to leave the region
        the simulator was sampled in, and the reported recipe is a clipped
        answer, not an interior optimum.
        """
        u = self.unit()
        return float(torch.minimum(u, 1 - u).min())


def build_cond(values: torch.Tensor, geom: dict, dt, norm: dict, device) -> torch.Tensor:
    """Differentiable version of `data.cond_vector` + standardisation.

    Must stay numerically identical to the training-time path; if it drifts, the
    operator is being conditioned on a different vector at design time than at
    fit time and the whole exercise is measuring that mismatch instead.
    """
    mean = torch.tensor(norm["cond_mean"], dtype=torch.float32, device=device)
    std = torch.tensor(norm["cond_std"], dtype=torch.float32, device=device)
    c = torch.stack([
        torch.log(values[0]),
        torch.log(values[1]),
        values[2],
        values[3],
        torch.tensor(float(geom["trench_width"]), device=device),
        torch.tensor(float(geom["mask_height"]), device=device),
        (torch.log(dt) if torch.is_tensor(dt)
         else torch.tensor(float(np.log(dt)), device=device)),
    ])
    return ((c - mean) / std)[None]


def design(
    model,
    target_phi: torch.Tensor,
    phi0: torch.Tensor,
    geom: dict,
    dt: float,
    norm: dict,
    n_steps: int,
    init=None,
    iters: int = 300,
    lr: float = 0.05,
    band_um: float = 1.5,
    device="cuda:0",
    seed: int = 0,
    optimise_dt: bool = False,
):
    """Descend on the recipe to match `target_phi` after `n_steps` operator steps.

    Loss is taken in a band around the target interface: matching the far field
    of an SDF is free and would let a bad recipe post a small loss.

    `optimise_dt` decides whether **total etch time is known**. With it False the
    caller supplies the target's own dt, so `T = n_steps * dt` is pinned to the
    ground truth -- and since dt was itself derived from a probe of the true
    recipe's etch rate, that hands the optimiser the degree of freedom that sets
    depth, which is the dominant term in shape error. That was expected to be the
    easy protocol and to bound the achievable error from below. **Measured, it is
    the worse of the two** (0.0098 against 0.0061): pinning T is a constraint, and
    the constraint costs more than the information it hands over, because with T
    free the optimiser can trade rate against time along a family of processes
    that reach the same profile. With it True, dt is searched inside
    the range seen in training alongside the recipe, which is the problem a real
    target poses: a profile arrives with no duration attached.
    """
    torch.manual_seed(seed)
    p = RecipeParam(init=init, device=device).to(device)
    params = list(p.parameters())
    z_dt = None
    if optimise_dt:
        lo, hi = norm["dt_lo"], norm["dt_hi"]
        u0 = (np.log(dt) - np.log(lo)) / (np.log(hi) - np.log(lo))
        z_dt = torch.nn.Parameter(
            torch.logit(torch.tensor(float(np.clip(u0, 1e-3, 1 - 1e-3)), device=device)))
        params.append(z_dt)
    opt = torch.optim.Adam(params, lr=lr)
    scale = norm["sdf_scale_um"]
    band = (target_phi.abs() * scale < band_um).float()
    hist = []
    best = (float("inf"), None, float(dt))
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        if z_dt is not None:
            lo, hi = norm["dt_lo"], norm["dt_hi"]
            dt_t = torch.exp(np.log(lo) + torch.sigmoid(z_dt) * (np.log(hi) - np.log(lo)))
        else:
            dt_t = dt
        cond = build_cond(p.values(), geom, dt_t, norm, device)
        phi = phi0
        for _ in range(n_steps):
            phi = model(phi, cond)
        num = (((phi - target_phi) ** 2) * band).sum().sqrt()
        den = ((target_phi ** 2) * band).sum().sqrt().clamp_min(1e-8)
        loss = num / den
        loss.backward()
        opt.step()
        v = float(loss)
        hist.append(v)
        if v < best[0]:
            best = (v, p.values().detach().cpu().numpy().copy(),
                    float(dt_t) if z_dt is not None else float(dt))
    return {
        "loss_history": hist,
        "final_surrogate_loss": hist[-1],
        "best_surrogate_loss": best[0],
        "recipe_values": best[1].tolist(),
        "dt": best[2],
        "dt_was_optimised": bool(optimise_dt),
        "dt_given": float(dt),
        "recipe_keys": p.keys,
        "box_margin": p.margin(),
        "iters": iters,
        "lr": lr,
    }


def recipe_from_values(values, geom: dict) -> Recipe:
    kw = dict(zip(DESIGN_KEYS, [float(v) for v in values]))
    kw.update(trench_width=float(geom["trench_width"]), mask_height=float(geom["mask_height"]))
    return Recipe(**kw)
