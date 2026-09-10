"""Recipe-conditioned 2-D FNO for etch surface evolution.

`SpectralConv2d` and `FNOBlock` are taken unchanged from
`pde-neural-operator/pdeno/fno2d.py` (real-stored complex weights, so every
gradient stays real and DDP can compress the all-reduce to bf16; FFTs forced to
fp32 because `torch.fft` has no half kernels). The one substantive change is the
conditioning: that model selects a PDE family with a discrete `nn.Embedding`,
while an etch recipe is a point in a continuous box, so the embedding is
replaced by an MLP over the standardised recipe vector. Everything else --
lifting, block stack, projection, coordinate channels -- is the reference
architecture, so a regression here is a regression in the conditioning, not in
the operator.

The model maps (phi_t, recipe, dt) -> phi_{t+dt}, predicting the *residual*
phi_{t+dt} - phi_t. Residual form matters for more than optimisation: the
identity map is a strong predictor of an SDF over one step, so a model that
predicted phi directly could score well while having learned nothing. Predicting
the residual makes that failure visible instead of hiding it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """R^{C_in x H x W} -> R^{C_out x H x W} via a truncated Fourier multiply."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_lo = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))
        self.w_hi = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes1, modes2, 2))

    def forward(self, x):  # (B, C, H, W)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if x.dtype in (torch.bfloat16, torch.float16):
                x = x.float()
            B, _, H, W = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            m1 = min(self.modes1, H // 2)
            m2 = min(self.modes2, W // 2 + 1)
            w_lo = torch.view_as_complex(self.w_lo.contiguous())
            w_hi = torch.view_as_complex(self.w_hi.contiguous())
            out_dtype = torch.promote_types(x_ft.dtype, w_lo.dtype)
            out_ft = torch.zeros(
                B, self.out_ch, H, W // 2 + 1, dtype=out_dtype, device=x.device
            )
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w_lo[:, :, :m1, :m2]
            )
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], w_hi[:, :, :m1, :m2]
            )
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").to(x.dtype)


class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2, norm=True):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(8, width) if norm else nn.Identity()

    def forward(self, x):
        return x + F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class EtchOperator(nn.Module):
    """phi_{t+dt} = phi_t + G(phi_t, recipe, dt)."""

    def __init__(
        self,
        cond_dim: int = 7,
        width: int = 64,
        modes: int = 20,
        n_layers: int = 4,
        cond_ch: int = 24,
        norm: bool = True,
    ):
        super().__init__()
        self.width, self.modes, self.n_layers = width, modes, n_layers
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.GELU(), nn.Linear(64, cond_ch)
        )
        self.lift = nn.Sequential(
            nn.Conv2d(1 + 2 + cond_ch, width, 1), nn.GELU(), nn.Conv2d(width, width, 1)
        )
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes, modes, norm=norm) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, 1, 1)
        )

    @staticmethod
    def coords(B, H, W, device, dtype):
        y = torch.linspace(0, 1, H, device=device, dtype=dtype)
        x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([X, Y]).expand(B, 2, H, W)

    def residual(self, phi, cond):
        """phi: (B,1,H,W) normalised SDF. cond: (B, cond_dim) standardised."""
        B, _, H, W = phi.shape
        c = self.cond(cond)[:, :, None, None].expand(-1, -1, H, W).to(phi.dtype)
        x = torch.cat([phi, self.coords(B, H, W, phi.device, phi.dtype), c], dim=1)
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(x)

    def forward(self, phi, cond):
        return phi + self.residual(phi, cond)

    def rollout(self, phi0, cond, n_steps):
        """Autoregressive rollout. Returns (B, n_steps, 1, H, W)."""
        out = []
        phi = phi0
        for _ in range(n_steps):
            phi = self.forward(phi, cond)
            out.append(phi)
        return torch.stack(out, dim=1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def rel_l2(pred, target, reduce="mean"):
    """Per-sample relative L2 error, the standard operator-learning metric."""
    num = (pred - target).flatten(1).norm(dim=1)
    den = target.flatten(1).norm(dim=1).clamp_min(1e-8)
    err = num / den
    return err.mean() if reduce == "mean" else err


def band_rel_l2(pred, target, band_mask, reduce="mean"):
    """rel-L2 restricted to a mask -- the headline reading of the KPI.

    The far field of an SDF is a smooth ramp with a huge norm, so whole-window
    rel-L2 is dominated by the part of the field that carries no interface
    information. Restricting to a band around the zero set measures the surface,
    which is the thing the KPI is about.
    """
    m = band_mask.flatten(1).float()
    num = (((pred - target).flatten(1) ** 2) * m).sum(dim=1).sqrt()
    den = ((target.flatten(1) ** 2) * m).sum(dim=1).sqrt().clamp_min(1e-8)
    err = num / den
    return err.mean() if reduce == "mean" else err


class CompactCNN(nn.Module):
    """A deliberately small conditioned operator that emits the same field.

    **Why this exists.** `runs/cost_floor.json` priced a two-layer 3x3
    convolution at **370 us / 748x**, within 1.34x of the 1000x clause, and that
    row is the only architecture this repo has measured anywhere near it. But
    that row is `Conv2d(1,8,3) -> GELU -> Conv2d(8,1,3)` on a bare field: **no
    recipe embedding, no coordinate channels, no broadcast**. It cannot be a
    surrogate for this task, because the task is to advance a surface *given a
    recipe*, and its cost therefore understates any real model.

    This class is the honest version of that row. It keeps every structural
    element of `EtchOperator` that the conditioning needs -- the recipe MLP, the
    broadcast to H x W, the two coordinate channels, the residual form -- and
    replaces only the spectral blocks with 3x3 convolutions. So a cost
    difference between this and `EtchOperator` at the same width is a difference
    between spectral and local mixing, not between a surrogate and a toy.

    The interface matches `EtchOperator` exactly (`residual`, `forward`,
    `rollout`, `param_count`) so it drops into the existing rollout, training
    and benchmarking machinery untouched.
    """

    def __init__(
        self,
        cond_dim: int = 7,
        width: int = 8,
        n_layers: int = 2,
        cond_ch: int = 8,
        kernel: int = 3,
    ):
        super().__init__()
        self.width, self.n_layers, self.cond_ch = width, n_layers, cond_ch
        self.kernel = kernel
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, cond_ch)
        )
        self.lift = nn.Conv2d(1 + 2 + cond_ch, width, 1)
        pad = kernel // 2
        self.body = nn.ModuleList(
            [nn.Conv2d(width, width, kernel, padding=pad) for _ in range(n_layers)]
        )
        self.proj = nn.Conv2d(width, 1, 1)

    def residual(self, phi, cond):
        B, _, H, W = phi.shape
        c = self.cond(cond)[:, :, None, None].expand(-1, -1, H, W).to(phi.dtype)
        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype), c],
                      dim=1)
        x = self.lift(x)
        for conv in self.body:
            x = torch.nn.functional.gelu(conv(x))
        return self.proj(x)

    def forward(self, phi, cond):
        return phi + self.residual(phi, cond)

    def rollout(self, phi0, cond, n_steps):
        out = []
        phi = phi0
        for _ in range(n_steps):
            phi = self.forward(phi, cond)
            out.append(phi)
        return torch.stack(out, dim=1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
