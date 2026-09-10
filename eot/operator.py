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


class MultiScaleOperator(nn.Module):
    """All spectral mixing on a coarse grid, plus a pointwise full-resolution path.

    **The measurement this is built from.** `runs/shrink.json` puts
    `EtchOperator(width=8, modes=4, n_layers=2)` at stride 10 -- one application
    per wafer -- at **0.04717** in-distribution terminal band rel-L2, inside
    clause 1 on the point estimate, on every one of 3 seeds, and on the
    trajectory bootstrap's upper bound (0.04964). It costs **1777 us/wafer**
    against the 277 us budget clause 2 allows: short by **6.4x**. That is the
    smallest gap between the two clauses this repo has measured on a single
    model, and it is a cost gap, not an accuracy gap.

    **Why a coarse body should be nearly free of accuracy cost.** That model
    truncates its spectral convolutions to `modes=4`. At 128x128 there are 64
    representable modes per axis, so its body provably cannot carry any
    structure above Fourier mode 4 -- `SpectralConv2d` zeroes the rest before
    the inverse transform. Downsampling the body's input by s=4 to 32x32 leaves
    Nyquist at mode 16, still 4x above the highest mode the body uses, so the
    downsample removes **no frequency band the body was able to represent**,
    while dividing every per-pixel cost inside the body -- both FFTs and both
    pointwise convolutions -- by s^2 = 16.

    **What the downsample does cost.** The `lift` and `proj` paths of
    `EtchOperator` are pointwise at full resolution and are not mode-limited, so
    they can carry interface-local high-frequency structure that a coarse body
    cannot. Those are kept at full resolution here, as 1x1 convolutions --
    `runs/cost_floor.json` prices a full-resolution 1x1 field map at **10 us**,
    3.6% of the budget -- and the coarse body is added back by bilinear
    upsampling. So the architecture is a cheap local path plus a cheap global
    path, and the hypothesis it tests is that the expensive thing it removes,
    high-resolution *global* mixing, is not what the accuracy depends on.

    **Conditioning enters the full-resolution path by FiLM, not by concatenation.**
    `runs/cnn_cost.json` measures the conditioning floor -- broadcasting a
    `cond_ch`-channel recipe embedding to 128x128 and concatenating it -- at
    **101 us, 36% of the entire budget**, before any spatial processing at all.
    The recipe is spatially constant, so broadcasting it as feature channels
    pays H*W*cond_ch writes to carry cond_ch numbers. A per-channel affine
    modulation of the lifted features carries the same information for
    `width_full` multiply-adds on tensors the path already materialised, and
    keeps the 1x1 convolution's input at 3 channels instead of 3+cond_ch. The
    coarse path still concatenates a broadcast embedding, because at 32x32 that
    costs 1/16th as much and the concatenation form is what the FNO body was
    trained under in every other arm.

    The interface matches `EtchOperator` (`residual`, `forward`, `rollout`,
    `param_count`) so it drops into the existing trainer, evaluator, rollout and
    benchmark machinery with no changes to them.
    """

    def __init__(
        self,
        cond_dim: int = 7,
        width: int = 8,
        modes: int = 4,
        n_layers: int = 2,
        cond_ch: int = 8,
        width_full: int = 8,
        scale: int = 4,
        norm: bool = True,
        act: str = "gelu",
        n_local: int = 1,
    ):
        super().__init__()
        self.width, self.modes, self.n_layers = width, modes, n_layers
        self.scale, self.width_full, self.cond_ch = scale, width_full, cond_ch
        self.act_name = act
        # `runs/arch_cost.json`: one GELU over 8x128x128 costs 152.7 us -- 55%
        # of the entire clause-2 budget -- against 13.2 us for a ReLU over the
        # same tensor, an 11.6x difference on identical memory traffic, because
        # torch's default GELU is erf-based. At full resolution the choice of
        # nonlinearity is a budget decision, so it is a constructor argument
        # rather than a hard-coded call.
        self._act = {"gelu": torch.nn.functional.gelu,
                     "relu": torch.nn.functional.relu}[act]
        self.cond = nn.Sequential(
            nn.Linear(cond_dim, 64), nn.GELU(), nn.Linear(64, cond_ch)
        )
        # Full-resolution pointwise path: 3 input channels (phi + 2 coords),
        # FiLM-modulated by the recipe embedding.
        self.local_in = nn.Conv2d(3, width_full, 1)
        self.film = nn.Linear(cond_ch, 2 * width_full)
        # Extra 1x1 layers make the full-resolution path a per-pixel MLP in
        # (phi, x, y, recipe) rather than a single affine-plus-activation. With
        # no spatial mixing the width and depth here are the only capacity the
        # pointwise architecture has, and at full resolution both cost linearly,
        # so this is the knob the clause-2 budget is actually spent on.
        self.n_local = n_local
        self.local_hidden = nn.ModuleList(
            [nn.Conv2d(width_full, width_full, 1) for _ in range(max(0, n_local - 1))]
        )
        self.local_out = nn.Conv2d(width_full, 1, 1)
        # Coarse path: the same FNO body every other arm in this repo uses.
        # `scale=0` removes it entirely, leaving a purely POINTWISE operator --
        # a per-pixel function of (phi, x, y, recipe) with no spatial mixing at
        # any resolution. That is the only architecture measured under the
        # clause-2 budget (67.7 us linear / 157.0 us with a full-resolution
        # GELU, i.e. 4086x / 1762x in runs/arch_cost.json), so its accuracy is
        # what decides whether the two clauses can hold at once.
        self.coarse = scale > 0
        if self.coarse:
            assert modes <= 128 // (2 * scale), (
                f"modes={modes} exceeds Nyquist on the coarse grid at "
                f"scale={scale}; the downsample would then discard modes the "
                "body can represent, which is precisely the thing this "
                "architecture claims it does not do"
            )
            self.lift = nn.Sequential(
                nn.Conv2d(3 + cond_ch, width, 1), nn.GELU(),
                nn.Conv2d(width, width, 1)
            )
            self.blocks = nn.ModuleList(
                [FNOBlock(width, modes, modes, norm=norm) for _ in range(n_layers)]
            )
            self.proj = nn.Sequential(
                nn.Conv2d(width, 2 * width, 1), nn.GELU(),
                nn.Conv2d(2 * width, 1, 1)
            )

    def residual(self, phi, cond):
        B, _, H, W = phi.shape
        emb = self.cond(cond).to(phi.dtype)
        x = torch.cat([phi, EtchOperator.coords(B, H, W, phi.device, phi.dtype)], dim=1)

        h = self.local_in(x)
        g, b = self.film(emb).chunk(2, dim=1)
        h = self._act(h * (1.0 + g[:, :, None, None]) + b[:, :, None, None])
        for lin in self.local_hidden:
            h = self._act(lin(h))
        local = self.local_out(h)
        if not self.coarse:
            return local

        s = self.scale
        xc = torch.nn.functional.avg_pool2d(x, s)
        c = emb[:, :, None, None].expand(-1, -1, H // s, W // s)
        z = self.lift(torch.cat([xc, c], dim=1))
        for blk in self.blocks:
            z = blk(z)
        coarse = torch.nn.functional.interpolate(
            self.proj(z).float(), size=(H, W), mode="bilinear", align_corners=False
        ).to(phi.dtype)
        return local + coarse

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


def build_from_cfg(cfg: dict, cond_dim: int):
    """Rebuild the model a run's `args.json` describes.

    Every script that scores a checkpoint used to construct `EtchOperator`
    directly from `cfg["width"]`, `cfg["modes"]` and `cfg["layers"]`. That was
    correct while `EtchOperator` was the only architecture, and it becomes a
    silent hazard the moment a second one exists: a `MultiScaleOperator`
    checkpoint has `local_in`/`film` keys and no `blocks.*` at all when
    `scale=0`, so `load_state_dict` would raise -- loudly, which is the good
    case -- while a `scale`-mismatched multiscale checkpoint has identical key
    names and identical shapes and would load *silently* into a model that
    downsamples by a different factor and computes something else.

    Runs written before `--arch` existed have no `arch` key. They are all
    `EtchOperator`, so the default is `fno` and those runs keep scoring exactly
    as they did; `tests/test_arch.py` pins that.
    """
    arch = cfg.get("arch", "fno")
    if arch == "fno":
        return EtchOperator(cond_dim=cond_dim, width=cfg["width"],
                            modes=cfg["modes"], n_layers=cfg["layers"])
    if arch == "multiscale":
        return MultiScaleOperator(
            cond_dim=cond_dim, width=cfg["width"], modes=cfg["modes"],
            n_layers=cfg["layers"], width_full=cfg["width_full"],
            scale=cfg["scale"], n_local=cfg.get("n_local", 1),
            act=cfg.get("act", "gelu"))
    if arch == "specprop":
        return SpectralPropagator(cond_dim=cond_dim, modes=cfg["modes"],
                                  modes_a=cfg.get("modes_a", 16))
    raise ValueError(f"unknown arch {arch!r} in args.json")


class SpectralPropagator(nn.Module):
    """A conditioned linear propagator in Fourier space: ~4 full-resolution ops.

    **The measurement this is built from.** `runs/arch_cost.json` shows clause
    2's cost is per-operation, not arithmetic: the spectral body of
    `EtchOperator(width=8, modes=4, n_layers=2)` carries ~0.1 MFLOP on 8x32x32
    tensors and costs 645 us, roughly 80x what this box's measured ~13 GFLOP/s
    implies, i.e. about 20 torch operations at ~30 us each. Three attacks on
    clause 2 have failed and none of them touched that count:

    * shrinking the tensors (H15: predicted 16x, measured 2.26x -- a downsample
      does not shrink a per-operation cost);
    * fusing them (`torch.compile`: 2.0-3.3x *slower* on one CPU thread at every
      size, including a 26.2M-parameter control);
    * removing spatial mixing entirely (H16: the cheap pointwise arm reaches
      474.5x at rel-L2 0.261, and the accurate ones are Pareto-dominated by the
      FNO on both axes).

    So this class attacks the count. It computes

        phi_next = phi + irfft2( rfft2(phi) * H(recipe) + A(recipe) )

    which is one `rfft2`, one masked multiply-add over the retained modes, one
    `irfft2` and one add at full resolution -- four, against the FNO's ~20 -- and
    the transforms act on **one** channel rather than `width` of them.

    **Why this form is the physics and not just a cheap shape.** A level set
    advancing at normal speed V with |grad phi| = 1 updates as phi - V*dt: a
    pointwise shift. `H` is a linear propagator (the advection, plus the
    curvature-driven smoothing a level set performs), and `A` supplies the
    spatially varying rate that mask shadowing produces.

    **A motivation that was in this docstring and is WRONG, kept as a warning.**
    It said H16's pointwise result showed "this is most of the problem", because
    a purely pointwise model reaches 0.05052 -- "within 3% of clause 1". 3% is
    the distance to the *threshold*, not to a model that mixes spatially. The
    comparison that bears on the physics is against the spatial models: pointwise
    is 1.07x worse than `fno_w8m4L2` (0.04717) and **2.67x** worse than the
    8-seed K=1 anchor (0.01890). Spatial mixing buys a factor of 2.67 in error,
    not 3%, so the inference was overstated by about 9x and is no argument for
    this architecture. A concurrent instance caught it (commit 49bcc9a).

    The justification that survives is the operation count, which is measured:
    ~20 operations at ~30 us each against a 511 us budget. If anything the
    accuracy prediction should move PESSIMISTICALLY on the corrected reading --
    a model merely linear in phi has 2.67x to make up, not 3%.

    What it provably cannot represent is any part of the advance that depends
    *nonlinearly* on phi, which is what an undercut is: the advance beneath an
    overhang depends on the mask above it. `runs/surface_representable.json`
    finds an undercut in 249 of 250 trajectories, so this is the model's
    expected failure mode and the reason its accuracy is a real question rather
    than a formality.

    **`modes_a` may exceed `modes` at no operation cost, which is not obvious.**
    The `irfft2` costs the same whatever fraction of the spectrum is non-zero,
    so the additive term can carry a far sharper field than the multiplicative
    one for free. The mask geometry's sharp features are exactly what needs
    those modes, and they enter additively.

    Interface matches `EtchOperator` (`residual`, `forward`, `rollout`,
    `param_count`) so it drops into the existing trainer and benchmarks.
    """

    def __init__(
        self,
        cond_dim: int = 7,
        modes: int = 4,
        modes_a: int = 16,
        hidden: int = 64,
        n_grid: int = 128,
    ):
        super().__init__()
        self.modes, self.modes_a, self.n_grid = modes, modes_a, n_grid
        # rfft2 of a real (H, W) field has shape (H, W//2 + 1), so the second
        # axis is already half-spectrum and must not be truncated symmetrically.
        assert modes <= n_grid // 2 and modes_a <= n_grid // 2, (modes, modes_a)
        self.h_head = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * (2 * modes) * modes),
        )
        self.a_head = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * (2 * modes_a) * modes_a),
        )
        # The conditioning MLPs run on a (B, cond_dim) tensor, so their cost is
        # independent of the grid and does not enter the per-pixel budget.
        # Initialise both heads small: at init the operator is the identity plus
        # a small perturbation, which is the right starting point for a residual
        # update and keeps the first rollouts stable.
        for head in (self.h_head, self.a_head):
            nn.init.zeros_(head[-1].bias)
            nn.init.normal_(head[-1].weight, std=1e-3)

    def _coeffs(self, cond, m, head):
        """(B, 2m, m) complex coefficients from the recipe.

        Laid out as the low-frequency corner block of the rfft2 spectrum: `m`
        positive and `m` negative frequencies on the first axis (which is a full
        spectrum) and `m` on the second (which is already half).
        """
        B = cond.shape[0]
        # `.float()` is required, not defensive: the trainer runs under
        # torch.autocast(bfloat16), so the head emits bfloat16 and
        # torch.complex accepts only Half/Float/Double. The transforms are
        # float32 anyway -- torch.fft has no bfloat16 kernel -- so the whole
        # spectral path is deliberately outside the autocast dtype.
        raw = head(cond).float().view(B, 2, 2 * m, m)
        return torch.complex(raw[:, 0], raw[:, 1])

    def residual(self, phi, cond):
        B, _, H, W = phi.shape
        f = torch.fft.rfft2(phi.squeeze(1).float())          # (B, H, W//2+1)
        out = torch.zeros_like(f)
        m, ma = self.modes, self.modes_a
        hc = self._coeffs(cond.float(), m, self.h_head)
        ac = self._coeffs(cond.float(), ma, self.a_head)
        # Multiplicative term on the retained low modes only: everything outside
        # is annihilated, exactly as SpectralConv2d does.
        out[:, :m, :m] = f[:, :m, :m] * hc[:, :m]
        out[:, -m:, :m] = f[:, -m:, :m] * hc[:, m:]
        # Additive term, on a wider band, for free.
        out[:, :ma, :ma] = out[:, :ma, :ma] + ac[:, :ma]
        out[:, -ma:, :ma] = out[:, -ma:, :ma] + ac[:, ma:]
        r = torch.fft.irfft2(out, s=(H, W))
        return r.unsqueeze(1).to(phi.dtype)

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
