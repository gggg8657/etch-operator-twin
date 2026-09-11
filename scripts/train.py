"""Train the recipe-conditioned etch operator.

    python scripts/train.py --run runs/base --epochs 60

One-step training by default; `--rollout-steps k` switches on pushforward
finetuning, where the model is unrolled k steps and the loss is taken on all of
them, which is what stops autoregressive drift from compounding.

Everything measured lands in `<run>/log.jsonl` and `<run>/eval.json`. Nothing is
printed as a result that is not also written there.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import (PairDataset, TrajDataset, band_mask,  # noqa: E402
                      fit_norm, norm_path)
from eot import runlock  # noqa: E402
from eot.operator import (EtchOperator, MultiScaleOperator,  # noqa: E402
                          SpectralPropagator, band_rel_l2, rel_l2)  # noqa: E402


def evaluate(model, loader, device, scale, band_um, rollout=True, blind=False):
    """One-step and full-rollout error, against the persistence null.

    Persistence -- predict no change at all -- is reported next to every number
    because an SDF over one timestep is mostly unchanged, so a model can post a
    small rel-L2 while having learned nothing. A number that does not beat this
    column is not a result.
    """
    model.eval()
    acc = {k: [] for k in ["one_full", "one_band", "roll_full", "roll_band",
                           "p_one_full", "p_one_band", "p_roll_full", "p_roll_band"]}
    with torch.no_grad():
        for phi0, cond, traj in loader:
            phi0, cond, traj = phi0.to(device), cond.to(device), traj.to(device)
            if blind:
                cond = torch.zeros_like(cond)
            T = traj.shape[1]
            # --- one step, teacher-forced over every t
            ins = torch.cat([phi0[:, None], traj[:, :-1]], dim=1)  # (B,T,1,H,W)
            B = ins.shape[0]
            flat_in = ins.reshape(B * T, *ins.shape[2:])
            flat_tg = traj.reshape(B * T, *traj.shape[2:])
            flat_cd = cond[:, None].expand(-1, T, -1).reshape(B * T, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(flat_in, flat_cd).float()
            m = band_mask(flat_tg * scale, band_um)
            acc["one_full"] += rel_l2(pred, flat_tg, "none").tolist()
            acc["one_band"] += band_rel_l2(pred, flat_tg, m, "none").tolist()
            acc["p_one_full"] += rel_l2(flat_in, flat_tg, "none").tolist()
            acc["p_one_band"] += band_rel_l2(flat_in, flat_tg, m, "none").tolist()
            if rollout:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rl = model.rollout(phi0, cond, T).float()
                mm = band_mask(traj * scale, band_um)
                acc["roll_full"] += rel_l2(rl.flatten(0, 1), traj.flatten(0, 1), "none").tolist()
                acc["roll_band"] += band_rel_l2(rl.flatten(0, 1), traj.flatten(0, 1),
                                                mm.flatten(0, 1), "none").tolist()
                pz = phi0[:, None].expand(-1, T, -1, -1, -1)
                acc["p_roll_full"] += rel_l2(pz.flatten(0, 1), traj.flatten(0, 1), "none").tolist()
                acc["p_roll_band"] += band_rel_l2(pz.flatten(0, 1), traj.flatten(0, 1),
                                                  mm.flatten(0, 1), "none").tolist()
    model.train()
    return {k: (float(np.mean(v)) if v else None) for k, v in acc.items()}


def build_model(a, cond_dim):
    """The single place a model is constructed from parsed arguments.

    Extracted so `tests/test_flag_wiring.py` can exercise the real path. The
    H22 sweep trained nine arms with `--state-modes` silently dropped, because
    the construction lived inline in `main()` and nothing could reach it: the
    flag parsed, was recorded into args.json by `vars(a)`, and never met the
    constructor. A test against a COPY of this logic would not have caught it.

    Every architecture-shaping constructor parameter of every model class must
    appear here; `test_every_constructor_parameter_is_reachable_from_build_model`
    fails otherwise. That test was added after `a_rank` was found on
    `SpectralPropagator` with a cost script, a measured 1.59x, and no way to
    train it -- the same defect as H22 arriving from the opposite direction,
    and the earlier meta-test could not see it because it only checked flags
    this function already mentioned.

    Zero means "class default" for the integer widths, so no existing arm's
    geometry changes and every trained checkpoint still loads.
    """
    if a.arch == "fno":
        kw = dict(cond_dim=cond_dim, width=a.width, modes=a.modes,
                  n_layers=a.layers, norm=not a.no_norm)
        if a.cond_ch:
            kw["cond_ch"] = a.cond_ch
        return EtchOperator(**kw)
    if a.arch == "specprop":
        kw = dict(cond_dim=cond_dim, modes=a.modes, modes_a=a.modes_a,
                  state_modes=a.state_modes, a_rank=a.a_rank)
        if a.hidden:
            kw["hidden"] = a.hidden
        return SpectralPropagator(**kw)
    kw = dict(cond_dim=cond_dim, width=a.width, modes=a.modes,
              n_layers=a.layers, width_full=a.width_full, scale=a.scale,
              n_local=a.n_local, act=a.act, norm=not a.no_norm)
    if a.cond_ch:
        kw["cond_ch"] = a.cond_ch
    return MultiScaleOperator(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--run", default="runs/base")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    # Architecture. `fno` is EtchOperator, the model every existing run in this
    # repo was trained with, and it stays the default so no stored args.json
    # changes meaning. `multiscale` is MultiScaleOperator: a full-resolution
    # pointwise path plus an optional spectral body on a coarser grid, added
    # because runs/arch_cost.json prices the full-resolution spectral body 6-10x
    # over the clause-2 budget while the pointwise path alone fits inside it.
    ap.add_argument("--arch", choices=("fno", "multiscale", "specprop"),
                    default="fno")
    # H19. The operator needs its horizon channel supplied before it can be
    # queried at all. In `dt` mode that channel is log(K*dt), and a caller whose
    # query is a target DEPTH must run eot.solver.probe_rate to obtain it -- a
    # real solver call, measured at 39.9 ms -- which caps the speedup clause at
    # solver/probe = 12.8x for EVERY architecture, including the specprop rows
    # that clear 1000x on cost (runs/bench_workload.json,
    # runs/arch_cost_h18.json). In `depth` mode the channel is the depth the
    # application must advance, which IS the query, so no probe is needed and
    # the ceiling does not apply.
    #
    # Only well-posed at one application per wafer: the etch rate falls as the
    # trench deepens, so no single depth describes a multi-application rollout.
    # eot.data._cond_for raises rather than approximating.
    ap.add_argument("--cond", choices=("dt", "depth"), default="dt",
                    help="horizon conditioning channel: log(K*dt), or the "
                         "achieved etch depth from scripts/derive_depth.py")
    ap.add_argument("--hidden", type=int, default=0,
                    help="specprop only: width of the coefficient heads' "
                         "bottleneck. 0 = the class default (64). This scales "
                         "the dominant stage's MAC term linearly "
                         "(runs/bandwidth_bound.json fits stage_us ~ "
                         "3.17e-3*n_out + 1.26e-4*hidden*n_out, R^2 0.987), so "
                         "it is the other lever on the same cost. Untested for "
                         "accuracy; wired so that it CAN be tested.")
    ap.add_argument("--cond-ch", type=int, default=0,
                    help="fno/multiscale only: conditioning channels. 0 = the "
                         "class default (24 for fno, 8 for multiscale).")
    ap.add_argument("--no-norm", action="store_true",
                    help="fno/multiscale only: drop the GroupNorm in each "
                         "block. Default keeps it, which is what every "
                         "existing checkpoint used.")
    ap.add_argument("--a-rank", type=int, default=0,
                    help="specprop only: factorise the additive coefficient "
                         "head as a rank-r outer product instead of a dense "
                         "hidden->2*(2*ma)*ma projection. 0 = dense, which is "
                         "what every checkpoint before 2026-09-11 used. The "
                         "dense head is 98.0%% of the model's parameters and "
                         "46.3%% of its forward (runs/specprop_profile.json), "
                         "and runs/rank_cost.json prices r=4 at 1.59x on the "
                         "whole forward -- but NO accuracy has been measured "
                         "for any rank, and a rank constraint asserts the "
                         "additive spectral response is separable in the two "
                         "frequency axes, which nothing has shown.")
    ap.add_argument("--state-modes", type=int, default=0,
                    help="specprop only: feed a state_modes x state_modes "
                         "spectral summary of the CURRENT field into the "
                         "additive head, making it a NONLINEAR function of phi "
                         "at no full-grid cost (it reuses the rfft2 already "
                         "computed). 0 keeps the recipe-only head, which is "
                         "exactly linear in phi.")
    ap.add_argument("--modes-a", type=int, default=16,
                    help="specprop only: modes for the ADDITIVE spectral term. "
                         "May exceed --modes at no inference cost, because the "
                         "irfft2 costs the same whatever fraction of the "
                         "spectrum is non-zero.")
    ap.add_argument("--scale", type=int, default=4,
                    help="multiscale only: spatial downsample of the spectral "
                         "body. 0 removes the body, leaving a purely pointwise "
                         "operator with no spatial mixing at any resolution.")
    ap.add_argument("--width-full", type=int, default=8,
                    help="multiscale only: channels in the full-resolution path")
    ap.add_argument("--n-local", type=int, default=1,
                    help="multiscale only: 1x1 layers in the full-resolution "
                         "path, i.e. the depth of the per-pixel MLP")
    ap.add_argument("--act", choices=("gelu", "relu"), default="gelu",
                    help="multiscale only: full-resolution nonlinearity. One "
                         "GELU at 128x128 costs 152.7 us against a ReLU's 13.2 "
                         "us on the same tensor (runs/arch_cost.json), i.e. 55% "
                         "of the entire clause-2 budget, so this is a budget "
                         "decision and not a taste one.")
    ap.add_argument("--rollout-steps", type=int, default=0,
                    help="0 = one-step training; k>0 = pushforward over k steps")
    ap.add_argument("--stride", type=int, default=1,
                    help="K: how many dataset timesteps ONE application of the "
                         "operator advances. K>1 divides the number of "
                         "applications per wafer by K, so both the cost and the "
                         "number of compoundings fall by K. The conditioning's "
                         "log_dt channel becomes log(K*dt); nothing else changes.")
    ap.add_argument("--strides", default=None,
                    help="H12: comma-separated strides to train ONE operator on "
                         "jointly, e.g. '1,2,5,10'. The conditioning already "
                         "carries log(K*dt), so a single network can serve every "
                         "horizon, and every intermediate state becomes an input "
                         "at some stride. This exists because a stride-10 arm on "
                         "a T=10 trajectory has exactly ONE start offset "
                         "(T-K+1 = T/K = 1), so it cannot be given input-state "
                         "diversity by --overlap-pairs the way K=2 and K=5 can. "
                         "Evaluation still happens at --eval-stride.")
    ap.add_argument("--eval-stride", type=int, default=None,
                    help="horizon to validate and checkpoint on; defaults to "
                         "--stride, or to max(--strides) when that is given, "
                         "because the deployment horizon is the one that matters")
    ap.add_argument("--overlap-pairs", action="store_true",
                    help="data-matched control: train on every legal start offset "
                         "(T-K+1 per trajectory) instead of the K non-overlapping "
                         "ones, so the pair count stays ~T at any K. Separates a "
                         "horizon effect from a training-set-size effect.")
    ap.add_argument("--band-weight", type=float, default=0.0,
                    help="extra loss weight on the narrow band; 0 = plain rel-L2")
    ap.add_argument("--blind", action="store_true",
                    help="zero the conditioning vector: same capacity, no recipe "
                         "and no dt. The ablation arm for 'does the conditioning "
                         "carry anything', differing from the main arm in exactly "
                         "this one thing.")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="append to an existing run directory (see the guard above)")
    ap.add_argument("--device", default="cuda:0")
    # The splits are already resident numpy arrays, so worker processes buy
    # nothing but shared-memory pressure -- two arms plus a generation job
    # exhausted /dev/shm and killed a run with a DataLoader bus error. 0 keeps
    # indexing in process, where it is a slice of an in-memory array anyway.
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    # Before the lock, so a lease violation cannot even take a run directory.
    runlock.assert_gpu_lease(a.device)
    runlock.acquire(run, what="train")
    # A run directory belongs to exactly one process. Two trainers once shared
    # runs/base -- a launch fired from a compound command I believed had aborted
    # -- and interleaved their epochs into one log.jsonl while overwriting each
    # other's best.pt. The weights stayed coherent (each process had its own
    # model) but the checkpoint's provenance did not, and two evaluations of
    # "the same run" differed by 3x. Refuse rather than race.
    #
    # `acquire` above is the guard that actually enforces this: its flock dies
    # with its holder, so reaching this line proves no live process is writing
    # `run`. A leftover log.jsonl under a free lock is therefore an orphan of a
    # killed trainer, not a race -- archive it and start clean rather than
    # refusing (a blanket refusal stranded four kcurve arms) or appending
    # (appending is what produced the interleaved log in the first place).
    if (run / "log.jsonl").exists() and not a.force:
        runlock.reclaim_orphan(run, what="train")
    data = Path(a.data)

    norm_p = norm_path(data, a.cond)
    if not norm_p.exists():
        norm_p.write_text(json.dumps(fit_norm(data / "train.npz",
                                              cond_mode=a.cond), indent=2))
    norm = json.loads(norm_p.read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]

    strides = [int(x) for x in a.strides.split(",")] if a.strides else [a.stride]
    eval_stride = a.eval_stride or (max(strides) if a.strides else a.stride)
    # One PairDataset per stride, concatenated. Each sets its own conditioning
    # via eot.data.standardise(..., stride=K), so log_dt carries log(K*dt) for
    # its own pairs and the network is told which horizon each sample is.
    parts = [PairDataset(data / "train.npz", norm, stride=k,
                         overlap=a.overlap_pairs) for k in strides]
    tr_pairs = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    tr_traj = TrajDataset(data / "train.npz", norm, stride=eval_stride)
    va_traj = TrajDataset(data / "val.npz", norm, stride=eval_stride)
    device = torch.device(a.device)

    model = build_model(a, len(norm["cond_keys"])).to(device)
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    if a.rollout_steps > 0:
        loader = DataLoader(tr_traj, batch_size=max(a.batch // 4, 1), shuffle=True,
                            num_workers=a.num_workers, drop_last=True)
    else:
        loader = DataLoader(tr_pairs, batch_size=a.batch, shuffle=True,
                            num_workers=a.num_workers, drop_last=True)
    va_loader = DataLoader(va_traj, batch_size=16, shuffle=False, num_workers=0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(loader), pct_start=0.1)

    args_p = run / "args.json"
    args_p.write_text(json.dumps({**vars(a), "params": model.param_count(),
                                  "n_train_pairs": len(tr_pairs),
                                  "trained_strides": strides,
                                  "eval_stride": eval_stride,
                                  "pairs_per_stride": [len(q) for q in parts],
                                  "pair_starts": (parts[0].starts if len(parts) == 1
                                                  else {str(k): q.starts for k, q
                                                        in zip(strides, parts)}),
                                  "applications_per_wafer": len(va_traj.times),
                                  "gpu": torch.cuda.get_device_name(device)}, indent=2))
    logf = (run / "log.jsonl").open("a")
    best = float("inf")
    t_start = time.perf_counter()

    for ep in range(a.epochs):
        losses = []
        t0 = time.perf_counter()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            if a.rollout_steps > 0:
                phi0, cond, traj = [x.to(device, non_blocking=True) for x in batch]
                if a.blind:
                    cond = torch.zeros_like(cond)
                k = min(a.rollout_steps, traj.shape[1])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model.rollout(phi0, cond, k).float()
                tgt = traj[:, :k]
                loss = rel_l2(pred.flatten(0, 1), tgt.flatten(0, 1))
                if a.band_weight > 0:
                    m = band_mask(tgt * scale, band_um)
                    loss = loss + a.band_weight * band_rel_l2(
                        pred.flatten(0, 1), tgt.flatten(0, 1), m.flatten(0, 1))
            else:
                x, cond, y = [t.to(device, non_blocking=True) for t in batch]
                if a.blind:
                    cond = torch.zeros_like(cond)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x, cond).float()
                loss = rel_l2(pred, y)
                if a.band_weight > 0:
                    loss = loss + a.band_weight * band_rel_l2(
                        pred, y, band_mask(y * scale, band_um))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        rec = {"epoch": ep, "train_loss": float(np.mean(losses)),
               "lr": sched.get_last_lr()[0], "epoch_s": time.perf_counter() - t0}
        if ep % 5 == 4 or ep == a.epochs - 1:
            rec.update(evaluate(model, va_loader, device, scale, band_um, blind=a.blind))
            if rec["roll_band"] is not None and rec["roll_band"] < best:
                best = rec["roll_band"]
                torch.save(model.state_dict(), run / "best.pt")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    torch.save(model.state_dict(), run / "last.pt")
    final = evaluate(model, va_loader, device, scale, band_um, blind=a.blind)
    final.update({"best_val_roll_band": best, "train_wall_s": time.perf_counter() - t_start,
                  "params": model.param_count()})
    (run / "val_eval.json").write_text(json.dumps(final, indent=2))
    runlock.mark_done(run, epochs=a.epochs, seed=a.seed,
                      best_val_roll_band=best, train_wall_s=final["train_wall_s"])
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
