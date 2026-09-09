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
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot.data import PairDataset, TrajDataset, band_mask, fit_norm  # noqa: E402
from eot.operator import EtchOperator, band_rel_l2, rel_l2  # noqa: E402


def evaluate(model, loader, device, scale, band_um, rollout=True):
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
    ap.add_argument("--rollout-steps", type=int, default=0,
                    help="0 = one-step training; k>0 = pushforward over k steps")
    ap.add_argument("--band-weight", type=float, default=0.0,
                    help="extra loss weight on the narrow band; 0 = plain rel-L2")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)

    norm_p = data / "norm.json"
    if not norm_p.exists():
        norm_p.write_text(json.dumps(fit_norm(data / "train.npz"), indent=2))
    norm = json.loads(norm_p.read_text())
    scale, band_um = norm["sdf_scale_um"], norm["band_um"]

    tr_pairs = PairDataset(data / "train.npz", norm)
    tr_traj = TrajDataset(data / "train.npz", norm)
    va_traj = TrajDataset(data / "val.npz", norm)
    device = torch.device(a.device)

    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=a.width,
                         modes=a.modes, n_layers=a.layers).to(device)
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)

    if a.rollout_steps > 0:
        loader = DataLoader(tr_traj, batch_size=max(a.batch // 4, 1), shuffle=True,
                            num_workers=4, drop_last=True, persistent_workers=True)
    else:
        loader = DataLoader(tr_pairs, batch_size=a.batch, shuffle=True,
                            num_workers=4, drop_last=True, persistent_workers=True)
    va_loader = DataLoader(va_traj, batch_size=16, shuffle=False, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(loader), pct_start=0.1)

    args_p = run / "args.json"
    args_p.write_text(json.dumps({**vars(a), "params": model.param_count(),
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
            rec.update(evaluate(model, va_loader, device, scale, band_um))
            if rec["roll_band"] is not None and rec["roll_band"] < best:
                best = rec["roll_band"]
                torch.save(model.state_dict(), run / "best.pt")
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    torch.save(model.state_dict(), run / "last.pt")
    final = evaluate(model, va_loader, device, scale, band_um)
    final.update({"best_val_roll_band": best, "train_wall_s": time.perf_counter() - t_start,
                  "params": model.param_count()})
    (run / "val_eval.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
