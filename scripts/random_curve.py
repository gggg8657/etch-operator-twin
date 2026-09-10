"""H3: is gradient descent's win over random search a budget artefact?

`design.py` ran GD at 3 restarts x 200 iters (600 rollouts forward AND backward)
against random search at 256 forward rollouts, while its docstring claimed the
budgets were "comparable". They are not -- counting a backward pass at ~2x a
forward, GD had ~1800 forward-equivalents against random search's 256, a factor
of 7. This script replaces the single random-search point with a curve, so the
comparison cannot hide in the budget.

Method, and two choices that make the baseline as strong as possible:

* **Nested prefixes.** One stream of `--max-budget` candidates per target; the
  best-of-first-N is read off for each N. The curve is then monotone in the
  surrogate ranking and free of independent-draw noise, and only the budgets at
  which the argmin actually changes cost a simulation.
* **Random search keeps the true etch time.** `design.py` gave its random arm the
  target's own dt under both protocols, so in the T-free comparison the baseline
  knows T and GD does not. Kept deliberately: it makes the baseline stronger than
  the method it is being compared to.

Candidate seeds follow `design.py` exactly (`seed + 10_000*ti + j`), so the N=256
point must reproduce `runs/design_*.json`'s `random_search` value. That agreement
is asserted in the output as `n256_matches_design_json`. One consequence of
reusing the formula: for j >= 10_000 the streams of consecutive targets overlap,
so the same recipes recur across targets. Within a target all candidates are
distinct, which is what the search needs; across targets it only correlates the
aggregate slightly, and it is worth that to keep the cross-check exact.

    python scripts/random_curve.py --run runs/seed1 --design runs/design_Tfree.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eot import runlock  # noqa: E402
from eot import solver as S  # noqa: E402
from eot.data import TrajDataset, cond_vector  # noqa: E402
from eot.inverse import DESIGN_KEYS, recipe_from_values  # noqa: E402
from eot.metrics import shape_error  # noqa: E402
from eot.operator import EtchOperator  # noqa: E402
from scripts.design import simulate_recipe  # noqa: E402  (one simulator path, not two)

BUDGETS = [64, 256, 1024, 1800, 4096, 16384]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/seed1")
    ap.add_argument("--data", default="data")
    ap.add_argument("--design", default="runs/design_Tfree.json",
                    help="the design run this curve is the baseline for; its "
                         "targets, seed and GD budget are read from it")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/random_curve.json")
    a = ap.parse_args()

    budgets = sorted(int(x) for x in a.budgets.split(","))
    run = Path(a.run)
    runlock.acquire(Path(a.out), what="random-curve")
    design_doc = json.loads(Path(a.design).read_text())
    cfg = json.loads((run / "args.json").read_text())
    norm = json.loads((Path(a.data) / "norm.json").read_text())
    gen = json.loads((Path(a.data) / "gen_report.json").read_text())
    grid_delta, n_grid = gen["grid_delta"], gen["grid_n"]
    scale = norm["sdf_scale_um"]
    device = torch.device(a.device)

    ds = TrajDataset(Path(a.data) / "test.npz", norm)
    model = EtchOperator(cond_dim=len(norm["cond_keys"]), width=cfg["width"],
                         modes=cfg["modes"], n_layers=cfg["layers"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    gx, gy = S.grid_axes(n_grid)
    # the same targets design.py used, taken from its own output rather than
    # re-derived, so a change to its selection cannot silently desynchronise them
    target_rows = design_doc["targets"]
    rows, rank_seconds = [], 0.0

    for ti, drow in enumerate(target_rows):
        idx = int(drow["target_index"])
        phi0_np, tgt_np = ds.sdf[idx, 0], ds.sdf[idx, -1]
        n_steps = ds.sdf.shape[1] - 1
        dt = float(ds.dt[idx])
        geom = drow["geometry"]
        phi0 = torch.from_numpy(phi0_np)[None, None].float().to(device) / scale
        tgt = torch.from_numpy(tgt_np)[None, None].float().to(device) / scale
        band = (tgt.abs() * scale < norm["band_um"]).float()

        cands = np.stack([
            np.array([getattr(S.sample_recipe(np.random.default_rng(a.seed + 10_000 * ti + j),
                                              vary_geometry=False), k)
                      for k in DESIGN_KEYS])
            for j in range(max(budgets))
        ])
        cvec = cond_vector(
            np.concatenate([cands, np.tile([[geom["trench_width"], geom["mask_height"]]],
                                           (len(cands), 1))], axis=1),
            np.full(len(cands), dt))
        cstd = ((cvec - np.array(norm["cond_mean"], np.float32))
                / np.array(norm["cond_std"], np.float32))

        t0 = time.perf_counter()
        losses = []
        with torch.no_grad():
            for s in range(0, len(cands), a.chunk):
                cb = torch.from_numpy(cstd[s:s + a.chunk]).float().to(device)
                p = phi0.expand(cb.shape[0], -1, -1, -1)
                for _ in range(n_steps):
                    p = model(p, cb)
                num = (((p - tgt) ** 2) * band).flatten(1).sum(1).sqrt()
                den = ((tgt ** 2) * band).flatten(1).sum(1).sqrt().clamp_min(1e-8)
                losses.append((num / den).cpu().numpy())
        torch.cuda.synchronize()
        rank_seconds += time.perf_counter() - t0
        losses = np.concatenate(losses)

        # best-of-prefix, simulating each distinct winner exactly once
        sim_cache: dict[int, dict] = {}
        per_budget = {}
        for b in budgets:
            j = int(np.argmin(losses[:b]))
            if j not in sim_cache:
                final, _, st = simulate_recipe(recipe_from_values(cands[j], geom),
                                               dt, n_steps, grid_delta, n_grid)
                sim_cache[j] = (dict(shape_error(final, tgt_np, phi0_np, gx, gy),
                                     sim_status=st)
                                if final is not None else dict(st))
            per_budget[str(b)] = dict(sim_cache[j],
                                      winner_index=j,
                                      surrogate_loss=float(losses[j]),
                                      recipe=dict(zip(DESIGN_KEYS, cands[j].tolist())))
        rows.append({"target_index": idx, "dt": dt, "budgets": per_budget})
        print(f"[{ti+1}/{len(target_rows)}] idx={idx} " + " ".join(
            f"N={b}:{per_budget[str(b)].get('area_error_vs_removed', float('nan')):.4f}"
            for b in budgets), flush=True)

    def agg(b):
        v = [r["budgets"][str(b)].get("area_error_vs_removed") for r in rows]
        v = np.array([x for x in v if x is not None], dtype=float)
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "n": int(v.size), "n_failed_simulation": len(rows) - int(v.size)}

    gd0 = design_doc["targets"][0]["operator_gd_surrogate"]
    iters, restarts = int(gd0["iters"]), int(gd0["restarts"])
    # A backward pass through the spectral stack costs about twice a forward, so
    # GD's compute in forward-equivalents is restarts*iters*(1 + 2).
    gd_fwd_equiv = restarts * iters * 3
    gd_mean = design_doc["summary"]["area_error_vs_removed"]["operator_gd"]["mean"]
    curve = {str(b): agg(b) for b in budgets}
    reached = [b for b in budgets if curve[str(b)]["mean"] <= gd_mean]

    design_rand = design_doc["summary"]["area_error_vs_removed"]["random_search"]["mean"]
    n256 = curve.get("256", {}).get("mean")

    out = {
        "hypothesis": "H3: GD's advantage over random search is not a budget "
                      "artefact; random search at up to 16384 candidates will not "
                      "reach GD's simulated area error.",
        "protocol": {
            "design_source": a.design,
            "run": str(run),
            "dt_optimised_in_design": design_doc["summary"]["dt_optimised"],
            "random_search_uses_true_etch_time": True,
            "nested_prefixes": True,
            "metric": "area_error_vs_removed, measured in ViennaPS",
            "n_targets": len(rows),
        },
        "compute": {
            "gd_restarts": restarts, "gd_iters": iters,
            "gd_rollouts_forward": restarts * iters,
            "gd_rollouts_backward": restarts * iters,
            "gd_forward_equivalents": gd_fwd_equiv,
            "backward_cost_assumed_in_forwards": 2.0,
            "random_forward_equivalents_per_candidate": 1.0,
            "budget_matched_to_gd": gd_fwd_equiv,
            "ranking_seconds_total": rank_seconds,
            "ranking_seconds_per_candidate": rank_seconds / (len(rows) * max(budgets)),
        },
        "gd_reference": {"mean_area_error": gd_mean},
        "curve": curve,
        "verdict": {
            "budgets_reaching_gd": reached,
            "h3_falsified": bool(reached),
            "random_at_matched_budget": curve[str(gd_fwd_equiv)]["mean"]
            if str(gd_fwd_equiv) in curve else None,
            "ratio_gd_over_random_at_max_budget":
                curve[str(max(budgets))]["mean"] / gd_mean if gd_mean > 0 else None,
        },
        "cross_check": {
            # The N=256 prefix is design.py's own candidate set, so this point
            # should reproduce its random_search mean. It reproduces it to
            # within a tolerance rather than exactly, and the reason is worth
            # recording: the spectral forward pass is not bitwise deterministic,
            # so two rankings of the same 256 candidates can disagree on the
            # argmin whenever the top two are nearly tied, and the winner then
            # simulates to a slightly different profile. design.py's own two
            # invocations already differed this way (0.02324 vs 0.02328 for
            # random arms that are constructed identically in both protocols).
            # A hard equality here would fail for that reason and nothing else.
            "design_json_random_search_mean": design_rand,
            "curve_n256_mean": n256,
            "abs_delta": abs(n256 - design_rand) if n256 is not None else None,
            "tolerance": 1e-3,
            "n256_matches_design_json": (n256 is not None
                                         and abs(n256 - design_rand) < 1e-3),
        },
        "targets": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ["compute", "gd_reference", "curve", "verdict", "cross_check"]},
                     indent=2))


if __name__ == "__main__":
    main()
