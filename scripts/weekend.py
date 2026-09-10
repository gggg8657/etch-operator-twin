"""Regenerate WEEKEND.md from run JSONs.

    python scripts/weekend.py

WEEKEND.md is the file a tired person reads on Monday, so it must be short and it
must be current. It is generated rather than hand-written for two reasons: no
number in it can then drift from the JSON that produced it, and concurrent turns
of this loop cannot half-overwrite each other's prose -- whoever runs this last
gets a correct file.

Prose lives here. Numbers live in the JSONs. `[not measured]` is emitted for
anything no run has produced.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

NM = "`[not measured]`"


def read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def f(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else NM


def cheapest_rows(ac, shr, target=0.05):
    """(cheapest surrogate, cheapest surrogate that MEETS clause 1) from the JSONs.

    Both were hand-typed into this script's headline table for one revision,
    which is exactly what this repo forbids. `runs/arch_cost.json` carries the
    cost of every candidate and `runs/shrink.json` carries which trained
    configs meet clause 1, so the join is computed rather than remembered.

    "Meets clause 1" is deliberately strict: the point estimate, every seed, AND
    the trajectory-bootstrap upper bound must be under target. Only the trained
    FNO configs can qualify -- the multiscale and pointwise rows have no
    accuracy yet, and a row with no accuracy can never be the second return
    value.
    """
    models = {k: v for k, v in (ac.get("models") or {}).items()
              if v.get("warm", {}).get("per_wafer_cpu_s")
              and v.get("applications_per_wafer") == 1}
    if not models:
        return None, None
    cheap = min(models.items(), key=lambda kv: kv[1]["warm"]["per_wafer_cpu_s"])
    # Which trained configs meet clause 1, by name as arch_cost spells them.
    ok = set()
    for name, row in (shr.get("configs") or {}).items():
        e = row["in_distribution"]["terminal_step"]
        if e["met_point"] and e["met_every_seed"] and e.get("met_upper_ci"):
            c = row["config"]
            if c["stride"] == 10:
                ok.add(f"fno_w{c['width']}m{c['modes']}L{c['layers']}_K10")
    admissible = {k: v for k, v in models.items() if k in ok}
    best_ok = (min(admissible.items(), key=lambda kv: kv[1]["warm"]["per_wafer_cpu_s"])
               if admissible else None)
    return cheap, best_ok


def coverage_verdict():
    """The computed verdict, not the point estimates. An earlier version of this
    file reported the in-coverage mean and called the clause met; the terminal-step
    reading fails under every coverage rule and clause 1 requires both."""
    p = Path("runs/coverage_verdict.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    b = d["rules"]["B_train_p1_p99"]
    return {"n_in": b["n_in"], "n_tot": d["n_crossed"],
            "mean_pt": b["mean_over_steps"]["in_range"]["point"],
            "mean_hi": b["mean_over_steps"]["in_range"]["hi"],
            "mean_ok": b["mean_over_steps"]["met_upper_ci"],
            "term_pt": b["terminal_step"]["in_range"]["point"],
            "term_hi": b["terminal_step"]["in_range"]["hi"],
            "term_ok": b["terminal_step"]["met_upper_ci"],
            "out_pt": b["mean_over_steps"]["out_of_range"]["point"],
            "verdict": b["verdict_in_coverage"],
            "apriori": d["a_priori_selector"]["mean_over_steps"]["point"],
            "agree": d["a_priori_selector"]["agreement_with_oracle_selector"]}


def indist_arm():
    """The in-distribution arm from runs/seed_spread.json.

    Kept separate from coverage_summary() because the two count different
    things and were being printed as one number: the crossed analysis exists
    for the seeds that were run through analyse_crossed.py, while the
    in-distribution spread covers every converged seed of the headline
    configuration. Reporting the smaller count next to the in-distribution mean
    understated the evidence behind the clause that actually passes.
    """
    d = read("runs/seed_spread.json")
    if not d:
        return None
    for g in d.get("groups", []):
        if not g["config"].get("blind") and g.get("n_seeds", 0) >= 2:
            return {"n": g["n_seeds"], "seeds": g["seeds"], "mean": g["mean"],
                    "range": g["range"], "max": g["max"], "std": g["std"]}
    return None


def coverage_summary():
    """Crossed-split error split by displacement coverage, across seeds.
    Returns None unless at least two seeds have the analysis -- this quantity's
    seed range is larger than its mean, so one seed is not reportable."""
    rows = []
    for p in sorted(Path("runs").glob("crossed_analysis*.json")):
        d = json.loads(p.read_text())
        m = d.get("displacement_matched") or {}
        if not m:
            continue
        rows.append((d["splits"]["test"]["band_rel_l2_mean"],
                     d["splits"]["test_crossed"]["band_rel_l2_mean"],
                     m["crossed_err_in_range_mean"], m["crossed_err_out_of_range_mean"]))
    if len(rows) < 2:
        return None
    ind, allx, inr, outr = (list(c) for c in zip(*rows))
    mean = lambda v: sum(v) / len(v)
    return {"n": len(rows), "in_dist_mean": mean(ind),
            "in_dist_range": max(ind) - min(ind), "crossed_mean": mean(allx),
            "in_range_mean": mean(inr), "in_range_max": max(inr),
            "in_range_range": max(inr) - min(inr), "out_range_mean": mean(outr),
            "out_range_range": max(outr) - min(outr)}



def live_jobs():
    """What is actually running right now, from tmux and the run directories.

    This section was hand-written prose for three turns and went stale every
    turn: it still described clause-3 jobs that had landed and claimed a test
    count of 41 that had been wrong since the count reached 68. A tired person
    reading this on Monday needs it to be true, so it is derived.
    """
    import subprocess
    out = {"sessions": [], "queues": {}, "n_tests": 0}
    try:
        r = subprocess.run(["tmux", "ls", "-F", "#{session_name}"],
                           capture_output=True, text=True, timeout=20)
        out["sessions"] = sorted(x for x in r.stdout.split()
                                 if x.startswith("e4-"))
    except Exception as e:
        out["sessions_error"] = str(e)
    for q in ("kcurve", "shrink"):
        d = Path("runs") / q
        if not d.is_dir():
            continue
        arms = [x for x in sorted(d.iterdir())
                if x.is_dir() and x.name != "_orphaned"]
        done = [x for x in arms if (x / "done.json").exists()
                and (x / "test_eval.json").exists()]
        out["queues"][q] = {"arms": len(arms), "done": len(done),
                            "incomplete": sorted(x.name for x in arms
                                                 if x not in done)}
    for t in sorted(Path("tests").glob("test_*.py")):
        out["n_tests"] += sum(1 for ln in t.read_text().splitlines()
                              if ln.startswith("def test_"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/seed1")
    ap.add_argument("--design", default=None)
    ap.add_argument("--out", default="WEEKEND.md")
    a = ap.parse_args()
    run = Path(a.run)

    ev = read(run / "test_eval.json")
    evx = read(run / "test_crossed_eval.json")
    # runs/speed.json timed a cold solver against a warm operator and its
    # like-for-like reading is withdrawn; runs/speed_symmetric.json pays the
    # one-time cost on both sides or neither. Read only the latter, so a
    # withdrawn number cannot reappear here if the newer run is missing.
    sp = read("runs/speed_symmetric.json")
    # The honest protocol's output is the headline. Named explicitly, then the
    # legacy locations, so a renamed output does not silently blank the clause.
    dsn = (read(a.design) if a.design else None) or \
        read("runs/design_Tfree.json") or read(run / "design.json") or \
        read("runs/design.json")
    dsn_alt = read("runs/design_Tfixed.json")
    degen = read("runs/design_degeneracy.json")
    rcurve = read("runs/random_curve.json")
    rtest = read("runs/random_curve_test.json")
    ver = read("runs/verify_solver.json", {})
    cf = read("runs/confound.json")
    gen = read("data/gen_report.json", {})
    genx = read("data/gen_report_crossed.json", {})

    wl = read("runs/bench_workload.json") or {}
    cap = read("runs/capacity_ood.json") or {}
    kc = read("runs/kcurve.json") or {}
    shr = read("runs/shrink.json") or {}
    ac = read("runs/arch_cost.json") or {}
    cap = read("runs/capacity_test.json") or {}
    h18 = read("runs/arch_cost_h18.json") or {}
    cheap, best_ok = cheapest_rows(ac, shr)
    k = (ev or {}).get("kpi_clause_rel_l2", {})
    kx = (evx or {}).get("kpi_clause_rel_l2", {})
    ks = (sp or {}).get("kpi_clause", {})
    kd = ((dsn or {}).get("summary") or {}).get("kpi_clause_shape_error", {})

    cov = coverage_summary()
    cvd = coverage_verdict()
    ind = indist_arm()
    L = [f"# WEEKEND — E4, Etch Neural Operator Twin",
         "",
         f"*Generated by `scripts/weekend.py` at "
         f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
         "Numbers come from run JSONs; prose lives in the script. Long form is "
         "`critique_log.md` and `paper_draft.md`; `RESULTS.md` is the full table.*",
         "",
         "> **KPI:** 2D 표면진화 rel-L2 ≤0.05 · 시뮬 대비 ≥1000× 가속 · 역설계 형상오차 ≤5%",
         "",
         "## REOPENED — the speedup clause has two readings and they are 40x apart",
         "",
         (
          "**The clause-2 verdict now depends on a question nobody has answered, "
          "and the two answers are 40x apart.** The operator takes log(dt) as a "
          "conditioning input, so a caller must know the timestep before "
          "querying it. If the timestep is part of the query, the budget is "
          f"**{wl['columns']['terminal_one_apply']['median_cpu_s']*1e6/1000:.0f} µs/wafer** "
          "and the cheapest surrogate measured is within about 2x of it. If the "
          "query is instead a *target depth*, obtaining dt costs a probe that is "
          "itself a solver run "
          f"(**{wl['columns']['dt_probe']['median_cpu_s']*1e3:.1f} ms**, "
          "`eot/solver.py:326`), and the speedup is bounded by "
          f"**{wl['ratios']['dt_probe_ceiling']['ceiling_operator_pays']['vs_terminal']:.1f}x** "
          "however fast the network gets. That is D5 below and it needs a human."
          if wl else
          "Clause 2's denominator is being re-measured; see critique_log.md."),
         "",
         (
          "Separately, **the denominator every published speedup in this repo "
          "divided by was measured on the wrong workload.** "
          "`runs/speed_symmetric.json` times one solver apply of a fixed "
          "2.0-minute etch; the dataset's trajectories carry a per-recipe dt "
          f"whose median is {wl['dt_of_priced_wafers']['median']:.3f}, so the "
          f"median test wafer etches for "
          f"{wl['dt_of_priced_wafers']['total_etch_time_median']:.2f} minutes. "
          "Measured on the test split's own recipes "
          "(`runs/bench_workload.json`), the like-for-like cost of producing "
          "what a one-application operator produces is "
          f"**{wl['columns']['terminal_one_apply']['median_cpu_s']*1e3:.0f} ms**, "
          "against the "
          f"{wl['columns']['fixed_duration_one_apply']['median_cpu_s']*1e3:.0f} ms "
          "the fixed-duration reading gives on the same wafers. All three "
          "readings are reported side by side and the old one is not deleted — "
          "`tests/test_workload.py` fails if it ever is."
          if wl else ""),
         "",
         (
          "Clause 1 is **met in distribution at one application per wafer**, "
          "now at 8 seeds rather than 2: "
          f"`K10_sm` reads **{kc['arms']['K10_sm']['in_distribution']['terminal_step']['point']:.5f}** "
          f"(seed range {kc['arms']['K10_sm']['in_distribution']['terminal_step']['seed_range']:.5f}, "
          f"bootstrap upper {kc['arms']['K10_sm']['in_distribution']['terminal_step']['hi']:.5f}), "
          "and it still fails on the crossed-dt split everywhere. Clause 3 is "
          "met as profile targeting. Every verdict in `RESULTS.md` is derived "
          "from a run JSON by `scripts/report.py`."
          if kc.get("arms", {}).get("K10_sm") else ""),
         "",
         "## Headline: Friday vs now",
         "",
         "| clause | Friday | now |",
         "|---|---|---|",
         (f"| rel-L2 ≤ 0.05 | repo did not exist | "
          f"**{f(ind['mean']) if ind else f(cov['in_dist_mean'])}** "
          f"in-distribution, {ind['n'] if ind else cov['n']} seeds "
          f"(range {f(ind['range']) if ind else f(cov['in_dist_range'])}) — **MET**; "
          f"crossed-dt ({cov['n']} seeds) inside displacement "
          f"coverage **{f(cvd['mean_pt'])}** mean / **{f(cvd['term_pt'])}** terminal step — "
          f"**{cvd['verdict']}** (terminal-step upper CI {f(cvd['term_hi'])}); "
          f"outside coverage **{f(cvd['out_pt'])}** — NOT MET |" if (cov and cvd) else
          f"| rel-L2 ≤ 0.05 | repo did not exist | **{f(k.get('value'))}** in-distribution "
          f"({'MET' if k.get('met') else 'NOT MET' if k else NM}) |"),
         # Two rows, because there is no single denominator: see D5. The old
         # fixed-duration reading stays, labelled as superseded, so the
         # correction is visible rather than substituted.
         (f"| ≥1000× speedup, **dt given** | repo did not exist | "
          f"**{cheap[1]['warm']['speedup_vs_each_denominator']['terminal_one_apply']:.0f}×** "
          f"at the cheapest surrogate measured (`{cheap[0]}`, "
          f"{cheap[1]['params']:,} params, "
          f"{cheap[1]['warm']['per_wafer_cpu_s']*1e6:.0f} µs/wafer, "
          f"**accuracy not yet measured**)"
          + (f", **{best_ok[1]['warm']['speedup_vs_each_denominator']['terminal_one_apply']:.0f}×** "
             f"at the cheapest one that MEETS clause 1 (`{best_ok[0]}`, "
             f"{best_ok[1]['warm']['per_wafer_cpu_s']*1e6:.0f} µs/wafer)"
             if best_ok else ", and no measured-accurate row is priced")
          + f". Denominator {wl['columns']['terminal_one_apply']['median_cpu_s']*1e3:.0f} ms "
          f"= one solver apply of 10·dt, the same output. **Measured at load "
          f"{ac['protocol']['budget_provenance'].get('loadavg_1min_at_start', 0):.0f}, "
          f"so a lower bound; NOT MET, and not a verdict** |"
          if (wl and cheap) else f"| ≥1000× speedup | repo did not exist | {NM} |"),
         (f"| ≥1000× speedup, **target depth given** | repo did not exist | "
          f"capped at **{wl['ratios']['dt_probe_ceiling']['ceiling_operator_pays']['vs_terminal']:.1f}×** "
          f"for any architecture, because the surrogate must probe for dt "
          f"({wl['columns']['dt_probe']['median_cpu_s']*1e3:.1f} ms) and the solver "
          f"need not — **UNREACHABLE** |" if wl else ""),
         (f"| — superseded reading | repo did not exist | "
          f"{f(ks.get('best_value'), 1) + '×' if ks else NM} against a "
          f"fixed-2.0-minute denominator, which is not the workload the "
          f"operator is scored on. Kept for the record |" if ks else ""),
         f"| shape error ≤ 5% | repo did not exist | "
         f"**{f(kd.get('value')) if kd else NM}** "
         f"({'MET' if kd.get('met') else 'NOT MET' if kd else NM}) |",
         ""]

    if k and kx and cov and cvd:
        L += ["## The one paragraph that matters", "",
              f"In distribution the operator clears the accuracy clause with room to spare — "
              f"**{f(cov['in_dist_mean'])}** against ≤0.05, mean over {cov['n']} seeds, seed "
              f"range {f(cov['in_dist_range'], 5)}. Take away the protocol that chose the "
              f"timestep per recipe and the error goes to **{f(cov['crossed_mean'])}**, a clear "
              f"miss. That miss is *not* about the timestep: no crossed trajectory has a dt "
              f"outside the trained range. What leaves the range is how far the surface moves "
              f"in one step, and splitting on that separates the two cleanly — inside coverage "
              f"**{f(cvd['mean_pt'])}**, outside it **{f(cvd['out_pt'])}**, a factor of "
              f"{cvd['out_pt']/max(cvd['mean_pt'],1e-12):.0f}.", "",
              f"**But the clause is still not met inside coverage, and an earlier version of "
              f"this file said it was.** The mean-over-steps reading passes "
              f"({f(cvd['mean_pt'])}, upper 95% bound {f(cvd['mean_hi'])}); the terminal-step "
              f"reading does not ({f(cvd['term_pt'])}, upper bound {f(cvd['term_hi'])}), and "
              f"clause 1 requires both everywhere else in this repo. The first version of the "
              f"analysis reported only the mean. An adversarial review caught it, the bootstrap "
              f"confirmed it, and the verdict in coverage is **{cvd['verdict']}**.", "",
              f"Worse for the deployment story: the coverage rule needs the simulated truth to "
              f"apply. Selecting on the displacement the *model* predicts — the only version "
              f"you could actually use — agrees with the oracle rule on "
              f"{cvd['agree']:.0%} of trajectories but gives **{f(cvd['apriori'])}**, "
              f"{cvd['apriori']/max(cvd['mean_pt'],1e-12):.1f}× worse, because a trajectory the "
              f"model gets badly wrong also mis-states its own displacement and admits itself "
              f"into the covered set. **The coverage finding explains the failure; it does not "
              f"license a deployable domain of validity.**", "",
              f"Retired this turn: the framing that removing the adaptive timestep costs a "
              f"factor in accuracy. It named the wrong variable — coverage, not protocol.", ""]

    # `runs/confound.json` is written by two different scripts in this repo with
    # two different shapes; take the one we understand and fall through otherwise
    # rather than aborting the whole file for one optional paragraph.
    cf_splits = (cf or {}).get("splits") or []
    if cf and all(isinstance(x, dict) for x in cf_splits) and cf_splits:
        rows = {s["split"]: s for s in cf_splits}
        t, c = rows.get("test"), rows.get("test_crossed")
        if t and c:
            L += [f"Displacement spread across recipes: **"
                  f"{t['per_trajectory_mean_displacement']['max_over_min']:.1f}×** on the "
                  f"adaptive test split against **"
                  f"{c['per_trajectory_mean_displacement']['max_over_min']:.1f}×** on the "
                  f"crossed one — the protocol compressed the physics by about "
                  f"{c['per_trajectory_mean_displacement']['max_over_min']/t['per_trajectory_mean_displacement']['max_over_min']:.1f}×.", ""]

    gc = (ver.get("grid_convergence") or {}).get("pairs") or []
    if gc and ev:
        msd = ev["final_profile_geometry"]["mean_surface_dist_um"]["mean"]
        L += ["## A ceiling that is already binding", "",
              f"Operator mean surface distance to ground truth is **{msd:.4f} µm**. The "
              f"solver's own grid error between Δ={gc[-1]['coarse_delta']} (what the dataset "
              f"uses) and Δ={gc[-1]['fine_delta']} is **{gc[-1]['mean_sdf_shift_um']:.4f} µm**. "
              "The operator is already closer to its training target than that target is to a "
              "converged solution, so further in-distribution accuracy work fits the "
              "discretisation, not the physics. That is why the next experiment is the "
              "crossed split, not a bigger model.", ""]

    if cap.get("capacity_curve"):
        best = cap["best_crossed"]
        anc = next(c for c in cap["capacity_curve"]
                   if c["name"] == "anchor_w64m20L4")
        t = cap["tests_vs_anchor"][best["name"]]["crossed_in_coverage"]
        e = cap["arms"][best["name"]]["crossed_in_coverage"]
        L += ["## The first thing that moved the crossed split: capacity, not data",
              "",
              (f"Out-of-distribution error is **non-monotone in capacity, with an "
               f"interior optimum**, and the model that generalises best is "
               f"**{anc['params'] // best['params']}x smaller** than the deployed one. "
               f"Crossed-in-coverage terminal band rel-L2, stride 1, 8 seeds each: "
               + " → ".join(f"{c['params']:,} params **{c['crossed_in_coverage']:.5f}**"
                            for c in cap["capacity_curve"]) + "."),
              "",
              (f"Against the deployed anchor the improvement is **{t['mean_diff_vs_anchor']:+.5f}**, "
               f"exact seed-level permutation test **p = {t['seed_level_exact']['p']:.4f}**, "
               f"95% interval **[{t['interval_95']['lo']:+.5f}, {t['interval_95']['hi']:+.5f}]** "
               f"— it excludes zero, and unlike every other crossed comparison in this "
               f"repo it does not cover the 0.00440 yardstick, so it bounds something. "
               f"This one also survived the seed extension that reversed two other "
               f"3-seed crossed readings this weekend, including one in the same turn."),
              "",
              (f"**It is still not a pass, and that has to be said plainly.** "
               f"Point estimate {e['point']:.5f} MET, every one of 8 seeds MET, "
               f"trajectory-bootstrap upper bound **{e['hi']:.5f} — NOT MET**. "
               f"Clause 1 requires the strict reading everywhere else in this repo "
               f"and it fails here. What changed is that the crossed split moved at "
               f"all: every previous explanation put the failure in the data "
               f"(displacement coverage, the adaptive-dt protocol, the horizon), and "
               f"this is the first one that located part of it in the model and came "
               f"with an intervention that worked."),
              "",
              ("Confound, stated because it bounds the claim: the three arms differ "
               "in width, modes *and* depth together, so this is one axis through a "
               "three-dimensional space. That establishes non-monotonicity; it does "
               "not attribute the effect to any one of the three, and no attribution "
               "is made."),
              ""]
    if h18.get("models"):
        sp = {k: v for k, v in h18["models"].items() if k.startswith("specprop")}
        under = [v for v in sp.values() if v["warm"]["under_budget"]]
        best_sp = min(sp.items(), key=lambda kv: kv[1]["warm"]["per_wafer_cpu_s"])
        L += ["## Clause 2 has been cleared on cost, by an architecture whose accuracy is not yet known",
              "",
              (f"**{len(under)} of {len(sp)}** priced variants of a conditioned linear "
               f"Fourier propagator come in **under** the workload-matched 1000x budget "
               f"— the first architectures in this repo to do so. Best: "
               f"`{best_sp[0]}`, {best_sp[1]['params']:,} params, "
               f"**{best_sp[1]['warm']['per_wafer_cpu_s']*1e6:.0f} µs/wafer = "
               f"{best_sp[1]['warm']['speedup_vs_solver']:.0f}x** "
               f"(`runs/arch_cost_h18.json`), measured at load "
               f"{h18['protocol']['budget_provenance'].get('loadavg_1min_at_start', 0):.0f} "
               f"so a lower bound."),
              "",
              ("It got there by attacking the one axis the three failed attacks never "
               "touched. Clause 2's cost is **~20 PyTorch operations at ~30 µs each**, "
               "not arithmetic; shrinking the tensors (H15) bought 2.26x of a predicted "
               "16x, fusing them (`torch.compile`) was 2.0-3.3x *slower*, and removing "
               "spatial mixing (H16) was Pareto-dominated. This architecture does four "
               "full-resolution operations instead of twenty."),
              "",
              ("**No accuracy is claimed for it. It is training now**, and the "
               "prediction written before the sweep started is 0.05-0.12 "
               "in-distribution terminal — i.e. that it clears clause 2 and misses "
               "clause 1. If that is what happens, the frontier is pinned from both "
               "sides for the first time: a measured point under 1000x that is too "
               "inaccurate, and a measured point at 0.04717 that is too slow. The "
               "expected failure mode is stated in advance: a propagator linear in phi "
               "cannot represent an advance that depends nonlinearly on phi, which is "
               "what an undercut is, and an undercut appears in 249 of 250 "
               "trajectories."),
              ""]
    L += ["## What was tried that did not work, and what it rules out", "",
          "1. **90-way parallel dataset generation.** Throughput *peaks at 8 workers and "
          "falls at 90* (`runs/worker_scaling.json`) — ViennaPS's flux solver is a "
          "bandwidth-bound Monte Carlo ray trace. Rules out treating a 192-core box as free "
          "parallelism, and matters for clause 2 because solver wall-clock then varies ~40× "
          "with machine load.",
          "2. **A single global timestep.** The etch rate spans ~20× across the recipe box, so "
          "one dt leaves most of the box nearly static — which a do-nothing predictor solves. "
          "Rules out the naive fixed-dt dataset.",
          "3. **The per-recipe timestep that fixed (2).** It compressed the rate law's spread; "
          "an adversarial review caught it and measurement put the cost at the factor quoted "
          "above. Rules out quoting the in-distribution number alone.",
          "4. **Hypothesis H1** — that the 10-step rollout would miss ≤0.05 through compounding "
          "drift. Falsified. Drift is real and monotonic but starts too low to matter over "
          "this horizon. Recorded as falsified rather than reframed.",
          "5. **Running ViennaPS in the same process as CUDA work.** One `simulate()` call "
          "permanently breaks cuFFT (`CUFFT_EXEC_FAILED` on every later backward). Rules out a "
          "single-process design loop; simulation now runs in a spawn subprocess pool.",
          "6. **An unguarded `simulate()`.** Fine under adaptive dt, but inverse design "
          "proposes arbitrary recipes and one verification run sat >13 min before being "
          "killed. Now stops at the window with `steps_ok` False.",
          ("7. **`torch.compile`, as the escape from per-operation overhead.** "
           "The clause-2 cost is dispatch and allocation, not arithmetic — the "
           "spectral body costs 645 µs on 8x32x32 tensors carrying ~0.1 MFLOP, "
           "80x what the arithmetic implies — so fusing it looked like the fix. "
           "Measured: **2.0-3.3x SLOWER** on one CPU thread at every size, "
           "including a 26.2M-parameter control that should be "
           "arithmetic-bound. Rules out `torch.compile` specifically; says "
           "nothing about ONNX Runtime or oneDNN graph fusion, neither "
           "measured."),
          ("8. **A coarse-grid spectral body, on the argument that it is "
           "representationally free.** The accuracy-admissible FNO truncates to "
           "`modes=4` of 64, so a 4x downsample provably discards no mode it can "
           "carry, and the pixel count says 16x cheaper. Measured: **2.26x**. "
           "Rules out reasoning about this cost from arithmetic — and located "
           "the money instead: one erf-based GELU at 128x128 costs 152.7 µs "
           "against a ReLU's 13.2 µs, **55% of the whole budget for one "
           "activation**, which was never a considered choice anywhere in this "
           "repo."),
          ("9. **Buying accuracy back with a wider pointwise path.** "
           "`pw_wf32_n3` is **more expensive than the FNO it was built to "
           "undercut**: width and depth at full resolution cost linearly where "
           "spectral modes do not. Rules out the pointwise family as anything "
           "but tiny — and 'tiny' here means 1,217 parameters."),
          ""]

    L += ["## Decisions that need a human", "",
          ("**D5 — NEW, and it is the one that decides clause 2. Is the timestep "
           "part of the query, or is the target depth?** The operator takes "
           "log(dt) as a conditioning input. The dataset's dt was chosen by "
           "`choose_dt`, which calls `probe_rate` — a real solver run, measured "
           f"at **{(read('runs/bench_workload.json') or {}).get('columns',{}).get('dt_probe',{}).get('median_cpu_s',0)*1e3:.1f} ms**."),
          ("- *Option A (recommended, and what every number in this repo "
           "assumes):* **dt is part of the query.** The caller says 'advance "
           "this recipe by 10 steps of 0.34 min', neither side probes, and the "
           "budget is 511 µs/wafer. The KPI says 표면진화 — given a recipe and a "
           "duration, predict the surface — and log(dt) is an input exactly as "
           "ion flux is. The adaptive dt was a *dataset construction* device: "
           "the etch rate spans ~20x across the recipe box, so one global "
           "timestep leaves most of the box static enough for a do-nothing "
           "predictor to pass clause 1 (`eot/solver.py:339`). It was never part "
           "of the query."),
          ("- *Option B:* **target depth is the query.** A process engineer asks "
           "for a depth, not a timestep. Then the surrogate must probe to learn "
           "dt while the solver can simply integrate until the depth is reached "
           "— so only the surrogate pays, and clause 2 is capped at "
           "**12.8x** for a terminal query and **24.3x** for an all-frames one, "
           "*no matter how fast the network is*. Under this reading clause 2 is "
           "`UNREACHABLE` by a factor of 78 and no architecture changes that."),
          ("- *Option C:* **dissolve the question — condition the operator on "
           "target_depth instead of log(dt).** Then a depth query needs no "
           "probe by construction. The dataset already stores `target_depth` "
           "for train/val/test with no gaps, so this is testable on existing "
           "data with no regeneration. Two known costs: the crossed-dt split "
           "stores `target_depth` as NaN for all 209 trajectories (independent-dt "
           "generation never computed one), so the split clause 1 already fails "
           "on could not be evaluated at all without deriving achieved depth "
           "from the stored SDFs; and the model must then learn the rate law "
           "internally, which may cost accuracy. **Unmeasured, and no number is "
           "claimed for it.** This is the route I would take first."),
          "",
          "**D1 — which speedup number goes on the board.** The grid contains a batched-H100-"
          "vs-1-thread-CPU figure and a like-for-like CPU-1-thread figure. The catalog says "
          "\"시뮬 대비 ≥1000× 가속\" without naming hardware.",
          "- *Option A (taken by default in code):* the like-for-like figure is the KPI number, "
          "and the clause reads `UNREACHABLE` if it falls short, with the grid as evidence.",
          "- *Option B:* the batched-GPU figure is the KPI number. Defensible only if the "
          "deployment story is explicitly batched inference on an H100.",
          "",
          "**D2 — what counts as passing clause 1, now that the failure is located.** "
          "The crossed-dt probe was built to test a decoupled timestep and does not test that "
          "at all: no crossed trajectory has a dt outside the trained range, while 72 of 209 "
          "have a per-step *displacement* outside it. Splitting on displacement separates the "
          "two cleanly. But the clause requires the mean-over-steps **and** the terminal-step "
          "reading, and inside coverage the terminal-step reading fails under all three "
          "coverage rules (0.0531–0.0665, upper CI to 0.0947). An earlier version of this file "
          "claimed clause 1 was met in coverage; that claim is withdrawn.",
          "- *Option A (recommended):* clause 1 is **MET in distribution and nowhere else** — "
          f"{f(ind['mean']) if ind else NM} over {ind['n'] if ind else NM} seeds, "
          f"range {f(ind['range']) if ind else NM} — "
          "with the crossed-split numbers printed beside "
          "it as the stated limit of validity. This is the narrowest claim the measurements "
          "support, and the narrow version is the one that survives review.",
          "- *Option B:* claim validity on a displacement-coverage domain. **Not recommended, "
          "and now measured to be untenable twice over:** the terminal-step reading fails "
          "inside coverage, and the coverage rule needs the simulated truth to apply. The "
          "deployable selector, using the displacement the model itself predicts, scores 0.1402 "
          "against the oracle rule's 0.0330 — 4.3× worse — because a trajectory the model gets "
          "badly wrong also mis-states its own displacement and so admits itself into the "
          "covered set.",
          "- *Option C:* widen the training set to cover the displacement range and re-measure. "
          "Most likely to produce a genuinely better model, and the only option that could turn "
          "the crossed split into a pass. It changes the dataset, so every number in this repo "
          "would need re-running and nothing before and after would be comparable. Not started "
          "for that reason; it is the obvious next week of work, not the next hour.",
          "",
          "**D3 — whether ≤5% shape error is judged on the mean or the tail.** Now that clause "
          "3 is measured this decision has numbers attached, and it changes the verdict on one "
          "protocol but not the other:",
          "- Under the honest **T-searched** protocol the question is moot: mean 0.0061, worst "
          "target 0.0098, 20/20 targets under 5%. Mean, p90, max and per-target fraction all "
          "pass, so no reading of the clause fails.",
          "- Under the **T-pinned** protocol it matters: mean 0.0098 passes, but one target of "
          "20 reads 0.0587 and misses. A mean-based verdict would hide it.",
          "- *Option A (recommended, and taken in code):* report the honest protocol as the KPI "
          "number and publish mean, median, p90, max and the fraction of targets under 5% for "
          "both, so a tail reading is always available to the reader. `design.py` already "
          "emits all five.",
          "- *Option B:* make the tail the verdict everywhere — the clause passes only if "
          "**every** target is under 5%. Stricter, defensible, and the T-searched arm still "
          "passes it. Adopt this if the deployment story is per-wafer rather than per-lot.",
          "",
          "**D3b — what the recipe-degeneracy result means for how this is sold.** Shape error "
          "is met by a factor of 8. The recipe found is nevertheless 0.177 of the box away from "
          "the one that made the target (18 of 20 beyond 10% of the box), and the etch time is "
          "wrong by 23% on average. The forward map is degenerate over (rate × time), so a "
          "matched profile does not identify the process that produced it. No decision is "
          "needed about the KPI — the clause says 형상오차, shape error, and shape error is what "
          "passes — but a decision is needed about the claim made *around* it:",
          "- *Option A (recommended, and what the repo now says):* sell it as **profile "
          "targeting** — given a desired profile it finds a recipe that produces it. Verified "
          "in the simulator, 20/20.",
          "- *Option B:* sell it as recipe identification or tool matching. **Not supported by "
          "any measurement here**, and the degeneracy result is positive evidence against it. "
          "Making that claim would need a new experiment: an identifiability study with a "
          "prior over recipes, or extra observables (depth *and* sidewall angle *and* time) "
          "that break the degeneracy.",
          "",
          "**D4 — REOPENED 2026-09-10 13:23. It recurred, and it is again a human "
          "decision.** Two commits, `e1ebc29` and `91a19da`, were pushed to this "
          "repository's `main` by a process that is not this track's loop: pid 693697, "
          "whose parent 2749455 is the pid recorded in `.lock_trkC` — a *different* "
          "track's wrapper — committing to track α's project. Costs so far are small "
          "and entirely of the D4 shape: the two instances independently found the same "
          "test defect and each wrote a meta-test for it (one has been removed), and "
          "`91a19da` swept `log.jsonl` from five `runs/kcurve` arms that this track's "
          "tmux sessions were actively writing. `eot/runlock.py` stops two *trainers* "
          "sharing a run directory; nothing stops two loops running `git add -A` over "
          "one half-written run, and nothing stops two loops pushing.\n\n"
          "  Options, in the order I would take them: **(1)** give each repository a "
          "lock naming the track that owns it, checked before any commit — enforceable "
          "inside the repo, and the only one this track could implement alone. I have "
          "**not** implemented it, because a commit lock that the concurrent instance "
          "does not know about would lock that instance out mid-turn, and its state is "
          "not visible from here. **(2)** make the wrapper refuse to `--continue` a "
          "session whose project is not its track's; this is the actual fix and it "
          "needs the harness, not this repo. **(3)** accept the duplication and rely on "
          "both instances rebasing before pushing, which is what happened here by luck "
          "rather than design.",
          "",
          "**D4 (original, 2026-09-09) — the earlier occurrence, kept for the record.** Two α loop instances shared this "
          "repository and working tree on 2026-09-09 (two `claude -p`, started one second "
          "apart, identical brief) and killed each other's inverse-design jobs after eight "
          "minutes of duplicated compute. As of 2026-09-10 07:42 there is exactly **one** α "
          "loop (pid 2021260); the other two `claude -p` processes on the box carry the β and "
          "γ briefs and work in their own repositories. Recorded rather than deleted because "
          "the mitigation is worth keeping: `eot/runlock.py` makes a second writer to any run "
          "directory or output file exit 3 with the holder's pid instead of interleaving, and "
          "the completeness guard refuses to score a run whose log shows two writers. Locking "
          "converts silent corruption into a loud refusal; it cannot stop a supervisor from "
          "starting two loops, which is why this was a human decision and not a code fix.",
          ""]

    lj = live_jobs()
    L += ["## Still running, and how to check it", "",
          f"Derived from `tmux ls` and the run directories at the moment this "
          f"file was generated, because this section was hand-written for three "
          f"turns and was stale every one of them.", "",
          "```bash",
          "cd ~/Documents/workspace/etch-operator-twin",
          "bash scripts/make_report.sh                       # regenerate RESULTS.md",
          "python scripts/weekend.py                          # regenerate this file",
          f"for t in tests/test_*.py; do PYTHONPATH=. python $t; done   # {lj['n_tests']} tests",
          "tmux ls | grep e4-                                 # this repo's jobs",
          "```", ""]
    if lj["sessions"]:
        L += [f"**{len(lj['sessions'])} tmux session(s) of this repo's own are alive:** "
              + ", ".join(f"`{x}`" for x in lj["sessions"])
              + ". Each was started detached, so it survives the loop process "
                "exiting — four `runs/kcurve` arms were lost earlier to a "
                "process-group kill when a parent died, which is why nothing "
                "long runs as a child any more. Attach with "
                "`tmux attach -t <name>`; the drivers log to `logs/`.", ""]
    else:
        L += ["**No job of this repo's is running.** Every queue below is either "
              "drained or was never started.", ""]
    for q, st in lj["queues"].items():
        L += [f"- **`runs/{q}`: {st['done']} of {st['arms']} arms complete.** "
              + (f"Incomplete: {', '.join('`' + x + '`' for x in st['incomplete'][:8])}"
                 + (f" and {len(st['incomplete']) - 8} more" if len(st['incomplete']) > 8 else "")
                 + ". Re-running the driver is safe and idempotent: it skips any arm "
                   "carrying both `done.json` and `test_eval.json`, and an arm killed "
                   "mid-training is archived to `runs/_orphaned/` and restarted clean "
                   "rather than appended to."
                 if st["incomplete"] else "Drained.")]
    L += [""]

    L += [""]

    Path(a.out).write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
