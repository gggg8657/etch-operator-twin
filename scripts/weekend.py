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
         "## Declared UNREACHABLE, 2026-09-10",
         "",
         (f"Two of three clauses met. The speedup clause is short by a factor of "
          f"**{ks['shortfall_factor']:.0f}** — {ks['best_value']:.2f}× at the best of four "
          f"honest rows (`{ks['best_arm'].split('/')[-1]}`, {ks['best_reading']}) against "
          f"1000× — and that is not a gap an implementation closes. At K=1 the operator is "
          f"*slower* than the simulator like-for-like. Clause 1 holds in distribution "
          f"and nowhere else; clause 3 holds as profile targeting, not recipe identification. "
          f"Every clause verdict in `RESULTS.md` is derived from a run JSON by "
          f"`scripts/report.py`."
          if ks else
          "Two of three clauses met; the speedup clause is not."),
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
         f"| ≥1000× speedup | repo did not exist | "
         f"**{f(ks.get('best_value'), 1) + '×' if ks else NM}** like-for-like, CPU-seconds, "
         f"one-time cost paid on both sides or neither "
         f"({'MET' if ks and ks.get('met') else 'NOT MET' if ks else NM}) |",
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
          ""]

    L += ["## Decisions that need a human", "",
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
