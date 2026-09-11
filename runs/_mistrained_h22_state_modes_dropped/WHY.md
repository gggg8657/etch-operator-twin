# Nine arms that trained the wrong model, kept rather than deleted

`scripts/train.py` parsed `--state-modes`, recorded it in each run's
`args.json` via `vars(a)`, and never passed it to `SpectralPropagator`. All
nine arms below therefore trained the `state_modes=0` anchor.

Proof, not inference:

* every one carries `params = 1070144`, the anchor's exact parameter count;
* `best_val_roll_band` matches the corresponding anchor seed to all seventeen
  digits — `m4_ma64_sm{4,8,16}_s1` all read `0.08605794188876947`, which is
  `m4_ma64_s1`'s value.

Kept for two reasons. First, their `args.json` misreports what was trained, so
deleting them would erase the evidence for a defect that a future sweep could
repeat. Second, the three triples are three independent invocations of an
identical configuration on different days, which is the run-to-run
reproducibility check the seed-count discipline asks for and this repo had
never run: **the spread is exactly 0.000000**. All seed spread in this repo is
attributable to the seed, with no nondeterminism underneath it.

Fixed at `scripts/train.py` (`build_model`), guarded by
`tests/test_flag_wiring.py`, which fails with this exact diagnostic if the
wiring regresses.
