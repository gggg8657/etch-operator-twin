#!/usr/bin/env bash
# The canonical RESULTS.md invocation.
#
# Which JSON a section of RESULTS.md reads is decided by CLI flags, so the
# document was not reproducible from the repo alone -- a different --design
# would silently change the clause-3 headline. This script is the invocation of
# record, and report.py writes the argv it was given into the document.
#
# Clause 3's headline is the `mid` initialisation: target-independent (it starts
# at the geometric centre of the trained dt range) and deterministic. The leaky
# `target` arm and the stochastic `random` arm are both reported beside it in
# the dt-initialisation table -- `random` and `mid` were pre-registered together
# in one loop in scripts/run_dt_honest.sh at commit eb82a5c, before either ran,
# so neither is a protocol chosen after seeing a result.
set -eu
PY=${PY:-~/miniforge3/envs/pdeno/bin/python}
$PY scripts/report.py \
  --run runs/seed1 \
  --design runs/design_Tfree_dtinit_mid.json \
  --design-alt runs/design_Tfixed.json \
  --design-dtinit runs/design_Tfree.json \
                  runs/design_Tfree_dtinit_mid.json \
                  runs/design_Tfree_dtinit_random.json \
  --design-restart-curve runs/design_Tfree_restartcurve.json \
  --out RESULTS.md "$@"
