"""The config key must separate every architecture it can be handed.

A key that omits a distinguishing field pools two architectures into one seed
group and reports their pooled mean as if it were one arm. That happened: the
first version of `shrink_report.config_key` had no `specprop` branch, so
specprop runs fell through to the multiscale branch, `modes_a` was not in the
key, and two models 7.7x apart in parameter count (9,344 and 71,744) were
reported as one two-seed arm.

**The guard in `shrink_report.main` cannot catch this**, because it asserts that
all configs in a group produce the same key using the same `config_key`
function. A key blind to a field is self-consistently blind. So the property has
to be asserted from outside: changing any field that defines an architecture
must change the key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.shrink_report import (ARCH_FIELDS, LEGACY_DEFAULTS,  # noqa: E402
                                   config_key)

BASE = {
    "fno": {"arch": "fno", "width": 8, "modes": 4, "layers": 2, "stride": 10},
    "multiscale": {"arch": "multiscale", "width": 8, "modes": 4, "layers": 2,
                   "stride": 10, "width_full": 8, "scale": 4, "n_local": 1,
                   "act": "gelu"},
    "specprop": {"arch": "specprop", "modes": 4, "modes_a": 16, "stride": 10,
                 "state_modes": 0},
}


def test_every_arch_field_changes_the_key():
    for arch, base in BASE.items():
        k0 = config_key(base)
        for f in ARCH_FIELDS[arch]:
            v = base[f]
            alt = {**base, f: ("relu" if v == "gelu" else
                               "gelu" if v == "relu" else v + 1)}
            assert config_key(alt) != k0, \
                f"{arch}: changing {f} ({v!r}) did not change the key {k0!r}"


def test_the_specific_pair_that_was_pooled():
    a = config_key({"arch": "specprop", "modes": 4, "modes_a": 4, "stride": 10})
    b = config_key({"arch": "specprop", "modes": 4, "modes_a": 16, "stride": 10})
    assert a != b, f"m4_ma4 and m4_ma16 both key to {a!r}"


def test_architectures_never_collide_with_each_other():
    keys = {arch: config_key(base) for arch, base in BASE.items()}
    assert len(set(keys.values())) == len(keys), keys


def test_missing_field_raises_instead_of_defaulting():
    """A specprop args.json without modes_a must fail loudly. Silently
    defaulting is how the pooling bug produced a plausible-looking number."""
    try:
        config_key({"arch": "specprop", "modes": 4, "stride": 10})
    except ValueError:
        return
    raise AssertionError("a specprop config with no modes_a was accepted")


def test_unknown_arch_raises():
    try:
        config_key({"arch": "diffusion", "width": 8})
    except ValueError:
        return
    raise AssertionError("an unknown arch was given a key")


def test_real_stored_runs_key_uniquely_per_directory():
    """On the actual run tree: two directories with different names must never
    share a key, which is the invariant the pooling bug violated."""
    import json
    seen = {}
    for sub in ("shrink", "specprop", "ladder"):
        for p in sorted(Path("runs", sub).glob("*_s*")):
            aj = p / "args.json"
            if not aj.exists():
                continue
            cfg = json.loads(aj.read_text())
            k = config_key(cfg)
            stem = p.name.rsplit("_s", 1)[0]
            if k in seen and seen[k] != stem:
                raise AssertionError(
                    f"{p.name} and {seen[k]}_s* both key to {k!r}")
            seen[k] = stem


def test_a_field_added_after_runs_existed_defaults_without_renaming_them():
    """`LEGACY_DEFAULTS` is an exemption from the raise, so it needs pinning.

    `config_key` raises on a missing field on purpose -- defaulting is what
    pooled a 9,344-parameter model with a 71,744-parameter one. But
    `state_modes` was added after five specprop runs were already trained and
    scored, and those runs have a definite value for it: the behaviour before
    the flag existed, which is 0. So the exemption has to do three things at
    once, and all three are asserted here rather than trusted:

      1. a config with the field absent must not raise;
      2. it must key IDENTICALLY to one that sets the field explicitly to the
         legacy value -- they are the same architecture and belong in one seed
         group;
      3. it must key DIFFERENTLY from any non-legacy value, or the exemption has
         reintroduced the pooling bug it was carved out of.
    """
    legacy = {"arch": "specprop", "modes": 4, "modes_a": 64, "stride": 10}
    explicit = {**legacy, "state_modes": 0}
    stated = {**legacy, "state_modes": 8}

    k_legacy = config_key(legacy)                      # (1) must not raise
    assert k_legacy == config_key(explicit), (          # (2)
        f"{k_legacy!r} != {config_key(explicit)!r}: a run predating the flag "
        f"and a run setting it to the legacy value are the same architecture"
    )
    assert k_legacy != config_key(stated), (            # (3)
        f"state_modes=8 keys to {k_legacy!r}, the same as a legacy run"
    )
    # And the published rows must not have been renamed by adding the field.
    assert k_legacy == "sp_m4ma64_K10", k_legacy


def test_only_declared_legacy_fields_are_exempt_from_the_raise():
    """Every other missing field must still raise.

    The exemption is one dict with a written reason per entry. If it silently
    grew to cover a field that was merely forgotten, the original pooling bug
    comes straight back.
    """
    for arch, base in BASE.items():
        for f in ARCH_FIELDS[arch]:
            if f in LEGACY_DEFAULTS:
                continue
            missing = {k: v for k, v in base.items() if k != f}
            try:
                config_key(missing)
            except ValueError:
                continue
            raise AssertionError(
                f"{arch}: a config missing {f!r} was keyed instead of refused"
            )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} tests passed")
