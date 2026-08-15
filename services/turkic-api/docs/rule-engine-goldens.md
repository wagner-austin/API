# Re-measuring the rule-engine goldens

This service transliterates with a pure-Python interpreter of ICU's transform
language — `src/turkic_api/core/rule_lexer.py`, `rule_parser.py` and
`rule_engine.py` — rather than with ICU itself, so that deployment needs no
libicu and so that this service depends on nothing outside this repository. The rule files it interprets are written for ICU, which
leaves one question that has to be answered and kept answered: **does this
engine read them the way ICU does?**

It cannot be answered at test time, because ICU is deliberately absent. So it is
answered out of band and frozen: `tests/golden_sweep.py` holds a SHA-256 per rule
file over a table of probe inputs and their outputs, measured while PyICU was
available. Reproducing a digest means reproducing, output for output, what ICU
produced.

Regenerate the digests **only** when the rules change, and never by pasting in
whatever the engine currently prints — that would certify the engine against
itself and quietly discard the only evidence it matches ICU.

## When this is needed

- The rule files were re-vendored from `turkic-transliteration` (see
  `src/turkic_api/core/rules/PROVENANCE.md`).
- A rule file was added.
- The engine gained support for a construct the rules now use.

A change to comments alone does not need it: the digests are over transliterated
output, and comments do not transliterate. The per-file SHA-256 in
`PROVENANCE.md` *does* change, because that one is over the file's bytes.

## Measuring

PyICU is not a dependency of this service and must not become one. Install it
into a throwaway environment.

```bash
python -m venv /tmp/icu-check
/tmp/icu-check/bin/pip install pyicu          # Windows: use a wheel from
                                              # github.com/cgohlke/pyicu-build
/tmp/icu-check/bin/pip install -e .
```

Then run both engines over the same probes and compare, per rule file:

```python
"""Compare this engine against PyICU and print digests when they agree."""

import hashlib
from pathlib import Path

import icu

from tests.golden_sweep import sweep_probes
from turkic_api.core.rule_engine import apply_rules
from turkic_api.core.rule_parser import load_rules

RULES = Path("src/turkic_api/core/rules")

digests: dict[str, str] = {}
mismatches = 0
for path in sorted(RULES.glob("*.rules")):
    ruleset = load_rules(path.name)
    reference = icu.Transliterator.createFromRules(
        path.name, path.read_text(encoding="utf-8"), 0
    )
    digest = hashlib.sha256()
    for probe in sweep_probes(ruleset):
        ours = apply_rules(probe, ruleset)
        theirs = reference.transliterate(probe)
        if ours != theirs:
            mismatches += 1
            print(f"{path.name}: {probe!r} -> ours {ours!r}, icu {theirs!r}")
        digest.update(probe.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(ours.encode("utf-8"))
        digest.update(b"\x1e")
    digests[path.name] = digest.hexdigest()

if mismatches:
    raise SystemExit(f"{mismatches} mismatches; the engine does not agree with ICU")
for name, value in sorted(digests.items()):
    print(f'    "{name}": "{value}",')
```

**A single mismatch means stop.** The engine and ICU disagree, and the rules
cannot be adopted until the engine is corrected — pasting the new digests would
freeze the disagreement instead of fixing it. Both times this has run, the
disagreement was real. On 2026-08-14 the engine read `}` as a *left* context
where ICU reads it as a right one, and matched contexts against the source where
ICU matches the converted output. On 2026-08-15 a re-vendored `ar_lat.rules`
brought the first negated set, which the engine had been reading as a literal
`^` among the members, and with it the first rule whose match turns on a text
boundary; both were implemented from what PyICU actually did rather than from
the specification.

Paste the printed digests into `SWEEP_DIGESTS`, update this module's docstring
with the PyICU version and the date, and delete the throwaway environment.

## What the probes cover, and what they rest on

`sweep_probes` derives its probes from each rule set's own alphabet, so they
track the rules rather than being a fixed list: every single character, every
ordered pair, and per-rule probes built from each rule's context window — once
satisfied, and once with a single context position broken.

Pairs are what make this adequate, and they are adequate because of a property
of *these* rule files: no context is more than one element wide and no source
pattern is longer than two. Exhaustive pairs therefore exercise every
before-context (a pair produces the converted output a left context matches
against) and every after-context. The per-rule probes cover the three- and
four-character windows pairs cannot reach.

If a future rule file uses a wider context, that reasoning stops holding and the
probe set has to grow with it. `test_sweep_covers_every_rule` will not catch
that — it only checks the probe count scales with the rule count.

A larger sweep was run once as a cross-check: adding a full cube over every rule
file's context characters gave 449,326 probes with zero mismatches. It is not
kept, because `fi_ipa` alone took 26 seconds of it.
