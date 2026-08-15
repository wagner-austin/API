"""Frozen certification that the pure-Python engine agrees with ICU.

This service reimplements ICU's transform language in pure Python so that
deployment needs no ICU C++ library, and it carries no dependency on the
upstream ``turkic-transliteration`` project. Those two facts together mean
the two engines cannot be compared at test time — there is no ICU here to
compare against.

So the comparison was made once, out of band, and its result frozen. For
every vendored rule file, a deterministic probe set was run through both
this engine and PyICU 78.3, and the SHA-256 of the whole probe-to-output
table was recorded below. The digests therefore certify ICU agreement
without ICU being present: reproducing a digest means reproducing, output
for output, what ICU produced on the day it was measured.

Verified on 2026-08-14 against PyICU 78.3, rules vendored from
turkic-transliteration ``dd5fc51``. Two sweeps were run:

* The frozen sweep below — 61,535 probes across 13 rule files, 0
  mismatches. It is what the test suite re-checks.
* A larger exhaustive sweep, adding a full cube over every rule file's
  context characters — 449,326 probes, 0 mismatches. That one is too slow
  to keep (26s for ``fi_ipa`` alone), so it is recorded here as a
  one-time result rather than run on every commit.

The probe set is derived from each rule set's *own* alphabet, so it tracks
the rules rather than being a fixed list. That is deliberate: a rule
change moves the probes and the digest together, and a stale digest fails
loudly instead of silently testing the wrong thing.

Coverage rests on a property of these particular rule files: every context
is exactly one element wide, and no source pattern is longer than two.
Exhaustive pairs over the alphabet therefore already exercise every
before-context (whose match is against converted output, which a pair
produces) and every after-context. The per-rule probes cover what pairs
cannot: rules whose full window is three or four characters wide, each
tried once satisfied and once with a single context position broken.

See ``docs/rule-engine-goldens.md`` for how to re-measure the digests when
the rules are re-vendored.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Final

from turkic_api.core.rule_engine import apply_rules
from turkic_api.core.rule_parser import Rule, RuleSet

# How many members of a character set stand in for it in a targeted probe.
# Two is enough to show a set is read as a set rather than as one literal.
_REPRESENTATIVES: Final = 2

_FIELD_SEPARATOR: Final = b"\x1f"
_RECORD_SEPARATOR: Final = b"\x1e"

# SHA-256 over each rule file's probe-to-output table. Measured against
# PyICU 78.3; see this module's docstring.
SWEEP_DIGESTS: Final[dict[str, str]] = {
    "ar_lat.rules": "c249cae9998798aa36b6c178cec7588a050e7f18beecfedc6d0937deee09d9c4",
    "az_ipa.rules": "bcd33aed22d6dfc5d8349298b9f8c90de5d7226af4be38c6621d5d62ead96776",
    "fi_ipa.rules": "e73378c2eb1b66f3b34ec877d4d581665af5d1e3691e64d34c8d0d02bc0c324a",
    "kk_ipa.rules": "10a50804b0c00e49b5910ca46a573a26e89503aafdbbda6f60fb7645b02ecd0c",
    "kk_lat.rules": "2cdfaf1c5e9b1b49c3ec79beccf86e2355f7b2e1e130baf2f8f0a50e6fcfe34f",
    "ky_ipa.rules": "09e44f292298a09f340ed8c3bf9e56bd45cf9d40482537c1582b37e354f6ab40",
    "ky_lat.rules": "3828c33ade262585900eba926325bf6c77d4e820468ebbf0b768455a856a7bb3",
    "ru_ipa.rules": "5dfe8cf7dda5cab469256e09fa124f97d4e5b927822378433981cf1290d8f46d",
    "tr_ipa.rules": "49304f19db2cd3572a026d5ba3a3a6a6616004de26865ad7efe931356e271522",
    "tr_lat.rules": "cde1a9ba53a597dc4aac52b98bd21ef15c4a9908a5ca40631d32eeaa31c8b31d",
    "ug_ipa.rules": "ef379d45c3018d63ced79c77e0f27120245e34fc06ea3e67fae411bbe96b554a",
    "uz_ipa.rules": "a37b53541c69bede17fee674d252d6e8107388e34f3900a41827be55b9527295",
    "uzc_ipa.rules": "90b3a65c02662e433cf4f5c684b4ab3c7f3f50aa83ac9593a4e58ae2bae53ad8",
}


def rule_alphabet(ruleset: RuleSet) -> list[str]:
    """Return every character any rule can consume, sorted.

    Args:
        ruleset: The parsed rule file.

    Returns:
        The source alphabet, in code-point order so the probe set is
        reproducible.
    """
    chars: set[str] = set()
    for rule in ruleset.rules:
        for element in rule.source:
            chars |= element
    return sorted(chars)


def _representatives(element: frozenset[str]) -> list[str]:
    """Return up to ``_REPRESENTATIVES`` members of ``element``, in order."""
    return sorted(element)[:_REPRESENTATIVES]


def _outsider(element: frozenset[str], alphabet: list[str]) -> str | None:
    """Return a character the element rejects, or None if it accepts all."""
    for char in alphabet:
        if char not in element:
            return char
    return None


def _rule_probes(rule: Rule, alphabet: list[str]) -> list[str]:
    """Build probes for one rule's own context window.

    Args:
        rule: The rule to probe.
        alphabet: The whole file's source alphabet, used to find a
            character each context position rejects.

    Returns:
        Probes that satisfy the rule's window, probes that break exactly
        one position of it, and the bare source with and without a repeat.
    """
    window = list(rule.before) + list(rule.source) + list(rule.after)
    probes = [
        "".join(combination)
        for combination in itertools.product(*(_representatives(e) for e in window))
    ]
    for position, element in enumerate(window):
        outsider = _outsider(element, alphabet)
        if outsider is None:
            continue
        broken = [_representatives(other)[0] for other in window]
        broken[position] = outsider
        probes.append("".join(broken))
    bare = "".join(_representatives(element)[0] for element in rule.source)
    probes.extend([bare, bare + bare])
    return probes


def sweep_probes(ruleset: RuleSet) -> list[str]:
    """Build the deterministic probe set for one rule file.

    Args:
        ruleset: The parsed rule file.

    Returns:
        Every single character, every ordered pair, and the targeted
        per-rule probes — deduplicated, order preserved, no empty probe.
    """
    alphabet = rule_alphabet(ruleset)
    probes: list[str] = list(alphabet)
    probes.extend("".join(pair) for pair in itertools.product(alphabet, repeat=2))
    for rule in ruleset.rules:
        probes.extend(_rule_probes(rule, alphabet))

    seen: set[str] = set()
    unique: list[str] = []
    for probe in probes:
        if probe and probe not in seen:
            seen.add(probe)
            unique.append(probe)
    return unique


def sweep_digest(ruleset: RuleSet) -> str:
    """Return the SHA-256 of a rule file's probe-to-output table.

    Args:
        ruleset: The parsed rule file.

    Returns:
        The hex digest, over ``probe US output RS`` records in probe order.
        Separators that cannot occur in transliterated text keep the
        framing unambiguous, so two different tables cannot hash alike by
        running their fields together.
    """
    digest = hashlib.sha256()
    for probe in sweep_probes(ruleset):
        digest.update(probe.encode("utf-8"))
        digest.update(_FIELD_SEPARATOR)
        digest.update(apply_rules(probe, ruleset).encode("utf-8"))
        digest.update(_RECORD_SEPARATOR)
    return digest.hexdigest()


__all__ = [
    "SWEEP_DIGESTS",
    "rule_alphabet",
    "sweep_digest",
    "sweep_probes",
]
