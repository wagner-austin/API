"""Applying rules to text.

A cursor walks the source left to right. At each position the rules are tried in
the order they were written and the first match wins; its output is emitted and
the cursor advances past what the rule consumed.

Three asymmetries matter, and getting any of them wrong is silent rather than
loud. They were established by probing PyICU 78.3, not by reading the
specification, and the frozen goldens pin every one of them:

* The **before context matches the already-converted output**, not the source.
  This is why the rule files write their left contexts over IPA characters
  (``$Vout {`` U ``> w``) rather than over Cyrillic.
* The **after context matches the untouched source**, ahead of the cursor.
* The **output is never re-examined**, so a replacement cannot trigger a second
  rule.
* A **context position past an end of the text matches a negated set** and no
  other kind of element, so ``ئ } [^$AsuVowel]`` in ``ar_lat.rules`` fires on a
  word-final hamza carrier as well as on one followed by a consonant.
* The **start anchor asks whether the output is empty**, not whether the source
  cursor is at zero. ``^`` is a before context like any other, and before
  contexts read the converted output, so a rule that deleted everything to the
  left leaves the next position initial. Measured on PyICU 78.3:
  ``x > ; ^ a > S ; a > A ;`` turns ``xa`` into ``S``, and ``bxa`` into ``bA``.

An earlier version of this engine read ``}`` as a *left* context and matched it
against the source. Two wrongs cancelled for the rule files of the time, and
every rule that depends on the real semantics would have done nothing at all.
"""

from __future__ import annotations

import unicodedata as ud

from turkic_api.core.rule_parser import Rule, RuleSet


def _matches(rule: Rule, out: list[str], source: str, cursor: int) -> bool:
    """Return whether ``rule`` applies at ``cursor``.

    Args:
        rule: The rule to try.
        out: Converted output so far, one character per entry.
        source: The text being transliterated.
        cursor: Offset in ``source`` of the next unconverted character.

    Returns:
        Whether the before context, the source elements, and the after
        context all match.
    """
    if rule.anchor_start and out:
        return False

    width = len(rule.before)
    if width:
        missing = max(0, width - len(out))
        window: list[str | None] = [None] * missing + list(out[len(out) - width + missing :])
        if any(
            not element.accepts(char) for element, char in zip(rule.before, window, strict=True)
        ):
            return False

    end = cursor + len(rule.source)
    if end > len(source):
        return False
    if any(not element.accepts(source[cursor + n]) for n, element in enumerate(rule.source)):
        return False

    return all(
        element.accepts(source[end + n] if end + n < len(source) else None)
        for n, element in enumerate(rule.after)
    )


def apply_rules(text: str, ruleset: RuleSet) -> str:
    """Transliterate ``text`` with ``ruleset``.

    The cursor moves left to right. At each position the rules are tried
    in order and the first match wins; its output is emitted and the
    cursor advances past what the rule consumed, so output is never
    re-examined. A position no rule matches emits its character
    unchanged.

    Args:
        text: Text to transliterate.
        ruleset: Rules to apply.

    Returns:
        The transliterated text, NFC-normalised when the rule file asked
        for it.
    """
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        for rule in ruleset.rules:
            if _matches(rule, out, text, cursor):
                out.extend(rule.output)
                cursor += len(rule.source)
                break
        else:
            out.append(text[cursor])
            cursor += 1

    result = "".join(out)
    return ud.normalize("NFC", result) if ruleset.normalize_nfc else result


__all__ = [
    "apply_rules",
]
