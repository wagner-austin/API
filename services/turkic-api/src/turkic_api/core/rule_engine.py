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
    if rule.anchor_start and cursor != 0:
        return False

    width = len(rule.before)
    if width:
        if len(out) < width:
            return False
        window = out[len(out) - width :]
        if any(char not in element for element, char in zip(rule.before, window, strict=True)):
            return False

    end = cursor + len(rule.source)
    if end > len(source):
        return False
    if any(source[cursor + n] not in element for n, element in enumerate(rule.source)):
        return False

    stop = end + len(rule.after)
    if stop > len(source):
        return False
    return not any(source[end + n] not in element for n, element in enumerate(rule.after))


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
