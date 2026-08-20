"""Capability declarations must be derived from their implementation.

A backend's ``supported_sizes`` names what that backend can actually build. When
it is written as a literal it is a hand-maintained copy of information the
implementation already holds, and the two drift silently -- nothing compares
them, and a test that asserts the literal against a copy of itself proves only
that the constant is what someone typed.

That drift was real, in both directions at once, in Model-Trainer:

* GPT2_CAPABILITIES advertised a ``"tiny"`` that MODEL_SIZES did not implement,
  so asking for the advertised size raised a bare ``KeyError`` from a dict index
  -- an untyped 500 for what is a caller mistake.
* The same table implemented an ``"xl"`` the registry never advertised.
* CHAR_LSTM_CAPABILITIES advertised only ``("small",)`` while its size lookup
  accepted ``tiny`` and ``medium`` too, hiding two working sizes from every
  caller that consulted capabilities.

The fix is to derive rather than to check: ``tuple(MODEL_SIZES)`` cannot disagree
with ``MODEL_SIZES``. This rule keeps the literal form from coming back.

An empty tuple is allowed and is not a copy of anything: it is the honest claim
of a backend that has no size table at all, such as one whose size is determined
by a hub model id, or a null-object backend that supports nothing.
"""

from __future__ import annotations

from pathlib import Path

from monorepo_guards import Rule, Violation
from monorepo_guards.util import read_lines

_KEY = '"supported_sizes":'


def _is_hardcoded_literal(value: str) -> bool:
    """Is this declaration a hand-written tuple rather than a derived one?

    Args:
        value: The text to the right of the ``"supported_sizes":`` key, already
            stripped of surrounding whitespace and any trailing comma.

    Returns:
        True when the value opens a tuple literal with content. ``()`` is empty
        and allowed; ``tuple(...)``/``frozenset(...)``/a name does not open with
        a parenthesis and is therefore derived.
    """
    if not value.startswith("("):
        return False
    # A multi-line literal leaves just "(" on this line; still a literal.
    return value != "()"


class CapabilityDerivationRule(Rule):
    name = "capability-sizes"

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for path in files:
            as_posix = path.as_posix()
            # Tests legitimately construct capability fixtures by hand; the rule
            # is about the declarations that ship, not about the ones under test.
            if "/tests/" in as_posix:
                continue
            for idx, line in enumerate(read_lines(path), start=1):
                if _KEY not in line:
                    continue
                value = line.split(_KEY, 1)[1].strip()
                # Drop a trailing comma and any trailing comment before judging.
                if "#" in value:
                    value = value.split("#", 1)[0].strip()
                value = value.rstrip(",").strip()
                if _is_hardcoded_literal(value):
                    out.append(
                        Violation(
                            file=path,
                            line_no=idx,
                            kind="capability-sizes-hardcoded",
                            line=line.strip(),
                        )
                    )
        return out


__all__ = ["CapabilityDerivationRule"]
