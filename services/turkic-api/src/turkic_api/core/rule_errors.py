"""Refusing a rule file, with the position that caused it.

Both the lexer and the parser refuse input, and both need to say where. The
error type and the position arithmetic live here so that neither has to import
the other to raise.
"""

from __future__ import annotations

from typing import Final

ERR_RULE_PARSE: Final = "TURKIC_TRANSLITEVAL_001_RULE_PARSE"


class RuleParseError(ValueError):
    """Raised when a rule file uses syntax this interpreter does not accept.

    Refusing to parse is deliberate. A rule that is accepted but
    misunderstood transliterates silently and wrongly, which is far more
    expensive to discover than a file that will not load.

    Args:
        line (int): One-based line the offending statement started on.
        reason (str): What could not be read.
        statement (str): The offending source text.
    """

    def __init__(self, line: int, reason: str, statement: str) -> None:
        """Render ``code: line N: reason :: statement`` as the string form."""
        self.code = ERR_RULE_PARSE
        self.line = line
        self.reason = reason
        self.statement = statement
        super().__init__(f"{ERR_RULE_PARSE}: line {line}: {reason} :: {statement}")


def _line_of(text: str, index: int) -> int:
    """Return the one-based line number containing ``index``."""
    return text.count("\n", 0, index) + 1


def fail(text: str, index: int, reason: str) -> RuleParseError:
    """Build a parse error naming the line and the statement it sits on."""
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    statement = text[start:] if end == -1 else text[start:end]
    return RuleParseError(_line_of(text, index), reason, statement.strip())


__all__ = [
    "ERR_RULE_PARSE",
    "RuleParseError",
    "fail",
]
