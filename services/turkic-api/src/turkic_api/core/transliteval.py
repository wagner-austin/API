from __future__ import annotations

import re
import unicodedata as ud
from pathlib import Path
from re import Match
from typing import Final, NamedTuple

from typing_extensions import TypedDict

_RULES_DIR: Final[Path] = Path(__file__).with_suffix("").parent / "rules"


class Rule(TypedDict):
    lhs: str
    rhs: str
    prev_chars: frozenset[str] | None
    prev_literal: str
    word_initial: bool
    lhs_macro_chars: frozenset[str] | None
    lhs_prefix: str


class MatchResult(NamedTuple):
    matched: bool
    chars_consumed: int


class RuleParseError(ValueError):
    def __init__(self, line_number: int, line_text: str, reason: str) -> None:
        self.line_number = line_number
        self.line_text = line_text
        self.reason = reason
        super().__init__(f"line {line_number}: {reason} :: {line_text}")


def _parse_macro_chars(body: str) -> frozenset[str]:
    return frozenset([ch for ch in body if ch != " "])


def _iter_rule_specs(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for ch in text:
        if ch == ";":
            segment = "".join(current).strip()
            if segment:
                parts.append(segment)
            current = []
        else:
            current.append(ch)
    segment = "".join(current).strip()
    if segment:
        parts.append(segment)
    return parts


def _rule(
    lhs: str,
    rhs: str,
    *,
    prev_chars: frozenset[str] | None = None,
    prev_literal: str = "",
    word_initial: bool = False,
    lhs_macro_chars: frozenset[str] | None = None,
    lhs_prefix: str = "",
) -> Rule:
    lhs_norm = lhs.replace(" ", "")
    rhs_norm = rhs.replace(" ", "")
    if lhs_norm == "":
        raise ValueError("rule lhs must be non-empty")
    return {
        "lhs": lhs_norm,
        "rhs": rhs_norm,
        "prev_chars": prev_chars,
        "prev_literal": prev_literal,
        "word_initial": word_initial,
        "lhs_macro_chars": lhs_macro_chars,
        "lhs_prefix": lhs_prefix,
    }


def _group(
    match: Match[str],
    name: str,
    line_number: int,
    stmt: str,
) -> str:
    value = match.group(name)
    if not isinstance(value, str):
        raise RuleParseError(line_number, stmt, f"group '{name}' not a string")
    return value


def _require_macro(
    macros: dict[str, frozenset[str]],
    name: str,
    line_number: int,
    stmt: str,
) -> frozenset[str]:
    macro = macros.get(name)
    if macro is None:
        raise RuleParseError(line_number, stmt, f"macro '{name}' not defined")
    return macro


def _parse_rule_stmt(
    stmt: str,
    macros: dict[str, frozenset[str]],
    line_number: int,
) -> Rule | None:
    if stmt.startswith("::"):
        return None

    macro_def: Match[str] | None = re.match(
        r"^\$(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[(?P<body>.*)\]$", stmt
    )
    if macro_def:
        name = macro_def.group("name")
        body = macro_def.group("body")
        macros[name] = _parse_macro_chars(body)
        return None

    word_initial: Match[str] | None = re.match(r"^\^\s*(?P<lhs>.+?)\s*>\s*(?P<rhs>.*)$", stmt)
    if word_initial:
        lhs_word = _group(word_initial, "lhs", line_number, stmt).rstrip()
        rhs_word = _group(word_initial, "rhs", line_number, stmt).strip()
        return _rule(lhs_word, rhs_word, word_initial=True)

    macro_ctx: Match[str] | None = re.match(
        r"^\$(?P<ctx>[A-Za-z_][A-Za-z0-9_]*)\s*}\s*(?P<lhs>.+?)\s*>\s*(?P<rhs>.*)$",
        stmt,
    )
    if macro_ctx:
        ctx = _group(macro_ctx, "ctx", line_number, stmt)
        lhs_ctx = _group(macro_ctx, "lhs", line_number, stmt).rstrip()
        rhs_ctx = _group(macro_ctx, "rhs", line_number, stmt).strip()
        prev_chars = _require_macro(macros, ctx, line_number, stmt)
        return _rule(lhs_ctx, rhs_ctx, prev_chars=prev_chars)

    literal_ctx: Match[str] | None = re.match(
        r"^(?P<lit>[^$][^}]*)}\s*(?P<lhs>.+?)\s*>\s*(?P<rhs>.*)$",
        stmt,
    )
    if literal_ctx:
        prev_literal = _group(literal_ctx, "lit", line_number, stmt).replace(" ", "")
        lhs_lit = _group(literal_ctx, "lhs", line_number, stmt).rstrip()
        rhs_lit = _group(literal_ctx, "rhs", line_number, stmt).strip()
        return _rule(lhs_lit, rhs_lit, prev_literal=prev_literal)

    icu_left_ctx: Match[str] | None = re.match(
        r"^\$(?P<ctx>[A-Za-z_][A-Za-z0-9_]*)\s*\{\s*(?P<lhs>.+?)\s*\}\s*>\s*(?P<rhs>.*)$",
        stmt,
    )
    if icu_left_ctx:
        ctx = _group(icu_left_ctx, "ctx", line_number, stmt)
        lhs_icu = _group(icu_left_ctx, "lhs", line_number, stmt).rstrip()
        rhs_icu = _group(icu_left_ctx, "rhs", line_number, stmt).strip()
        prev_chars = _require_macro(macros, ctx, line_number, stmt)
        return _rule(lhs_icu, rhs_icu, prev_chars=prev_chars)

    main: Match[str] | None = re.match(r"^(?P<lhs>.+?)\s*>\s*(?P<rhs>.*)$", stmt)
    if main:
        lhs_raw = _group(main, "lhs", line_number, stmt).rstrip()
        rhs_raw = _group(main, "rhs", line_number, stmt).strip()
        macro_lhs: Match[str] | None = re.match(
            r"^(?P<prefix>.+?)\s+\$(?P<macro>[A-Za-z_][A-Za-z0-9_]*)$",
            lhs_raw,
        )
        if macro_lhs:
            macro_name = _group(macro_lhs, "macro", line_number, stmt)
            macro_chars = _require_macro(macros, macro_name, line_number, stmt)
            prefix = _group(macro_lhs, "prefix", line_number, stmt).replace(" ", "")
            return _rule(
                lhs_raw,
                rhs_raw,
                lhs_macro_chars=macro_chars,
                lhs_prefix=prefix,
            )
        return _rule(lhs_raw, rhs_raw)

    raise RuleParseError(line_number, stmt, "unrecognized rule syntax")


def load_rules(name: str) -> list[Rule]:
    path = _RULES_DIR / name
    text = path.read_text(encoding="utf-8")
    macros: dict[str, frozenset[str]] = {}
    logical_lines = _gather_logical_lines(text.splitlines())
    rules: list[Rule] = []

    for idx, line in logical_lines:
        if line.startswith("#"):
            continue
        for stmt in _iter_rule_specs(line):
            if not stmt or stmt.startswith("#"):
                continue
            rule = _parse_rule_stmt(stmt, macros, idx)
            if rule is not None:
                rules.append(rule)
    return rules


def _gather_logical_lines(raw_lines: list[str]) -> list[tuple[int, str]]:
    logical_lines: list[tuple[int, str]] = []
    pending_macro: list[str] = []
    pending_start_idx = 0
    for idx, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if pending_macro:
            pending_macro.append(line)
            if "]" in line:
                combined = " ".join(pending_macro)
                logical_lines.append((pending_start_idx, combined))
                pending_macro = []
                pending_start_idx = 0
            continue
        if re.match(r"^\$[A-Za-z_][A-Za-z0-9_]*\s*=\s*\[", line) and "]" not in line:
            pending_macro = [line]
            pending_start_idx = idx
            continue
        logical_lines.append((idx, line))

    if pending_macro:
        raise RuleParseError(
            pending_start_idx,
            " ".join(pending_macro),
            "macro definition missing closing ']'",
        )
    return logical_lines


def _rule_match_at(rule: Rule, text: str, position: int) -> MatchResult:
    if rule["word_initial"] and position != 0:
        return MatchResult(False, 0)
    if not _prev_literal_matches(rule, text, position):
        return MatchResult(False, 0)
    if not _prev_chars_match(rule, text, position):
        return MatchResult(False, 0)
    macro_chars = rule["lhs_macro_chars"]
    if macro_chars is not None:
        return _match_macro(rule, text, position, macro_chars)
    if text.startswith(rule["lhs"], position):
        return MatchResult(True, len(rule["lhs"]))
    return MatchResult(False, 0)


def _prev_literal_matches(rule: Rule, text: str, position: int) -> bool:
    literal = rule["prev_literal"]
    if not literal:
        return True
    lit_len = len(literal)
    if position < lit_len:
        return False
    return text[position - lit_len : position] == literal


def _prev_chars_match(rule: Rule, text: str, position: int) -> bool:
    prev_chars = rule["prev_chars"]
    if prev_chars is None:
        return True
    if position == 0:
        return False
    return text[position - 1] in prev_chars


def _match_macro(rule: Rule, text: str, position: int, macro_chars: frozenset[str]) -> MatchResult:
    prefix = rule["lhs_prefix"]
    if not text.startswith(prefix, position):
        return MatchResult(False, 0)
    prefix_end = position + len(prefix)
    if prefix_end >= len(text):
        return MatchResult(False, 0)
    if text[prefix_end] in macro_chars:
        return MatchResult(True, len(prefix) + 1)
    return MatchResult(False, 0)


def _truncate_output(out: list[str], chars: int) -> None:
    remaining = chars
    while remaining > 0 and out:
        last = out.pop()
        if len(last) <= remaining:
            remaining -= len(last)
            continue
        out.append(last[:-remaining])
        remaining = 0


def apply_rules(text: str, rules: list[Rule]) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        replaced = False
        for rule in rules:
            match = _rule_match_at(rule, text, i)
            if match.matched:
                if rule["prev_literal"]:
                    _truncate_output(out, len(rule["prev_literal"]))
                replacement = rule["rhs"]
                if rule["prev_literal"]:
                    replacement = replacement + rule["lhs"]
                if replacement:
                    out.append(replacement)
                i += match.chars_consumed
                replaced = True
                break
        if not replaced:
            out.append(text[i])
            i += 1
    return ud.normalize("NFC", "".join(out))
