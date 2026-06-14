"""Project invariant: every action_lab / sniffer diagnostic is structured.

Diagnostic events live on the ``DIAGNOSTIC`` channel of the runtime event
stream, not in printf-rendered ``log.info`` lines. This invariant guards
against regression to the old "text-format diagnostic" style by scanning
the source tree for the banned patterns:

* A ``log.info(...)`` call whose first argument is a string literal
  containing ``"DIAGNOSTIC"`` (e.g. ``"TELEPORT_DIAGNOSTIC ..."``).
* A ``log.info(...)`` call whose first argument matches the ASCII pattern
  ``MAP_POSITIONS``, ``ACTION_PHASE_OVERLAP``, or
  ``COMMAND_DISPATCH_FAILURE`` -- the named diagnostics migrated away
  from text in this turn.

Any future caller who wants to emit a diagnostic must use
:func:`tankpit_bot.runtime_logging.emit_diagnostic` with a documented
``diagnostic_kind`` and a strict-typed payload.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BANNED_DIAGNOSTIC_SUBSTRINGS: tuple[str, ...] = (
    "DIAGNOSTIC",
    "MAP_POSITIONS",
    "ACTION_PHASE_OVERLAP",
    "COMMAND_DISPATCH_FAILURE",
)


def _iter_log_info_string_literals(
    tree: ast.AST,
) -> list[tuple[int, str]]:
    """Return ``(lineno, literal)`` pairs for every ``log.info("...")`` call.

    Only the FIRST positional argument is examined -- printf-style
    ``log.info`` calls put the format string there, which is what
    consumers query for diagnostic patterns.
    """
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "info"
            and isinstance(func.value, ast.Name)
            and func.value.id == "log"
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            findings.append((node.lineno, first.value))
    return findings


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def _scan_paths() -> list[Path]:
    """Return every ``.py`` source file in ``src/`` that this invariant covers."""
    root = _project_root() / "src" / "tankpit_bot"
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_text_format_diagnostic_log_lines_remain() -> None:
    """No ``log.info("...DIAGNOSTIC...")`` (or related) lines remain in src/.

    Catches accidental reintroduction of text-format diagnostics. New
    diagnostic events must call
    :func:`tankpit_bot.runtime_logging.emit_diagnostic`.
    """
    violations: list[str] = []
    for path in _scan_paths():
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        tree = ast.parse(source, filename=str(path))
        for lineno, literal in _iter_log_info_string_literals(tree):
            for banned in _BANNED_DIAGNOSTIC_SUBSTRINGS:
                if banned in literal:
                    violations.append(
                        f"{path.relative_to(_project_root())}:{lineno} "
                        f"log.info({literal!r}) contains banned diagnostic "
                        f"substring {banned!r} -- use emit_diagnostic(...) instead"
                    )
    assert violations == [], "\n".join(violations)


def test_invariant_scanner_detects_banned_substring(tmp_path: Path) -> None:
    """The scanner picks up a banned substring in a synthetic source file.

    Exercises the scanner against a tiny fixture so the positive path of
    :func:`_iter_log_info_string_literals` has explicit coverage and a
    future broken scanner would itself fail this test.
    """
    fixture = "log.info('TELEPORT_DIAGNOSTIC target=%d', 1)\n"
    tree = ast.parse(fixture)
    findings = _iter_log_info_string_literals(tree)
    assert findings == [(1, "TELEPORT_DIAGNOSTIC target=%d")]


def test_invariant_scanner_ignores_unrelated_log_calls() -> None:
    """Non-``log.info`` calls and non-string first args do not register."""
    fixture = """
log.debug("DIAGNOSTIC something")
log.info(some_variable)
other_logger.info("DIAGNOSTIC text")
log.info("benign text without banned substring")
"""
    tree = ast.parse(fixture)
    findings = _iter_log_info_string_literals(tree)
    assert findings == [(5, "benign text without banned substring")]
