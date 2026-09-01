"""Run the monorepo's own checkers over a generated file.

The instrument's whole premise is that house style has no external
benchmark. Nothing off the shelf knows that this repo wants TypedDicts with
encode/decode pairs, ``_test_hooks`` seams and no ``Any``. The repo's own
checkers do, so they are the metric: a completion passes if the tools the
operator already runs would accept it.

The three are run in increasing order of what they presuppose. ``ruff``
needs only that the file parses, ``mypy`` needs it to resolve, and the
monorepo guards encode architecture on top of both. Reporting them
separately is what distinguishes "the model cannot write Python" from "the
model writes Python that is not this repo's Python".
"""

from __future__ import annotations

from pathlib import Path

from code_style_eval.contracts.outcomes import (
    CHECKERS,
    CheckOutcome,
    ItemOutcome,
)
from code_style_eval.core._test_hooks import Hooks


def checker_command(checker: str, interpreter: str, target: Path) -> tuple[str, ...]:
    """Compose the argv for one checker over one target.

    Every checker is invoked through ``-m`` on a named interpreter rather
    than by bare name. A bare ``ruff`` resolves against PATH, which is how a
    sweep silently scores generated code with a different tool version than
    the repo pins.

    Args:
        checker: Which checker to run.
        interpreter: Path to the Python interpreter that has the checkers.
        target: File or directory to check.

    Returns:
        The argv.

    Raises:
        ValueError: If the checker is not one of :data:`CHECKERS`.
    """
    if checker == "ruff":
        return (interpreter, "-m", "ruff", "check", str(target))
    if checker == "mypy":
        return (interpreter, "-m", "mypy", str(target))
    if checker == "guards":
        return (interpreter, "-m", "scripts.guard")
    raise ValueError(f"unknown checker '{checker}'; known checkers: {CHECKERS}")


def _first_line(text: str) -> str:
    """Return the first non-blank line of some output.

    Args:
        text: The output to summarise.

    Returns:
        The first non-blank line, stripped, or the empty string.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def run_check(checker: str, interpreter: str, target: Path, cwd: Path) -> CheckOutcome:
    """Run one checker over one target and record its verdict.

    Args:
        checker: Which checker to run.
        interpreter: Interpreter that has the checkers installed.
        target: File to check.
        cwd: Directory to run the checker in, which is what makes the
            monorepo guards see the package they are checking.

    Returns:
        The outcome.

    Raises:
        ValueError: If the checker is not one of :data:`CHECKERS`.
    """
    command = checker_command(checker, interpreter, target)
    finished = Hooks.run_checker(command, cwd)
    detail = _first_line(finished.stdout) or _first_line(finished.stderr)
    return CheckOutcome(
        checker=("ruff" if checker == "ruff" else ("mypy" if checker == "mypy" else "guards")),
        passed=finished.returncode == 0,
        exit_code=finished.returncode,
        detail="" if finished.returncode == 0 else detail,
    )


def score_item(*, item_id: str, arm: str, interpreter: str, target: Path, cwd: Path) -> ItemOutcome:
    """Run every checker over one generated file.

    All three run even after one fails. A completion that ruff rejects may
    still be interesting to mypy, and stopping early would make the
    per-checker rates depend on the order rather than on the code.

    Args:
        item_id: The held-out file this completion was generated for.
        arm: Which model produced it.
        interpreter: Interpreter that has the checkers installed.
        target: The generated file.
        cwd: Directory to run the checkers in.

    Returns:
        The item's outcome across every checker.
    """
    checks = tuple(run_check(checker, interpreter, target, cwd) for checker in CHECKERS)
    return ItemOutcome(
        item_id=item_id,
        arm=arm,
        checks=checks,
        all_passed=all(check["passed"] for check in checks),
    )


__all__ = ["checker_command", "run_check", "score_item"]
