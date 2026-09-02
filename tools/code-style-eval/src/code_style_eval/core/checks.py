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


def checker_command(checker: str, interpreter: str, target: Path, root: Path) -> tuple[str, ...]:
    """Compose the argv for one checker over one item.

    Every checker is invoked through ``-m`` on a named interpreter rather
    than by bare name. A bare ``ruff`` resolves against PATH, which is how a
    sweep silently scores generated code with a different tool version than
    the repo pins.

    ``ruff`` and ``mypy`` take the FILE. The monorepo guards take the item's
    ROOT through ``--root``, because they are scoped to a tree rather than to
    a file. Passing them no root at all would scope them to whatever package
    the process happens to be running in, which is the same answer for every
    item in the sweep.

    Args:
        checker: Which checker to run.
        interpreter: Path to the Python interpreter that has the checkers.
        target: The generated file.
        root: The item's own guard root, holding only that file.

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
        return (interpreter, "-m", "scripts.guard", "--root", str(root))
    raise ValueError(f"unknown checker '{checker}'; known checkers: {CHECKERS}")


def _first_line(text: str) -> str:
    """Return the first line of some output that carries a finding.

    A line ending in a colon is an introducer: it announces what follows
    instead of saying it. Both checkers that fail here emit one.
    ``scripts.guard`` opens its violation list with "Guard checks failed:",
    so reading the first non-blank line gave every guard failure in a whole
    sweep the identical detail. ``mypy`` wraps to the console width, which in
    a pipe splits its diagnostic so the first line ends at "error:" and the
    message lands on the second. Skipping introducers reaches the finding in
    both cases; for mypy it trades the file:line locator for the message,
    which is the better half given the outcome already records the item.

    If nothing but introducers is present the first of them is returned
    rather than the empty string: a checker that said only "Guard checks
    failed:" still said something, and reporting nothing would read as a
    silent pass.

    Args:
        text: The output to summarise.

    Returns:
        The first line carrying a finding, stripped, or the empty string when
        the output is blank.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if not line.endswith(":"):
            return line
    return lines[0] if lines else ""


def run_check(checker: str, interpreter: str, target: Path, root: Path, cwd: Path) -> CheckOutcome:
    """Run one checker over one item and record its verdict.

    Args:
        checker: Which checker to run.
        interpreter: Interpreter that has the checkers installed.
        target: The generated file.
        root: The item's own guard root.
        cwd: Directory to run the checker in. It is the package through whose
            ``scripts/guard.py`` the guards are invoked, NOT the tree they
            check -- ``--root`` decides that.

    Returns:
        The outcome.

    Raises:
        ValueError: If the checker is not one of :data:`CHECKERS`.
    """
    command = checker_command(checker, interpreter, target, root)
    finished = Hooks.run_checker(command, cwd)
    # stderr FIRST. The guards write a rule-count summary to stdout and the
    # violations themselves to stderr, so reading stdout first gave every
    # guard failure the same detail -- the constant banner line "Guard rule
    # summary:" -- which indexes nothing. ruff and mypy report findings on
    # stdout and leave stderr empty on an ordinary failure, so they are
    # unaffected; when mypy does write to stderr it has crashed, and the
    # crash is the more useful line of the two.
    detail = _first_line(finished.stderr) or _first_line(finished.stdout)
    return CheckOutcome(
        checker=("ruff" if checker == "ruff" else ("mypy" if checker == "mypy" else "guards")),
        passed=finished.returncode == 0,
        exit_code=finished.returncode,
        detail="" if finished.returncode == 0 else detail,
    )


def score_item(
    *, item_id: str, arm: str, interpreter: str, target: Path, root: Path, cwd: Path
) -> ItemOutcome:
    """Run every checker over one generated file.

    All three run even after one fails. A completion that ruff rejects may
    still be interesting to mypy, and stopping early would make the
    per-checker rates depend on the order rather than on the code.

    Args:
        item_id: The held-out file this completion was generated for.
        arm: Which model produced it.
        interpreter: Interpreter that has the checkers installed.
        target: The generated file.
        root: The item's own guard root.
        cwd: Directory the checkers are invoked from.

    Returns:
        The item's outcome across every checker.
    """
    checks = tuple(run_check(checker, interpreter, target, root, cwd) for checker in CHECKERS)
    return ItemOutcome(
        item_id=item_id,
        arm=arm,
        checks=checks,
        all_passed=all(check["passed"] for check in checks),
    )


__all__ = ["checker_command", "run_check", "score_item"]
