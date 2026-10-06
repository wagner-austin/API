"""``concurrent TARGET...``: several of one package's make targets side by side.

WHY. A package's ``make check`` is held to five minutes (MCPs board task
1b152218, :mod:`maketools.budget_run`), and lint and the suite were run one
after the other although neither reads what the other writes once the
dependencies are synced and the sources formatted. clients/TankpitBot on
sedona, 2026-10-05 (board task 28e47ae3): 371 s of tests after about 154 s
of guard and mypy, so the check paid both in full. Run beside the suite, the
guard and mypy cost the check nothing while the suite is the longer.

WHY NOT ``make -j``. GNU make runs prerequisites in parallel only under
``-j``, and reading its interleaved output needs ``--output-sync``, which
GNU Make 3.81 does not have; pendragon and serendipity run 3.81, sedona,
loki and the hub 4.4.1. This runs ``make TARGET`` once per target as
concurrent children, through :data:`maketools._test_hooks.run_concurrently`,
and prints each child's whole output as one block in the order the targets
were named, headed by its verdict and its seconds, so a failure reads as the
named target's own output and not as lines threaded through two others.

THE TARGETS MUST BE INDEPENDENT. Each child is a separate make with no
memory of its siblings, so a target listed here must not need another to
have run first; a package runs whatever they share (``poetry sync``, a
formatter that rewrites files) before this command, in its own recipe.

ITS OWN LINES NEVER READ AS A SUITE'S COUNTS. tools/fleet's verdict
(``fleet.core.verdict``) takes the LAST ``N passed`` and ``N failed`` in a
check's transcript as the suite's totals, and this command's lines are the
last a check prints before the budget verdict. So they spell no number
before either word: on 2026-10-05 the summary read ``3 of 3 passed`` and
loki's verdict for TankpitBot at dec6ae3f2 reported ``tests=3p`` for a suite
of over seven thousand.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools import _test_hooks
from maketools.workspace import FANOUT_WALL_SECONDS

#: Fewer than this is a plain ``make TARGET``, which needs no command.
MINIMUM_TARGETS: Final[int] = 2


def run_targets_concurrently(package_directory: Path, targets: Sequence[str]) -> int:
    """Run ``make TARGET`` for every target at once, then report each.

    Args:
        package_directory: The package whose Makefile holds the targets.
        targets: The targets, each runnable on its own.

    Returns:
        0 when every target passed, else the exit code of the first one
        named that failed.
    """
    outcomes = _test_hooks.run_concurrently(
        [["make", target] for target in targets],
        cwd=package_directory,
        env=_test_hooks.environ(),
        timeout_seconds=FANOUT_WALL_SECONDS,
    )
    failed: list[tuple[str, int]] = []
    timings: list[str] = []
    for target, outcome in zip(targets, outcomes, strict=True):
        seconds = int(outcome["seconds"])
        code = outcome["returncode"]
        verdict = "passed" if code == 0 else f"FAILED with exit {code}"
        _test_hooks.write_line("")
        _test_hooks.write_line(f"=== concurrent: {target} {verdict} after {seconds}s")
        _test_hooks.write_line(outcome["output"].rstrip("\n"))
        timings.append(f"{target} {seconds}s")
        if code != 0:
            failed.append((target, code))
    summary = ", ".join(timings)
    if failed:
        names = ", ".join(target for target, _ in failed)
        _test_hooks.write_error(
            f"concurrent: {len(failed)} of {len(targets)} target(s) failed: {names} ({summary})"
        )
        return failed[0][1]
    _test_hooks.write_line("")
    _test_hooks.write_line(f"concurrent: every target succeeded ({summary})")
    return 0


def command_concurrent(arguments: Sequence[str]) -> int:
    """``concurrent TARGET TARGET...``: run this package's targets side by side.

    Args:
        arguments: Two or more make target names.

    Returns:
        :func:`run_targets_concurrently`'s exit code.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` for fewer than two targets or for an
            argument that make would read as a flag.
    """
    if len(arguments) < MINIMUM_TARGETS:
        raise AppError(
            MaketoolsErrorCode.USAGE,
            f"concurrent needs at least {MINIMUM_TARGETS} targets, got {list(arguments)}; "
            "one target is plain make",
        )
    flags = [argument for argument in arguments if argument.startswith("-")]
    if flags:
        raise AppError(
            MaketoolsErrorCode.USAGE, f"concurrent takes target names only, not flags: {flags}"
        )
    return run_targets_concurrently(Path.cwd(), arguments)


__all__ = ["MINIMUM_TARGETS", "command_concurrent", "run_targets_concurrently"]
