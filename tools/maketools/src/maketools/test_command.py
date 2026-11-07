"""The ``test`` command's flags: what every package's ``make test`` hands the launcher.

Held apart from :mod:`maketools.cli`, which names it in its command table,
because the command line it parses grew with each item of
:mod:`maketools.test_run` and the table's file reached the 600-line
ceiling when ``--mutation`` (MCPs board task e52e5148) was added.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.test_run import Runner, run_tests

#: The flag that skips the pre-run sweep; for debugging only.
NO_SWEEP_FLAG: Final[str] = "--no-sweep"

#: The flag that runs a suite without xdist.
SERIAL_FLAG: Final[str] = "--serial"

#: The flag that names how pytest is reached.
RUNNER_FLAG: Final[str] = "--runner"

#: The flag that runs a suite under a HOME of its own (test_run item 6).
OWN_HOME_FLAG: Final[str] = "--own-home"

#: The flag that refuses the Hugging Face hub to a suite (test_run item 7).
HUB_OFFLINE_FLAG: Final[str] = "--hub-offline"

#: The flag that mutation tests the src lines the commit adds after a
#: passing suite (test_run item 8).
MUTATION_FLAG: Final[str] = "--mutation"

#: Every flag that takes no value.
SWITCHES: Final[tuple[str, ...]] = (
    NO_SWEEP_FLAG,
    SERIAL_FLAG,
    OWN_HOME_FLAG,
    HUB_OFFLINE_FLAG,
    MUTATION_FLAG,
)


def require_runner(value: str) -> Runner:
    """Narrow a ``--runner`` value.

    Args:
        value: The text after the flag.

    Returns:
        The runner.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` when it is neither runner.
    """
    if value == Runner.POETRY:
        return Runner.POETRY
    if value == Runner.VENV:
        return Runner.VENV
    raise AppError(MaketoolsErrorCode.USAGE, f"{RUNNER_FLAG} must be poetry or venv, got {value!r}")


def command_test(arguments: Sequence[str]) -> int:
    """``test [--no-sweep] [--serial] [--own-home] [--hub-offline] [--mutation] [--runner R] ...``.

    Args:
        arguments: The launcher's flags first, then anything for pytest.

    Returns:
        pytest's exit status, or the mutation step's after a passing suite.

    Raises:
        AppError: ``MAKETOOLS_USAGE`` on an unknown runner.
    """
    given: set[str] = set()
    runner = Runner.POETRY
    remaining = list(arguments)
    while remaining and remaining[0] in (*SWITCHES, RUNNER_FLAG):
        flag = remaining.pop(0)
        if flag == RUNNER_FLAG:
            runner = require_runner(remaining.pop(0) if remaining else "")
        else:
            given.add(flag)
    return run_tests(
        Path.cwd(),
        remaining,
        sweep=NO_SWEEP_FLAG not in given,
        serial=SERIAL_FLAG in given,
        runner=runner,
        own_home=OWN_HOME_FLAG in given,
        hub_offline=HUB_OFFLINE_FLAG in given,
        mutation=MUTATION_FLAG in given,
    )


__all__ = [
    "HUB_OFFLINE_FLAG",
    "MUTATION_FLAG",
    "NO_SWEEP_FLAG",
    "OWN_HOME_FLAG",
    "RUNNER_FLAG",
    "SERIAL_FLAG",
    "SWITCHES",
    "command_test",
    "require_runner",
]
