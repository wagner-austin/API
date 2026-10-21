"""CLI: record which commit each of the hub pump's wake bridges ran, and how it exited.

Usage:
    poetry run -- python -m fleet.cli.publisher_ticks \\
        hpc-wake=0=/home/corvis/PROJECTS/API/tools/hpc-wake \\
        ci-wake=0=/home/corvis/PROJECTS/API/tools/ci-wake ...

The hpc-wake pump (``tools/hpc-wake/scripts/run_cycle.py``) runs every wake
bridge from a checkout on the hub, then runs this once with one
``<publisher>=<exit code>=<directory>`` token per bridge it ran. For each it
reads the checkout the directory belongs to (:mod:`fleet.core.checkout`) and
records the tick through MCPs ``dispatch_publisher_tick``
(:func:`fleet.core.tick_report.record_publisher_tick`), so the audit's query
checks can read which commit the hub ran (MCPs board task 9f1367d5); until
this the answer was in files on the hub that no check reads.

THE CHECKOUT IS READ WHEN THE TICK ENDS, after the bridges ran: one align
landing inside the same three minutes would be recorded against the commit
it moved to, and the next tick records the commit that then runs.

IT RUNS AS ONE OF THE PUMP'S OWN ROWS, in the fleet package's environment,
with the credentials the pump hands every bridge (``FLEET_MCP_API_KEY`` and
``CORVIS_TENANT_ID`` from its ``runs/env.ps1``), so a record that could not be
made is a red row in the pump's health record like any bridge's, never a
tick that crashed. It acts under the dispatch identity of the hub runner on
the machine it runs on (:func:`fleet.cli.tick.runner_identity`): it is the
fleet package acting for the hub, and the queue already knows that writer.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence
from typing import Final, TypedDict

from platform_core.cli_boundary import run_cli
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.cli.tick import runner_identity
from fleet.contracts.dispatch import DispatchLane
from fleet.contracts.publisher_tick import PublisherTick
from fleet.core import _test_hooks, queue, tick_report
from fleet.core.checkout import read_checkout

_log = get_logger(__name__)

#: What separates a token's publisher, exit code and directory.
TOKEN_SEPARATOR: Final = "="

#: The prefix of every refusal of the arguments.
USAGE_CODE: Final = "FLEET_PUBLISHER_TICKS_USAGE"


class RanPublisher(TypedDict):
    """One bridge the pump ran this tick.

    Attributes:
        publisher: Its name in the pump's table.
        exit_code: Its exit code.
        directory: The directory it ran in, inside its checkout.
    """

    publisher: str
    exit_code: int
    directory: pathlib.Path


def parse_ran(token: str) -> RanPublisher:
    """Read one ``<publisher>=<exit code>=<directory>`` token.

    The directory is last so a path holding the separator still reads whole;
    the exit code may be negative, as a signal death is reported.

    Args:
        token: One argument.

    Returns:
        The bridge.

    Raises:
        ValueError: ``FLEET_PUBLISHER_TICKS_USAGE`` naming the token when it
            has fewer than three parts, an empty name or directory, or an
            exit code that is not an integer.
    """
    parts = token.split(TOKEN_SEPARATOR, 2)
    if len(parts) == 3 and parts[0] != "" and parts[2] != "":
        publisher, exit_text, directory = parts
        digits = exit_text.removeprefix("-")
        if digits.isdigit():
            return RanPublisher(
                publisher=publisher, exit_code=int(exit_text), directory=pathlib.Path(directory)
            )
    raise ValueError(
        f"{USAGE_CODE}: {token!r} is not <publisher>=<exit code>=<directory>, the form "
        "the hpc-wake pump passes for each bridge it ran"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Record every bridge the pump ran.

    Args:
        argv: One token per bridge, at least one. Defaults to the process
            arguments.

    Returns:
        0 once every tick is recorded.

    Raises:
        ValueError: ``FLEET_PUBLISHER_TICKS_USAGE`` for no tokens or a
            malformed one, before anything is read or sent.
        AppError: ``QUEUE_CREDENTIALS_MISSING`` from
            :func:`fleet.core.queue.load_credentials`; ``CONFIG_ERROR`` from
            :func:`fleet.core.checkout.read_checkout` for a directory that is
            not a checkout of a known repository; and any refusal of the
            queue's, each before the later ticks are sent.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    if not tokens:
        raise ValueError(f"{USAGE_CODE}: name at least one <publisher>=<exit code>=<directory>")
    ran = [parse_ran(token) for token in tokens]
    credentials = queue.load_credentials()
    host = _test_hooks.hostname()
    label, session = runner_identity(DispatchLane.HUB, host)
    identity = queue.identity_arguments(label, session, str(pathlib.Path.cwd()))
    for bridge in ran:
        checkout = read_checkout(bridge["directory"])
        counted = tick_report.record_publisher_tick(
            credentials,
            PublisherTick(
                host=host,
                publisher=bridge["publisher"],
                repo=checkout["repo"],
                head_sha=checkout["head_sha"],
                dirty=checkout["dirty"],
                exit_code=bridge["exit_code"],
            ),
            identity=identity,
        )
        _log.info(
            "%s ran %s at %s%s, exit %d: tick %d in this state",
            bridge["publisher"],
            checkout["repo"],
            checkout["head_sha"],
            " (dirty)" if checkout["dirty"] else "",
            bridge["exit_code"],
            counted,
        )
    return 0


def entrypoint() -> None:
    """Module entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-publisher-ticks",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["TOKEN_SEPARATOR", "USAGE_CODE", "RanPublisher", "entrypoint", "main", "parse_ran"]


# Without this, `python -m fleet.cli.publisher_ticks` imports the module,
# records nothing and exits 0, which the pump would read as a recorded tick.
if __name__ == "__main__":
    entrypoint()
