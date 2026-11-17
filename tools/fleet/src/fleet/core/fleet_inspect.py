"""One queued inspection, run by the machine holding the control duty (MCPs board task 3e9c99af).

WHAT AN INSPECTION IS. An ``inspect`` row (MCPs mig 987) names one roster
machine as its requested node and one probe of the committed catalog as its
project, ``probes/<id>``, and nothing else: no path, argument or command
line from the submitter reaches the queue. MCPs ``fleet-mcp``'s
``fleet-inspect`` (``inspect-cli.ts``, MCPs board task df8d446b) reads the
roster, refuses a node it does not name, a client and a disabled node,
stages that probe's fixed read-only script through the fleet audit's ssh
transport and prints the inspection as one JSON line. This runner hands it
the node and the probe id as two argv elements and nothing more.

THE RELEASE'S OWN COMMAND, VERIFIED. The command runs from the installed
sealed ``fleet-audit`` release, the one the control duty's fleet audit runs
from, which already reaches every node over ssh: its own verifier checks its
seal before anything in it runs (:mod:`fleet.core.sealed_release`), and the
release's root is handed to the CLI, which stages its scripts from there and
reads the roster from the workspace the row names.

HOW THE JOB CLOSES. Exit 0 with exactly one line holding a JSON object is
the answer, reported with ``dispatch_report``'s ``inspected``, which closes
the job ``passed`` or ``failed`` by the probe's own exit status and stores
the inspection in the same transaction. The CLI's other exits close it
here: 2 is a refusal before anything was copied (an unknown, client or
disabled node, or a malformed invocation), ``refused`` with its own words;
1, any other status and the deadline are ``failed`` with the output's tail,
since the probe was never answered. A run that exits 0 without exactly one
JSON object is ``refused``, naming why its answer was not taken.
"""

from __future__ import annotations

import pathlib
from enum import IntEnum
from typing import Final, TypedDict

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    load_json_str,
    narrow_json_to_dict,
)

from fleet.contracts.dispatch import ClosingStatus, DispatchJob
from fleet.core import _test_hooks
from fleet.core.release_cli import ReleaseCliOutcome, output_tail, release_cli_argv
from fleet.core.sealed_release import SealedRelease

#: The release kind the command runs from, as MCPs ``release-kinds.json`` names it.
KIND: Final = "fleet-audit"

#: The command inside a release: fleet-mcp's built inspect CLI.
CLI: Final = pathlib.PurePosixPath("fleet-mcp/dist/src/inspect-cli.js")

#: What an inspect job's project opens with, before the probe's id (MCPs mig 987).
PROJECT_PREFIX: Final = "probes/"

#: The command's deadline: the catalog's longest probe (45 s) and the 15 s
#: the CLI gives its ssh beyond it, the copy before it and the roster read
#: before that, inside the runner's lease and the tick's wall.
INSPECT_TIMEOUT_SECONDS: Final[int] = 180


class CliExit(IntEnum):
    """The CLI's exits this runner reads, as fleet-mcp's inspect-cli.ts
    exports them; any other status is a run that was never answered."""

    #: It printed the inspection.
    ANSWERED = 0
    #: It refused before copying anything.
    REFUSED = 2


#: Detail prefix when a runner without the inspect lane's flags claims one.
NO_RELEASE_CODE: Final = "INSPECT_NO_RELEASE"

#: Detail prefix when the job names no machine or no catalog probe.
JOB_MALFORMED_CODE: Final = "INSPECT_JOB_MALFORMED"

#: Detail prefix when the CLI exited 0 without exactly one JSON object.
ANSWER_MALFORMED_CODE: Final = "INSPECT_ANSWER_MALFORMED"


class InspectPlace(TypedDict):
    """Where this machine's inspections run from.

    Attributes:
        release: The installed sealed fleet-audit release's checkout.
        nodejs: The node binary that runs its CLI.
        workspace: The MCPs workspace whose registry names the roster.
    """

    release: pathlib.Path
    nodejs: str
    workspace: pathlib.Path


class InspectTarget(TypedDict):
    """What an inspect job asks: one machine and one catalog probe.

    Attributes:
        node: The roster machine, the job's requested node.
        probe: The catalog probe's id, the project after ``probes/``.
    """

    node: str
    probe: str


def inspect_target(job: DispatchJob) -> InspectTarget | str:
    """Read the machine and the probe a claimed inspect job names.

    Args:
        job: The claimed job.

    Returns:
        Its machine and probe, or a :data:`JOB_MALFORMED_CODE` refusal when
        it names no machine or its project is not ``probes/<id>``.
    """
    node = job["requested_node"]
    project = job["project"]
    if node is None or not project.startswith(PROJECT_PREFIX):
        return (
            f"{JOB_MALFORMED_CODE}: an inspect job names its machine as the requested node and "
            f"its probe as {PROJECT_PREFIX}<id>; this one names {node!r} and {project!r}, "
            "so nothing ran"
        )
    return InspectTarget(node=node, probe=project.removeprefix(PROJECT_PREFIX))


def inspect_argv(
    place: InspectPlace, release: SealedRelease, target: InspectTarget
) -> tuple[str, ...]:
    """The release's own ``fleet-inspect`` of one machine and probe.

    Args:
        place: Where inspections run from.
        release: The verified release.
        target: The machine and the probe.

    Returns:
        Node importing the release's CLI, entered with the release's root,
        and run with the workspace, the machine and the probe.
    """
    root = release["root"]
    return release_cli_argv(
        place["nodejs"],
        root / CLI,
        (str(root),),
        "--workspace",
        str(place["workspace"]),
        "--node",
        target["node"],
        "--probe",
        target["probe"],
    )


def run_inspection(argv: tuple[str, ...], *, timeout_seconds: int) -> _test_hooks.CommandResult:
    """Run one ``fleet-inspect`` with no variable added to its environment.

    Args:
        argv: What :func:`inspect_argv` composed.
        timeout_seconds: Its deadline; a job runs it with
            :data:`INSPECT_TIMEOUT_SECONDS`.

    Returns:
        Its exit status and captured streams, or the timed-out result after
        ``timeout_seconds``.
    """
    return _test_hooks.run(argv, timeout_seconds=timeout_seconds, set_env=())


def answered_inspection(result: _test_hooks.CommandResult) -> JSONObject | None:
    """The inspection a finished run printed, when it printed one.

    Args:
        result: The run's exit status and streams.

    Returns:
        The one JSON object of an exit-0 run whose stdout is exactly one
        line, passed on unread for the queue to validate; None for any
        other run, which :func:`inspect_outcome` closes.
    """
    match result["returncode"], result["stdout"].splitlines():
        case CliExit.ANSWERED, [line]:
            try:
                return narrow_json_to_dict(load_json_str(line))
            except (InvalidJsonError, JSONTypeError):
                return None
        case _:
            return None


def inspect_outcome(
    result: _test_hooks.CommandResult, release: SealedRelease, target: InspectTarget
) -> ReleaseCliOutcome:
    """Read a run that printed no inspection into the job's close.

    Args:
        result: The run's exit status and streams.
        release: The release it ran from.
        target: The machine and probe it was asked.

    Returns:
        ``refused`` with the CLI's own words for its refusal, or with this
        runner's for an exit-0 run without exactly one JSON object;
        ``failed`` with the output's tail otherwise. Every detail names the
        probe, the machine, the release and the revision.
    """
    opening = (
        f"fleet-inspect {target['probe']} of {target['node']} from sealed release "
        f"{release['release']} (revision {release['revision']})"
    )
    match result["returncode"]:
        case CliExit.ANSWERED:
            return ReleaseCliOutcome(
                status=ClosingStatus.REFUSED,
                exit_code=None,
                detail=(
                    f"{ANSWER_MALFORMED_CODE}: {opening} exited 0 and printed "
                    f"{len(result['stdout'].splitlines())} line(s) where it prints one JSON "
                    "object, so this runner reports no inspection"
                ),
            )
        case CliExit.REFUSED:
            return ReleaseCliOutcome(
                status=ClosingStatus.REFUSED,
                exit_code=None,
                detail=f"{opening} refused: {result['stderr'].strip()}",
            )
        case returncode:
            return ReleaseCliOutcome(
                status=ClosingStatus.FAILED,
                exit_code=returncode,
                detail=f"{opening} exited {returncode}: {output_tail(result)}",
            )


__all__ = [
    "ANSWER_MALFORMED_CODE",
    "CLI",
    "INSPECT_TIMEOUT_SECONDS",
    "JOB_MALFORMED_CODE",
    "KIND",
    "NO_RELEASE_CODE",
    "PROJECT_PREFIX",
    "CliExit",
    "InspectPlace",
    "InspectTarget",
    "answered_inspection",
    "inspect_argv",
    "inspect_outcome",
    "inspect_target",
    "run_inspection",
]
