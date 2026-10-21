"""One queued netboot arm retirement, run by the netboot duty's machine (MCPs board task 7b7570c5).

WHAT A RETIREMENT IS. A ``netboot-retire`` row (MCPs mig 833) names one
roster machine and says why. A live netboot arm refuses its machine a second
arm until it expires, two hours after it was made; on 2026-10-21 pendragon's
install was cut off mid-download by a hub runner restart and its re-arm was
refused ``NETBOOT_ARM_LIVE`` for forty minutes. MCPs ``fleet-mcp``'s
``fleet-netboot retire`` ends such an arm: it finds the machine's live arm by
its roster boot MAC, revokes the arm's minted Tailscale key and reads back
that the key can enrol nothing, records the retirement in the served root,
from which moment the server serves the arm nothing, and prints one
``NETBOOT RETIRED`` line per arm.

THE SAME RELEASE, THE SAME CLIENT. It runs from the installed sealed netboot
release with the tailnet writer client in its environment alone, through
:func:`fleet.core.netboot_arm.ready` and :func:`fleet.core.netboot_arm.run_cli`,
exactly as an arm and the adopt pass do. The reason travels as one argv
element, never through a shell.

HOW THE JOB CLOSES. Exit 0 with at least one ``NETBOOT RETIRED`` line is
``passed``, the detail those lines. A refusal the command raises before
calling Tailscale (the machine is not in the roster or declares no boot MAC,
it has no live arm, or the reason is outside its shape) is ``refused`` with
the command's own text, since nothing was revoked or recorded. Any other exit
is ``failed`` with the output's tail. An exit 0 that printed no retired line,
or output carrying a tailnet key, is closed ``refused`` by this runner, with
a detail saying it exited 0 and why its answer was not taken, as for an arm.
"""

from __future__ import annotations

from typing import Final

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks, netboot_arm
from fleet.core.sealed_release import SealedRelease

#: The line the command prints for each arm it retired.
RETIRED_PREFIX: Final = "NETBOOT RETIRED "

#: The command's refusals that leave nothing revoked and nothing recorded,
#: as ``fleet-mcp`` names them (``boot-mac.ts``, ``netboot-retire.ts`` and
#: ``netboot-retirements.ts``).
RETIRE_REFUSALS: Final[tuple[str, ...]] = (
    "NETBOOT_ARM_MACHINE_UNKNOWN",
    "NETBOOT_ARM_MACHINE_NOT_ARMABLE",
    "NETBOOT_RETIRE_NO_LIVE_ARM",
    "NETBOOT_RETIRE_REASON_INVALID",
)

#: The retirement's deadline: one token exchange, then a DELETE and a GET of
#: the arm's key, each a Tailscale request with its own wall.
RETIRE_TIMEOUT_SECONDS: Final[int] = 300

#: Detail prefix when the job carries no reason, which the queue's own pin
#: (MCPs mig 833) refuses before a row exists.
NO_REASON_CODE: Final = "NETBOOT_RETIRE_NO_REASON"

#: Detail prefix when the command exited 0 without a retired line.
ANSWER_MALFORMED_CODE: Final = "NETBOOT_RETIRE_ANSWER_MALFORMED"

#: Detail prefix when the command's output carries a tailnet key.
KEY_IN_OUTPUT_CODE: Final = "NETBOOT_RETIRE_OUTPUT_CARRIES_KEY"


def retire_argv(
    place: netboot_arm.NetbootPlace, release: SealedRelease, machine: str, reason: str
) -> tuple[str, ...]:
    """The release's own ``fleet-netboot retire`` of one roster machine's live arm.

    Args:
        place: Where the netboot duty's commands run from.
        release: The verified release.
        machine: The roster name the job asks to retire the arm of.
        reason: Why, as the job carries it; one argv element.

    Returns:
        Node running the release's CLI's ``retire`` with the served root,
        whose arms and retirement records it reads and writes, the registry
        this tick loaded, whose roster names the machine's boot MAC, the
        machine and the reason.
    """
    return netboot_arm.cli_argv(
        place,
        release,
        "retire",
        "--root",
        str(place["served_root"]),
        "--registry",
        str(place["registry"]),
        "--node",
        machine,
        "--reason",
        reason,
    )


def retire_outcome(
    result: _test_hooks.CommandResult, release: SealedRelease, machine: str
) -> netboot_arm.ArmOutcome:
    """Read a finished retirement into the job's close.

    Args:
        result: The command's exit status and streams.
        release: The release it ran from.
        machine: The machine whose arm it retired.

    Returns:
        ``passed`` with the retired lines; ``refused`` with the command's own
        refusal, or with this runner's when it exited 0 without a retired
        line or its output carries a key; ``failed`` with the output's tail
        otherwise. Every detail this runner writes names the machine, the
        release and the revision.
    """
    opening = (
        f"fleet-netboot retire of {machine}'s arm from sealed release {release['release']} "
        f"(revision {release['revision']})"
    )
    returncode = result["returncode"]
    if netboot_arm.carries_key(result):
        return netboot_arm.ArmOutcome(
            status=ClosingStatus.REFUSED if returncode == 0 else ClosingStatus.FAILED,
            exit_code=None if returncode == 0 else returncode,
            detail=(
                f"{KEY_IN_OUTPUT_CODE}: {opening} exited {returncode} and its output carries "
                f"a tailnet key ({netboot_arm.KEY_MARK}...), so none of it is copied here; the "
                "served root's retirement records on the netboot host are the record"
            ),
        )
    if returncode == 0:
        stdout = result["stdout"].splitlines()
        retired = [line for line in stdout if line.startswith(RETIRED_PREFIX)]
        if retired:
            return netboot_arm.ArmOutcome(
                status=ClosingStatus.PASSED, exit_code=0, detail=f"{opening}: {'; '.join(retired)}"
            )
        return netboot_arm.ArmOutcome(
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"{ANSWER_MALFORMED_CODE}: {opening} exited 0 and printed no "
                f"'{RETIRED_PREFIX.strip()}' line where it prints one per arm retired, so this "
                "runner does not report it retired"
            ),
        )
    error = netboot_arm.cli_error(result["stderr"])
    if any(error.startswith(f"{code}:") for code in RETIRE_REFUSALS):
        return netboot_arm.ArmOutcome(status=ClosingStatus.REFUSED, exit_code=None, detail=error)
    return netboot_arm.ArmOutcome(
        status=ClosingStatus.FAILED,
        exit_code=returncode,
        detail=f"{opening} exited {returncode}: {netboot_arm.output_tail(result)}",
    )


__all__ = [
    "ANSWER_MALFORMED_CODE",
    "KEY_IN_OUTPUT_CODE",
    "NO_REASON_CODE",
    "RETIRED_PREFIX",
    "RETIRE_REFUSALS",
    "RETIRE_TIMEOUT_SECONDS",
    "retire_argv",
    "retire_outcome",
]
