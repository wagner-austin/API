"""The netboot duty's adopt pass, run every netboot tick (MCPs board task 49d366ac).

WHAT ADOPTING IS. A machine the netboot installer reinstalled reaches the
tailnet as a NEW device, under a random address, while its old record still
holds the roster's name and address. Every peer, every ssh alias and the
roster name the machine by that address, so until the new device takes it
the reinstalled machine is unreachable by everything that knows it. MCPs
``fleet-mcp``'s ``fleet-netboot adopt`` reads the arms that checked in
within the last day and have no adoption record, finds the device each
checked in from, deletes the other devices holding the roster's address,
moves the new one to that address, names it, and records the adoption in
the served root, printing one ``NETBOOT ADOPTED`` line for each, or one
``NETBOOT ADOPT nothing pending`` line counting the arms it read.

WHY EVERY TICK, AND NOT A JOB. Until this pass an adoption was typed by
hand on diphtheria after someone noticed the check-in. A check-in is the
machine's own act, which no session queues, so the duty that served the
boot finishes it: each netboot tick runs ``adopt`` once, after its claim.
A pending adoption a step of which Tailscale refused is left without a
record, so the next tick tries it again.

THE SAME RELEASE, THE SAME KEY. ``adopt`` runs from the installed sealed
netboot release with the tailnet API key in its environment alone, through
:func:`fleet.core.netboot_arm.ready` and :func:`fleet.core.netboot_arm.run_cli`,
exactly as an arm does. The serve process never holds the key.

WHAT THE TICK LOGS. The pass returns the lines the tick logs, one per line
``adopt`` printed, each naming the release and revision it ran from; a
refusal before it ran, a non-zero exit with its output's tail, an exit 0
that printed nothing, or output carrying a tailnet key, which is never
copied, is one line saying so. None of them fails the tick: the tick's exit
says whether the runner worked (:mod:`fleet.cli.agent`).
"""

from __future__ import annotations

from typing import Final

from fleet.core import _test_hooks, netboot_arm
from fleet.core.sealed_release import SealedRelease

#: The pass's deadline: one device list, then a delete, a readdress and a
#: rename per pending check-in, each a Tailscale request with its own wall.
ADOPT_TIMEOUT_SECONDS: Final[int] = 300

#: Line prefix when ``adopt`` exited 0 and printed nothing.
ANSWER_EMPTY_CODE: Final = "NETBOOT_ADOPT_ANSWER_EMPTY"

#: Line prefix when ``adopt``'s output carries a tailnet key.
KEY_IN_OUTPUT_CODE: Final = "NETBOOT_ADOPT_OUTPUT_CARRIES_KEY"


def adopt_argv(place: netboot_arm.NetbootPlace, release: SealedRelease) -> tuple[str, ...]:
    """The release's own ``fleet-netboot adopt`` over the served root.

    Args:
        place: Where the netboot duty's commands run from.
        release: The verified release.

    Returns:
        Node running the release's CLI's ``adopt`` with the served root,
        whose arms and adoption records it reads, and the registry this
        tick loaded, whose roster names each machine's address.
    """
    return netboot_arm.cli_argv(
        place,
        release,
        "adopt",
        "--root",
        str(place["served_root"]),
        "--registry",
        str(place["registry"]),
    )


def adopt_report(result: _test_hooks.CommandResult, release: SealedRelease) -> tuple[str, ...]:
    """Read a finished ``adopt`` into the lines the tick logs.

    Args:
        result: Its exit status and streams.
        release: The release it ran from.

    Returns:
        One line per line it printed when it exited 0; otherwise one line,
        naming the key in its output without copying any of it, its empty
        answer, or its exit status and output's tail. Every line names the
        release and the revision.
    """
    opening = (
        f"fleet-netboot adopt from sealed release {release['release']} "
        f"(revision {release['revision']})"
    )
    returncode = result["returncode"]
    if netboot_arm.carries_key(result):
        return (
            f"{KEY_IN_OUTPUT_CODE}: {opening} exited {returncode} and its output carries a "
            f"tailnet key ({netboot_arm.KEY_MARK}...), so none of it is logged; the served "
            "root's adoption records on the netboot host are the record",
        )
    if returncode != 0:
        return (f"{opening} exited {returncode}: {netboot_arm.output_tail(result)}",)
    printed = tuple(
        f"{opening}: {line}" for line in result["stdout"].splitlines() if line.strip() != ""
    )
    if printed:
        return printed
    return (
        f"{ANSWER_EMPTY_CODE}: {opening} exited 0 and printed nothing, where it prints a line "
        "for each adoption or one saying none is pending",
    )


def adopt_pass(place: netboot_arm.NetbootPlace) -> tuple[str, ...]:
    """Adopt every machine that checked in and has not been adopted.

    Args:
        place: Where the netboot duty's commands run from.

    Returns:
        The lines the tick logs: the refusal when the release's seal does
        not hold or the env file gives no key, with nothing run; otherwise
        :func:`adopt_report`'s reading of the run.
    """
    ready = netboot_arm.ready(place)
    if isinstance(ready, str):
        return (f"fleet-netboot adopt did not run: {ready}",)
    result = netboot_arm.run_cli(
        adopt_argv(place, ready["release"]),
        ready["writer"],
        wifi=None,
        timeout_seconds=ADOPT_TIMEOUT_SECONDS,
    )
    return adopt_report(result, ready["release"])


__all__ = [
    "ADOPT_TIMEOUT_SECONDS",
    "ANSWER_EMPTY_CODE",
    "KEY_IN_OUTPUT_CODE",
    "adopt_argv",
    "adopt_pass",
    "adopt_report",
]
