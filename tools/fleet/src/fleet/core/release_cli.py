"""A sealed MCPs release's own node CLI, as a duty's runner runs one and reads its close.

Every command a duty runs from an installed release is fleet-mcp's built
CLI module, imported by node and entered through its ``runFromProcess``:
the netboot duty's ``fleet-netboot arm``, ``retire`` and ``adopt``
(:mod:`fleet.core.netboot_arm`) and the control duty's ``fleet-inspect``
(:mod:`fleet.core.fleet_inspect`, MCPs board task 3e9c99af). How such a
command is composed, how much of a failed one's output a job's detail
carries, and the close a finished one gives its job are this module's, once:
the netboot lane held them until the inspect lane needed the same three.

THE MODULE IS ENTERED, NEVER RUN AS A SCRIPT. A built module whose entry
reads ``import.meta`` for its own place cannot be checked by its package's
mutation testing (fleet-mcp's ``bin.ts`` rule), so each CLI exports
``runFromProcess(process, ...)``, which reads its arguments from
``process.argv[1]`` on under ``node -e``; an entry that needs the release's
root is handed it as a further argument, so the module never derives it.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Final, TypedDict

from platform_core.json_utils import dump_json_str

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks

#: How much of a failed command's output its job's detail carries.
DETAIL_TAIL_CHARS: Final[int] = 600


class ReleaseCliOutcome(TypedDict):
    """How a finished release command closes its job.

    Attributes:
        status: ``passed``, ``failed`` or ``refused``.
        exit_code: The command's exit status, or None for ``refused``.
        detail: The line the submitter reads.
    """

    status: ClosingStatus
    exit_code: int | None
    detail: str


def release_cli_argv(
    nodejs: str, module: pathlib.Path, entered_with: Sequence[str], *arguments: str
) -> tuple[str, ...]:
    """Node importing a release's built CLI module and entering it.

    Args:
        nodejs: The node binary.
        module: The built module inside the verified release.
        entered_with: What ``runFromProcess`` is handed after ``process``,
            each as a JSON string literal; empty for a CLI that needs nothing
            more.
        *arguments: The command and its flags, which the CLI reads from
            ``process.argv[1]`` on.

    Returns:
        ``node --input-type=module -e <importer> -- <arguments>``. The
        ``--`` ends node's own options: without it node reads a leading
        flag as one of its own, as it refused fleet-inspect's
        ``--workspace`` with ``bad option`` (MCPs board task c8978826), and
        node consumes it, so ``process.argv[1]`` is the first argument
        whatever it is.
    """
    handed = "".join(f", {dump_json_str(value)}" for value in entered_with)
    importer = (
        f"const m = await import({module.as_uri()!r}); await m.runFromProcess(process{handed});"
    )
    return (nodejs, "--input-type=module", "-e", importer, "--", *arguments)


def output_tail(result: _test_hooks.CommandResult) -> str:
    """The end of a failed command's output, for the line that reports it.

    Args:
        result: The command's exit status and streams.

    Returns:
        The last :data:`DETAIL_TAIL_CHARS` characters of its stdout then
        stderr, stripped.
    """
    return (result["stdout"] + result["stderr"]).strip()[-DETAIL_TAIL_CHARS:]


__all__ = [
    "DETAIL_TAIL_CHARS",
    "ReleaseCliOutcome",
    "output_tail",
    "release_cli_argv",
]
