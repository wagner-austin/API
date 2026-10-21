"""One queued netboot arm, run by the machine holding the netboot duty (MCPs board task 83102d9d).

WHAT AN ARM IS. A ``netboot-arm`` row (MCPs migs 807 and 809) names one
roster machine. Arming it makes that machine's next network boot wipe its
disk and install Ubuntu, reach the tailnet with a single-use key minted for
it, and check in. MCPs ``fleet-mcp``'s ``fleet-netboot arm`` does all of it:
it reads the machine's boot MAC and the keys allowed to reach it from the
roster by name, builds the served tree, mints the key and writes the arm
into the served root, and prints one ``NETBOOT ARMED`` line. Until this
lane, that command was typed by hand on diphtheria (the operator,
2026-10-18: "everything needs to gothrough our fleet or event loop or job
queue syetem").

THE RELEASE'S OWN COMMAND, VERIFIED. The arm runs from the installed sealed
netboot release, the same one the hub runner's ``netboot-serve`` row serves
from: the row's ``{release}`` names it, and its own verifier checks its seal
before anything in it runs (:mod:`fleet.core.sealed_release`). The revision
the seal names is the arm's ``--commit``, so every arm records the commit it
was built from. The roster it reads is the registry this tick loaded, the
rolled commit's ``tools/fleet/registry`` (MCPs board task e447de72); until
then it was named as the release's ``fleet-mcp/fleet-nodes.json``, a file
the netboot release never shipped.

THE CREDENTIAL STAYS WHERE IT IS. The arm mints with the tailnet writer
OAuth client the workspace env file already holds for the netboot execution
lane, ``TAILNET_WRITER_CLIENT_ID`` and ``TAILNET_WRITER_CLIENT_SECRET``,
which replaced the personal ``TAILSCALE_API_KEY`` (MCPs board task
3b4c0931; fleet-mcp ``tailnet-oauth.ts`` exchanges them for a token). They
are read here and handed to the arm's environment and to nothing else: no
argv, no log, no detail. The arm prints the minted key's id and never the
key; a line of output carrying a tailnet key or secret (``tskey-``) means
that contract broke, and then none of the output is copied into the job's
detail.

SHARED WITH THE ADOPT PASS. :func:`ready`, :func:`cli_argv`,
:func:`run_cli`, :func:`carries_key` and :func:`output_tail` are how every
``fleet-netboot`` command the duty runs is verified, composed, run and read;
:mod:`fleet.core.netboot_adopt` runs ``adopt`` through them each tick.

HOW THE JOB CLOSES. Exit 0 with exactly one ``NETBOOT ARMED`` line is
``passed``, the detail that line. A refusal the arm raises before minting
anything (the machine is not in the roster, it declares no boot MAC, or it
already has a live arm) is ``refused`` with the arm's own text, since
nothing was armed. Any other exit is ``failed`` with the output's tail. The
queue pairs ``passed`` with exit 0, ``failed`` with a non-zero exit and
``refused`` with none, so an arm that exits 0 without its one line, or
whose output carries a key, is closed ``refused`` by this runner, with a
detail saying it exited 0 and why its answer was not taken.
"""

from __future__ import annotations

import pathlib
from typing import Final, TypedDict

from platform_core.env_file import parse_env_file
from platform_core.errors import AppError

from fleet.contracts.dispatch import ClosingStatus
from fleet.core import _test_hooks
from fleet.core.sealed_release import SealedRelease, verify_release

#: The release kind an arm runs, as MCPs ``release-kinds.json`` names it.
KIND: Final = "netboot"

#: The arm's command inside a release: fleet-mcp's built netboot CLI.
CLI: Final = pathlib.PurePosixPath("fleet-mcp/dist/src/netboot-cli.js")

#: The installer definition inside a release.
DEFINITION: Final = pathlib.PurePosixPath("fleet-mcp/netboot")

#: Where the arm fetches the ISO, inside the served root so a build's
#: rename stays on one file system.
CACHE: Final = "cache"

#: The variables the duty's commands read the tailnet writer client from,
#: id then secret, and the env file lines that carry them.
WRITER_NAMES: Final[tuple[str, str]] = ("TAILNET_WRITER_CLIENT_ID", "TAILNET_WRITER_CLIENT_SECRET")

#: What every tailnet credential begins with: a client secret, an access
#: token and a minted auth key alike.
KEY_MARK: Final = "tskey-"

#: The line the arm prints once it has armed a machine.
ARMED_PREFIX: Final = "NETBOOT ARMED "

#: How the arm's CLI prefixes the error it exits with.
CLI_ERROR_PREFIX: Final = "fleet-netboot: "

#: The arm's refusals that leave nothing armed and nothing minted, as
#: ``fleet-mcp`` names them (``boot-mac.ts`` and ``netboot-arm.ts``).
ARM_REFUSALS: Final[tuple[str, ...]] = (
    "NETBOOT_ARM_MACHINE_UNKNOWN",
    "NETBOOT_ARM_MACHINE_NOT_ARMABLE",
    "NETBOOT_ARM_LIVE",
)

#: The arm's deadline. The first arm on a host fetches the installer ISO,
#: a few GB, and builds the served tree; later arms reuse both and finish
#: in seconds. Inside the runner's one-hour lease and the tick's wall.
ARM_TIMEOUT_SECONDS: Final[int] = 1800

#: How much of a failed arm's output its detail carries.
DETAIL_TAIL_CHARS: Final[int] = 600

#: Detail prefix when the env file holds no usable key.
NO_KEY_CODE: Final = "NETBOOT_ARM_NO_KEY"

#: Detail prefix when the job names no machine to arm.
NO_MACHINE_CODE: Final = "NETBOOT_ARM_NO_MACHINE"

#: Detail prefix when a runner that is not the netboot duty's claims an arm.
NO_RELEASE_CODE: Final = "NETBOOT_ARM_NO_RELEASE"

#: Detail prefix when the arm exited 0 without exactly one armed line.
ANSWER_MALFORMED_CODE: Final = "NETBOOT_ARM_ANSWER_MALFORMED"

#: Detail prefix when the arm's output carries a tailnet key.
KEY_IN_OUTPUT_CODE: Final = "NETBOOT_ARM_OUTPUT_CARRIES_KEY"


class NetbootPlace(TypedDict):
    """Where this machine's arms run from and write to.

    Attributes:
        release: The installed sealed netboot release's checkout.
        nodejs: The node binary that runs its CLI.
        served_root: The tree ``netboot-serve`` answers from.
        env_file: The workspace env file holding the tailnet writer client.
        registry: The registry directory this tick loaded, whose
            ``machines`` roster names each machine's boot MAC (MCPs board
            task e447de72).
    """

    release: pathlib.Path
    nodejs: str
    served_root: pathlib.Path
    env_file: pathlib.Path
    registry: pathlib.Path


class TailnetWriter(TypedDict):
    """The tailnet writer OAuth client, read for one command's environment.

    Attributes:
        client_id: The client's id.
        client_secret: The client's secret. Never logged, never in an argv
            or a detail.
    """

    client_id: str
    client_secret: str


class ArmOutcome(TypedDict):
    """How a finished arm closes its job.

    Attributes:
        status: ``passed``, ``failed`` or ``refused``.
        exit_code: The arm's exit status, or None for ``refused``.
        detail: The line the submitter reads.
    """

    status: ClosingStatus
    exit_code: int | None
    detail: str


class Ready(TypedDict):
    """What every ``fleet-netboot`` command the duty runs needs before it runs.

    Attributes:
        release: The installed release, its seal verified.
        writer: The tailnet writer client.
    """

    release: SealedRelease
    writer: TailnetWriter


def ready(place: NetbootPlace) -> Ready | str:
    """Verify the installed release's seal, then read the writer client.

    Args:
        place: Where the netboot duty's commands run from.

    Returns:
        The verified release and the writer client, or the first refusal:
        the verifier's when the seal does not hold,
        :func:`read_tailnet_writer`'s when the env file gives no client.
        Nothing in the release has run.
    """
    release = verify_release(place["release"], KIND)
    if isinstance(release, str):
        return release
    writer = read_tailnet_writer(place["env_file"])
    if isinstance(writer, str):
        return writer
    return Ready(release=release, writer=writer)


def read_tailnet_writer(env_file: pathlib.Path) -> TailnetWriter | str:
    """Read the tailnet writer client from the workspace env file.

    Args:
        env_file: The stack's ``.env``, read by
            :func:`platform_core.env_file.parse_env_file`, the one reader of
            it, in the grammar its owner holds.

    Returns:
        The client, or a ``CODE: message`` refusal naming the file and
        never a line of it, when it is absent, holds a line that is not
        ``KEY=VALUE``, or sets no id or no secret or an empty one, naming
        each line it lacks. The retired ``TAILSCALE_API_KEY`` is never read.
    """
    if not _test_hooks.file_exists(env_file):
        return f"{NO_KEY_CODE}: {env_file} does not exist, so no key can be minted; nothing ran"
    try:
        values = parse_env_file(_test_hooks.read_text(env_file), source=str(env_file))
    except AppError as unreadable:
        return f"{NO_KEY_CODE}: {unreadable.message}; nothing ran"
    client_id, client_secret = (values.get(name, "") for name in WRITER_NAMES)
    missing = [
        name
        for name, value in zip(WRITER_NAMES, (client_id, client_secret), strict=True)
        if value == ""
    ]
    if missing:
        return (
            f"{NO_KEY_CODE}: {env_file} sets no {' or '.join(missing)}, or sets it empty, and the "
            "netboot duty's commands run with the tailnet writer client; nothing ran"
        )
    return TailnetWriter(client_id=client_id, client_secret=client_secret)


def cli_argv(place: NetbootPlace, release: SealedRelease, *arguments: str) -> tuple[str, ...]:
    """The release's own ``fleet-netboot``, run with ``arguments``.

    Args:
        place: Where the netboot duty's commands run from.
        release: The verified release.
        *arguments: The command and its flags.

    Returns:
        Node importing the release's CLI and running it with ``arguments``;
        the CLI reads them from ``process.argv[1]`` on under ``-e``.
    """
    importer = (
        f"const m = await import({(release['root'] / CLI).as_uri()!r}); "
        "await m.runFromProcess(process);"
    )
    return (place["nodejs"], "--input-type=module", "-e", importer, *arguments)


def arm_argv(place: NetbootPlace, release: SealedRelease, machine: str) -> tuple[str, ...]:
    """The release's own ``fleet-netboot arm`` of one roster machine.

    Args:
        place: Where arms run from and write to.
        release: The verified release.
        machine: The roster name the job asks to arm.

    Returns:
        Node importing the release's CLI and running ``arm`` with its
        definition, the served root, a cache inside it, the seal's
        revision as the commit, the machine and the loaded registry.
    """
    root = release["root"]
    return cli_argv(
        place,
        release,
        "arm",
        "--definition",
        str(root / DEFINITION),
        "--root",
        str(place["served_root"]),
        "--cache",
        str(place["served_root"] / CACHE),
        "--commit",
        release["revision"],
        "--node",
        machine,
        "--registry",
        str(place["registry"]),
    )


def run_cli(
    argv: tuple[str, ...], writer: TailnetWriter, *, timeout_seconds: int
) -> _test_hooks.CommandResult:
    """Run one ``fleet-netboot`` command with the writer client in its environment alone.

    Args:
        argv: What :func:`cli_argv` composed.
        writer: The tailnet writer client.
        timeout_seconds: The command's deadline.

    Returns:
        Its exit status and captured streams, or the timed-out result after
        ``timeout_seconds``.
    """
    client_id_name, client_secret_name = WRITER_NAMES
    return _test_hooks.run(
        argv,
        timeout_seconds=timeout_seconds,
        set_env=(
            (client_id_name, writer["client_id"]),
            (client_secret_name, writer["client_secret"]),
        ),
    )


def cli_error(stderr: str) -> str:
    """The error a ``fleet-netboot`` command exited with, without its prefix.

    Args:
        stderr: The command's standard error.

    Returns:
        The text after the last ``fleet-netboot: `` line's prefix, or ``""``
        when it printed none.
    """
    errors = [
        line.removeprefix(CLI_ERROR_PREFIX)
        for line in stderr.splitlines()
        if line.startswith(CLI_ERROR_PREFIX)
    ]
    return errors[-1] if errors else ""


def carries_key(result: _test_hooks.CommandResult) -> bool:
    """Whether a command's output carries a tailnet key, which it never prints.

    Args:
        result: The command's exit status and streams.

    Returns:
        True when either stream holds :data:`KEY_MARK`; then none of the
        output may be copied anywhere.
    """
    return KEY_MARK in result["stdout"] or KEY_MARK in result["stderr"]


def output_tail(result: _test_hooks.CommandResult) -> str:
    """The end of a failed command's output, for the line that reports it.

    Args:
        result: The command's exit status and streams.

    Returns:
        The last :data:`DETAIL_TAIL_CHARS` characters of its stdout then
        stderr, stripped.
    """
    return (result["stdout"] + result["stderr"]).strip()[-DETAIL_TAIL_CHARS:]


def arm_outcome(
    result: _test_hooks.CommandResult, release: SealedRelease, machine: str
) -> ArmOutcome:
    """Read a finished arm into the job's close.

    Args:
        result: The arm's exit status and streams.
        release: The release it ran from.
        machine: The machine it armed.

    Returns:
        ``passed`` with the armed line; ``refused`` with the arm's own
        refusal, or with this runner's when it exited 0 without exactly one
        armed line or its output carries a key; ``failed`` with the output's
        tail otherwise. Every detail names the machine, the release and the
        revision.
    """
    opening = (
        f"fleet-netboot arm of {machine} from sealed release {release['release']} "
        f"(revision {release['revision']})"
    )
    returncode = result["returncode"]
    if carries_key(result):
        return ArmOutcome(
            status=ClosingStatus.REFUSED if returncode == 0 else ClosingStatus.FAILED,
            exit_code=None if returncode == 0 else returncode,
            detail=(
                f"{KEY_IN_OUTPUT_CODE}: {opening} exited {returncode} and its output carries "
                f"a tailnet key ({KEY_MARK}...), so none of it is copied here; the served "
                "root's arm store on the netboot host is the record"
            ),
        )
    if returncode == 0:
        armed = [line for line in result["stdout"].splitlines() if line.startswith(ARMED_PREFIX)]
        if len(armed) == 1:
            return ArmOutcome(
                status=ClosingStatus.PASSED, exit_code=0, detail=f"{opening}: {armed[0]}"
            )
        return ArmOutcome(
            status=ClosingStatus.REFUSED,
            exit_code=None,
            detail=(
                f"{ANSWER_MALFORMED_CODE}: {opening} exited 0 and printed {len(armed)} "
                f"'{ARMED_PREFIX.strip()}' line(s) where it prints exactly one, so this runner "
                "does not report it armed"
            ),
        )
    error = cli_error(result["stderr"])
    if any(error.startswith(f"{code}:") for code in ARM_REFUSALS):
        return ArmOutcome(status=ClosingStatus.REFUSED, exit_code=None, detail=error)
    return ArmOutcome(
        status=ClosingStatus.FAILED,
        exit_code=returncode,
        detail=f"{opening} exited {returncode}: {output_tail(result)}",
    )


__all__ = [
    "ANSWER_MALFORMED_CODE",
    "ARMED_PREFIX",
    "ARM_REFUSALS",
    "ARM_TIMEOUT_SECONDS",
    "CACHE",
    "CLI",
    "CLI_ERROR_PREFIX",
    "DEFINITION",
    "DETAIL_TAIL_CHARS",
    "KEY_IN_OUTPUT_CODE",
    "KEY_MARK",
    "KIND",
    "NO_KEY_CODE",
    "NO_MACHINE_CODE",
    "NO_RELEASE_CODE",
    "WRITER_NAMES",
    "ArmOutcome",
    "NetbootPlace",
    "Ready",
    "TailnetWriter",
    "arm_argv",
    "arm_outcome",
    "carries_key",
    "cli_argv",
    "cli_error",
    "output_tail",
    "read_tailnet_writer",
    "ready",
    "run_cli",
]
