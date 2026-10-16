"""Asking a run's environment whether it can start the module the run names.

THE HOLE, exercised twice in two days (board task 2c8161a3). The registered
``mi`` image, v23 (sha256 ``55651342e15d``), carries the commit it was built
from at ``/opt/env/GIT_COMMIT``: ``fab1d991``. That commit has no
``model_trainer.cli.cartridge_benchmark`` and no
``cartridge_composition_sweep``, so a run document naming either under that
image asserts a command its image cannot run, and only a session declining to
commit such a document stopped one reaching the queue. Onboarding is enforced
(``PROJECT_UNIMAGED``) and identity is enforced (the ``.sif`` is pinned by
sha256), but whether the image is new enough for the command was convention.
Job 55898551 is the same failure arriving: it named a plan its image predated
by twenty-three minutes and died in 5m32s.

THE QUESTION IS ASKED OF THE ENVIRONMENT ITSELF, at preflight. For every
``python -m <module>`` the command runs, the environment's own interpreter
resolves the module the way ``python -m`` will, from the directory the
command enters first (its last ``cd`` before the invocation) or the job's
working directory otherwise, and says PRESENT or ABSENT. Nothing is imported:
only the top-level name goes through ``importlib.util.find_spec``, which does
not execute it, and the rest is a file test under the package's search
locations. So a ``model_trainer`` whose ``__init__`` loads torch costs this
probe nothing, and ``scripts.optimize`` run from a cluster checkout is found
where it lives rather than refused for not being installed.

HARD STALENESS IS REFUSED, SOFT DRIFT IS NOT. A module the environment lacks
is a run that dies on its first line, and ``ENTRYPOINT_NOT_INSTALLED`` says so
before a job id exists, naming the commit the image was built from. A module
the environment HAS, but whose source has changed since the image's commit,
is admitted: that is what reproducing an old command under the old image IS,
and it must stay legal. The image's commit is reported either way, so a
reader can tell which code the run will execute.

WHAT IS NOT EXAMINED: a command that starts no ``-m`` module (``python
train.py``, a shell pipeline) names no entrypoint this check can resolve, and
the reading says it examined none rather than passing silently.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.image import ImageReference
from hpc3.core import image_exec, remote

_MARK = "ENTRYPOINT|"
_COMMIT_MARK = "ENTRYPOINT-COMMIT|"
_PRESENT = "PRESENT"
_ABSENT = "ABSENT"

_INVOCATION = re.compile(
    r"(?:^|[\s\"';&|(])(?:cd\s+(?P<directory>[^\s\"';&|()]+)"
    r"|(?:[^\s\"';&|()]*/)?python(?:3(?:\.\d+)?)?\s+-m\s+"
    r"(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*))"
)
"""One ``cd <directory>`` or one ``[path/]python[3[.x]] -m <module>``.

Read left to right in one pass, so each invocation is paired with the last
``cd`` before it. The directory is a single unquoted word, which is every
``cd`` the committed run documents carry (``/pub/wagnera3/api/...``).
"""

_PROBE_SOURCE = (
    "import importlib.util as u,os,sys;f=os.path.isfile;j=os.path.join;"
    "g=lambda s,r:s is not None and ("
    "(r==[] and (s.submodule_search_locations is None or "
    "any(f(j(d,'__main__.py')) for d in s.submodule_search_locations))) or "
    "(r!=[] and any(f(j(d,*r)+'.py') or f(j(d,*r,'__main__.py')) "
    "for d in (s.submodule_search_locations or []))));"
    f"[print('{_MARK}'+('{_PRESENT}' if g(u.find_spec(m.split('.')[0]),m.split('.')[1:]) "
    f"else '{_ABSENT}')+'|'+m) for m in sys.argv[1:]]"
)
"""Print ``ENTRYPOINT|PRESENT|<module>`` or ``ENTRYPOINT|ABSENT|<module>`` per argument.

The rule ``python -m`` applies: a top-level module, or a package carrying
``__main__.py``; below the top level, ``<rest>.py`` or
``<rest>/__main__.py`` under one of the top-level package's search
locations. Only the top-level name is resolved with ``find_spec``, which does
not import it. One ``-c`` expression with no newline and no double quote, so
it survives ``ssh``, ``sh -c`` and the double quotes it is wrapped in, as
:mod:`hpc3.core.env_probe`'s probe does.
"""


class EntrypointReading(TypedDict):
    """What the environment said about the modules a run starts.

    Attributes:
        image_commit: The commit stamped into the environment at
            ``<env_path>/GIT_COMMIT``, or empty when it carries no stamp.
        modules: Every module the command starts with ``python -m``, in the
            order it starts them, each one resolved PRESENT. Empty when the
            command starts none, which is a reading of zero modules, not a
            pass over them.
    """

    image_commit: str
    modules: list[str]


def _by_directory(found: Sequence[tuple[str, str]]) -> dict[str, list[str]]:
    """Group modules by the directory they resolve in, keeping first-seen order.

    The probe starts one interpreter per directory and answers group by
    group, so this grouping is both how the probe is built and the order its
    verdicts arrive in.

    Args:
        found: ``(directory, module)`` pairs.

    Returns:
        Modules keyed by directory.
    """
    groups: dict[str, list[str]] = {}
    for directory, module in found:
        groups.setdefault(directory, []).append(module)
    return groups


def invocations(command: str, *, working_directory: str) -> list[tuple[str, str]]:
    """Find every module a command starts, with the directory it starts in.

    Args:
        command: The run's command line.
        working_directory: The job's working directory, used for an
            invocation no ``cd`` precedes.

    Returns:
        ``(directory, module)`` pairs in command order, without repeats.
    """
    directory = working_directory
    found: list[tuple[str, str]] = []
    for match in _INVOCATION.finditer(command):
        entered = match.group("directory")
        if entered is not None:
            directory = entered
            continue
        pair = (directory, match.group("module"))
        if pair not in found:
            found.append(pair)
    return found


def probe_command(env_path: str, found: Sequence[tuple[str, str]]) -> str:
    """Build the command that resolves each module in its own directory.

    Args:
        env_path: The environment whose interpreter the job runs.
        found: ``(directory, module)`` pairs from :func:`invocations`, at
            least one.

    Returns:
        One shell line printing the environment's commit stamp, then one
        marked line per module in order. Modules sharing a directory share
        one interpreter start. The steps are chained with ``&&``, so a
        ``cd`` that fails ends the line with that failure's status, which
        :func:`~hpc3.core.remote.run_remote` raises, rather than an answer
        that could be read as "absent" (measured on HPC3 2026-10-16: ``cd:
        can't cd to /pub/wagnera3/mi/jobs``, exit 2).
    """
    python = shlex.quote(f"{env_path}/bin/python")
    steps = [
        f"printf '{_COMMIT_MARK}%s\\n' \"$(cat {shlex.quote(env_path + '/GIT_COMMIT')} "
        '2>/dev/null)"'
    ]
    steps.extend(
        f'(cd {directory} && {python} -c "{_PROBE_SOURCE}" '
        f"{' '.join(shlex.quote(module) for module in modules)})"
        for directory, modules in _by_directory(found).items()
    )
    return " && ".join(steps)


def parse_probe(output: str, found: Sequence[tuple[str, str]]) -> tuple[str, dict[str, bool]]:
    """Read the commit stamp and each module's verdict.

    Args:
        output: The probe's standard output.
        found: The pairs the probe was built from.

    Returns:
        The commit stamp (empty when unstamped) and, per module, whether it
        resolved.

    Raises:
        AppError: With ``ENTRYPOINT_PROBE_UNREADABLE`` when the output lacks
            the stamp line, or carries a different number of verdicts than
            modules asked about, or a verdict for another module or with
            another word. A short answer means the interpreter printed
            something other than the verdicts, and reading it as "absent"
            would refuse a run for a probe fault, while reading it as
            "present" would admit one unchecked.
    """
    lines = [line.strip() for line in output.splitlines()]
    stamps = [line[len(_COMMIT_MARK) :] for line in lines if line.startswith(_COMMIT_MARK)]
    verdicts = [line[len(_MARK) :].split("|") for line in lines if line.startswith(_MARK)]
    expected = [module for _, module in found]
    answer_order = [module for modules in _by_directory(found).values() for module in modules]
    if len(stamps) != 1 or [verdict[-1] for verdict in verdicts] != answer_order:
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE,
            f"The environment's answer about {expected} could not be read: expected "
            f"one {_COMMIT_MARK!r} line and one {_MARK!r} line per module, got "
            f"{output.strip()!r}.",
        )
    present: dict[str, bool] = {}
    for verdict in verdicts:
        if len(verdict) != 2 or verdict[0] not in (_PRESENT, _ABSENT):
            raise AppError(
                Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE,
                f"The environment answered {_MARK + '|'.join(verdict)!r}, which is "
                f"neither {_PRESENT} nor {_ABSENT} for one module.",
            )
        present[verdict[1]] = present.get(verdict[1], True) and verdict[0] == _PRESENT
    return stamps[0], present


def check_entrypoints(
    host: str,
    commands: Sequence[str],
    *,
    env_path: str,
    image: ImageReference | None,
    working_directory: str,
) -> EntrypointReading:
    """Refuse a run whose environment cannot start a module its command starts.

    Args:
        host: SSH destination.
        commands: Every command the submission runs: one for a job, one per
            member for an array.
        env_path: The environment the commands run in, inside ``image`` when
            there is one.
        image: The image the environment lives in, or None for a cluster
            directory.
        working_directory: The job's working directory, where ``python -m``
            looks first when the command enters no other.

    Returns:
        The environment's commit stamp and the modules resolved. No round
        trip is made when the commands start no module.

    Raises:
        AppError: With ``ENTRYPOINT_NOT_INSTALLED`` naming every absent
            module and the commit the environment was built from,
            ``ENTRYPOINT_PROBE_UNREADABLE`` when its answer cannot be read,
            or ``REMOTE_COMMAND_FAILED`` when the probe exits non-zero, a
            directory the command enters being absent among the causes.
    """
    found: list[tuple[str, str]] = []
    for command in commands:
        for pair in invocations(command, working_directory=working_directory):
            if pair not in found:
                found.append(pair)
    if found == []:
        return EntrypointReading(image_commit="", modules=[])
    probe = probe_command(env_path, found)
    if image is not None:
        probe = image_exec.run_inside_image(image, probe)
    commit, present = parse_probe(remote.run_remote(host, probe), found)
    absent = [module for module, ok in present.items() if not ok]
    if absent != []:
        built = "carries no commit stamp" if commit == "" else f"was built from {commit}"
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED,
            f"{env_path} {image_exec.describe_location(image, host)} cannot start "
            f"{', '.join(absent)}: the environment {built}, which does not carry "
            "that module, so the job would die on its first line. Pin an image "
            "built from a commit that has it, or rebuild this one.",
        )
    return EntrypointReading(image_commit=commit, modules=list(present))


__all__ = [
    "EntrypointReading",
    "check_entrypoints",
    "invocations",
    "parse_probe",
    "probe_command",
]
