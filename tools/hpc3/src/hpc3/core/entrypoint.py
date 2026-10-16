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
working directory otherwise, and says PRESENT or ABSENT, and for a present
one WHERE it resolved: inside the environment's own prefix, which is the
code the image was built with, or outside it, which is the command's own
checkout. Nothing is imported: only the top-level name goes through
``importlib.util.find_spec``, which does not execute it, and the rest is a
file test under the package's search locations. So a ``model_trainer``
whose ``__init__`` loads torch costs this probe nothing, and
``scripts.optimize`` run from a cluster checkout is found where it lives
rather than refused for not being installed.

AN IMAGE MUST SAY WHICH COMMIT BUILT IT. The same round trip reads
``<env_path>/GIT_COMMIT``, which :mod:`hpc3.contracts.image_spec` requires
every built image to carry. An image whose stamp cannot be read, or reads
empty, is refused ``ENTRYPOINT_IMAGE_UNSTAMPED`` with the reason the read
gave: without it nothing can say which code the run executes, so whether its
numbers come from older code than the submitter's is unknown. A directory
environment is not read for one; it is identified by no build.

HARD STALENESS IS REFUSED HERE; SOFT DRIFT IS MEASURED NEXT. A module the
environment lacks is a run that dies on its first line, and
``ENTRYPOINT_NOT_INSTALLED`` says so before a job id exists, naming the
commit the image was built from. A module the environment HAS is admitted,
since reproducing an old command under its old image must stay legal, and
:mod:`hpc3.core.image_drift` then compares its source between that commit
and the submitter's HEAD, so the run is warned about and recorded with
whatever moved.

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
_IN_ENVIRONMENT = "ENV"
_OUTSIDE = "OUTSIDE"
_NOWHERE = "-"

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
    "r=os.path.realpath;p=r(sys.prefix);"
    "g=lambda s,t:s is not None and ("
    "(t==[] and (s.submodule_search_locations is None or "
    "any(f(j(d,'__main__.py')) for d in s.submodule_search_locations))) or "
    "(t!=[] and any(f(j(d,*t)+'.py') or f(j(d,*t,'__main__.py')) "
    "for d in (s.submodule_search_locations or []))));"
    "l=lambda s:r(list(s.submodule_search_locations)[0] "
    "if s.submodule_search_locations else os.path.dirname(s.origin));"
    f"w=lambda x:'{_IN_ENVIRONMENT}' if x==p or x.startswith(p+os.sep) else '{_OUTSIDE}';"
    f"v=lambda m,s:'{_PRESENT}|'+w(l(s))+'|'+m+'|'+l(s) if g(s,m.split('.')[1:]) "
    f"else '{_ABSENT}|{_NOWHERE}|'+m+'|';"
    f"[print('{_MARK}'+v(m,u.find_spec(m.split('.')[0]))) for m in sys.argv[1:]]"
)
"""Print one ``ENTRYPOINT|<verdict>|<where>|<module>|<location>`` line per argument.

``<verdict>`` is the rule ``python -m`` applies: a top-level module, or a
package carrying ``__main__.py``; below the top level, ``<rest>.py`` or
``<rest>/__main__.py`` under one of the top-level package's search
locations. For a PRESENT module, ``<location>`` is the real path the
top-level name resolved to, and ``<where>`` says whether it lies under the
interpreter's own ``sys.prefix``: an image's installed code, or not. An
ABSENT module has neither. Only the top-level name is resolved with
``find_spec``, which does not import it. One ``-c`` expression with no
newline and no double quote, so it survives ``ssh``, ``sh -c`` and the
double quotes it is wrapped in, as :mod:`hpc3.core.env_probe`'s probe does.
"""


class ModuleReading(TypedDict):
    """Where one module a run starts resolved in its environment.

    Attributes:
        directory: The directory the command starts it from.
        module: The dotted module name.
        location: The real path its top-level name resolved to.
        in_environment: Whether that path lies under the environment's own
            prefix, which makes it the code the environment was built with
            rather than the command's own checkout.
    """

    directory: str
    module: str
    location: str
    in_environment: bool


class EntrypointReading(TypedDict):
    """What the environment said about the modules a run starts.

    Attributes:
        image_commit: The commit stamped into the image's environment at
            ``<env_path>/GIT_COMMIT``, or None when the run has no image, so
            no build commit identifies its environment.
        modules: Every ``(directory, module)`` the commands start with
            ``python -m``, in the order they start them, each resolved
            PRESENT. Empty when the commands start none, which is a reading
            of zero modules, not a pass over them.
    """

    image_commit: str | None
    modules: list[ModuleReading]


class ProbeAnswer(TypedDict):
    """The probe's output, read but not yet judged.

    Attributes:
        stamp_status: Exit status of reading the stamp, or None when it was
            not read.
        stamp: What reading the stamp printed: the commit, or the reason it
            could not be read.
        modules: Per ``(directory, module)`` in probe order, its reading,
            or None when it is absent.
    """

    stamp_status: int | None
    stamp: str
    modules: list[tuple[tuple[str, str], ModuleReading | None]]


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


def probe_command(env_path: str, found: Sequence[tuple[str, str]], *, read_stamp: bool) -> str:
    """Build the command that reads the stamp and resolves each module.

    Args:
        env_path: The environment whose interpreter the job runs.
        found: ``(directory, module)`` pairs from :func:`invocations`.
        read_stamp: Whether to read ``<env_path>/GIT_COMMIT``: true for an
            image, whose build wrote it. At least one of this and ``found``
            asks something.

    Returns:
        One shell line printing ``ENTRYPOINT-COMMIT|<status>|<text>`` when
        the stamp is read, with ``cat``'s own complaint as the text when it
        fails, then one marked line per module in order. Modules sharing a
        directory share one interpreter start. The steps are chained with
        ``&&``, so a ``cd`` that fails ends the line with that failure's
        status, which :func:`~hpc3.core.remote.run_remote` raises, rather
        than an answer that could be read as "absent" (measured on HPC3
        2026-10-16: ``cd: can't cd to /pub/wagnera3/mi/jobs``, exit 2).
    """
    python = shlex.quote(f"{env_path}/bin/python")
    steps: list[str] = []
    if read_stamp:
        steps.append(
            f"{{ s=$(cat {shlex.quote(env_path + '/GIT_COMMIT')} 2>&1); "
            f'printf \'{_COMMIT_MARK}%s|%s\\n\' "$?" "$s"; }}'
        )
    steps.extend(
        f'(cd {directory} && {python} -c "{_PROBE_SOURCE}" '
        f"{' '.join(shlex.quote(module) for module in modules)})"
        for directory, modules in _by_directory(found).items()
    )
    return " && ".join(steps)


def _unreadable(found: Sequence[tuple[str, str]], output: str, why: str) -> AppError[Hpc3ErrorCode]:
    """Build the refusal for an answer this module cannot read.

    Args:
        found: The pairs the probe asked about.
        output: Everything the probe printed.
        why: What was wrong with it.

    Returns:
        An ``ENTRYPOINT_PROBE_UNREADABLE`` error quoting the output.
    """
    return AppError(
        Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE,
        f"The environment's answer about {[module for _, module in found]} could not be "
        f"read: {why}; it printed {output.strip()!r}.",
    )


def _read_verdict(
    pair: tuple[str, str], fields: list[str], *, found: Sequence[tuple[str, str]], output: str
) -> ModuleReading | None:
    """Read one module's verdict line, already split after its mark.

    Args:
        pair: The ``(directory, module)`` the line must answer.
        fields: ``[verdict, where, module, location]``.
        found: Every pair the probe asked about, for the refusal.
        output: Everything the probe printed, for the refusal.

    Returns:
        The reading for a PRESENT module, or None for an ABSENT one.

    Raises:
        AppError: With ``ENTRYPOINT_PROBE_UNREADABLE`` when the line answers
            another module, or its fields do not agree with its verdict: a
            present module names where it resolved, an absent one nothing.
    """
    if len(fields) != 4 or fields[2] != pair[1]:
        raise _unreadable(
            found, output, f"a verdict line answers {fields!r} where {pair[1]!r} was asked"
        )
    verdict, where, _, location = fields
    if verdict == _ABSENT and where == _NOWHERE and location == "":
        return None
    if verdict == _PRESENT and where in (_IN_ENVIRONMENT, _OUTSIDE) and location != "":
        return ModuleReading(
            directory=pair[0],
            module=pair[1],
            location=location,
            in_environment=where == _IN_ENVIRONMENT,
        )
    raise _unreadable(
        found, output, f"{'|'.join(fields)!r} is not a {_PRESENT} or {_ABSENT} verdict"
    )


def parse_probe(output: str, found: Sequence[tuple[str, str]], *, read_stamp: bool) -> ProbeAnswer:
    """Read the stamp line and each module's verdict, judging neither.

    Args:
        output: The probe's standard output.
        found: The pairs the probe was built from.
        read_stamp: Whether the probe read the stamp.

    Returns:
        The stamp's status and text, and each pair's reading in the order
        the probe answered them.

    Raises:
        AppError: With ``ENTRYPOINT_PROBE_UNREADABLE`` when the stamp line is
            missing, repeated or unexpected, when its status is not a number,
            or when the verdicts are not one well-formed line per pair in
            the order asked. A short answer means the interpreter printed
            something other than the verdicts, and reading it as "absent"
            would refuse a run for a probe fault, while reading it as
            "present" would admit one unchecked.
    """
    lines = [line.strip() for line in output.splitlines()]
    stamps = [
        line[len(_COMMIT_MARK) :].split("|", 1) for line in lines if line.startswith(_COMMIT_MARK)
    ]
    verdicts = [line[len(_MARK) :].split("|", 3) for line in lines if line.startswith(_MARK)]
    if len(stamps) != (1 if read_stamp else 0):
        raise _unreadable(found, output, f"expected {int(read_stamp)} {_COMMIT_MARK!r} line(s)")
    if read_stamp and (len(stamps[0]) != 2 or not stamps[0][0].isdigit()):
        raise _unreadable(found, output, "the stamp line carries no exit status")
    answer_order = [
        (directory, module)
        for directory, modules in _by_directory(found).items()
        for module in modules
    ]
    if len(verdicts) != len(answer_order):
        raise _unreadable(found, output, f"expected {len(answer_order)} {_MARK!r} line(s)")
    modules = [
        (pair, _read_verdict(pair, fields, found=found, output=output))
        for pair, fields in zip(answer_order, verdicts, strict=True)
    ]
    return ProbeAnswer(
        stamp_status=int(stamps[0][0]) if read_stamp else None,
        stamp=stamps[0][1].strip() if read_stamp else "",
        modules=modules,
    )


def check_entrypoints(
    host: str,
    commands: Sequence[str],
    *,
    env_path: str,
    image: ImageReference | None,
    working_directory: str,
) -> EntrypointReading:
    """Refuse a run whose image cannot say its commit or cannot start its module.

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
        The image's commit and where each module resolved. No round trip is
        made when there is no image and the commands start no module.

    Raises:
        AppError: With ``ENTRYPOINT_IMAGE_UNSTAMPED`` when the image's stamp
            cannot be read or reads empty, ``ENTRYPOINT_NOT_INSTALLED``
            naming every absent module and the commit the environment was
            built from, ``ENTRYPOINT_PROBE_UNREADABLE`` when its answer
            cannot be read, or ``REMOTE_COMMAND_FAILED`` when the probe exits
            non-zero, a directory the command enters being absent among the
            causes.
    """
    found: list[tuple[str, str]] = []
    for command in commands:
        for pair in invocations(command, working_directory=working_directory):
            if pair not in found:
                found.append(pair)
    read_stamp = image is not None
    if not read_stamp and found == []:
        return EntrypointReading(image_commit=None, modules=[])
    probe = probe_command(env_path, found, read_stamp=read_stamp)
    if image is not None:
        probe = image_exec.run_inside_image(image, probe)
    answer = parse_probe(remote.run_remote(host, probe), found, read_stamp=read_stamp)
    where = f"{env_path} {image_exec.describe_location(image, host)}"
    if read_stamp and (answer["stamp_status"] != 0 or answer["stamp"] == ""):
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_IMAGE_UNSTAMPED,
            f"{where} cannot say which commit built it: reading GIT_COMMIT exited "
            f"{answer['stamp_status']} with {answer['stamp']!r}. Without it nothing can "
            "say which code this run executes. Rebuild the image with hpc3-image-build, "
            "which writes the stamp.",
        )
    absent = [pair[1] for pair, reading in answer["modules"] if reading is None]
    if absent != []:
        built = (
            "is a directory environment" if image is None else f"was built from {answer['stamp']}"
        )
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED,
            f"{where} cannot start {', '.join(absent)}: the environment {built}, which "
            "does not carry that module, so the job would die on its first line. Pin an "
            "image built from a commit that has it, or rebuild this one.",
        )
    return EntrypointReading(
        image_commit=answer["stamp"] if read_stamp else None,
        modules=[reading for _, reading in answer["modules"] if reading is not None],
    )


__all__ = [
    "EntrypointReading",
    "ModuleReading",
    "ProbeAnswer",
    "check_entrypoints",
    "invocations",
    "parse_probe",
    "probe_command",
]
