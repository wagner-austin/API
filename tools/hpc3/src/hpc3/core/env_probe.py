"""Asking an environment what is actually installed in it, and by which interpreter.

:func:`~hpc3.core.preflight.check_env_path` proves a directory exists. That
catches a typo'd path and nothing else, and the failure it cannot catch is the
expensive one: two environments built for the same project, one pinned to the
stack a published result used and one on current releases, differing by a few
characters in the path. Both directories exist. Both pass a ``test -d``. The
run completes, the number is plausible, and it is not comparable to the
results it was meant to extend, because the library underneath it changed a
major version.

So a project may declare exactly what its environment must contain, and
preflight asks the environment itself rather than trusting the path. One extra
SSH round trip, against a job that may run for ten hours.

Versions come from ``importlib.metadata``, which reports installed
distributions rather than importable modules, so a package that is present but
broken is still reported and a package that is absent is simply missing from
the listing. Names are compared after PEP 503 normalisation, because
``importlib.metadata`` reports whatever the distribution called itself and
``Torch``, ``torch`` and ``TORCH`` are one package.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.image import ImageReference
from hpc3.contracts.pins import normalise_name
from hpc3.core import image_exec, remote
from hpc3.core.interpreter import (
    IDENTITY_IMPORTS,
    IDENTITY_LINES,
    InterpreterIdentity,
    check_interpreter_home,
    split_identity,
)

_PROBE_SOURCE = (
    f"{IDENTITY_IMPORTS},importlib.metadata as m;"
    "t=lambda d:next((l.split(':',1)[1].strip() "
    "for l in (d.read_text('WHEEL') or '').splitlines() if l.startswith('Tag:')),'');"
    f"print(chr(10).join({IDENTITY_LINES}+["
    "(d.metadata['Name'] or '?')+'=='+(d.version or '?')+'=='+t(d) "
    "for d in m.distributions()]))"
)
"""Print the interpreter's identity, then every distribution as ``Name==version==wheel_tag``.

THE INTERPRETER IS ASKED ABOUT ITSELF FIRST. Until 2026-10-05 this listed
distributions only, so an environment whose interpreter belonged to another
project -- ``envs/tankpit``, a venv over ``envs/cleargbm`` -- passed every
check. The identity lines of :mod:`hpc3.core.interpreter` now lead the output,
in the same round trip, and a probe whose answer does not start with them is
unreadable rather than "nothing to say about the interpreter".

THE WHEEL TAG IS READ, NOT ASSUMED. Capture used to synthesise every
first-party wheel filename as ``py3-none-any``, which is right for a pure
Python distribution and wrong for a compiled one -- and ``cleargbm_rs`` is
compiled, so its real wheel is ``cp311-cp311-linux_x86_64``. The spec named a
file that does not exist and the build would have failed on it.

The tag comes from the distribution's own ``WHEEL`` metadata. It is EMPTY
when there is none, which is the ordinary case for a conda-installed package
and for anything not installed from a wheel; that is not an error here,
because only first-party distributions become wheels. Capture refuses an
empty tag for those, where it matters, rather than this probe refusing it for
every package where it does not.

Built as a single ``-c`` expression with no newline in it: the probe travels
as one argument through ``ssh`` to a remote shell, and an embedded newline
would be split by that shell rather than by Python. ``chr(10)`` produces the
separator without ever putting one in the command text.
"""


def probe_command(env_path: str) -> str:
    """Build the command that lists an environment's installed distributions.

    Args:
        env_path: Absolute path to the environment on the cluster.

    Returns:
        A shell command running that environment's own interpreter. The
        environment's ``python`` is invoked by absolute path rather than
        through ``PATH``, so the answer describes the environment named in the
        spec and not whichever one the login shell happens to activate.
    """
    return f"'{env_path}/bin/python' -c \"{_PROBE_SOURCE}\""


class InstalledDistribution(TypedDict):
    """What the probe reports about one installed distribution.

    Attributes:
        version: Exact version string.
        wheel_tag: The distribution's own PEP 425 compatibility tag, read from
            its ``WHEEL`` metadata, or empty when it has none. Empty is
            ordinary -- a conda-installed package has no ``WHEEL`` file -- and
            is only an error for a distribution that has to become a wheel.
    """

    version: str
    wheel_tag: str


class EnvironmentReport(TypedDict):
    """Everything one probe of an environment reports.

    Attributes:
        interpreter: What the environment's interpreter says about itself.
        installed: Every distribution it reports, keyed by normalised name.
    """

    interpreter: InterpreterIdentity
    installed: dict[str, InstalledDistribution]


def parse_probe(output: str) -> EnvironmentReport:
    """Parse the probe's output into the interpreter and its distributions.

    Args:
        output: The probe command's standard output.

    Returns:
        The interpreter's identity and every distribution the environment
        reports, keyed by normalised name.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the output does not open
            with the interpreter's identity, or if no line after it carries
            the ``name==version`` separator. An interpreter that printed a
            traceback, or a path that is a directory but not an environment,
            lands here rather than being read as "nothing is installed" --
            which would make every pin fail with a misleading message.
    """
    interpreter, lines = split_identity(output)
    installed: dict[str, InstalledDistribution] = {}
    for line in lines:
        name, separator, rest = line.partition("==")
        if separator == "":
            continue
        version, _, wheel_tag = rest.partition("==")
        installed[normalise_name(name)] = InstalledDistribution(
            version=version.strip(), wheel_tag=wheel_tag.strip()
        )

    if installed == {}:
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            f"The environment reported no installed distributions; "
            f"its interpreter printed {output.strip()!r}. "
            "The path exists but does not look like a Python environment.",
        )
    return EnvironmentReport(interpreter=interpreter, installed=installed)


def check_pins(
    installed: Mapping[str, InstalledDistribution], pinned: Mapping[str, str], *, env_path: str
) -> None:
    """Refuse an environment that does not match what the project declared.

    Args:
        installed: What the environment reports, keyed by normalised name.
        pinned: What the project requires, keyed by normalised name.
        env_path: The environment's path, named in the error so the message
            says which environment was wrong rather than only that one was.

    Raises:
        AppError: With ``ENV_PACKAGE_MISMATCH`` on the first package that is
            absent or at the wrong version. Absent and wrong-version are one
            code deliberately: both mean "this environment is not the one the
            result was produced with", the caller's next step is identical,
            and the message distinguishes them.
    """
    for name in sorted(pinned):
        required = pinned[name]
        actual = installed.get(name)
        if actual is None:
            raise AppError(
                Hpc3ErrorCode.ENV_PACKAGE_MISMATCH,
                f"{env_path} does not have {name} installed, but this project "
                f"pins {name}=={required}. Results from this environment would "
                "not be comparable to results from the pinned one.",
            )
        if actual["version"] != required:
            raise AppError(
                Hpc3ErrorCode.ENV_PACKAGE_MISMATCH,
                f"{env_path} has {name}=={actual['version']}, but this project pins "
                f"{name}=={required}. A version difference under a published "
                "comparison is a confound, not a detail.",
            )


def probe_environment(
    host: str, env_path: str, *, image: ImageReference | None
) -> EnvironmentReport:
    """Ask an environment what it is, refusing one that borrowed its interpreter.

    ``image`` is keyword-only and has NO default. A default of None would let
    a call site probe the host for a container environment by omission, and
    the answer would be an empty listing read as "torch is missing" -- a
    confident, wrong diagnosis of the image rather than of the probe. It also
    selects which shape of the borrowed-interpreter rule applies.

    Args:
        host: SSH destination.
        env_path: Absolute path to the environment, on the cluster for a host
            environment and inside the image for an image one.
        image: The image the environment lives inside, or None when it is a
            cluster directory.

    Returns:
        The interpreter's identity and the installed distributions, the
        interpreter already proven to belong to the environment.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the environment's answer
            cannot be read, ``ENV_INTERPRETER_BORROWED`` if its interpreter
            belongs to another installation, or ``REMOTE_COMMAND_FAILED`` if
            the probe could not be run at all.
    """
    command = probe_command(env_path)
    if image is not None:
        command = image_exec.run_inside_image(image, command)
    report = parse_probe(remote.run_remote(host, command))
    check_interpreter_home(report["interpreter"], env_path=env_path, image=image)
    return report


def verify_environment(
    host: str,
    env_path: str,
    pinned: Mapping[str, str],
    *,
    image: ImageReference | None,
) -> None:
    """Hold an environment to its own interpreter and to the project's pins.

    The probe runs whether or not the project pins anything. It used to
    return before the round trip when ``pinned`` was empty, which made the
    projects with the least other evidence -- ``rusted`` pins nothing -- the
    ones whose interpreter nobody ever asked about.

    Args:
        host: SSH destination.
        env_path: Absolute path to the environment, on the cluster for a host
            run and inside the image for an image run.
        pinned: Required versions, keyed by normalised name. Empty means the
            project declared no pins; the interpreter is still checked.
        image: The image the environment lives inside, or None when it is a
            cluster directory.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the environment's answer
            cannot be read, ``ENV_INTERPRETER_BORROWED`` if its interpreter
            belongs to another installation, ``ENV_PACKAGE_MISMATCH`` if its
            packages do not match, or ``REMOTE_COMMAND_FAILED`` if the probe
            could not be run at all.
    """
    report = probe_environment(host, env_path, image=image)
    check_pins(report["installed"], pinned, env_path=env_path)


__all__ = [
    "EnvironmentReport",
    "InstalledDistribution",
    "check_pins",
    "parse_probe",
    "probe_command",
    "probe_environment",
    "verify_environment",
]
