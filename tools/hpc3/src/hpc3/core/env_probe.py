"""Asking an environment what is actually installed in it.

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

from hpc3.contracts.pins import normalise_name
from hpc3.core import remote

_PROBE_SOURCE = (
    "import importlib.metadata as m;"
    "print(chr(10).join("
    "(d.metadata['Name'] or '?')+'=='+(d.version or '?') for d in m.distributions()))"
)
"""Print every installed distribution as ``Name==version``, one per line.

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


def parse_installed(output: str) -> dict[str, str]:
    """Parse the probe's output into normalised name to version.

    Args:
        output: The probe command's standard output.

    Returns:
        Every distribution the environment reports, keyed by normalised name.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if no line carries the
            ``name==version`` separator. An interpreter that printed a
            traceback, or a path that is a directory but not an environment,
            lands here rather than being read as "nothing is installed" --
            which would make every pin fail with a misleading message.
    """
    installed: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        name, separator, version = stripped.partition("==")
        if separator == "":
            continue
        installed[normalise_name(name)] = version.strip()

    if installed == {}:
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            f"The environment reported no installed distributions; "
            f"its interpreter printed {output.strip()!r}. "
            "The path exists but does not look like a Python environment.",
        )
    return installed


def check_pins(installed: Mapping[str, str], pinned: Mapping[str, str], *, env_path: str) -> None:
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
        if actual != required:
            raise AppError(
                Hpc3ErrorCode.ENV_PACKAGE_MISMATCH,
                f"{env_path} has {name}=={actual}, but this project pins "
                f"{name}=={required}. A version difference under a published "
                "comparison is a confound, not a detail.",
            )


def verify_env_packages(host: str, env_path: str, pinned: Mapping[str, str]) -> None:
    """Ask an environment what it contains and hold it to the project's pins.

    Args:
        host: SSH destination.
        env_path: Absolute path to the environment on the cluster.
        pinned: Required versions, keyed by normalised name. Empty means the
            project declared no pins, and no round trip is made.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the environment's answer
            cannot be read, ``ENV_PACKAGE_MISMATCH`` if it does not match, or
            ``REMOTE_COMMAND_FAILED`` if the probe could not be run at all.
    """
    if pinned == {}:
        return
    output = remote.run_remote(host, probe_command(env_path))
    check_pins(parse_installed(output), pinned, env_path=env_path)


__all__ = [
    "check_pins",
    "parse_installed",
    "probe_command",
    "verify_env_packages",
]
