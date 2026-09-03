"""Creating the first environment, which every later command assumes exists.

THE STEP THAT WAS MISSING. The image flow starts at
:mod:`hpc3.cli.image_capture`, which PROBES a live environment over SSH. A
project being onboarded has nothing to probe, so the four documented commands
begin one step after the beginning. What filled the gap was improvisation, and
the improvisation left a defect: ``/pub/wagnera3/envs/tankpit`` is a venv whose
interpreter is a symlink into ``/pub/wagnera3/envs/cleargbm``, because
borrowing a working interpreter from another project was the only move
available. Delete that project's environment and this one stops having a
Python ([[interpreter-availability]]).

WHY CONDA AND NOT ``module load python``. The cluster's module system offers
``python/2.7.17``, ``3.8.0``, ``3.10.2`` and ``3.14.3``, its system interpreter
is 3.9, and every project in this monorepo requires ^3.11 -- so the interpreter
this stack runs on exists under no ``python`` module at all. It is available
through ``miniconda3``, which ``module -t avail python`` will never show you.
Measured 2026-09-03: ``conda create -p <path> python=3.11`` resolves
``python-3.11.16`` from conda-forge, the same build the three existing
environments already run.

WHAT THIS REFUSES, AND WHY IT IS NOT ANOTHER GATE. Three refusals, all about
what this command itself just created: a path that already holds something, an
interpreter that is not the version asked for, and an environment that borrowed
its interpreter from somewhere else. None of them can fire at a project that
already works. That is deliberate -- this package had thirty-four refusals and
no command that creates anything, and adding a thirty-fifth gate on the running
path would have deepened exactly the asymmetry this command exists to close.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.core import remote

CONDA_MODULE = "miniconda3/24.9.2"
"""The module that puts ``conda`` on PATH, measured on hpc3 2026-09-03.

Pinned to an exact version rather than the bare ``miniconda3`` name for the
reason every other pin here exists: the bare name resolves to whatever the
cluster currently defaults to, and an environment built by a different conda
is a different environment. ``module show`` reports this one prepends
``/opt/apps/miniconda3/24.9.2/bin`` to PATH, which is the whole mechanism.
"""


class InterpreterIdentity(TypedDict):
    """What a created environment's own interpreter reports about itself.

    Attributes:
        version: ``major.minor``, e.g. ``"3.11"``. The patch level is
            deliberately not carried: a project declares the language version
            it needs, and conda picks the newest patch for it, so comparing
            patches would refuse a correct environment for being current.
        base_prefix: ``sys.base_prefix``. For a self-contained environment
            this is the environment's own path; for a venv it is whatever
            installation the venv was created FROM, which is how a borrowed
            interpreter becomes visible.
    """

    version: str
    base_prefix: str


_IDENTITY_PROBE = "import sys;print('%d.%d' % sys.version_info[:2]);print(sys.base_prefix)"
"""Ask an interpreter its version and where it actually lives.

Two lines rather than a parsed banner. ``python -V`` prints a string that has
changed format before and carries no prefix at all, and ``sys.base_prefix`` is
the only field that distinguishes an environment which owns its interpreter
from one pointing at somebody else's.

Written as a single ``-c`` expression with no embedded newline: the probe
travels as one argument through ``ssh`` to a remote shell, which would split on
a real newline before Python ever saw it.
"""


def identity_command(env_path: str) -> str:
    """Build the command that asks an environment's interpreter about itself.

    Args:
        env_path: Absolute path to the environment on the cluster.

    Returns:
        A shell command running that environment's own interpreter by
        absolute path rather than through ``PATH``, so the answer describes
        the environment named here and not whichever one a login shell
        activates.
    """
    return f"'{env_path}/bin/python' -c \"{_IDENTITY_PROBE}\""


def create_command(env_path: str, python_version: str) -> str:
    """Build the command that creates the environment.

    ``module load`` and ``conda create`` are ONE command line, joined by
    ``&&``. They cannot be separate calls: each :func:`~hpc3.core.remote.run_remote`
    is its own SSH session with its own shell, so a PATH change made in the
    first would be gone before the second ran. That failure is not
    hypothetical -- it is the same shape as piping ``module load`` into
    ``head``, which puts the load in a subshell and reports ``conda: command
    not found`` from a cluster where conda is present.

    Args:
        env_path: Absolute path to create the environment at.
        python_version: Language version to install, e.g. ``"3.11"``.

    Returns:
        The command line. ``-y`` because there is no terminal to answer a
        prompt on, and a prompt over ``BatchMode=yes`` ssh hangs rather than
        failing.
    """
    return (
        f"module load {CONDA_MODULE} && conda create -y -p '{env_path}' 'python={python_version}'"
    )


def parse_identity(output: str) -> InterpreterIdentity:
    """Read the identity probe's two lines.

    Args:
        output: The probe command's standard output.

    Returns:
        What the interpreter reported about itself.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the output does not carry
            two non-empty lines. A traceback, an empty answer, or a directory
            that is not an environment lands here rather than being read as a
            version of ``""``, which would then be compared against the
            requested version and produce a mismatch message blaming conda for
            a probe that never ran.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip() != ""]
    if len(lines) != 2:
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            f"The interpreter did not report a version and a base prefix; "
            f"it printed {output.strip()!r}.",
        )
    return InterpreterIdentity(version=lines[0], base_prefix=lines[1])


def check_identity(identity: InterpreterIdentity, *, env_path: str, python_version: str) -> None:
    """Hold a freshly created environment to what was asked for.

    Args:
        identity: What the environment's interpreter reported.
        env_path: Where the environment was created.
        python_version: The version that was requested.

    Raises:
        AppError: With ``BOOTSTRAP_PYTHON_MISMATCH`` if the interpreter is
            not the requested version, or
            ``BOOTSTRAP_ENV_NOT_SELF_CONTAINED`` if it belongs to a different
            installation. The second is the one worth having: an environment
            that borrowed its interpreter works perfectly until the
            environment it borrowed FROM is deleted, and nothing else in this
            package would ever notice.
    """
    if identity["version"] != python_version:
        raise AppError(
            Hpc3ErrorCode.BOOTSTRAP_PYTHON_MISMATCH,
            f"{env_path} reports Python {identity['version']}, but "
            f"{python_version} was requested. The environment was created and "
            "is not the one asked for; nothing downstream checks the "
            "interpreter, so this is the only place it can be caught.",
        )
    if identity["base_prefix"] != env_path:
        raise AppError(
            Hpc3ErrorCode.BOOTSTRAP_ENV_NOT_SELF_CONTAINED,
            f"{env_path} runs an interpreter belonging to "
            f"{identity['base_prefix']}. An environment that borrows another "
            "installation's interpreter stops working the day that "
            "installation is moved or deleted, and no run document records "
            "the dependency.",
        )


def refuse_existing(host: str, env_path: str) -> None:
    """Refuse to build on top of whatever is already at the path.

    Args:
        host: SSH destination.
        env_path: Absolute path the environment would be created at.

    Raises:
        AppError: With ``BOOTSTRAP_ENV_EXISTS`` if anything is there.
            Deliberately not "reuse it if it looks right": an existing
            directory is somebody's environment, possibly one an image spec
            names as its source, and silently installing into it would change
            what a built image claims to have come from.
        AppError: With ``REMOTE_COMMAND_FAILED`` if the check could not run.
    """
    answer = remote.run_remote(host, f"test -e '{env_path}' && echo present || echo absent").strip()
    if answer == "present":
        raise AppError(
            Hpc3ErrorCode.BOOTSTRAP_ENV_EXISTS,
            f"{env_path} already exists on {host}. Bootstrap creates a first "
            "environment and will not write into an existing one -- it may be "
            "the source an image spec already names. Remove it deliberately, "
            "or choose another path.",
        )


def bootstrap_environment(host: str, env_path: str, python_version: str) -> InterpreterIdentity:
    """Create a self-contained environment and prove it is what was asked for.

    The order is load-bearing. The existence check comes first so a mistyped
    path fails before anything is built; the identity check comes last because
    it is the only step that reads the environment rather than the request,
    and an environment that was created but is wrong is exactly the state this
    command exists to make impossible to leave behind.

    Args:
        host: SSH destination.
        env_path: Absolute path to create the environment at.
        python_version: Language version to install, e.g. ``"3.11"``.

    Returns:
        The created environment's verified interpreter identity.

    Raises:
        AppError: With ``BOOTSTRAP_ENV_EXISTS`` if the path is occupied,
            ``REMOTE_COMMAND_FAILED`` if conda fails,
            ``ENV_PROBE_UNREADABLE`` if the new interpreter cannot be read, or
            ``BOOTSTRAP_PYTHON_MISMATCH`` / ``BOOTSTRAP_ENV_NOT_SELF_CONTAINED``
            if it is not the environment that was requested.
    """
    refuse_existing(host, env_path)
    _ = remote.run_remote(host, create_command(env_path, python_version))
    identity = parse_identity(remote.run_remote(host, identity_command(env_path)))
    check_identity(identity, env_path=env_path, python_version=python_version)
    return identity


__all__ = [
    "CONDA_MODULE",
    "InterpreterIdentity",
    "bootstrap_environment",
    "check_identity",
    "create_command",
    "identity_command",
    "parse_identity",
    "refuse_existing",
]
