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
its interpreter from somewhere else. The first two cannot fire at a project that
already works -- this package had thirty-four refusals and no command that
creates anything, and this command exists to close that asymmetry. The third is
the one exception, and deliberately so: a borrowed interpreter is a defect in a
running project too, so the rule lives in :mod:`hpc3.core.interpreter` and
preflight and capture hold every environment to it as well.
"""

from __future__ import annotations

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.core import remote
from hpc3.core.interpreter import (
    InterpreterIdentity,
    check_interpreter_home,
    identity_command,
    parse_identity,
)

CONDA_MODULE = "miniconda3/24.9.2"
"""The module that puts ``conda`` on PATH, measured on hpc3 2026-09-03.

Pinned to an exact version rather than the bare ``miniconda3`` name for the
reason every other pin here exists: the bare name resolves to whatever the
cluster currently defaults to, and an environment built by a different conda
is a different environment. ``module show`` reports this one prepends
``/opt/apps/miniconda3/24.9.2/bin`` to PATH, which is the whole mechanism.
"""


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


def check_identity(identity: InterpreterIdentity, *, env_path: str, python_version: str) -> None:
    """Hold a freshly created environment to what was asked for.

    Args:
        identity: What the environment's interpreter reported.
        env_path: Where the environment was created.
        python_version: The version that was requested.

    Raises:
        AppError: With ``BOOTSTRAP_PYTHON_MISMATCH`` if the interpreter is
            not the requested version, or ``ENV_INTERPRETER_BORROWED`` from
            :func:`~hpc3.core.interpreter.check_interpreter_home` if it
            belongs to a different installation -- the same rule preflight
            and capture hold every host environment to, applied here to the
            one this command just built.
    """
    if identity["version"] != python_version:
        raise AppError(
            Hpc3ErrorCode.BOOTSTRAP_PYTHON_MISMATCH,
            f"{env_path} reports Python {identity['version']}, but "
            f"{python_version} was requested. The environment was created and "
            "is not the one asked for, and no run document declares the "
            "version a later check could hold it to.",
        )
    check_interpreter_home(identity, env_path=env_path, image=None)


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
            ``BOOTSTRAP_PYTHON_MISMATCH`` / ``ENV_INTERPRETER_BORROWED`` if it
            is not the environment that was requested.
    """
    refuse_existing(host, env_path)
    _ = remote.run_remote(host, create_command(env_path, python_version))
    identity = parse_identity(remote.run_remote(host, identity_command(env_path)))
    check_identity(identity, env_path=env_path, python_version=python_version)
    return identity


__all__ = [
    "CONDA_MODULE",
    "bootstrap_environment",
    "check_identity",
    "create_command",
    "refuse_existing",
]
