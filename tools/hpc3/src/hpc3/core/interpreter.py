"""Asking an environment's interpreter where it came from.

THE DEFECT THIS EXISTS FOR. ``/pub/wagnera3/envs/tankpit`` is a venv whose
interpreter is a symlink into ``/pub/wagnera3/envs/cleargbm``: its
``sys.base_prefix`` is the other project's environment (measured 2026-09-03 on
login-i15 and again 2026-10-05 on login-i16, [[interpreter-availability]]).
Every check that looks at an environment kept passing -- the directory exists
and its packages are pinned -- because none of them asked the interpreter about
itself. Delete cleargbm's environment and tankpit's stops having a Python, with
a failed job as the first signal.

ONE RULE, TWO SHAPES, BECAUSE THE TWO KINDS OF ENVIRONMENT DIFFER. Measured
2026-10-05 inside all eight registered images: ``/opt/env`` is a venv whose
``bin/python`` is a symlink to ``/usr/local/bin/python``, so its
``sys.base_prefix`` is ``/usr/local``, not ``/opt/env``. That is not borrowing:
``/usr/local`` is inside the same digest-pinned image. So:

* A HOST environment owns its interpreter only when ``sys.base_prefix`` IS the
  environment, which is what ``conda create -p`` produces and what the three
  self-contained host environments report.
* An IMAGE environment owns its interpreter when the interpreter lives on the
  image's own root filesystem. The bind list cannot decide this: inside the
  container ``/dfs6b`` is mounted although only ``/pub/wagnera3`` is bound
  (``/proc/mounts``, 2026-10-05), so a venv pointing at
  ``/dfs6b/pub/wagnera3/envs/cleargbm`` would pass any check on binds. Device
  numbers do decide it -- ``/``, ``/usr/local`` and ``/opt/env`` report one
  ``st_dev`` (the overlay root) and ``/dfs6b/...`` another (BeeGFS) -- so the
  probe asks the interpreter whether its base is on the root's device.
"""

from __future__ import annotations

import re

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.image import ImageReference


class InterpreterIdentity(TypedDict):
    """What an environment's own interpreter reports about itself.

    Attributes:
        version: ``major.minor``, e.g. ``"3.11"``. The patch level is
            deliberately not carried: a project declares the language version
            it needs, and conda picks the newest patch for it, so comparing
            patches would refuse a correct environment for being current.
        base_prefix: ``sys.base_prefix``. For a self-contained environment
            this is the environment's own path; for a venv it is whatever
            installation the venv was created FROM, which is how a borrowed
            interpreter becomes visible.
        base_on_root_filesystem: Whether ``sys.base_prefix`` sits on the same
            device as ``/``. Inside an image that is the difference between an
            interpreter the image carries and one reached through a mount; on
            a host it is not consulted, since a host environment's base is
            held to its own path instead.
    """

    version: str
    base_prefix: str
    base_on_root_filesystem: bool


IDENTITY_IMPORTS = "import os,sys"
"""The modules :data:`IDENTITY_LINES` reads, as a statement to prefix it with."""

IDENTITY_LINES = (
    "['%d.%d'%sys.version_info[:2],sys.base_prefix,"
    "str(os.stat(sys.base_prefix).st_dev==os.stat('/').st_dev)]"
)
"""A Python list expression of the identity's three lines, in field order.

An EXPRESSION rather than a statement so the distribution probe in
:mod:`hpc3.core.env_probe` can concatenate it in front of its own listing and
answer both questions in one SSH round trip. Single-quoted throughout because
it travels inside a double-quoted ``-c`` argument, and newline-free because the
remote shell would split a real newline before Python saw it.
"""

IDENTITY_LINE_COUNT = 3
"""How many lines :data:`IDENTITY_LINES` prints, one per identity field."""

_VERSION = re.compile(r"\d+\.\d+")
_BOOLEANS = {"True": True, "False": False}


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
    source = f"{IDENTITY_IMPORTS};print(chr(10).join({IDENTITY_LINES}))"
    return f"'{env_path}/bin/python' -c \"{source}\""


def split_identity(output: str) -> tuple[InterpreterIdentity, list[str]]:
    """Read the identity from the head of a probe's output.

    Args:
        output: Standard output of a command that printed
            :data:`IDENTITY_LINES` first.

    Returns:
        The identity, and every non-blank line after it, stripped, for the
        caller that printed more than the identity.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the head is not a
            ``major.minor`` version, a base prefix and ``True`` or ``False``.
            The prefix is not held to a path syntax: it is whatever the
            interpreter's platform spells, and the lines around it already
            tell an identity from anything else. A traceback, an empty
            answer, or a directory that is not an environment lands here
            rather than being read as a version of ``""`` and then blamed on
            whatever it is compared against.
    """
    lines = [line.strip() for line in output.splitlines() if line.strip() != ""]
    head = lines[:IDENTITY_LINE_COUNT]
    if (
        len(head) != IDENTITY_LINE_COUNT
        or _VERSION.fullmatch(head[0]) is None
        or head[2] not in _BOOLEANS
    ):
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            "The interpreter did not report a version, a base prefix and "
            f"whether that prefix is on the root filesystem; it printed {output.strip()!r}.",
        )
    identity = InterpreterIdentity(
        version=head[0], base_prefix=head[1], base_on_root_filesystem=_BOOLEANS[head[2]]
    )
    return identity, lines[IDENTITY_LINE_COUNT:]


def parse_identity(output: str) -> InterpreterIdentity:
    """Read the output of :func:`identity_command`, which prints nothing else.

    Args:
        output: The identity command's standard output.

    Returns:
        What the interpreter reported about itself.

    Raises:
        AppError: With ``ENV_PROBE_UNREADABLE`` if the identity cannot be
            read, or if anything follows it -- this command prints exactly the
            identity, so an extra line means it was not this command's answer.
    """
    identity, rest = split_identity(output)
    if rest != []:
        raise AppError(
            Hpc3ErrorCode.ENV_PROBE_UNREADABLE,
            f"The interpreter printed more than its identity: {output.strip()!r}.",
        )
    return identity


def check_interpreter_home(
    identity: InterpreterIdentity, *, env_path: str, image: ImageReference | None
) -> None:
    """Refuse an environment that runs another installation's interpreter.

    ``image`` is keyword-only with no default, for the reason
    :func:`~hpc3.core.env_probe.verify_environment` gives: the rule differs by
    where the environment lives, and an omitted image would hold a container
    environment to the host rule and refuse every registered project.

    Args:
        identity: What the environment's interpreter reported.
        env_path: The environment that was probed.
        image: The image the environment lives inside, or None for a host
            directory.

    Raises:
        AppError: With ``ENV_INTERPRETER_BORROWED`` if a host environment's
            base prefix is not the environment itself, or an image
            environment's base prefix is not on the image's root filesystem.
            Either way the environment works until the installation it
            borrowed from is moved or deleted, and nothing else records the
            dependency.
    """
    if image is None and identity["base_prefix"] != env_path:
        raise AppError(
            Hpc3ErrorCode.ENV_INTERPRETER_BORROWED,
            f"{env_path} runs Python {identity['version']} belonging to "
            f"{identity['base_prefix']}. An environment that borrows another "
            "installation's interpreter stops working the day that installation "
            "is moved or deleted; build it with hpc3-bootstrap so it owns one.",
        )
    if image is not None and not identity["base_on_root_filesystem"]:
        raise AppError(
            Hpc3ErrorCode.ENV_INTERPRETER_BORROWED,
            f"{env_path} inside {image['path']} runs Python {identity['version']} "
            f"from {identity['base_prefix']}, which is a mounted filesystem rather "
            "than the image. The image's digest does not cover that interpreter, "
            "and the job breaks the day the directory behind the mount changes.",
        )


__all__ = [
    "IDENTITY_IMPORTS",
    "IDENTITY_LINES",
    "IDENTITY_LINE_COUNT",
    "InterpreterIdentity",
    "check_interpreter_home",
    "identity_command",
    "parse_identity",
    "split_identity",
]
