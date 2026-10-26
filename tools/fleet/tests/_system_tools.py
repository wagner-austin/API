"""This machine's own programs, for a case that runs a dialect's line with a PATH of its own.

Until MCPs board task c2e3f31f the cases linked ``/usr/bin`` and ``/bin``,
which is where an Ubuntu node keeps its programs. A NixOS node keeps them in
its system profile and its setuid wrappers and has nothing in ``/usr/bin``
but ``env``, so on loki every such case ran the probe with no ``head`` or
``sed`` and read nothing. The directories are the isolated build's own
:data:`fleet.core.linux_isolated_build.EXEC_PATH`, the system's programs on
either.
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.core.linux_isolated_build import EXEC_PATH

#: The system's program directories this machine has, in EXEC_PATH's order.
SYSTEM_BINARIES: Final[tuple[pathlib.Path, ...]] = tuple(
    directory for directory in map(pathlib.Path, EXEC_PATH.split(":")) if directory.is_dir()
)


def link_system_tools(tools: pathlib.Path, placed: frozenset[str]) -> None:
    """Link every program in :data:`SYSTEM_BINARIES` but ``placed`` into ``tools``.

    A name in more than one directory is linked once, from the first, as the
    shell would find it.

    Args:
        tools: The empty directory the links go in.
        placed: The programs the case places itself, never linked from the node.
    """
    for directory in SYSTEM_BINARIES:
        for program in sorted(directory.iterdir()):
            link = tools / program.name
            if program.name not in placed and not program.is_dir() and not link.is_symlink():
                link.symlink_to(program)
