"""Which operating system a decision is being made for.

One constant and one predicate, in their own module because the things that
need them sit at different layers and neither may import the other.
:mod:`rw_bot.validation` is foundational -- the harness imports it, not the
reverse -- and it needs the answer to decide whether a path is absolute.
:mod:`rw_bot.harness.jvm` needs it to decide what the JDK's tools are called.
A copy in each is two definitions of one fact, and the second one to change is
the bug.

The platform is always an ARGUMENT here, never a read of the running machine.
Everything downstream of it is a pure function, so both answers stay reachable
from either platform -- which is what lets a Windows workstation prove the
rules a Linux node will run under, and what keeps the branches coverable when
the suite can only execute on one of them.
"""

from __future__ import annotations

from pathlib import PurePath, PurePosixPath, PureWindowsPath

#: The ``sys.platform`` value naming Windows.
#:
#: Every other value is POSIX for every purpose this package has, which is why
#: the predicate below is a single equality rather than a table: macOS and
#: Linux spell executable names, classpath separators and path roots the same
#: way as each other and differently from this one.
WINDOWS = "win32"

#: How Windows separates the entries of a path LIST.
#:
#: One constant for both of this package's path lists, because the operating
#: system has only one convention and Java honours it: a JVM classpath and a
#: ``PYTHONPATH`` are separated by the same character, and Java's own
#: ``File.pathSeparator`` is this. Two constants would be two things to get
#: right and one of them would eventually be the wrong one.
WINDOWS_PATH_LIST_SEPARATOR = ";"

#: The same, everywhere else. Distinct from the Windows spelling by exactly
#: the character Windows spends on drive letters, which is why they differ.
POSIX_PATH_LIST_SEPARATOR = ":"


def is_windows(platform: str) -> bool:
    """Report whether a platform uses Windows conventions.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        True for Windows, False for everything else.
    """
    return platform == WINDOWS


def pure_path(platform: str) -> type[PurePath]:
    """Return the path flavour a platform spells paths in.

    WHY THIS IS NOT ``pathlib.Path``. ``Path`` resolves to the flavour of the
    interpreter that is RUNNING, so composing a launch with it makes the
    command depend on where it was composed rather than on where it will run.
    In production the two agree and nothing shows; in a test that asks a
    Windows box what a Linux launch looks like, the answer comes back with
    backslashes -- and the cross-platform checks that are this package's whole
    verification strategy quietly stop meaning anything.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :class:`pathlib.PureWindowsPath` or :class:`pathlib.PurePosixPath`.
    """
    if is_windows(platform):
        return PureWindowsPath
    return PurePosixPath


def path_list_separator(platform: str) -> str:
    """Return the character joining the entries of a path list.

    Args:
        platform: A ``sys.platform`` value.

    Returns:
        :data:`WINDOWS_PATH_LIST_SEPARATOR` or
        :data:`POSIX_PATH_LIST_SEPARATOR`.
    """
    if is_windows(platform):
        return WINDOWS_PATH_LIST_SEPARATOR
    return POSIX_PATH_LIST_SEPARATOR


__all__ = [
    "POSIX_PATH_LIST_SEPARATOR",
    "WINDOWS",
    "WINDOWS_PATH_LIST_SEPARATOR",
    "is_windows",
    "path_list_separator",
    "pure_path",
]
