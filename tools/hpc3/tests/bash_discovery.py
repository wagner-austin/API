"""Locating a POSIX bash on a machine where ``bash`` is not one program.

Extracted from ``conftest.py`` when it crossed the size ceiling: this trio is
the one block there with a single role -- resolving a real shell out of the
three programs Windows calls ``bash`` -- and a single consumer,
``test_image_sbatch``. The reasoning lives on :func:`posix_bash`.
"""

from __future__ import annotations

import pathlib
import shutil

_WINDOWS_SYSTEM_DIRS = frozenset({"system32", "syswow64"})
"""Where the WSL launcher lives, and nothing that is a shell does."""


def is_wsl_launcher(bash_path: str) -> bool:
    """Report whether a resolved ``bash`` is the WSL launcher rather than a shell.

    Args:
        bash_path: Path ``shutil.which`` returned.

    Returns:
        True when it sits in a Windows system directory. Matched on the path
        component rather than against ``SystemRoot`` so this reads no
        environment -- the guard bans that, and the directory NAME is the
        thing that identifies the launcher anyway. Both spellings are
        checked: a 32-bit process, which is what ``make`` recipes run in
        here, sees the same launcher through ``SysWOW64``.

        Split with ``PureWindowsPath`` rather than ``Path``. ``Path`` takes
        the flavour of the machine it runs on, and on Linux a backslash is an
        ordinary character -- so ``C:\\Windows\\System32\\bash.exe`` parses to
        ONE part, no component equals ``system32``, and this answered False
        for the launcher it exists to recognise. The three tests below caught
        that the first time this package's suite was run in CI on 2026-09-04.
        The argument is a Windows path by definition (the WSL launcher exists
        nowhere else), so parsing it with Windows rules is right on every
        platform, and a POSIX path still splits correctly because
        ``PureWindowsPath`` accepts ``/`` as a separator too.
    """
    return any(
        part.lower() in _WINDOWS_SYSTEM_DIRS for part in pathlib.PureWindowsPath(bash_path).parts
    )


def bash_beside_git(git_path: str) -> str | None:
    """Locate the bash that Git for Windows installs alongside ``git``.

    The second place to look, and on this machine the only one that works:
    PowerShell's PATH carries ``C:\\Program Files\\Git\\cmd`` but not
    ``…\\Git\\usr\\bin``, so excluding the WSL launcher leaves no bash on PATH
    at all. ``git`` itself is always reachable -- this is a git repository and
    the lint step shells out to it -- and Git for Windows ships a real MSYS
    bash two directories up from ``git.exe``.

    Derived from ``git``'s own location rather than hardcoded to
    ``C:\\Program Files``: a machine that installed Git elsewhere is a machine
    where a hardcoded path silently finds nothing.

    Args:
        git_path: Absolute path to ``git``.

    Returns:
        Absolute path to the bash beside it, or None if there is none --
        which is the normal case off Windows, where the PATH search already
        succeeded.
    """
    root = pathlib.Path(git_path).parent.parent
    for relative in ("usr/bin/bash.exe", "bin/bash.exe"):
        candidate = root / relative
        if candidate.is_file():
            return str(candidate)
    return None


def posix_bash() -> str:
    """Locate a bash that will actually parse a script handed to it on stdin.

    ``bash`` IS NOT ONE PROGRAM ON THIS MACHINE, and which one you get depends
    on the shell that launched pytest. From a Git Bash session the first match
    on PATH is MSYS bash, a real shell. From PowerShell -- which is the shell
    the Makefile runs pytest in, so it is what ``make check`` uses -- the
    first match is ``C:\\Windows\\System32\\bash.exe``, the WSL launcher.
    That is not a shell: it forwards to a Linux distribution, and when the WSL
    service is not running it exits 1 with ``Error code:
    Bash/Service/0x8007274c`` and a UTF-16 message on stdout.

    Six tests in ``test_image_sbatch`` failed exactly that way during a
    ``make check`` whose subject was somewhere else entirely, having passed
    minutes earlier under pytest launched from Git Bash. They had been called
    flaky. They are not flaky: they resolve an interpreter through an
    ambient PATH, and the answer differs by caller.

    So the interpreter is chosen rather than inherited. ``git`` is asked
    first, because it is the one interpreter-adjacent program guaranteed to be
    here -- this is a git repository -- and on Windows it is the only route
    that finds a real shell at all: PowerShell's PATH carries
    ``…\\Git\\cmd`` but not ``…\\Git\\usr\\bin``, so a PATH search that merely
    skipped the launcher would find nothing. PATH is the fallback, which is
    what answers on POSIX, where git ships no bash beside itself.

    Returns:
        Absolute path to a usable bash.

    Raises:
        RuntimeError: If none is found, or the only one found is the launcher.
            Skipping instead would report green on a machine where the thing
            these tests check was never checked, which is the failure they
            exist to prevent wearing a hat.
    """
    git = shutil.which("git")
    beside = None if git is None else bash_beside_git(git)
    if beside is not None:
        return beside
    found = shutil.which("bash")
    if found is None or is_wsl_launcher(found):
        raise RuntimeError(
            "No POSIX bash found: none beside git, and the only one on PATH is "
            f"{found!r}. The tests that hand a rendered script to a real parser "
            "cannot run."
        )
    return found
