"""The bash that tests executing rendered shell lines run them under.

Support module, not a test module. On Windows it is Git's ``bin/bash.exe``,
found beside ``git --exec-path``, because ``bash`` on a Windows ``PATH`` may
be ``System32\\bash.exe``: the WSL launcher, which runs the file in whatever
distro is the default, or in none. Measured 2026-09-27 on sedona (MCPs board
task 140e7042): ``shutil.which("bash")`` answered that launcher, its distro
had no ``/bin/bash``, and two tests failed with ``execvpe(/bin/bash) failed:
No such file or directory`` while passing on the hub, whose PATH puts Git's
bash first. Every other host runs ``bash`` from ``PATH``.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

#: How long ``git --exec-path`` may take; it prints one line and exits.
GIT_EXEC_PATH_SECONDS = 30


def host_bash() -> str:
    """The bash executable for this test host.

    Returns:
        On Windows, the absolute path of Git's ``bin/bash.exe``; elsewhere,
        ``bash`` as ``PATH`` resolves it.

    Raises:
        AssertionError: When ``git --exec-path`` fails on Windows, naming its
            exit status and standard error, or when no bash is on ``PATH``
            elsewhere. A test that executes shell lines cannot run without
            one, and a skip would read as a pass.
    """
    if sys.platform == "win32":
        found = subprocess.run(
            ["git", "--exec-path"],
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_EXEC_PATH_SECONDS,
        )
        if found.returncode != 0:
            raise AssertionError(
                f"git --exec-path exited {found.returncode}: {found.stderr.strip()}"
            )
        return str(pathlib.Path(found.stdout.strip()).parents[2] / "bin" / "bash.exe")
    on_path = shutil.which("bash")
    if on_path is None:
        raise AssertionError("bash is required to execute the rendered provision lines")
    return on_path
