"""Run one MCPs maketools command as MCPs PUBLISHED it, never as a checkout holds it.

EVERY COMMIT NAMES ITS TASK (MCPs board task 691b0067), and a deploy ships
only what was run (465689f5). This repository's commit-msg and pre-push
hooks and its Makefile's gates ask MCPs' maketools, and the same file,
byte for byte, does it in API, corvis-stick and slime. MCPs is the
checkout beside this one, and
its working tree is shared: other sessions' uncommitted edits sit in it,
and it moves forward only when the fleet deploys a clean hub, so on a given
day it can lack the command or carry an uncommitted edit to it. A gate read
from there could be missing, or weakened by an edit nobody committed. So
the command is extracted from MCPs' origin/main, the published commit its
own pre-push already judged, which no editor mutates; the checkout supplies
only the board's and the fleet's credentials and addresses, which the
command reads from it.

MCPs' repository is named with ``--git-dir``, never ``-C``: inside a hook
git exports GIT_DIR for the repository being committed or pushed, and
GIT_DIR outranks ``-C``, so ``git -C ../MCPs archive`` read this repository
and found no packages/maketools (measured, on the hook's first real commit).

WHY PYTHON AND NOT SH. This was a sh script until a deploy ran it from a
PowerShell-launched make: on Windows every recipe runs under PowerShell
(scripts/make/shell.mk), whose PATH carries git and python but not sh, so
``make commit-tasks`` died with ``sh`` not found before it asked anything,
and only a make started from Git Bash, which hands its PATH down, worked.
Python is on the PATH of every caller: the recipes run it as ``$(PYTHON)``
and the hooks as ``python``, as the command itself always required. The
archive is unpacked with :mod:`zipfile` rather than a ``tar`` binary, which
differs between Git Bash and Windows. It is a zip and not a tar because
``TarFile.extractall`` confines its members only through a ``filter`` that
exists from Python 3.11.4, and a 3.11.1 node failed on it (MCPs board task
b61e9fdb); ``ZipFile.extractall`` has always dropped absolute paths and
``..`` from member names.

Arguments: the maketools command and its arguments. Exit status: the
command's own; 1 when MCPs' origin/main could not supply it, and 2 when no
command was named, each of which refuses whatever called this.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: The repository whose hooks and Makefile call this, and MCPs beside it.
REPO = Path(__file__).resolve().parent.parent
MCPS = REPO.parent / "MCPs"

#: The launcher inside the extracted package.
LAUNCHER = Path("packages", "maketools", "scripts", "run.py")


def main(arguments: list[str]) -> int:
    """Extract MCPs' published maketools and run one command with it.

    Args:
        arguments: The maketools command and its arguments.

    Returns:
        The command's exit status, 1 when MCPs' origin/main carries no
        packages/maketools, or 2 when no command was named.
    """
    if not arguments:
        sys.stderr.write("published-maketools: usage: published_maketools.py <command> [args...]\n")
        return 2
    archived = subprocess.run(
        [
            "git",
            f"--git-dir={MCPS / '.git'}",
            "archive",
            "--format=zip",
            "origin/main",
            "packages/maketools",
        ],
        capture_output=True,
        check=False,
    )
    if archived.returncode != 0:
        sys.stderr.write(archived.stderr.decode("utf-8", errors="replace"))
        sys.stderr.write(
            f"published-maketools: {MCPS} has no origin/main carrying packages/maketools, "
            f"so {arguments[0]} did not run and what called it is refused\n"
        )
        return 1
    with tempfile.TemporaryDirectory(prefix="published-maketools-") as extract:
        with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:
            archive.extractall(extract)
        return subprocess.run(
            [sys.executable, str(Path(extract) / LAUNCHER), *arguments], check=False
        ).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
