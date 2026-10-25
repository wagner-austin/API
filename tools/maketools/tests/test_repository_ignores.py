"""The repository's root ``.gitignore`` hides the check lock stamps, read by real git.

MCPs board task ae522e15. MCPs' maketools takes a per-package ``make test``
lock by writing a stamp under ``.check-locks/`` at the root of whichever
repository it checks, and on 2026-10-24 that directory stood untracked and
unignored in ziggurat's API checkout, where the session-start checkout sync
names every such path. Each case copies this repository's own root
``.gitignore`` into a scratch git repository holding a stamp and reads
``git status``, so the rule is judged by git's matching and not by its text.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from maketools.cli import repository_root

#: Where MCPs' maketools writes its lock stamps (its ``repo.LOCK_DIRECTORY_NAME``).
LOCK_DIRECTORY: str = ".check-locks"

#: A stamp as the lock writes it for ``tools/fleet``.
STAMP: str = "tools__fleet.lock"


def untracked(rules: str, scratch: Path) -> list[str]:
    """The untracked paths git reports in a scratch repository holding a stamp.

    Args:
        rules: The ``.gitignore`` text the repository carries.
        scratch: An empty directory to make the repository in.

    Returns:
        Each path ``git status`` lists as untracked, files listed one by one.
    """
    subprocess.run(("git", "init", "-q"), cwd=scratch, check=True)
    (scratch / ".gitignore").write_text(rules, encoding="utf-8", newline="\n")
    (scratch / LOCK_DIRECTORY).mkdir()
    (scratch / LOCK_DIRECTORY / STAMP).write_text("", encoding="utf-8")
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=scratch,
        capture_output=True,
        text=True,
        check=True,
    )
    return status.stdout.splitlines()


def test_the_root_gitignore_hides_a_check_lock_stamp(tmp_path: Path) -> None:
    rules = (repository_root() / ".gitignore").read_text(encoding="utf-8")
    assert untracked(rules, tmp_path) == ["?? .gitignore"]


def test_without_its_line_the_stamp_is_listed(tmp_path: Path) -> None:
    rules = (repository_root() / ".gitignore").read_text(encoding="utf-8")
    kept = [line for line in rules.splitlines() if LOCK_DIRECTORY not in line]
    assert len(kept) < len(rules.splitlines())
    assert untracked("\n".join(kept) + "\n", tmp_path) == [
        "?? .gitignore",
        f"?? {LOCK_DIRECTORY}/{STAMP}",
    ]
