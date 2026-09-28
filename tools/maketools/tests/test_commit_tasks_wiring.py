"""Every commit this repository pushes names the board task it serves.

MCPs board task 691b0067, b6eb30c8's R8 and R7. The question is MCPs'
maketools commit-tasks and commit-message-tasks, tested there against the
board's answers. What only this repository can pin is that its hooks and
its deploy still ASK it, with nothing around the call that lets a commit, a
push or a deploy through: a hook that lost the line, or gained an
``if [ -z "$SKIP" ]`` around it, would pass every other check and stop
enforcing the rule. The files read are this repository's own root hooks
and Makefile, which ``hooks install`` points git at.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: The repository root: tools/maketools/tests/ is three levels below it.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_VARIABLE: Final[re.Pattern[str]] = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def _executable(path: str) -> list[str]:
    """A file's lines that are not comments.

    Args:
        path: Repository-relative path.

    Returns:
        Its executable lines, line endings removed.
    """
    text = (ROOT / path).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def _variables_read(lines: list[str]) -> list[str]:
    """Every shell variable the lines read.

    Args:
        lines: Executable lines.

    Returns:
        The distinct names, sorted.
    """
    return sorted({match.group(1) for line in lines for match in _VARIABLE.finditer(line)})


def test_the_deploy_asks_about_the_commit_it_ships_before_infra() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert (
        "\ncommit-tasks:\n\t$(PYTHON) .githooks/published_maketools.py "
        "commit-tasks ../MCPs . HEAD^!\n" in makefile.replace("\r\n", "\n")
    )
    assert "\ninfra: commit-tasks executed\n" in makefile.replace("\r\n", "\n")


def test_commit_msg_asks_before_the_commit_exists_reading_only_the_repository() -> None:
    lines = _executable(".githooks/commit-msg")
    assert (
        'python "$REPO/.githooks/published_maketools.py" commit-message-tasks "$REPO/../MCPs" "$1"'
    ) in lines
    assert "set -eu" in lines
    assert _variables_read(lines) == ["REPO"]


def test_published_maketools_runs_origin_main_reading_no_environment() -> None:
    text = (ROOT / ".githooks" / "published_maketools.py").read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]
    # One argument per line, as MCPs board task 51723beb wrote it for chat's
    # line limit (chat 89520c8); the whole list, in order.
    archive_arguments = [
        '"git",',
        "f\"--git-dir={MCPS / '.git'}\",",
        '"archive",',
        '"--format=zip",',
        '"origin/main",',
        '"packages/maketools",',
    ]
    start = lines.index(archive_arguments[0])
    assert lines[start : start + len(archive_arguments)] == archive_arguments
    # A zip, unpacked by zipfile, which confines member paths on every
    # Python 3.11; tarfile needs a filter only 3.11.4 has (board task b61e9fdb).
    assert lines.count("with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:") == 1
    assert "tarfile" not in text
    assert (
        lines.count("[sys.executable, str(Path(extract) / LAUNCHER), *arguments], check=False") == 1
    )
    # Nothing a caller sets can change which command runs or skip it.
    assert [word for word in ("environ", "getenv", "argv[0]") if word in text] == []


def test_pre_push_asks_first_over_each_range_and_reads_only_what_git_hands_it() -> None:
    text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("# EVERY COMMIT NAMES ITS TASK")
    end = text.index("\ndone\n", start)
    stanza = [line for line in text[start:end].split("\n") if not line.lstrip().startswith("#")]
    assert (
        '        python "$REPO/.githooks/published_maketools.py" commit-tasks "$MCPS" "$REPO" '
        '"$ct_remote_sha..$ct_local_sha" < /dev/null'
    ) in stanza
    assert (
        '        python "$REPO/.githooks/published_maketools.py" commit-tasks "$MCPS" "$REPO" '
        '"$ct_local_sha" "^origin/main" < /dev/null'
    ) in stanza
    assert _variables_read(stanza) == [
        "MCPS",
        "PUSH_INPUT",
        "REPO",
        "ZERO",
        "ct_local_sha",
        "ct_remote_sha",
    ]
    # Before the GitHub check that ends a push to any other remote, so no
    # remote escapes it; only the mirror's own push ends earlier.
    assert start < text.index('case "$NWO" in')
    assert text.index("exit 0\nfi\n\n# Read git") < start
