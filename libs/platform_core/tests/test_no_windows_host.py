"""platform_core carries nothing that exists only for a Windows host (MCPs board task 960ac76a).

The fleet's last Windows machine left dispatch on 2026-10-25 (board task
e705f08c), and operator ruling 9c8844a5 removed what this package still kept
for one: ``process_job.py``, a kill-on-close job-object binding through
kernel32; ``process_group.py``, a setsid and killpg binding written around
the type checker's Windows ``os`` module; and ``env_assignments.py``, the
parser of the PowerShell credentials file. Each of them named the host it
served, in its code or its docstring, so a module, branch or docstring
brought back for one names it again. This case reads every source module,
the README and the Makefile for those names.

The tests are not read. A case that refuses a Windows path as input (a
cluster root, a stage destination, a flattened corpus id) keeps that input
because the refusal is Linux behaviour, and a TOML key named ``windows`` is
data.

A restored module that brings no test with it fails the coverage gate as
well, since coverage measures every file under ``src``, so this case is the
line that answers for one restored together with its test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

#: The package directory: ``tests``' parent.
PACKAGE: Final = Path(__file__).resolve().parents[1]

#: What code or prose written for a Windows host names: the platform, its
#: shell, the ``sys.platform`` value, the job-object API's library and the
#: ctypes loader for it, and the PowerShell environment drive.
WINDOWS_NAMES: Final = re.compile(
    r"windows|powershell|win32|kernel32|windll|\$env:", re.IGNORECASE
)

#: One line from each removed module, verbatim, so the pattern is shown to
#: catch what it exists to catch rather than assumed to.
REMOVED_LINES: Final = (
    '"""Windows kill-on-close ownership for launchers and nested worker processes.',
    ":mod:`platform_core.process_job` gives a Windows process a kill-on-close",
    '"""The untracked credentials file a scheduled task sources, read without PowerShell.',
)


def documents() -> list[Path]:
    """Every file this package ships text in that a Windows leftover could hide in.

    Returns:
        Each source module under ``src``, then the README and the Makefile.
    """
    return [*sorted((PACKAGE / "src").rglob("*.py")), PACKAGE / "README.md", PACKAGE / "Makefile"]


def test_the_pattern_catches_each_removed_module() -> None:
    assert [bool(WINDOWS_NAMES.search(line)) for line in REMOVED_LINES] == [True, True, True]


def test_no_source_module_readme_or_makefile_names_a_windows_host() -> None:
    read = documents()
    found = [
        f"{path.relative_to(PACKAGE)}:{number}: {line.strip()}"
        for path in read
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if WINDOWS_NAMES.search(line)
    ]
    assert found == [], f"{len(found)} line(s) in {len(read)} file(s) name a Windows host"
    assert PACKAGE / "src" / "platform_core" / "env_file.py" in read
