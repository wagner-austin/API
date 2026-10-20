"""Every push from this repository runs the tests it touches.

MCPs board task a7416f6f. The question is MCPs' maketools pushed-venvs and
pushed-tests, tested there against real git and real runners. What only this
repository can pin is that its pre-push still ASKS them, for every pushed
range, with nothing around the calls that lets a push through, and that the
roots they read are the ones this repository declares, the same file
``scripts/ci_select_packages.py`` reads for CI.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Final

#: The repository root: tools/maketools/tests/ is three levels below it.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_VARIABLE: Final[re.Pattern[str]] = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")

#: Where this repository declares its nested project roots.
DECLARATION: Final[Path] = ROOT / "scripts" / "make" / "project-roots.json"


def _stanza() -> list[str]:
    """The pre-push stanza that runs the push's tests, its comments removed.

    Returns:
        Its executable lines, from its header to the end of its loop.
    """
    text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("# EVERY PUSH RUNS THE TESTS IT TOUCHES")
    end = text.index("\ndone\n", start)
    return [line for line in text[start:end].split("\n") if not line.lstrip().startswith("#")]


def test_each_range_builds_its_venvs_then_runs_its_tests_as_mcps_published_them() -> None:
    stanza = _stanza()
    launch = '        "$HOOK_PYTHON" "$REPO/.githooks/published_maketools.py"'
    new_ref = '"$pt_local_sha" "^origin/main" < /dev/null'
    known_ref = '"$pt_remote_sha..$pt_local_sha" < /dev/null'
    assert [line for line in stanza if "published_maketools.py" in line] == [
        f'{launch} pushed-venvs "$REPO" {new_ref}',
        f'{launch} pushed-tests "$REPO" "$pt_local_sha" {new_ref}',
        f'{launch} pushed-venvs "$REPO" {known_ref}',
        f'{launch} pushed-tests "$REPO" "$pt_local_sha" {known_ref}',
    ]
    # Only a deleted ref and git's empty line are skipped; no variable or
    # author skips a range.
    assert sorted({line.strip() for line in stanza if "continue" in line}) == [
        '[ "$pt_local_sha" = "$ZERO" ] && continue',
        '[ -z "$pt_local_sha" ] && continue',
    ]
    variables = sorted({match.group(1) for line in stanza for match in _VARIABLE.finditer(line)})
    assert variables == [
        "HOOK_PYTHON",
        "PUSH_INPUT",
        "REPO",
        "ZERO",
        "pt_local_sha",
        "pt_remote_sha",
    ]


def test_the_tests_run_after_commit_tasks_and_before_any_remote_ends_the_hook() -> None:
    text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("# EVERY PUSH RUNS THE TESTS IT TOUCHES")
    assert text.index("# EVERY COMMIT NAMES ITS TASK") < start < text.index('case "$NWO" in')
    assert "set -eu" in text.split("\n")


def test_this_repository_declares_the_roots_ci_selects_its_packages_by() -> None:
    """The roots are stated once: the file pushed-tests reads from the tip is
    the one CI's package selection reads, run here as the workflow runs it.
    A file under each declared root selects its own package."""
    assert DECLARATION.read_text(encoding="utf-8") == (
        '{"nestedRoots": ["clients", "services", "libs", "tools"]}\n'
    )
    changed = [
        "clients/NavProbe/src/x.py",
        "services/turkic-api/src/x.py",
        "libs/platform_core/src/x.py",
        "tools/maketools/src/x.py",
    ]
    selected = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci_select_packages.py"), *changed],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for package in ("clients/NavProbe", "services/turkic-api", "libs/platform_core"):
        assert f'"{package}"' in selected
    source = (ROOT / "scripts" / "ci_select_packages.py").read_text(encoding="utf-8")
    assert '"clients", "services"' not in source
    assert "package_roots(repo)" in source
