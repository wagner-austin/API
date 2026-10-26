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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

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


def test_each_range_is_checked_on_the_fleet_from_a_session_host_or_here_elsewhere() -> None:
    # The session host's arm came with MCPs board task f8390a08 (API
    # 5e655da79): session-host decides, then each range goes to
    # pushed-fleet-check there and to pushed-venvs then pushed-tests anywhere
    # else, all as MCPs' origin/main carries them.
    stanza = _stanza()
    launch = '"$HOOK_PYTHON" "$REPO/.githooks/published_maketools.py"'
    new_ref = '"$pt_local_sha" "^origin/main" < /dev/null'
    known_ref = '"$pt_remote_sha..$pt_local_sha" < /dev/null'
    assert [line for line in stanza if "published_maketools.py" in line] == [
        f'{launch} session-host "$REPO" < /dev/null || session_host_status=$?',
        f'        {launch} pushed-fleet-check "$REPO" "$pt_local_sha" {new_ref}',
        f'        {launch} pushed-fleet-check "$REPO" "$pt_local_sha" {known_ref}',
        f'        {launch} pushed-venvs "$REPO" {new_ref}',
        f'        {launch} pushed-tests "$REPO" "$pt_local_sha" {new_ref}',
        f'        {launch} pushed-venvs "$REPO" {known_ref}',
        f'        {launch} pushed-tests "$REPO" "$pt_local_sha" {known_ref}',
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
        "session_host_status",
    ]


def test_the_tests_run_after_commit_tasks_and_before_any_remote_ends_the_hook() -> None:
    text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("# EVERY PUSH RUNS THE TESTS IT TOUCHES")
    assert text.index("# EVERY COMMIT NAMES ITS TASK") < start < text.index('case "$NWO" in')
    assert "set -eu" in text.split("\n")


def test_a_push_that_only_stages_a_tip_ends_the_hook_before_any_check() -> None:
    """pushed-tests stages a container suite's tip as refs/heads/pushed-tests/<tip>
    from inside pre-push, and that nested push runs this hook again; without
    the exit it ran pushed-tests over the same tip and staged it again, about
    45 processes deep on 2026-10-21 (MCPs board task 703fec5a). The exit is
    maketools' push-staging over every update git hands the hook, read before
    the first check, and only its 0 ends the hook: 3 goes on, anything else
    refuses."""
    text = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index("# A PUSH THAT ONLY STAGES A TIP FOR ITS FLEET RUN")
    end = text.index("\nfi\n", start)
    assert text.index('PUSH_INPUT="$(cat)"') < start < text.index("# EVERY COMMIT NAMES ITS TASK")
    lines = [
        line.strip()
        for line in text[start:end].split("\n")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines == [
        'STAGING_UPDATES=""',
        "while read -r _st_local_ref st_local_sha st_remote_ref _st_remote_sha; do",
        '[ -z "$st_local_sha" ] && continue',
        'STAGING_UPDATES="$STAGING_UPDATES $st_local_sha:$st_remote_ref"',
        "done <<EOF",
        "$PUSH_INPUT",
        "EOF",
        'if [ -n "$STAGING_UPDATES" ]; then',
        "staging_status=0",
        '"$HOOK_PYTHON" "$REPO/.githooks/published_maketools.py" push-staging '
        "$STAGING_UPDATES < /dev/null || staging_status=$?",
        'if [ "$staging_status" -eq 0 ]; then',
        "exit 0",
        "fi",
        'if [ "$staging_status" -ne 3 ]; then',
        'echo "=== PRE-PUSH BLOCKED: push-staging exited $staging_status, so whether this '
        'push only stages a tip is unknown ==="',
        "exit 1",
        "fi",
    ]


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


def _select_in(
    repo: Path, declaration: str, changed: list[str]
) -> subprocess.CompletedProcess[str]:
    """Run a copy of CI's selection in a repository declaring ``declaration``.

    The script finds its repository from its own path, so the copy selects
    among ``repo``'s packages by ``repo``'s declaration.

    Args:
        repo: A repository holding ``apps/web`` and ``clients/probe``, each gated.
        declaration: The text of its ``scripts/make/project-roots.json``.
        changed: The changed paths the selection is asked about.

    Returns:
        The finished run, its output captured.
    """
    (repo / "scripts" / "make").mkdir(parents=True, exist_ok=True)
    script = repo / "scripts" / "ci_select_packages.py"
    shutil.copyfile(ROOT / "scripts" / "ci_select_packages.py", script)
    (repo / "scripts" / "make" / "project-roots.json").write_text(declaration, encoding="utf-8")
    for package in ("apps/web", "clients/probe"):
        (repo / package).mkdir(parents=True, exist_ok=True)
        (repo / package / "Makefile").write_text("check:\n\ttrue\n", encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script), *changed],
        capture_output=True,
        text=True,
        check=False,
    )


def test_ci_selects_by_the_roots_its_repository_declares_not_by_this_one_s(tmp_path: Path) -> None:
    """A repository declaring ``apps`` alone gates apps/web and not
    clients/probe, though ``clients`` is one of this repository's roots: a
    selection that read any roots but the declared ones selects otherwise."""
    changed = ["apps/web/x.py", "clients/probe/x.py"]
    run = _select_in(tmp_path, '{"nestedRoots": ["apps"]}\n', changed)
    assert (run.returncode, run.stdout, run.stderr) == (0, '["apps/web"]', "")


@pytest.mark.parametrize(
    "declaration",
    [
        '{"nestedRoots": "apps"}',
        '{"nestedRoots": ["apps/web"]}',
        '{"nestedRoots": [".."]}',
        '{"nestedRoots": [3]}',
        '{"roots": ["apps"]}',
        '{"nestedRoots": ["apps"], "extra": []}',
        '["apps"]',
    ],
)
def test_ci_refuses_a_declaration_that_is_not_a_list_of_directory_names(
    tmp_path: Path, declaration: str
) -> None:
    run = _select_in(tmp_path, declaration, ["apps/web/x.py"])
    declared = tmp_path / "scripts" / "make" / "project-roots.json"
    refusal = f'ValueError: {declared} is not {{"nestedRoots": [<directory name>, ...]}}'
    assert (run.returncode, run.stdout) == (1, "")
    assert run.stderr.rstrip("\n").split("\n")[-1] == refusal
