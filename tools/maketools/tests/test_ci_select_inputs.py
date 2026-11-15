"""CI's selection answers every gated package's inputs (MCPs board task 05916ee9).

MCPs' push gate keys a kept fleet pass by a digest of the commit's entries
under these paths, so a path the selection checks a package for must be one
of the package's inputs, or a kept pass would stand for a change it never saw.
Each case runs a copy of ``scripts/ci_select_packages.py`` in a scratch
repository: ``web`` depends on ``core``, ``core`` on ``base``, and ``probe``
on nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
    require_str_list,
)

from tests._ci_selection_world import ci_repository, run_selection

PACKAGES: Final[dict[str, tuple[str, ...]]] = {
    "apps/web": ("libs/core",),
    "libs/core": ("libs/base",),
    "libs/base": (),
    "clients/probe": (),
}
DECLARATION: Final[str] = '{"nestedRoots": ["apps", "libs", "clients"]}\n'
GLOBALS: Final[list[str]] = [
    "monorepo-guards.toml",
    "scripts/",
    "tools/maketools/",
    ".github/workflows/packages.yml",
]

#: Paths a change could touch: each package's own file, the global paths and
#: two files no package owns.
CHANGES: Final[tuple[str, ...]] = (
    "apps/web/src/a.py",
    "libs/core/src/b.py",
    "libs/base/src/c.py",
    "clients/probe/src/d.py",
    "monorepo-guards.toml",
    "scripts/make/shell.mk",
    "tools/maketools/src/e.py",
    ".github/workflows/packages.yml",
    "docs/README.md",
    "wiki/page.md",
)


def _inputs(script: Path) -> dict[str, list[str]]:
    """The ``--inputs`` answer of a scratch repository's selection."""
    run = run_selection(script, ["--inputs"])
    assert (run.returncode, run.stderr) == (0, "")
    answer = narrow_json_to_dict(load_json_str(run.stdout))
    return {package: require_str_list(answer, package) for package in answer}


def test_a_package_reads_itself_its_dependencies_transitively_and_the_global_paths(
    tmp_path: Path,
) -> None:
    assert _inputs(ci_repository(tmp_path, DECLARATION, PACKAGES)) == {
        "apps/web": ["apps/web/", "libs/base/", "libs/core/", *GLOBALS],
        "clients/probe": ["clients/probe/", *GLOBALS],
        "libs/base": ["libs/base/", *GLOBALS],
        "libs/core": ["libs/base/", "libs/core/", *GLOBALS],
    }


@pytest.mark.parametrize("changed", CHANGES)
def test_every_package_a_change_selects_reads_the_changed_path(
    tmp_path: Path, changed: str
) -> None:
    script = ci_repository(tmp_path, DECLARATION, PACKAGES)
    run = run_selection(script, [changed])
    answer = narrow_json_to_list(load_json_str(run.stdout))
    selected = [narrow_json_to_str(entry) for entry in answer]
    inputs = _inputs(script)

    reading = sorted(
        package
        for package, paths in inputs.items()
        if any(changed == path or changed.startswith(path) for path in paths)
    )
    assert selected == reading
