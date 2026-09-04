"""Decide which packages a change requires CI to check.

WHY THIS EXISTS. Forty-three packages in this monorepo carry a ``check``
target, and until 2026-09-04 exactly TWO of them had that target run by CI --
``services/turkic-api`` and ``clients/NavProbe``. ``libs/platform_core``
appeared in a workflow path filter, but only as a TRIGGER for turkic-api's
suite; its own 875 tests ran nowhere. That is the same shape as the incident
recorded in the header of ``navprobe.yml``: a gate that sat unrun because
nothing executed it, discovered only when somebody finally looked.

One workflow with a matrix covers all forty-three, but a matrix cannot be
path-filtered per entry -- so this computes the entry list instead.

DEPENDENCY-AWARE ON PURPOSE. Selecting only the package whose own directory
changed would miss the case that matters most: every one of the forty-three
takes ``platform_core`` or ``monorepo_guards`` as a path dependency, so a
change there can break a package that was not touched. Each package's
``pyproject.toml`` declares those deps as
``name = { path = "../../libs/x", develop = true }``, which is machine
readable, so the reverse graph is derived rather than maintained by hand --
the same reasoning that removed the hardcoded project lists from
``tools/hpc3/tests/test_committed_runs.py``.

WHAT AN UNOWNED FILE SELECTS, and why it is not "everything". Only the paths
in :data:`GLOBAL_PATHS` fan out to all forty-three: the guard configuration
every package is checked against, the shared runner scripts, and this
workflow. Any other file no package owns -- a wiki page, a README, a doc --
selects NOTHING, because it is by construction not part of any package's
build. The alternative was tried on paper and rejected: this repository
commits documentation constantly, and making one typo run forty-three suites,
several of which install torch, buys nothing and would train people to ignore
the result.

THE GAP THAT LEAVES, stated rather than hidden: ``tools/hpc3/tests/
test_committed_runs.py`` reads ``docs/RESEARCH.md``, so a change to that file
alone can break hpc3's suite without selecting it. That is a real hole. It is
bounded -- the next change under ``tools/hpc3`` catches it -- and closing it
properly means a hand-maintained map of which package reads which document,
which is the kind of list this repository has spent the day deleting.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

#: Directories whose immediate children are candidate packages.
PACKAGE_ROOTS = ("clients", "services", "libs", "tools")

#: A package is one this repository gates, which is what a `check` target means.
CHECK_TARGET = re.compile(r"^check:", re.MULTILINE)

#: Poetry path dependency, e.g. `platform-core = { path = "../../libs/platform_core" }`.
PATH_DEPENDENCY = re.compile(r"""path\s*=\s*["']([^"']+)["']""")

#: Changes here can affect any package, so they select all of them.
GLOBAL_PATHS = ("monorepo-guards.toml", "scripts/", ".github/workflows/packages.yml")


def gated_packages(repo: pathlib.Path) -> list[str]:
    """Find every package this repository gates.

    Args:
        repo: Repository root.

    Returns:
        Repo-relative package directories carrying a ``check`` target, sorted.
        Membership is derived from the Makefile rather than from a list, so a
        new package joins CI by having a gate rather than by being added here.
    """
    found: list[str] = []
    for root in PACKAGE_ROOTS:
        for makefile in sorted((repo / root).glob("*/Makefile")):
            if CHECK_TARGET.search(makefile.read_text(encoding="utf-8", errors="replace")):
                found.append(makefile.parent.relative_to(repo).as_posix())
    return found


def direct_dependencies(repo: pathlib.Path, package: str) -> list[str]:
    """Read the in-repo packages a package depends on.

    Args:
        repo: Repository root.
        package: Repo-relative package directory.

    Returns:
        Repo-relative directories this package takes as path dependencies.
        Paths escaping the repository are dropped rather than raising: a
        dependency outside the tree cannot be a CI target, and refusing the
        whole run over one is worse than ignoring it.
    """
    manifest = repo / package / "pyproject.toml"
    if not manifest.is_file():
        return []
    resolved: list[str] = []
    for raw in PATH_DEPENDENCY.findall(manifest.read_text(encoding="utf-8", errors="replace")):
        candidate = (repo / package / raw).resolve()
        try:
            resolved.append(candidate.relative_to(repo.resolve()).as_posix())
        except ValueError:
            continue
    return resolved


def dependents_of(packages: list[str], repo: pathlib.Path) -> dict[str, set[str]]:
    """Build the reverse dependency graph.

    Args:
        packages: Every gated package.
        repo: Repository root.

    Returns:
        For each package, the packages that depend on it DIRECTLY. Transitive
        closure is taken later by :func:`select`, so this stays a plain edge
        map that a reader can check against one ``pyproject.toml``.
    """
    reverse: dict[str, set[str]] = {name: set() for name in packages}
    for name in packages:
        for dependency in direct_dependencies(repo, name):
            if dependency in reverse:
                reverse[dependency].add(name)
    return reverse


def owning_package(changed: str, packages: list[str]) -> str | None:
    """Find the package a changed file belongs to.

    Args:
        changed: Repo-relative path of a changed file.
        packages: Every gated package.

    Returns:
        The longest package prefix containing the file, or ``None`` when the
        file belongs to no package. Longest wins so a nested package is
        preferred over its parent.
    """
    owners = [name for name in packages if changed == name or changed.startswith(f"{name}/")]
    if owners == []:
        return None
    return max(owners, key=len)


def select(changed: list[str], packages: list[str], reverse: dict[str, set[str]]) -> list[str]:
    """Choose the packages CI must check for a set of changed files.

    Args:
        changed: Repo-relative paths of changed files.
        packages: Every gated package.
        reverse: Direct reverse dependency edges.

    Returns:
        Packages to check, sorted. A change to one of :data:`GLOBAL_PATHS`
        selects every package; a change to any other file no package owns
        selects none -- see the module docstring for why, and for the one
        gap that leaves.
    """
    selected: set[str] = set()
    for path in changed:
        if any(path == entry or path.startswith(entry) for entry in GLOBAL_PATHS):
            return sorted(packages)
        owner = owning_package(path, packages)
        if owner is not None:
            selected.add(owner)

    # Transitive closure over dependents: a change to platform_core must reach
    # everything that imports it, not only its immediate neighbours.
    pending = list(selected)
    while pending:
        current = pending.pop()
        for dependent in reverse.get(current, set()):
            if dependent not in selected:
                selected.add(dependent)
                pending.append(dependent)
    return sorted(selected)


def main(argv: list[str]) -> int:
    """Print the JSON matrix of packages to check.

    Args:
        argv: Changed file paths, repo-relative. Reads standard input, one
            path per line, when none are given.

    Returns:
        Exit code 0. The selection is written to standard output as a JSON
        array, which is what a workflow ``matrix`` consumes.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    changed = argv if argv else [line.strip() for line in sys.stdin if line.strip() != ""]
    packages = gated_packages(repo)
    sys.stdout.write(json.dumps(select(changed, packages, dependents_of(packages, repo))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
