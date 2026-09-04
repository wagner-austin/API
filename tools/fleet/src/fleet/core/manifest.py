"""Working out which directories a project's build actually needs.

A PROJECT IS NOT SELF-CONTAINED, AND THE FIRST REAL DISPATCH PROVED IT. On
2026-09-04 ``tools/fleet`` was staged to sedona as a single directory, which
is what "dispatch a project" reads as and is not enough to build:

* ``pyproject.toml`` declares ``platform-core = { path = "../../libs/platform_core" }``.
  Poetry resolves that path at lock time, so a tree without it cannot produce
  a lockfile, let alone install from one.
* every Makefile's ``test`` target calls ``..\\..\\scripts\\run-tests.ps1``,
  the launcher all forty-one packages share.
* every ``scripts/guard.py`` inserts ``<root>/libs/monorepo_guards/src`` onto
  ``sys.path`` before importing the rules, because that package is a
  dependency of four packages and cannot be imported by the other thirty-seven.
* those rules then read ``monorepo-guards.toml`` from the root, and raise
  ``FileNotFoundError`` without it -- which is how the SECOND dispatch failed,
  having fixed the first two omissions.

So the unit of staging is a SET of repo-relative paths, and this module
computes it.

WHY THE PATH DEPENDENCIES ARE READ FROM ``pyproject.toml`` RATHER THAN
DECLARED IN THE WORKSPACE. A ``depends`` field beside each project in
``fleet.json`` would be simpler to write and would be wrong within a month:
poetry already reads the authoritative list every time anybody builds, and a
second copy drifts silently in the direction of staging too little -- which
surfaces as a lockfile error on a node and reads as the project's fault.

WHY THE TWO SHARED PATHS ARE CONSTANTS AND NOT DISCOVERED. Neither is
discoverable. ``scripts/`` is named by a Makefile recipe and
``libs/monorepo_guards`` by a hard-coded ``parents[3]`` inside a shim that is
byte-identical in all forty-one packages -- so the monorepo asserts both as
facts about its own layout, and mirroring them here is quoting that assertion,
not duplicating a source of truth. Both are checked to exist on every call,
so a rename fails a dispatch loudly instead of staging a tree that cannot
build.
"""

from __future__ import annotations

import pathlib
from typing import Final

from monorepo_guards.external_inputs import GUARD_CONFIG_NAME, external_inputs
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONValue
from platform_core.toml_utils import loads_toml

from fleet.core import _test_hooks

#: Repo-relative DIRECTORIES every dispatch carries, whatever the project.
#:
#: ``scripts`` holds ``run-tests.ps1`` and the reaper it calls;
#: ``libs/monorepo_guards`` holds the rules ``scripts/guard.py`` imports by
#: absolute path. See the module docstring for why these are quoted here
#: rather than discovered.
SHARED_DIRECTORIES: Final[tuple[str, ...]] = ("scripts", "libs/monorepo_guards")

#: Repo-relative FILES every dispatch must carry, whatever the project.
#:
#: Spelled by :mod:`monorepo_guards` rather than here. The guard config is read
#: from the monorepo root by
#: :func:`monorepo_guards.config_loader._decode_monorepo_guard_config`, which
#: raises ``FileNotFoundError`` when it is absent. Found the expensive way on
#: 2026-09-04: a dispatch that carried both shared directories staged, locked,
#: installed platform-core from its own staged path, and then failed on the
#: guard step -- because the rules had arrived and the file naming which of
#: them to run had not.
#:
#: A separate constant from :data:`SHARED_DIRECTORIES` rather than one list of
#: paths, because the two are CHECKED differently: a file where a directory is
#: expected is a rename that has to fail loudly, and a single existence check
#: would accept either and stage something that cannot build.
#:
#: :func:`guard_inputs` would carry this file anyway, but only if it is there --
#: it reports what exists. Requiring it here is what turns its absence into a
#: refusal instead of a failure on somebody else's machine.
SHARED_FILES: Final[tuple[str, ...]] = (GUARD_CONFIG_NAME,)

#: Everything a dispatch carries beyond the project and its dependencies.
SHARED_PATHS: Final[tuple[str, ...]] = (*SHARED_DIRECTORIES, *SHARED_FILES)

#: The file a project declares its dependencies in.
MANIFEST_NAME = "pyproject.toml"


def build_tree(project_root: pathlib.Path, project: str) -> tuple[str, ...]:
    """Name every repo-relative directory a dispatch of this project needs.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path.

    Returns:
        The project, its transitive path dependencies, and
        :data:`SHARED_PATHS` -- deduplicated, in a stable order, with the
        project first. Ordered rather than a set because the result names tar
        members, and an archive whose member order changed between runs would
        make two identical trees produce different digests.

    Raises:
        AppError: With ``PROJECT_MANIFEST_MISSING`` when the project or one of
            its declared dependencies has no manifest,
            ``PROJECT_MANIFEST_UNREADABLE`` when one cannot be read as poetry
            metadata, or ``PROJECT_DEPENDENCY_ESCAPES_ROOT`` when a declared
            path resolves outside the monorepo.
    """
    ordered = list(_walk(project_root, project))
    for directory in SHARED_DIRECTORIES:
        _require_shared(project_root, directory, kind="directory")
        if directory not in ordered:
            ordered.append(directory)
    for name in SHARED_FILES:
        _require_shared(project_root, name, kind="file")
    for name in guard_inputs(project_root):
        if not _within(ordered, name):
            ordered.append(name)
    return tuple(ordered)


def guard_inputs(project_root: pathlib.Path) -> tuple[str, ...]:
    """Name the files ``make check`` reads from outside any one project.

    ASKED OF :mod:`monorepo_guards`, NOT DERIVED HERE. Three guard rules
    resolve their declaring module from the monorepo root by design, and the
    config rule scans every package manifest under the category directories,
    so running the guards over one package is not a self-contained act. A
    second copy of that list living here would drift towards naming too
    little, and too little surfaces on a remote node as three
    ``*-declaration-unresolved`` failures that read as the project's fault.

    Found the expensive way on 2026-09-04: a dispatch carrying the project,
    its path dependencies and both shared directories got all the way through
    ``poetry sync`` and failed on exactly that.

    Args:
        project_root: Absolute path to the monorepo root.

    Returns:
        Repo-relative paths, sorted, each of which exists. Files rather than
        directories: the declaring modules and manifests are small and the
        packages that own them are not, so a dispatch of ``tools/fleet``
        carries three source files rather than an ML service.
    """
    root = project_root.resolve()
    return tuple(path.resolve().relative_to(root).as_posix() for path in external_inputs(root))


def _within(ordered: list[str], candidate: str) -> bool:
    """Whether a path is already covered by something being staged.

    Args:
        ordered: The members collected so far.
        candidate: The path to test.

    Returns:
        True when it is already listed, or lies under a directory that is.
        Checked because the guard inputs include manifests of packages a
        dispatch may already be carrying whole -- ``libs/monorepo_guards``
        always, and the project itself -- and naming a file twice would put
        it in the archive twice.
    """
    return any(candidate == member or candidate.startswith(f"{member}/") for member in ordered)


def _walk(project_root: pathlib.Path, project: str) -> tuple[str, ...]:
    """Follow path dependencies from one project until they stop.

    Iterative rather than recursive, and carrying its own seen-set, because
    path dependencies in a monorepo form a graph rather than a tree: two
    projects commonly share one library, and a diamond would otherwise be
    walked twice and staged twice.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path to start from.

    Returns:
        The project and everything reachable from it, discovery-ordered.

    Raises:
        AppError: As :func:`build_tree` describes.
    """
    ordered: list[str] = []
    pending = [project]
    while pending:
        current = pending.pop(0)
        if current in ordered:
            continue
        ordered.append(current)
        pending.extend(path_dependencies(project_root, current))
    return tuple(ordered)


def path_dependencies(project_root: pathlib.Path, project: str) -> tuple[str, ...]:
    """Read one project's directly declared path dependencies.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path.

    Returns:
        Repo-relative paths, in the order the manifest declares them, from
        the main dependency table and from every dependency group. Groups are
        included because ``make check`` installs them: the recipe is
        ``poetry sync --with dev``, so a dev-group path dependency is as
        required for a build as a runtime one.

    Raises:
        AppError: As :func:`build_tree` describes.
    """
    document = _read_manifest(project_root, project)
    found: list[str] = []
    for table in _dependency_tables(document, project=project):
        for name, declared in table.items():
            relative = _declared_path(declared)
            if relative is None:
                continue
            found.append(_resolve(project_root, project, relative, dependency=name))
    return tuple(found)


def _read_manifest(project_root: pathlib.Path, project: str) -> dict[str, JSONValue]:
    """Read and parse one project's manifest.

    Args:
        project_root: Absolute path to the monorepo root.
        project: Repo-relative project path.

    Returns:
        The parsed document.

    Raises:
        AppError: With ``PROJECT_MANIFEST_MISSING`` when there is no manifest
            at that path.
        TOMLDecodeError: If the manifest is not valid TOML. Propagated rather
            than translated -- the parser's message carries the line and
            column, which is the whole diagnostic, and a manifest that does
            not parse has already failed every local ``poetry lock`` long
            before a dispatch reads it.
    """
    path = project_root / project / MANIFEST_NAME
    if not _test_hooks.file_exists(path):
        raise AppError(
            FleetErrorCode.PROJECT_MANIFEST_MISSING,
            f"{project!r} has no {MANIFEST_NAME} at {path}; a dispatch stages what a manifest "
            "declares, so a project without one cannot be built on a node",
        )
    return loads_toml(_test_hooks.read_text(path))


def _dependency_tables(
    document: dict[str, JSONValue], *, project: str
) -> tuple[dict[str, JSONValue], ...]:
    """Find every table in a manifest that can declare a path dependency.

    Args:
        document: The parsed manifest.
        project: Repo-relative project path, for messages.

    Returns:
        The main dependency table followed by each group's, skipping those
        that are absent. An absent table is not an error: a project with no
        dev group is ordinary, and a library with no dependencies at all is
        the base case this whole walk terminates on.

    Raises:
        AppError: With ``PROJECT_MANIFEST_UNREADABLE`` when something that
            must be a table is not one.
    """
    poetry = _table(_table(document, "tool", project=project), "poetry", project=project)
    tables = [_table(poetry, "dependencies", project=project)]
    groups = _table(poetry, "group", project=project)
    for name, group in groups.items():
        if not isinstance(group, dict):
            raise AppError(
                FleetErrorCode.PROJECT_MANIFEST_UNREADABLE,
                f"{project}: dependency group {name!r} is a {type(group).__name__}, not a table",
            )
        tables.append(_table(group, "dependencies", project=project))
    return tuple(tables)


def _table(document: dict[str, JSONValue], key: str, *, project: str) -> dict[str, JSONValue]:
    """Read one nested table, treating an absent one as empty.

    Args:
        document: The table to read from.
        key: The key to read.
        project: Repo-relative project path, for messages.

    Returns:
        The nested table, or an empty one when the key is absent.

    Raises:
        AppError: With ``PROJECT_MANIFEST_UNREADABLE`` when the key is
            present and holds something other than a table. Absent and
            wrong-typed are different: the first is a project that does not
            use the feature, the second is a manifest nobody can build from.
    """
    value = document.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AppError(
            FleetErrorCode.PROJECT_MANIFEST_UNREADABLE,
            f"{project}: {key!r} is a {type(value).__name__}, not a table",
        )
    return value


def _declared_path(declared: JSONValue) -> str | None:
    """Read the ``path`` of one dependency declaration, if it has one.

    Args:
        declared: The value beside a dependency's name. A plain string is a
            version constraint from PyPI; a table may carry ``path``.

    Returns:
        The declared path, or None when this dependency is not a local one.
        A table whose ``path`` is not a string returns None as well, because
        poetry would reject that manifest long before a dispatch saw it and
        this module's job is to find local paths, not to re-validate poetry.
    """
    if not isinstance(declared, dict):
        return None
    path = declared.get("path")
    if not isinstance(path, str):
        return None
    return path


def _resolve(project_root: pathlib.Path, project: str, relative: str, *, dependency: str) -> str:
    """Turn a manifest-relative dependency path into a repo-relative one.

    Args:
        project_root: Absolute path to the monorepo root.
        project: The project whose manifest declared it.
        relative: The path as the manifest spells it, relative to that
            manifest's own directory.
        dependency: The dependency's name, for messages.

    Returns:
        The path relative to the monorepo root, with forward slashes, which
        is the spelling tar members and the rest of this package use.

    Raises:
        AppError: With ``PROJECT_DEPENDENCY_ESCAPES_ROOT`` when it resolves
            outside the monorepo. Refused rather than staged: the archive is
            built with the root as its base, so an outside path has no name
            that could be extracted on a node, and a dispatch that silently
            dropped it would fail resolving the lockfile instead.
    """
    resolved = (project_root / project / relative).resolve()
    root = project_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise AppError(
            FleetErrorCode.PROJECT_DEPENDENCY_ESCAPES_ROOT,
            f"{project} declares {dependency!r} at {relative!r}, which resolves to {resolved} "
            f"outside {root}; a dispatch stages paths relative to the monorepo root and has no "
            "name to give this one on a node",
        )
    return resolved.relative_to(root).as_posix()


def _require_shared(project_root: pathlib.Path, relative: str, *, kind: str) -> None:
    """Refuse a shared path the monorepo no longer has.

    Args:
        project_root: Absolute path to the monorepo root.
        relative: The repo-relative path that must exist.
        kind: ``directory`` or ``file``, deciding which check applies and
            what the message says. Checked rather than accepting either,
            because a rename that turns one into the other is exactly the
            change that would stage something unbuildable.

    Raises:
        AppError: With ``PROJECT_MANIFEST_MISSING`` when it is absent or is
            the other kind. These paths are named by a Makefile recipe, by a
            hard-coded ``parents[3]`` in the guard shim, and by the guard
            config loader -- none of which this package can see. Checking
            here means a rename fails the dispatch that would have staged an
            unbuildable tree, rather than failing on a node with a message
            that reads as the project's fault.
    """
    path = project_root / relative
    present = (
        _test_hooks.directory_exists(path) if kind == "directory" else _test_hooks.file_exists(path)
    )
    if not present:
        raise AppError(
            FleetErrorCode.PROJECT_MANIFEST_MISSING,
            f"every dispatch carries the {kind} {relative!r} and it is not under "
            f"{project_root}; the Makefiles, the guard shim and the guard config loader name "
            f"it by that path, so a rename has to be made in {__name__} too rather than "
            "discovered on a node",
        )


__all__ = [
    "MANIFEST_NAME",
    "SHARED_DIRECTORIES",
    "SHARED_FILES",
    "SHARED_PATHS",
    "build_tree",
    "guard_inputs",
    "path_dependencies",
]
