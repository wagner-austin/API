"""Checks verify a package's ``poetry.lock``; only ``lock`` writes it (MCPs board task 972ea464).

WHY. Every package's ``lint`` and ``test`` recipes used to open with
``poetry lock``, which re-resolves the whole dependency graph on every run
even when nothing changed. For ``libs/covenant_nn`` that means querying
torch's index: on the GitHub runner, API CI run 46048435508 at 773acfb75
spent 292 s and 99 s in the two resolutions against a 26 s suite, and the
check overran its 300 s budget by 154 s. A check re-resolving also repairs
a stale lock where nobody sees it, so a commit's lock and the lock its
check tested could differ.

WHY ``poetry check --lock`` ALONE IS NOT ENOUGH. Poetry's freshness test
compares the lock's ``content-hash`` with the package's OWN
``pyproject.toml``. A path dependency's declarations are recorded in the
dependent's lock, but not in that hash: on 2026-11-27 a scratch copy of
``libs/covenant_ml`` given a new dependency left ``poetry check --lock`` in
``libs/covenant_nn`` at exit 0, and ``covenant_nn``'s committed lock still
recorded ``libs/cleargbm_rs``'s dev extras from before that package's
toolchain bump. Every re-resolving check had been papering over it.

SO A SECOND RECORD. ``paths-lock.json`` beside the lock holds, for every
directory package the lock names (the path dependencies, transitively), the
sha256 of the declarations a dependent's resolution reads from that
package's ``pyproject.toml``: its name, version, Python range, main
dependencies and extras (:data:`PROJECT_KEYS`, :data:`POETRY_KEYS`). A dev
group, a tool's configuration or a reordering leaves the digest alone,
because none of them changes what the dependent resolves.

TWO COMMANDS.

``lock-check`` runs in the recipes in place of ``poetry lock``. It
resolves nothing: poetry judges the package's own declarations, this module
judges its path dependencies' against the record, and either refusal is
``MAKETOOLS_LOCK_STALE`` naming what moved and the command that fixes it.

``lock`` is that command: ``poetry lock``, then the record rewritten from
the declarations the new lock was resolved against, then ``lock-check``
over what it wrote. It is the only writer of the record, so every digest
in one was computed from the files it names, never typed in.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
    narrow_json_to_str,
    require_dict,
    require_str,
)
from platform_core.toml_utils import loads_toml

from maketools import _test_hooks
from maketools.venvs import INSTALL_WALL_SECONDS

#: The record's file name, beside the package's ``poetry.lock``.
PATHS_LOCK_NAME: Final[str] = "paths-lock.json"

#: The lock poetry writes and this module reads.
POETRY_LOCK_NAME: Final[str] = "poetry.lock"

#: Every package's declarations file.
PYPROJECT_NAME: Final[str] = "pyproject.toml"

#: Poetry's own freshness test: the package's declarations against the lock's hash.
CHECK_ARGV: Final[tuple[str, ...]] = ("poetry", "check", "--lock")

#: The resolution ``lock`` runs.
LOCK_ARGV: Final[tuple[str, ...]] = ("poetry", "lock")

#: The ``[project]`` keys a dependent's resolution of this package reads.
PROJECT_KEYS: Final[tuple[str, ...]] = (
    "name",
    "version",
    "requires-python",
    "dependencies",
    "optional-dependencies",
)

#: The ``[tool.poetry]`` keys a dependent's resolution of this package reads.
POETRY_KEYS: Final[tuple[str, ...]] = ("name", "version", "dependencies", "extras")

#: The lock's source type for a path dependency.
DIRECTORY_SOURCE: Final[str] = "directory"

#: What a refusal tells the reader to run.
FIX: Final[str] = (
    "run tools/maketools/scripts/run.py lock in the package and commit "
    f"{POETRY_LOCK_NAME} and {PATHS_LOCK_NAME}"
)


class PathsLock(TypedDict):
    """The record of the path dependencies a lock was resolved against.

    Attributes:
        members: Each directory package's url as the lock spells it
            (relative to the package), mapped to the hex sha256 of its
            declarations (:func:`declaration_digest`).
    """

    members: dict[str, str]


def decode_paths_lock(text: str) -> PathsLock:
    """Read the record's text.

    Args:
        text: The file's contents.

    Returns:
        The record.

    Raises:
        InvalidJsonError: When the text is not JSON.
        JSONTypeError: When it is not an object whose ``members`` is an
            object of strings.
    """
    members = require_dict(narrow_json_to_dict(load_json_str(text)), "members")
    return PathsLock(members={url: narrow_json_to_str(digest) for url, digest in members.items()})


def encode_paths_lock(record: PathsLock) -> str:
    """Write the record's text, as :func:`decode_paths_lock` reads it.

    Args:
        record: The record.

    Returns:
        Indented JSON, members in url order, with a final newline.
    """
    members: dict[str, JSONValue] = {
        url: record["members"][url] for url in sorted(record["members"])
    }
    document: dict[str, JSONValue] = {"members": members}
    return dump_json_str(document, indent=2) + "\n"


def lock_members(lock_text: str) -> tuple[str, ...]:
    """The path dependencies a lock names.

    Args:
        lock_text: A ``poetry.lock``'s contents.

    Returns:
        The url of every package whose source is a directory, sorted.

    Raises:
        tomllib.TOMLDecodeError: When the text is not TOML.
        JSONTypeError: When ``package`` is not an array of tables, or a
            source is not a table with a string type and url.
    """
    found: list[str] = []
    for value in narrow_json_to_list(loads_toml(lock_text).get("package", [])):
        source = narrow_json_to_dict(value).get("source")
        if source is None:
            continue
        table = narrow_json_to_dict(source)
        if require_str(table, "type") == DIRECTORY_SOURCE:
            found.append(require_str(table, "url"))
    return tuple(sorted(found))


def _canonical(value: JSONValue) -> JSONValue:
    """The value with every object's keys in sorted order, recursively.

    Args:
        value: A decoded TOML or JSON value.

    Returns:
        The same value, ordered so its JSON text depends only on its content.
    """
    if isinstance(value, dict):
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _subset(table: JSONValue, keys: Sequence[str]) -> dict[str, JSONValue]:
    """The named keys a table declares.

    Args:
        table: A ``pyproject.toml`` table, or None when it is absent.
        keys: The keys to keep.

    Returns:
        Each named key the table holds, with its value.

    Raises:
        JSONTypeError: When the table is present and not a table.
    """
    if table is None:
        return {}
    declared = narrow_json_to_dict(table)
    return {key: declared[key] for key in keys if key in declared}


def declaration(pyproject_text: str) -> dict[str, JSONValue]:
    """The declarations a dependent's resolution reads from a package.

    Args:
        pyproject_text: The package's ``pyproject.toml``.

    Returns:
        ``project`` and ``tool.poetry`` holding the package's
        :data:`PROJECT_KEYS` and :data:`POETRY_KEYS`, each empty when the
        package declares none.

    Raises:
        tomllib.TOMLDecodeError: When the text is not TOML.
        JSONTypeError: When ``project``, ``tool`` or ``tool.poetry`` is
            present and not a table.
    """
    document = loads_toml(pyproject_text)
    tool = document.get("tool")
    poetry = None if tool is None else narrow_json_to_dict(tool).get("poetry")
    return {
        "project": _subset(document.get("project"), PROJECT_KEYS),
        "tool.poetry": _subset(poetry, POETRY_KEYS),
    }


def declaration_digest(pyproject_text: str) -> str:
    """The digest the record holds for one path dependency.

    Args:
        pyproject_text: The dependency's ``pyproject.toml``.

    Returns:
        The lower-case hex sha256 of its :func:`declaration`'s canonical JSON.
    """
    canonical = dump_json_str(_canonical(declaration(pyproject_text)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_paths(project: Path) -> PathsLock:
    """The record a lock of the package would carry if it were written now.

    Args:
        project: The package directory, holding ``poetry.lock``.

    Returns:
        Every path dependency the lock names, with the digest of its
        declarations as they stand.

    Raises:
        FileNotFoundError: When the package has no lock, or a path
            dependency the lock names has no ``pyproject.toml``.
    """
    lock_text = (project / POETRY_LOCK_NAME).read_text(encoding="utf-8")
    return PathsLock(
        members={
            url: declaration_digest((project / url / PYPROJECT_NAME).read_text(encoding="utf-8"))
            for url in lock_members(lock_text)
        }
    )


def stale_paths(recorded: PathsLock, current: PathsLock) -> list[str]:
    """Every difference between the record and the declarations as they stand.

    Args:
        recorded: What ``paths-lock.json`` holds.
        current: :func:`current_paths`.

    Returns:
        One line per path dependency that changed, appeared or went, in
        url order; empty when the record is current.
    """
    found: list[str] = []
    for url in sorted(recorded["members"].keys() | current["members"].keys()):
        before = recorded["members"].get(url)
        now = current["members"].get(url)
        if before is None:
            found.append(f"{url}: a path dependency the record does not name")
        elif now is None:
            found.append(f"{url}: recorded, but the lock no longer names it")
        elif before != now:
            found.append(f"{url}: its declared dependencies changed since the lock was written")
    return found


def _poetry_refusal(output: str, returncode: int) -> str:
    """What poetry said when it refused the lock.

    Args:
        output: Its standard output and error, joined.
        returncode: Its exit status.

    Returns:
        Its ``Error:`` lines, or its whole output under its exit status
        when it printed none.
    """
    errors = [line.strip() for line in output.splitlines() if line.startswith("Error:")]
    if errors:
        return " ".join(errors)
    return f"poetry check --lock exited {returncode}: {output.strip()}"


def check_lock(project: Path) -> None:
    """Verify the package's committed lock without resolving anything.

    Args:
        project: The package directory.

    Raises:
        AppError: ``MAKETOOLS_LOCK_STALE`` when poetry refuses the lock
            against the package's own declarations, the package has no
            ``paths-lock.json``, or the record differs from its path
            dependencies' declarations, naming each.
    """
    result = _test_hooks.run_capturing(CHECK_ARGV, cwd=project)
    if result["returncode"] != 0:
        refusal = _poetry_refusal(result["stdout"] + result["stderr"], result["returncode"])
        raise AppError(MaketoolsErrorCode.LOCK_STALE, f"{project.name}: {refusal}; {FIX}")
    record = project / PATHS_LOCK_NAME
    if not record.is_file():
        raise AppError(
            MaketoolsErrorCode.LOCK_STALE, f"{project.name}: no {PATHS_LOCK_NAME}; {FIX}"
        )
    current = current_paths(project)
    stale = stale_paths(decode_paths_lock(record.read_text(encoding="utf-8")), current)
    if stale:
        raise AppError(
            MaketoolsErrorCode.LOCK_STALE,
            f"{project.name}: {POETRY_LOCK_NAME} was resolved against other path "
            f"dependencies than these: {'; '.join(stale)}; {FIX}",
        )
    urls = ", ".join(current["members"])
    _test_hooks.write_line(
        f"lock-check: {POETRY_LOCK_NAME} is current for {PYPROJECT_NAME} and its "
        f"{len(current['members'])} path dependencies ({urls})"
    )


def write_lock(project: Path) -> int:
    """Resolve the package's lock, record its path dependencies, and verify both.

    Args:
        project: The package directory.

    Returns:
        ``poetry lock``'s exit status when it failed, with nothing recorded;
        else 0, after :func:`check_lock` has accepted what was written.

    Raises:
        AppError: ``MAKETOOLS_LOCK_STALE`` when the written lock and record
            do not verify.
        subprocess.TimeoutExpired: When the resolution outlives
            :data:`maketools.venvs.INSTALL_WALL_SECONDS`.
    """
    code = _test_hooks.run_inheriting(
        LOCK_ARGV,
        cwd=project,
        env=_test_hooks.environ(),
        new_session=False,
        timeout_seconds=INSTALL_WALL_SECONDS,
    )
    if code != 0:
        return code
    (project / PATHS_LOCK_NAME).write_text(
        encode_paths_lock(current_paths(project)), encoding="utf-8"
    )
    check_lock(project)
    return 0


__all__ = [
    "CHECK_ARGV",
    "DIRECTORY_SOURCE",
    "FIX",
    "LOCK_ARGV",
    "PATHS_LOCK_NAME",
    "POETRY_KEYS",
    "POETRY_LOCK_NAME",
    "PROJECT_KEYS",
    "PYPROJECT_NAME",
    "PathsLock",
    "check_lock",
    "current_paths",
    "declaration",
    "declaration_digest",
    "decode_paths_lock",
    "encode_paths_lock",
    "lock_members",
    "stale_paths",
    "write_lock",
]
