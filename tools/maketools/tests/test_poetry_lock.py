"""A check verifies a package's committed lock and never re-resolves it (MCPs board task 972ea464).

Expected digests are the sha256 of canonical JSON written out by hand here
(keys sorted, no whitespace), never copied from what the module printed.
The invariants are tested over every member of a generated family: each
irrelevant edit leaves the digest alone, each relevant key moves it, and
every record round-trips.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    InvalidJsonError,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)

from maketools.poetry_lock import (
    CHECK_ARGV,
    FIX,
    LOCK_ARGV,
    POETRY_KEYS,
    PROJECT_KEYS,
    PathsLock,
    check_lock,
    current_paths,
    declaration,
    declaration_digest,
    decode_paths_lock,
    encode_paths_lock,
    lock_members,
    stale_paths,
    write_lock,
)
from maketools.venvs import INSTALL_WALL_SECONDS
from tests.conftest import World, failed, ok

#: A lock naming a registry package, a legacy-index package and two path
#: dependencies, the second listed first, as poetry orders by name.
LOCK = """[[package]]
name = "a"
version = "0.1.0"

[package.source]
type = "directory"
url = "../b"

[[package]]
name = "numpy"
version = "2.3.0"

[[package]]
name = "torch"
version = "2.9.0"

[package.source]
type = "legacy"
url = "https://download.pytorch.org/whl/cu130"

[[package]]
name = "z"
version = "0.1.0"

[package.source]
type = "directory"
url = "../a"

[metadata]
content-hash = "0"
"""

#: A path dependency declaring every key a dependent's resolution reads,
#: in both tables, among keys it does not.
MEMBER = """[project]
name = "a"
version = "0.1.0"
description = "not read by a dependent's resolution"
requires-python = ">=3.11"
dependencies = ["numpy>=2"]
optional-dependencies = { gpu = ["torch"] }

[tool.poetry]
name = "a"
version = "0.1.0"
packages = [{ include = "a", from = "src" }]
dependencies = { python = "^3.11", numpy = "^2.3" }
extras = { gpu = ["torch"] }

[tool.poetry.group.dev.dependencies]
pytest = "^9.2.0"

[tool.ruff]
line-length = 100

[build-system]
requires = ["poetry-core>=2.0"]
"""

#: :data:`MEMBER`'s declaration, written out by hand.
MEMBER_DECLARATION: dict[str, JSONValue] = {
    "project": {
        "name": "a",
        "version": "0.1.0",
        "requires-python": ">=3.11",
        "dependencies": ["numpy>=2"],
        "optional-dependencies": {"gpu": ["torch"]},
    },
    "tool.poetry": {
        "name": "a",
        "version": "0.1.0",
        "dependencies": {"python": "^3.11", "numpy": "^2.3"},
        "extras": {"gpu": ["torch"]},
    },
}


#: :data:`MEMBER_DECLARATION` as canonical JSON, keys sorted at every level, by hand.
MEMBER_CANONICAL = (
    '{"project":{"dependencies":["numpy>=2"],"name":"a",'
    '"optional-dependencies":{"gpu":["torch"]},"requires-python":">=3.11","version":"0.1.0"},'
    '"tool.poetry":{"dependencies":{"numpy":"^2.3","python":"^3.11"},'
    '"extras":{"gpu":["torch"]},"name":"a","version":"0.1.0"}}'
)

#: The declaration of a package that declares neither table, by hand.
EMPTY_CANONICAL = '{"project":{},"tool.poetry":{}}'


def sha256(text: str) -> str:
    """The hex sha256 of a text's UTF-8 bytes.

    Args:
        text: Canonical JSON written out by hand.

    Returns:
        Its digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: :data:`MEMBER`'s digest, from its hand-written canonical JSON.
MEMBER_DIGEST = sha256(MEMBER_CANONICAL)


def package_tree(tmp_path: Path) -> Path:
    """A package whose lock names ``../a`` and ``../b``, both declared as :data:`MEMBER`.

    Args:
        tmp_path: The test's directory.

    Returns:
        The package directory.
    """
    project = tmp_path / "pkg"
    project.mkdir()
    (project / "poetry.lock").write_text(LOCK, encoding="utf-8")
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "pyproject.toml").write_text(MEMBER, encoding="utf-8")
    return project


def recorded(project: Path, members: dict[str, str]) -> None:
    """Write a record by hand.

    Args:
        project: The package directory.
        members: What the record holds.
    """
    document: dict[str, JSONValue] = {"members": {url: members[url] for url in members}}
    (project / "paths-lock.json").write_text(dump_json_str(document, indent=2), encoding="utf-8")


def members_of(text: str) -> dict[str, JSONValue]:
    """The ``members`` object of a record's text, in the order the text holds it.

    Args:
        text: The record's text.

    Returns:
        Its members.
    """
    return require_dict(narrow_json_to_dict(load_json_str(text)), "members")


def test_lock_members_are_the_directory_sources_in_url_order() -> None:
    assert lock_members(LOCK) == ("../a", "../b")


def test_a_lock_with_no_packages_names_no_members() -> None:
    assert lock_members("[metadata]\ncontent-hash = '0'\n") == ()


@pytest.mark.parametrize("kind", ["artifact", "custom", "dir", "directory-like", "file", "url"])
def test_only_the_directory_source_type_names_a_path_dependency(kind: str) -> None:
    # Values on either side of "directory" in sort order: the type is matched
    # exactly, never by ordering.
    lock = f'[[package]]\nname = "a"\n[package.source]\ntype = "{kind}"\nurl = "../a"\n'
    assert lock_members(lock) == ()


def test_a_directory_source_without_a_url_is_refused() -> None:
    with pytest.raises(JSONTypeError, match="Missing required field 'url'"):
        lock_members('[[package]]\nname = "a"\n[package.source]\ntype = "directory"\n')


def test_the_declaration_is_the_keys_a_dependent_reads() -> None:
    assert declaration(MEMBER) == MEMBER_DECLARATION
    assert PROJECT_KEYS == (
        "name",
        "version",
        "requires-python",
        "dependencies",
        "optional-dependencies",
    )
    assert POETRY_KEYS == ("name", "version", "dependencies", "extras")


def test_a_package_declaring_neither_table_declares_nothing() -> None:
    assert declaration("[tool.ruff]\nline-length = 100\n") == {"project": {}, "tool.poetry": {}}
    assert declaration_digest("") == sha256(EMPTY_CANONICAL)


def test_the_digest_is_of_the_canonical_json_written_by_hand() -> None:
    assert declaration_digest(MEMBER) == MEMBER_DIGEST


#: Edits a dependent's resolution does not read, each with the text it replaces.
IRRELEVANT = (
    ('description = "not read', 'description = "still not read'),
    ('pytest = "^9.2.0"', 'pytest = "^9.3.0"\nhypothesis = "^6"'),
    ("line-length = 100", "line-length = 120"),
    ('requires = ["poetry-core>=2.0"]', 'requires = ["poetry-core>=2.4"]'),
    ('packages = [{ include = "a", from = "src" }]', 'packages = [{ include = "b" }]'),
    (
        'dependencies = { python = "^3.11", numpy = "^2.3" }',
        'dependencies = { numpy = "^2.3", python = "^3.11" }',
    ),
)


@pytest.mark.parametrize(("before", "after"), IRRELEVANT)
def test_an_edit_no_dependent_reads_leaves_the_digest(before: str, after: str) -> None:
    assert before in MEMBER
    assert declaration_digest(MEMBER.replace(before, after)) == declaration_digest(MEMBER)


#: One edit per key a dependent's resolution reads, in each table.
RELEVANT = (
    ('[project]\nname = "a"', '[project]\nname = "a2"'),
    ('[project]\nname = "a"\nversion = "0.1.0"', '[project]\nname = "a"\nversion = "0.2.0"'),
    ('requires-python = ">=3.11"', 'requires-python = ">=3.12"'),
    ('dependencies = ["numpy>=2"]', 'dependencies = ["numpy>=2", "six"]'),
    ('optional-dependencies = { gpu = ["torch"] }', "optional-dependencies = {}"),
    ('[tool.poetry]\nname = "a"', '[tool.poetry]\nname = "a3"'),
    ('name = "a"\nversion = "0.1.0"\npackages', 'name = "a"\nversion = "0.3.0"\npackages'),
    ('numpy = "^2.3" }', 'numpy = "^2.4" }'),
    ('extras = { gpu = ["torch"] }', 'extras = { gpu = ["torch", "triton"] }'),
)


def test_every_key_read_has_an_edit_here() -> None:
    assert len(RELEVANT) == len(PROJECT_KEYS) + len(POETRY_KEYS)


@pytest.mark.parametrize(("before", "after"), RELEVANT)
def test_an_edit_a_dependent_reads_moves_the_digest(before: str, after: str) -> None:
    assert MEMBER.count(before) == 1
    assert declaration_digest(MEMBER.replace(before, after)) != declaration_digest(MEMBER)


#: Records over every subset of three members.
RECORDS = [
    PathsLock(members={url: f"{index:064x}" for index, url in enumerate(subset)})
    for size in range(4)
    for subset in itertools.combinations(("../b", "../../libs/a", "../a"), size)
]


@pytest.mark.parametrize("record", RECORDS)
def test_every_record_round_trips_in_url_order(record: PathsLock) -> None:
    text = encode_paths_lock(record)
    assert decode_paths_lock(text) == record
    assert list(members_of(text)) == sorted(record["members"])
    assert text.endswith("}\n")


@pytest.mark.parametrize(
    ("text", "error", "message"),
    [
        ("{}", JSONTypeError, "Missing required field 'members'"),
        ('{"members": []}', JSONTypeError, "Field 'members' must be an object"),
        ('{"members": {"../a": 1}}', JSONTypeError, "Expected JSON string, got int"),
        ("[]", JSONTypeError, "Expected JSON object, got list"),
        ("{", InvalidJsonError, "Invalid JSON payload"),
    ],
)
def test_decode_refuses_what_encode_never_writes(
    text: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        decode_paths_lock(text)


def test_current_paths_digests_each_member_the_lock_names(tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    assert current_paths(project) == PathsLock(
        members={"../a": MEMBER_DIGEST, "../b": MEMBER_DIGEST}
    )


def test_stale_paths_names_each_change_in_url_order() -> None:
    before = PathsLock(members={"../a": "1", "../b": "2", "../c": "3"})
    after = PathsLock(members={"../a": "1", "../b": "9", "../d": "4"})
    assert stale_paths(before, before) == []
    assert stale_paths(before, after) == [
        "../b: its declared dependencies changed since the lock was written",
        "../c: recorded, but the lock no longer names it",
        "../d: a path dependency the record does not name",
    ]


def test_a_current_lock_passes_resolving_nothing(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    recorded(project, {"../a": MEMBER_DIGEST, "../b": MEMBER_DIGEST})
    world.capturing_answers[CHECK_ARGV] = ok("Warning: [tool.poetry.name] is deprecated.\n")
    check_lock(project)
    assert world.captured == [CHECK_ARGV]
    assert world.inheriting_calls == []
    assert world.lines == [
        "lock-check: poetry.lock is current for pyproject.toml and its 2 path dependencies "
        "(../a, ../b)"
    ]


def stale(project: Path) -> str:
    """The refusal :func:`check_lock` raises for the package.

    Args:
        project: The package directory.

    Returns:
        Its message, after asserting its code.
    """
    with pytest.raises(AppError) as caught:
        check_lock(project)
    assert caught.value.code is MaketoolsErrorCode.LOCK_STALE
    return caught.value.message


def test_poetrys_refusal_is_named_by_its_error_lines(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = failed(
        1,
        "Warning: [tool.poetry.name] is deprecated.\n"
        "Error: pyproject.toml changed significantly since poetry.lock was last generated. "
        "Run `poetry lock` to fix the lock file.\n",
    )
    assert stale(project) == (
        "pkg: Error: pyproject.toml changed significantly since poetry.lock was last "
        f"generated. Run `poetry lock` to fix the lock file.; {FIX}"
    )


def test_error_lines_on_either_stream_are_all_named(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = {
        "returncode": 1,
        "stdout": "Error: one\n",
        "stderr": "Error: two\n",
    }
    assert stale(project) == f"pkg: Error: one Error: two; {FIX}"


def test_a_refusal_without_error_lines_carries_its_status_and_output(
    world: World, tmp_path: Path
) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = failed(2, "  poetry: no such command  \n")
    assert stale(project) == f"pkg: poetry check --lock exited 2: poetry: no such command; {FIX}"


def test_a_check_killed_by_a_signal_is_a_refusal(world: World, tmp_path: Path) -> None:
    # subprocess reports a child ended by signal N as exit status -N.
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = failed(-9, "")
    assert stale(project) == f"pkg: poetry check --lock exited -9: ; {FIX}"


def test_a_package_without_a_record_is_refused(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = ok("")
    assert stale(project) == f"pkg: no paths-lock.json; {FIX}"
    assert world.lines == []


def test_a_member_whose_declarations_moved_is_refused_by_name(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    recorded(project, {"../a": MEMBER_DIGEST, "../b": MEMBER_DIGEST})
    (tmp_path / "b" / "pyproject.toml").write_text(
        MEMBER.replace('dependencies = ["numpy>=2"]', 'dependencies = ["numpy>=2", "six"]'),
        encoding="utf-8",
    )
    world.capturing_answers[CHECK_ARGV] = ok("")
    assert stale(project) == (
        "pkg: poetry.lock was resolved against other path dependencies than these: "
        f"../b: its declared dependencies changed since the lock was written; {FIX}"
    )
    assert world.lines == []


def test_the_fix_names_the_writer_and_both_files() -> None:
    assert FIX == (
        "run tools/maketools/scripts/run.py lock in the package and commit "
        "poetry.lock and paths-lock.json"
    )


def test_write_lock_resolves_records_and_verifies(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = ok("")
    assert write_lock(project) == 0
    call = world.inheriting_calls[0]
    assert call["argv"] == LOCK_ARGV == ("poetry", "lock")
    assert call["cwd"] == project
    assert call["env"] == world.environment
    assert call["new_session"] is False
    assert call["timeout_seconds"] == INSTALL_WALL_SECONDS
    assert members_of((project / "paths-lock.json").read_text(encoding="utf-8")) == {
        "../a": MEMBER_DIGEST,
        "../b": MEMBER_DIGEST,
    }
    assert world.captured == [CHECK_ARGV]
    assert world.lines[-1].startswith("lock-check: poetry.lock is current")


def test_a_failed_resolution_returns_its_status_and_records_nothing(
    world: World, tmp_path: Path
) -> None:
    project = package_tree(tmp_path)
    world.inheriting_code = 3
    assert write_lock(project) == 3
    assert not (project / "paths-lock.json").exists()
    assert world.captured == []


def test_a_resolution_killed_by_a_signal_returns_its_status_and_records_nothing(
    world: World, tmp_path: Path
) -> None:
    project = package_tree(tmp_path)
    world.inheriting_code = -15
    assert write_lock(project) == -15
    assert not (project / "paths-lock.json").exists()
    assert world.captured == []


def test_a_written_lock_poetry_refuses_is_refused(world: World, tmp_path: Path) -> None:
    project = package_tree(tmp_path)
    world.capturing_answers[CHECK_ARGV] = failed(1, "Error: still stale\n")
    with pytest.raises(AppError, match="pkg: Error: still stale"):
        write_lock(project)
