"""The files this package lifts from MCPs, pinned by hash (MCPs board task 1b152218).

The lock's codec, the digest's line-ending rule, the check that fails a
hand-edited or missing lifted file by name, and the refresh, run against a
real git repository standing in for MCPs, that writes each file from a
commit's object store and re-pins the lock. The package's own lock is
checked for real, so a lifted file edited here fails this suite too.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from maketools.cli import dispatch, main
from maketools.lift import (
    LIFT_LOCK_NAME,
    LiftEntry,
    LiftLock,
    decode_lift_lock,
    drifted_lifts,
    encode_lift_lock,
    pinned_digest,
    read_lift_lock,
    refresh_lifts,
)
from tests.conftest import World

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

RULE = b"BUDGET_SECONDS = 300\n"


def _lock(sha256: str, revision: str = "a" * 40) -> LiftLock:
    """A one-entry lock.

    Args:
        sha256: The entry's pin.
        revision: The lock's revision.

    Returns:
        The lock.
    """
    return LiftLock(
        remote="https://github.com/wagner-austin/MCPs.git",
        revision=revision,
        entries=[
            LiftEntry(
                path="src/maketools/check_budget.py",
                source="packages/maketools/src/maketools/check_budget.py",
                sha256=sha256,
            )
        ],
    )


def _git(repo: Path, *arguments: str) -> str:
    """Run git in a fixture repository.

    Args:
        repo: The repository.
        arguments: git's arguments.

    Returns:
        Its standard output.
    """
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=True
    ).stdout


def _upstream(tmp_path: Path, content: bytes) -> tuple[Path, str]:
    """A real repository holding the rule at MCPs' path, committed.

    Args:
        tmp_path: Where to make it.
        content: The rule's bytes.

    Returns:
        The repository and its commit.
    """
    mcps = tmp_path / "MCPs"
    source = mcps / "packages" / "maketools" / "src" / "maketools" / "check_budget.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    _git(mcps.parent, "init", "--quiet", str(mcps))
    _git(mcps, "add", ".")
    _git(mcps, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "rule")
    return mcps, _git(mcps, "rev-parse", "HEAD").strip()


def test_the_lock_round_trips_through_its_codec() -> None:
    lock = _lock(pinned_digest(RULE))
    assert decode_lift_lock(encode_lift_lock(lock)) == lock
    assert encode_lift_lock(lock).endswith("]\n}\n")


def test_a_lock_of_the_wrong_shape_is_refused() -> None:
    with pytest.raises(JSONTypeError):
        decode_lift_lock('{"remote": "r", "revision": "v", "entries": [{"path": 1}]}')


def test_the_digest_is_taken_over_the_lf_form_so_a_windows_checkout_matches_its_blob() -> None:
    assert pinned_digest(b"a\r\nb\r\n") == pinned_digest(b"a\nb\n")
    assert pinned_digest(b"a\nb\n") != pinned_digest(b"a\nc\n")


def test_a_file_holding_its_pinned_bytes_has_not_drifted(tmp_path: Path) -> None:
    lifted = tmp_path / "src" / "maketools" / "check_budget.py"
    lifted.parent.mkdir(parents=True)
    lifted.write_bytes(RULE.replace(b"\n", b"\r\n"))
    assert drifted_lifts(tmp_path, _lock(pinned_digest(RULE))) == []


def test_an_edited_or_missing_lifted_file_is_named_with_its_pin(tmp_path: Path) -> None:
    lock = _lock(pinned_digest(RULE))
    (missing,) = drifted_lifts(tmp_path, lock)
    assert missing.startswith("src/maketools/check_budget.py: missing; it is lifted from MCPs")
    lifted = tmp_path / "src" / "maketools" / "check_budget.py"
    lifted.parent.mkdir(parents=True)
    lifted.write_bytes(b"BUDGET_SECONDS = 600\n")
    (edited,) = drifted_lifts(tmp_path, lock)
    assert f"is not the pinned {pinned_digest(RULE)}" in edited
    assert "never edit a lifted file here" in edited


def test_this_package_s_own_lifts_hold_their_pins() -> None:
    lock = read_lift_lock(PACKAGE_ROOT)
    assert len(lock["entries"]) == 1
    assert drifted_lifts(PACKAGE_ROOT, lock) == []


def test_lift_check_passes_this_package_and_says_how_many_it_read(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(PACKAGE_ROOT)
    assert dispatch(["lift-check"]) == 0
    (line,) = world.lines
    assert line.startswith("lift-check: 1 of 1 lifted file(s) hold their pinned bytes from MCPs ")


def test_lift_check_fails_a_drifted_package_naming_each_file(
    tmp_path: Path, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / LIFT_LOCK_NAME).write_text(
        encode_lift_lock(_lock(pinned_digest(RULE))), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert dispatch(["lift-check"]) == 1
    assert world.errors[0].startswith("src/maketools/check_budget.py: missing")
    assert world.errors[-1] == "lift-check: 1 of 1 lifted file(s) differ from their pins"


def test_lift_check_takes_no_arguments(world: World) -> None:
    with pytest.raises(AppError, match=r"lift-check takes no arguments, got \['now'\]"):
        dispatch(["lift-check", "now"])


def test_a_refresh_writes_each_file_from_the_commit_and_re_pins_the_lock(
    tmp_path: Path,
) -> None:
    mcps, commit = _upstream(tmp_path, RULE)
    package = tmp_path / "package"
    (package / "src" / "maketools").mkdir(parents=True)
    (package / LIFT_LOCK_NAME).write_text(encode_lift_lock(_lock("0" * 64)), encoding="utf-8")
    refreshed = refresh_lifts(package, mcps, "HEAD")
    assert refreshed == _lock(pinned_digest(RULE), revision=commit)
    assert read_lift_lock(package) == refreshed
    assert (package / "src" / "maketools" / "check_budget.py").read_bytes() == RULE
    assert drifted_lifts(package, refreshed) == []


def test_a_refresh_from_a_revision_mcps_lacks_is_refused_by_code_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mcps, _ = _upstream(tmp_path, RULE)
    package = tmp_path / "package"
    package.mkdir()
    before = encode_lift_lock(_lock("0" * 64))
    (package / LIFT_LOCK_NAME).write_text(before, encoding="utf-8")
    monkeypatch.chdir(package)
    assert main(["lift-refresh", str(mcps), "no-such-revision"]) == 1
    assert capsys.readouterr().err == (
        f"MAKETOOLS_LIFT: {mcps} has no commit 'no-such-revision' to lift from\n"
    )
    assert (package / LIFT_LOCK_NAME).read_text(encoding="utf-8") == before


def test_a_refresh_whose_source_is_absent_at_the_commit_writes_nothing(tmp_path: Path) -> None:
    mcps, commit = _upstream(tmp_path, RULE)
    package = tmp_path / "package"
    package.mkdir()
    absent = LiftLock(
        remote="r",
        revision="a" * 40,
        entries=[LiftEntry(path="moved.py", source="packages/maketools/moved.py", sha256="0")],
    )
    (package / LIFT_LOCK_NAME).write_text(encode_lift_lock(absent), encoding="utf-8")
    with pytest.raises(AppError, match=f"MCPs {commit} holds no packages/maketools/moved.py"):
        refresh_lifts(package, mcps, "HEAD")
    assert not (package / "moved.py").exists()


def test_lift_refresh_through_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mcps, commit = _upstream(tmp_path, RULE)
    package = tmp_path / "package"
    (package / "src" / "maketools").mkdir(parents=True)
    (package / LIFT_LOCK_NAME).write_text(encode_lift_lock(_lock("0" * 64)), encoding="utf-8")
    monkeypatch.chdir(package)
    assert dispatch(["lift-refresh", str(mcps), "HEAD"]) == 0
    assert read_lift_lock(package)["revision"] == commit
    assert capsys.readouterr().out == (
        f"lift-refresh: 1 lifted file(s) written from MCPs {commit} and pinned\n"
    )


def test_lift_refresh_takes_a_checkout_and_a_revision(world: World) -> None:
    with pytest.raises(AppError, match=r"lift-refresh takes MCPS REVISION, got \['../MCPs'\]"):
        dispatch(["lift-refresh", "../MCPs"])
