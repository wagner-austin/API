"""Tests for measuring TankPit's move from sedona to colossus.

The trees are real directories under ``tmp_path``, hashed for real; every
docker command goes through ``scripts._test_hooks.run_command``, here a
:class:`~tests.scripts._recorder.Recorder`, so each test asserts the exact
commands the host is asked.
"""

from __future__ import annotations

import os
import runpy
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from scripts.colossus_move import (
    LISTING,
    RELEASE,
    MovePlaces,
    colossus_host,
    compare_tree,
    created_at,
    file_sha256,
    hash_tree,
    read_listing,
    run,
    write_lines,
)
from scripts.fleet_remote import FleetHostError

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks
from tests.scripts._recorder import Recorder, fail, ok

#: sha256 of ``abc``, from FIPS 180-2's own example.
ABC: str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

#: sha256 of the empty string.
EMPTY: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

TAG = f"tankpit-fleet:{RELEASE}"
INSPECT = ["docker", "image", "inspect", "--format", "{{.Id}} {{.Created}}", TAG]
BY_NAME = ["docker", "ps", "-a", "--filter", "name=tankpit", "--format", "{{.Names}}"]
BY_IMAGE = ["docker", "ps", "-a", "--filter", f"ancestor={TAG}", "--format", "{{.Names}}"]

#: When the release in :func:`_ladder` was unpacked: 2026-10-18T12:22:31Z.
UNPACKED = int(datetime(2026, 10, 18, 12, 22, 31, tzinfo=UTC).timestamp())


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Put back every hook a test replaced.

    Yields:
        None, with the original hooks restored after.
    """
    run_command = script_hooks.run_command
    get_argv = core_hooks.get_argv
    yield
    script_hooks.run_command = run_command
    core_hooks.get_argv = get_argv


def _tree(root: Path) -> Path:
    """Write a host directory: one run file two levels down and one data file.

    Args:
        root: Where.

    Returns:
        ``root``.
    """
    (root / "runs" / "bot").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "runs" / "bot" / "events.jsonl").write_bytes(b"abc")
    (root / "data" / "tank_registry.json").write_bytes(b"")
    return root


def _listing(path: Path, lines: list[str]) -> Path:
    """Write a listing.

    Args:
        path: Where.
        lines: Its lines.

    Returns:
        ``path``.
    """
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _matching(path: Path) -> Path:
    """Write the listing :func:`_tree` hashes to.

    Args:
        path: Where.

    Returns:
        ``path``.
    """
    return _listing(path, [f"{EMPTY}  data/tank_registry.json", f"{ABC}  runs/bot/events.jsonl"])


def _ladder(root: Path) -> Path:
    """Write a release ladder holding :data:`RELEASE`, unpacked at :data:`UNPACKED`.

    Args:
        root: Where.

    Returns:
        The ladder.
    """
    release = root / "ladder" / RELEASE
    release.mkdir(parents=True)
    os.utime(release, (UNPACKED, UNPACKED))
    return root / "ladder"


def test_the_committed_listing_names_every_file_the_move_copied() -> None:
    """The listing both hosts hash to holds sedona's 614 files, two of them data."""
    hashes = read_listing(Path(__file__).resolve().parents[2] / LISTING)
    assert len(hashes) == 614
    assert sorted(path for path in hashes if path.startswith("data/")) == [
        "data/tank_registry.corrupt-20261017T0811Z.json",
        "data/tank_registry.json",
    ]


def test_a_listing_line_of_another_shape_is_refused(tmp_path: Path) -> None:
    """An uppercase hash is not what the listing holds."""
    listing = _listing(tmp_path / "listing.txt", [f"{ABC.upper()}  runs/a"])
    with pytest.raises(FleetHostError, match=r"^MOVE_LISTING_MALFORMED: .*listing\.txt: 'BA7816"):
        read_listing(listing)


def test_a_path_listed_twice_is_refused(tmp_path: Path) -> None:
    """A second line for one path would hide which hash is the truth."""
    listing = _listing(tmp_path / "listing.txt", [f"{ABC}  runs/a", f"{EMPTY}  runs/a"])
    with pytest.raises(FleetHostError, match=f"^MOVE_LISTING_MALFORMED: .*'{EMPTY}  runs/a'$"):
        read_listing(listing)


def test_a_file_hashes_to_its_published_sha256(tmp_path: Path) -> None:
    """The hash is sha256 of the bytes, past one read's megabyte as well."""
    small = tmp_path / "abc"
    small.write_bytes(b"abc")
    large = tmp_path / "large"
    large.write_bytes(b"a" * 1_000_000)
    assert file_sha256(small) == ABC
    assert file_sha256(large) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"


def test_every_file_under_the_copied_directories_is_hashed_by_its_slashed_path(
    tmp_path: Path,
) -> None:
    """Nested files are named relative to the host directory; other directories are not read."""
    root = _tree(tmp_path / "host")
    (root / "edge").mkdir()
    (root / "edge" / "nginx.conf").write_bytes(b"x")
    assert hash_tree(root) == {"runs/bot/events.jsonl": ABC, "data/tank_registry.json": EMPTY}


def test_a_host_directory_without_a_copied_directory_is_refused(tmp_path: Path) -> None:
    """A host with runs but no data is not the move's tree."""
    (tmp_path / "runs").mkdir()
    with pytest.raises(FleetHostError, match=r"^MOVE_TREE_MISSING: .*data is not a directory$"):
        hash_tree(tmp_path)


def test_a_tree_matching_its_listing_says_how_many_files_it_read(tmp_path: Path) -> None:
    """The line counts the files hashed against the files listed."""
    root = _tree(tmp_path / "host")
    listing = _matching(tmp_path / "listing.txt")
    assert compare_tree(root, listing) == (
        f"2 of 2 files under {root} (runs, data) hash to listing.txt"
    )


def test_a_tree_that_differs_names_what_is_missing_unlisted_and_changed(tmp_path: Path) -> None:
    """Each kind of difference is counted and named."""
    root = _tree(tmp_path / "host")
    (root / "runs" / "extra.log").write_bytes(b"abc")
    listing = _listing(
        tmp_path / "listing.txt",
        [f"{ABC}  data/tank_registry.json", f"{ABC}  runs/bot/events.jsonl", f"{ABC}  runs/gone"],
    )
    with pytest.raises(FleetHostError) as refused:
        compare_tree(root, listing)
    assert str(refused.value) == (
        f"MOVE_TREE_DIFFERS: {root} against listing.txt: missing 1 (runs/gone), "
        "unlisted 1 (runs/extra.log), changed 1 (data/tank_registry.json)"
    )


def test_a_long_difference_names_only_its_first_paths(tmp_path: Path) -> None:
    """Six missing files are counted and the first five named."""
    root = _tree(tmp_path / "host")
    gone = [f"runs/gone-{index}" for index in range(6)]
    listing = _listing(
        tmp_path / "listing.txt",
        [
            f"{EMPTY}  data/tank_registry.json",
            f"{ABC}  runs/bot/events.jsonl",
            *(f"{ABC}  {path}" for path in gone),
        ],
    )
    with pytest.raises(FleetHostError) as refused:
        compare_tree(root, listing)
    assert str(refused.value).endswith(f"missing 6 ({', '.join(gone[:5])}), unlisted 0, changed 0")


@pytest.mark.parametrize(
    ("created", "instant"),
    [
        ("2026-10-18T05:24:27.860717453-07:00", "2026-10-18T12:24:27+00:00"),
        ("2026-10-18T12:24:27Z", "2026-10-18T12:24:27+00:00"),
    ],
)
def test_dockers_created_instant_is_read_to_the_second_in_utc(created: str, instant: str) -> None:
    """A nanosecond fraction is dropped and the zone applied."""
    assert created_at(created).isoformat() == instant


def test_a_created_instant_of_another_shape_is_refused() -> None:
    """A date with no zone is not docker's."""
    with pytest.raises(FleetHostError, match=r"^MOVE_IMAGE_UNREADABLE: Created '2026-10-18'$"):
        created_at("2026-10-18")


def test_the_release_its_image_and_no_container_are_reported(tmp_path: Path) -> None:
    """The ladder, the daemon's image and both container listings are read, in that order."""
    recorder = Recorder({"inspect": ok("sha256:dc2b 2026-10-18T05:24:27.86-07:00\n")})
    script_hooks.run_command = recorder
    ladder = _ladder(tmp_path)
    assert colossus_host(ladder, tmp_path) == [
        f"release {RELEASE} on the ladder at {ladder}, unpacked 2026-10-18T12:22:31+00:00",
        f"image {TAG} sha256:dc2b created 2026-10-18T12:24:27+00:00",
        f"no container named tankpit and none from {TAG}",
    ]
    assert recorder.calls == [INSPECT, BY_NAME, BY_IMAGE]


def test_a_ladder_without_the_release_is_refused_before_docker_is_asked(tmp_path: Path) -> None:
    """No release, nothing else read."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    with pytest.raises(FleetHostError, match=f"^MOVE_RELEASE_MISSING: .*{RELEASE} is not a"):
        colossus_host(tmp_path, tmp_path)
    assert recorder.calls == []


def test_a_daemon_without_the_image_is_refused(tmp_path: Path) -> None:
    """docker's own refusal is named."""
    script_hooks.run_command = Recorder({"inspect": fail("No such image")})
    with pytest.raises(FleetHostError, match=r"^MOVE_IMAGE_MISSING: .* exited 1: No such image$"):
        colossus_host(_ladder(tmp_path), tmp_path)


def test_an_image_older_than_its_release_is_refused(tmp_path: Path) -> None:
    """An image created before the release was unpacked was not built from it."""
    script_hooks.run_command = Recorder({"inspect": ok("sha256:dc2b 2026-10-18T12:22:30Z")})
    with pytest.raises(FleetHostError) as refused:
        colossus_host(_ladder(tmp_path), tmp_path)
    assert str(refused.value) == (
        f"MOVE_IMAGE_PREDATES_RELEASE: {TAG} was created 2026-10-18T12:22:30+00:00, before "
        f"{tmp_path / 'ladder' / RELEASE} was unpacked at 2026-10-18T12:22:31+00:00"
    )


def test_a_container_named_tankpit_or_started_from_the_image_is_refused(tmp_path: Path) -> None:
    """Each container is named with the listing that found it."""
    script_hooks.run_command = Recorder(
        {
            "inspect": ok("sha256:dc2b 2026-10-18T12:24:27Z"),
            "name=tankpit": ok("tankpit-fleet-1\n"),
            "ancestor=": ok("tankpit-fleet-1\nhopeful_turing\n"),
        }
    )
    with pytest.raises(FleetHostError) as refused:
        colossus_host(_ladder(tmp_path), tmp_path)
    assert str(refused.value) == (
        "MOVE_CONTAINER_STARTED: tankpit-fleet-1 (name=tankpit), "
        f"tankpit-fleet-1 (ancestor={TAG}), hopeful_turing (ancestor={TAG})"
    )


def test_a_container_listing_docker_refuses_is_named(tmp_path: Path) -> None:
    """An unreadable listing is not an empty one."""
    script_hooks.run_command = Recorder(
        {"inspect": ok("sha256:dc2b 2026-10-18T12:24:27Z"), "ps": fail("permission denied")}
    )
    with pytest.raises(FleetHostError, match=r"^MOVE_CONTAINERS_UNREADABLE: .*permission denied$"):
        colossus_host(_ladder(tmp_path), tmp_path)


def _places(tmp_path: Path) -> MovePlaces:
    """Both hosts' places under ``tmp_path``, the listing beside them.

    Args:
        tmp_path: The test's directory, standing in for the package.

    Returns:
        The places.
    """
    (tmp_path / LISTING).parent.mkdir(parents=True)
    _matching(tmp_path / LISTING)
    return MovePlaces(
        sedona_tree=_tree(tmp_path / "sedona"),
        archive_tree=_tree(tmp_path / "archive"),
        releases=_ladder(tmp_path),
    )


def test_sedona_is_measured_against_the_packages_listing_only(tmp_path: Path) -> None:
    """``sedona`` hashes its tree and asks docker nothing."""
    recorder = Recorder({})
    script_hooks.run_command = recorder
    places = _places(tmp_path)
    assert run(["sedona"], tmp_path, places) == [
        f"2 of 2 files under {places['sedona_tree']} (runs, data) hash to {LISTING.name}"
    ]
    assert recorder.calls == []


def test_colossus_is_measured_on_its_copy_and_its_daemon(tmp_path: Path) -> None:
    """``colossus`` hashes the copy, then reads the release, the image and the containers."""
    script_hooks.run_command = Recorder({"inspect": ok("sha256:dc2b 2026-10-18T12:24:27Z")})
    places = _places(tmp_path)
    lines = run(["colossus"], tmp_path, places)
    assert lines[0] == (
        f"2 of 2 files under {places['archive_tree']} (runs, data) hash to {LISTING.name}"
    )
    assert lines[1:] == colossus_host(places["releases"], tmp_path)


def test_any_other_command_line_prints_the_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A host the module does not measure exits 2."""
    with pytest.raises(SystemExit) as exited:
        run(["sedona", "colossus"], tmp_path, _places(tmp_path))
    assert exited.value.code == 2
    assert capsys.readouterr().out == "usage: python -m scripts.colossus_move {sedona|colossus}\n"


def test_the_lines_are_printed_one_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Each line ends in a newline."""
    write_lines(["one", "two"])
    assert capsys.readouterr().out == "one\ntwo\n"


def test_the_module_runs_as_a_program(capsys: pytest.CaptureFixture[str]) -> None:
    """``python -m scripts.colossus_move`` with no host prints the usage."""
    core_hooks.get_argv = lambda: ["colossus_move"]
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.colossus_move", run_name="__main__")
    assert capsys.readouterr().out.startswith("usage:")
