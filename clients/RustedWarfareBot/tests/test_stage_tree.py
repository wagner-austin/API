"""Describing, packing and vouching for the game tree, end to end.

Real trees on a real filesystem and a real archive, because the walk, the
comparison and the tar are the whole of what is under test. Only the process
arguments and the two output files are faked, so what lands can be read back.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest
from platform_core.json_utils import load_json_str
from scripts.stage_tree import (
    EXIT_OK,
    REQUIRED_FLAGS,
    TARGET_PLATFORM,
    StageTreeError,
    check_reset_state_present,
    check_runtime_is_executable,
    depots,
    main,
    stageable_entries,
)

from rw_bot.harness.clone import VOLATILE_FILES
from rw_bot.harness.jvm import tool_path
from rw_bot.staged_tree import (
    APP_ID,
    ORIGIN_ASSEMBLED,
    ORIGIN_PINNED_STATE,
    ORIGIN_RENAMED,
    ORIGIN_VERIFIED,
    StagedTreeError,
    decode_staged_tree_spec,
)
from rw_bot.tree_archive import ELF_MAGIC
from rw_bot.tree_identity import TreeEntry
from tests.harness_fakes import FakeHost

_BUILD = "9902063"
_CONTENT_MANIFEST = "9090535937117498741"
_LINUX_MANIFEST = "223921525878913700"
_OUT = "runs/rusted-tree.json"
_DIGESTS = "runs/rusted-tree-digests.txt"
_MANIFEST = "runs/stage-rusted-tree.json"

#: Where the archive is staged to. An absolute cluster path, because that is
#: what the manifest's ``destination`` is required to be.
_DESTINATION = "/pub/wagnera3/rusted/staging"

#: A stand-in for the bundled runtime: the bytes that make a file a program.
_RUNTIME = ELF_MAGIC + b"\x02\x01\x01\x00 pretend this is the JRE"

#: The map every sweep plays, and the Steam file it is a byte copy of.
_MAP_BYTES = b"<map><tileset/></map>"
_SHELL_SAFE_MAP = "assets/maps/skirmish/[p2]duel_lake.tmx"
_STEAM_MAP = "assets/maps/skirmish/[p2]Lake (2p).tmx"


def _write(root: Path, relative: str, payload: bytes) -> None:
    """Write one file under a root, creating its parents.

    Args:
        root: Tree root.
        relative: Path under it, forward-slashed.
        payload: Bytes to write.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _reference(root: Path) -> Path:
    """Build a stand-in for Steam's installed copy.

    Carries the Windows runtime and the settings file the tree also has, so
    the volatile exclusion has something to exclude.

    Args:
        root: Directory to build.

    Returns:
        The reference root.
    """
    _write(root, "game-lib.jar", b"engine")
    _write(root, _STEAM_MAP, _MAP_BYTES)
    _write(root, "jvm64/bin/java.exe", b"MZ windows runtime")
    _write(root, "preferences.ini", b"nextBackgroundMap=4\n")
    return root


def _tree(root: Path, runtime: bytes = _RUNTIME) -> Path:
    """Build a stand-in for the assembled Linux tree.

    Args:
        root: Directory to build.
        runtime: Bytes of the bundled ``java``, so a test can make it
            something the archive would not mark executable.

    Returns:
        The tree root.
    """
    _write(root, "game-lib.jar", b"engine")
    _write(root, _STEAM_MAP, _MAP_BYTES)
    _write(root, _SHELL_SAFE_MAP, _MAP_BYTES)
    _write(root, tool_path("java", TARGET_PLATFORM), runtime)
    _write(root, "libsteam_api64.so", b"native")
    _write(root, "preferences.ini", b"nextBackgroundMap=9\n")
    return root


def _argv(tmp_path: Path, tree: Path, reference: Path) -> list[str]:
    """Build the command line the emitter is run with.

    Args:
        tmp_path: Where the archive goes.
        tree: The assembled tree.
        reference: Steam's installed copy.

    Returns:
        The arguments after the program name.
    """
    return [
        "--tree",
        str(tree),
        "--reference",
        str(reference),
        "--build-id",
        _BUILD,
        "--content-manifest",
        _CONTENT_MANIFEST,
        "--linux-manifest",
        _LINUX_MANIFEST,
        "--archive",
        str(tmp_path / "rusted-warfare-linux.tar"),
        "--out",
        _OUT,
        "--digests",
        _DIGESTS,
        "--manifest",
        _MANIFEST,
        "--destination",
        _DESTINATION,
    ]


def _run(tmp_path: Path, runtime: bytes = _RUNTIME) -> FakeHost:
    """Describe a sample tree and return the host holding the outputs.

    Args:
        tmp_path: Working directory.
        runtime: Bytes of the bundled ``java``.

    Returns:
        The host, after the emitter has run.
    """
    tree = _tree(tmp_path / "tree", runtime)
    reference = _reference(tmp_path / "reference")
    host = FakeHost()
    with host:
        assert main(_argv(tmp_path, tree, reference)) == EXIT_OK
    return host


class TestWhatAStagedTreeCarries:
    def test_the_settings_file_is_staged_because_every_clone_is_reset_from_it(self) -> None:
        """The instinct is to drop it -- the game rewrites it on every boot.
        That is wrong: ``reset_volatile`` copies it out of the SOURCE
        directory into each worker's copy, so a staged tree without it fails
        on the compute node, once per member."""
        kept = stageable_entries(
            (
                TreeEntry(path=VOLATILE_FILES[0], sha256="a" * 64, size_bytes=4),
                TreeEntry(path="game-lib.jar", sha256="b" * 64, size_bytes=4),
            )
        )
        assert [entry["path"] for entry in kept] == [VOLATILE_FILES[0], "game-lib.jar"]

    def test_a_tree_that_lost_it_is_refused_before_anything_is_packed(self) -> None:
        with pytest.raises(StageTreeError) as caught:
            check_reset_state_present(
                (TreeEntry(path="game-lib.jar", sha256="b" * 64, size_bytes=4),)
            )
        assert caught.value.code == "RW-STAGE-002"

    def test_a_tree_that_carries_it_passes(self) -> None:
        check_reset_state_present(
            (TreeEntry(path=VOLATILE_FILES[0], sha256="a" * 64, size_bytes=4),)
        )

    def test_the_trees_the_game_rebuilds_are_not_staged(self) -> None:
        kept = stageable_entries(
            (
                TreeEntry(path="saves/autosave.rwsave", sha256="a" * 64, size_bytes=4),
                TreeEntry(path="cache/mods-info.cachedata", sha256="b" * 64, size_bytes=4),
                TreeEntry(path="assets/maps/duel.tmx", sha256="c" * 64, size_bytes=4),
            )
        )
        assert [entry["path"] for entry in kept] == ["assets/maps/duel.tmx"]


class TestNamingTheDepots:
    def test_both_are_pinned_by_manifest_rather_than_by_id_alone(self) -> None:
        """A depot id names a moving target; the manifest GID is the pin."""
        named = depots(_CONTENT_MANIFEST, _LINUX_MANIFEST)
        assert [depot["manifest"] for depot in named] == [_CONTENT_MANIFEST, _LINUX_MANIFEST]
        assert [depot["depot_id"] for depot in named] == ["647961", "647963"]


class TestTheRuntimeMustBeRunnable:
    def test_a_packed_java_is_accepted(self) -> None:
        check_runtime_is_executable((tool_path("java", TARGET_PLATFORM), "other"))

    def test_a_java_that_would_extract_unexecutable_is_refused(self) -> None:
        """The tree was assembled on Windows, which has no executable bit, so
        the archive decides it. WSL2 mounts /mnt/c with everything 777, which
        is why the workstation proof of the Linux launch could never have
        shown this."""
        with pytest.raises(StageTreeError) as caught:
            check_runtime_is_executable(("libsteam_api64.so",))
        assert caught.value.code == "RW-STAGE-001"

    def test_a_runtime_that_is_not_a_program_stops_the_run(self, tmp_path: Path) -> None:
        with FakeHost(), pytest.raises(StageTreeError) as caught:
            main(
                _argv(
                    tmp_path,
                    _tree(tmp_path / "tree", b"not an ELF binary at all"),
                    _reference(tmp_path / "reference"),
                )
            )
        assert caught.value.code == "RW-STAGE-001"


class TestTheDocumentItWrites:
    def test_it_decodes_through_its_own_decoder(self, tmp_path: Path) -> None:
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        assert spec["app_id"] == APP_ID
        assert spec["build_id"] == _BUILD

    def test_every_origin_is_measured_rather_than_declared(self, tmp_path: Path) -> None:
        """The jar and the Steam-named map are verified against the reference,
        the shell-safe copy is a rename that names its source, and the runtime
        is stated as assembled because nothing here can check it."""
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        origins = {entry["path"]: entry["origin"] for entry in spec["entries"]}
        assert origins["game-lib.jar"] == ORIGIN_VERIFIED
        assert origins[_STEAM_MAP] == ORIGIN_VERIFIED
        assert origins[_SHELL_SAFE_MAP] == ORIGIN_RENAMED
        assert origins[tool_path("java", TARGET_PLATFORM)] == ORIGIN_ASSEMBLED

    def test_the_renamed_map_names_the_steam_file_it_copies(self, tmp_path: Path) -> None:
        """The finding that made this module necessary: every sweep plays the
        shell-safe copy, and rebuilding the tree from Steam alone would have
        deleted it."""
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        sources = {entry["path"]: entry["source"] for entry in spec["entries"]}
        assert sources[_SHELL_SAFE_MAP] == _STEAM_MAP

    def test_the_settings_file_is_carried_as_pinned_state(self, tmp_path: Path) -> None:
        """It differs from the reference and that says nothing -- the game
        rewrites it. What is staged is this workstation's copy, which makes it
        the experiment's starting state rather than whatever the first node
        left behind."""
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        origins = {entry["path"]: entry["origin"] for entry in spec["entries"]}
        assert origins[VOLATILE_FILES[0]] == ORIGIN_PINNED_STATE

    def test_the_archive_it_names_is_the_one_it_wrote(self, tmp_path: Path) -> None:
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        packed = tmp_path / "rusted-warfare-linux.tar"
        assert spec["archive_name"] == packed.name
        assert spec["archive_size_bytes"] == packed.stat().st_size

    def test_the_record_names_the_archive_digest(self, tmp_path: Path) -> None:
        host = _run(tmp_path)
        spec = decode_staged_tree_spec(load_json_str("\n".join(host.files[_OUT])))
        assert any(spec["archive_sha256"] in line for line in host.files[_DIGESTS])

    def test_it_reports_the_tally_and_the_identity(self, tmp_path: Path) -> None:
        host = _run(tmp_path)
        printed = "\n".join(host.printed)
        assert f"2 {ORIGIN_VERIFIED}" in printed
        assert f"1 {ORIGIN_RENAMED}" in printed
        assert "[tree] identity " in printed
        assert "1 member(s) marked executable" in printed


class TestWhenTheTwoCopiesDisagree:
    def test_a_file_that_differs_at_one_path_stops_the_run(self, tmp_path: Path) -> None:
        """No later check would catch it: each digest matches the file it came
        from, and the transfer matches the digest."""
        tree = _tree(tmp_path / "tree")
        reference = _reference(tmp_path / "reference")
        _write(reference, "game-lib.jar", b"a different engine")
        with FakeHost(), pytest.raises(StagedTreeError) as caught:
            main(_argv(tmp_path, tree, reference))
        assert caught.value.code == "RW-TREE-101"


class TestTheEntryPoint:
    def test_it_reads_the_process_arguments_when_given_none(self, tmp_path: Path) -> None:
        host = FakeHost()
        host.argv = _argv(tmp_path, _tree(tmp_path / "tree"), _reference(tmp_path / "reference"))
        with host:
            assert main(None) == EXIT_OK

    def test_a_missing_flag_is_refused(self, tmp_path: Path) -> None:
        argv = _argv(tmp_path, _tree(tmp_path / "tree"), _reference(tmp_path / "reference"))
        # Named from the flag table rather than written out, so adding a
        # required flag does not leave this asserting about the old last one.
        with FakeHost(), pytest.raises(ValueError, match=f"{REQUIRED_FLAGS[-1]} is required"):
            main(argv[:-2])

    def test_the_module_guard_runs_main(self, tmp_path: Path) -> None:
        host = FakeHost()
        host.argv = _argv(tmp_path, _tree(tmp_path / "tree"), _reference(tmp_path / "reference"))
        with host:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.stage_tree", run_name="__main__")
            assert caught.value.code == EXIT_OK
