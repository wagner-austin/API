"""Packing a game tree into one file whose bytes are a function of the tree.

Real files and a real tarfile, because what is under test is the archive's
BYTES: two packs of one tree must be the same file, and a member must extract
with the mode it was written with. A fake would be asserting against its own
idea of both.
"""

from __future__ import annotations

import tarfile
from hashlib import sha256
from pathlib import Path

from rw_bot.tree_archive import (
    ARCHIVE_GID,
    ARCHIVE_MODE_EXECUTABLE,
    ARCHIVE_MODE_FILE,
    ARCHIVE_MTIME,
    ARCHIVE_OWNER,
    ARCHIVE_UID,
    ELF_MAGIC,
    is_program,
    member_mode,
    write_archive,
)
from rw_bot.tree_identity import tree_entries

#: A file that begins the way every Linux program begins, and one that does
#: not. The second is taken from the real tree: ``jvm-linux/lib/classlist``
#: sits beside binaries in the same directory and is a text file.
PROGRAM = ELF_MAGIC + b"\x02\x01\x01\x00" + b"pretend this is a JVM"
NOT_A_PROGRAM = b"java/lang/Object\njava/lang/String\n"


def _tree(root: Path) -> Path:
    """Write a tree carrying a program, a plain file and a nested one.

    Args:
        root: Directory to build.

    Returns:
        The tree root.
    """
    (root / "jvm-linux" / "bin").mkdir(parents=True, exist_ok=True)
    (root / "assets" / "maps").mkdir(parents=True, exist_ok=True)
    (root / "jvm-linux" / "bin" / "java").write_bytes(PROGRAM)
    (root / "jvm-linux" / "classlist").write_bytes(NOT_A_PROGRAM)
    (root / "assets" / "maps" / "duel.tmx").write_bytes(b"map")
    (root / "game-lib.jar").write_bytes(b"engine")
    return root


class TestRecognisingAProgram:
    def test_the_elf_magic_is_what_makes_a_file_a_program(self) -> None:
        assert is_program(PROGRAM) is True
        assert is_program(NOT_A_PROGRAM) is False

    def test_a_file_shorter_than_the_magic_is_not_one(self) -> None:
        """An empty marker file sits in the real tree, and slicing past the
        end of it must not read as a match."""
        assert is_program(b"") is False
        assert is_program(b"\x7fE") is False

    def test_the_mode_follows_the_magic(self) -> None:
        assert member_mode(PROGRAM) == ARCHIVE_MODE_EXECUTABLE
        assert member_mode(NOT_A_PROGRAM) == ARCHIVE_MODE_FILE

    def test_the_rule_is_derived_rather_than_listed(self) -> None:
        """A hand-written list of which files are programs is a list that
        rots -- a JRE's helper binaries move between releases. A file that
        did not exist when the rule was written still gets the right mode."""
        assert member_mode(ELF_MAGIC + b"a helper nobody has heard of") == (ARCHIVE_MODE_EXECUTABLE)


class TestPacking:
    def test_the_members_are_the_entries_in_the_order_given(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "tree")
        entries = tree_entries(root)
        write_archive(root, entries, tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            assert archive.getnames() == [entry["path"] for entry in entries]

    def test_no_directory_is_written_because_a_directory_holds_no_bytes(
        self, tmp_path: Path
    ) -> None:
        """Extraction creates the parents it needs, so the archive holds
        exactly the set the document describes."""
        root = _tree(tmp_path / "tree")
        write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            assert [item.name for item in archive.getmembers() if not item.isfile()] == []

    def test_a_program_extracts_executable_and_a_plain_file_does_not(self, tmp_path: Path) -> None:
        """Without this the JVM extracts at 644 and every match on the
        cluster dies as permission denied."""
        root = _tree(tmp_path / "tree")
        write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            modes = {item.name: item.mode for item in archive.getmembers()}
        assert modes["jvm-linux/bin/java"] == ARCHIVE_MODE_EXECUTABLE
        assert modes["jvm-linux/classlist"] == ARCHIVE_MODE_FILE

    def test_the_executables_are_reported_so_a_caller_can_check_them(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "tree")
        result = write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        assert result["executables"] == ("jvm-linux/bin/java",)

    def test_every_member_carries_the_pinned_timestamp_and_owner(self, tmp_path: Path) -> None:
        """None of the three says anything about the tree, and all three would
        otherwise make two packs of one tree different files."""
        root = _tree(tmp_path / "tree")
        write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            for item in archive.getmembers():
                assert item.mtime == ARCHIVE_MTIME
                assert (item.uid, item.gid) == (ARCHIVE_UID, ARCHIVE_GID)
                assert (item.uname, item.gname) == (ARCHIVE_OWNER, ARCHIVE_OWNER)

    def test_the_contents_survive_the_round_trip(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "tree")
        write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            handle = archive.extractfile("jvm-linux/bin/java")
            if handle is None:
                raise AssertionError("the runtime member is not a regular file")
            assert handle.read() == PROGRAM

    def test_two_packs_of_one_tree_are_the_same_file(self, tmp_path: Path) -> None:
        """The property the whole header discipline exists for. Without it the
        archive's digest would be a fact about the machine that packed it."""
        root = _tree(tmp_path / "tree")
        entries = tree_entries(root)
        first = write_archive(root, entries, tmp_path / "first.tar")
        second = write_archive(root, entries, tmp_path / "second.tar")
        assert first["sha256"] == second["sha256"]
        assert (tmp_path / "first.tar").read_bytes() == (tmp_path / "second.tar").read_bytes()

    def test_the_same_tree_under_another_path_packs_alike(self, tmp_path: Path) -> None:
        """Identity travels with the content, which is what lets the archive
        be rebuilt somewhere else and compared."""
        here = write_archive(
            _tree(tmp_path / "a"), tree_entries(_tree(tmp_path / "a")), tmp_path / "a.tar"
        )
        there = write_archive(
            _tree(tmp_path / "b"), tree_entries(_tree(tmp_path / "b")), tmp_path / "b.tar"
        )
        assert here["sha256"] == there["sha256"]

    def test_one_changed_byte_changes_the_archive(self, tmp_path: Path) -> None:
        root = _tree(tmp_path / "tree")
        before = write_archive(root, tree_entries(root), tmp_path / "before.tar")
        (root / "game-lib.jar").write_bytes(b"engine, patched")
        after = write_archive(root, tree_entries(root), tmp_path / "after.tar")
        assert before["sha256"] != after["sha256"]

    def test_the_reported_digest_and_length_are_the_files_own(self, tmp_path: Path) -> None:
        """Read back off the finished archive rather than accumulated while
        writing, so what is reported is what a cluster-side ``sha256sum``
        will compute."""
        root = _tree(tmp_path / "tree")
        result = write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        packed = (tmp_path / "tree.tar").read_bytes()
        assert result["sha256"] == sha256(packed).hexdigest()
        assert result["size_bytes"] == len(packed)

    def test_a_long_nested_path_survives(self, tmp_path: Path) -> None:
        """The real tree runs to 105 characters under ``jvm-linux/lib/desktop``,
        past what a ustar header holds in one field."""
        root = _tree(tmp_path / "tree")
        deep = (
            "jvm-linux/lib/desktop/icons/HighContrastInverse/48x48/mimetypes/"
            "gnome-mime-application-x-java-jnlp-file.png"
        )
        target = root / deep
        target.parent.mkdir(parents=True)
        target.write_bytes(b"icon")
        write_archive(root, tree_entries(root), tmp_path / "tree.tar")
        with tarfile.open(tmp_path / "tree.tar") as archive:
            assert deep in archive.getnames()
