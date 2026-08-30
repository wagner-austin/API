"""What a directory of game content is, as a digest of the bytes in it.

Driven against real files on a real filesystem, because the thing under test
IS a filesystem walk. A fake tree would exercise the fake's idea of ordering,
of what a symlink is, and of which entries a walk yields -- which are exactly
the three things this module has to get right.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import JSONTypeError

from rw_bot.tree_identity import (
    LISTING_SEPARATOR,
    SHA256_HEX_LENGTH,
    TreeEntry,
    TreeIdentityError,
    decode_tree_entry,
    digest_record,
    digest_tree,
    encode_tree_entry,
    render_listing,
    tree_digest,
    tree_entries,
)

#: A file whose name sorts differently by byte than by locale, which is the
#: whole reason the order is defined on UTF-8 bytes. Under ``LC_ALL=C`` the
#: capital sorts before the lowercase; under a language-aware collation it
#: does not.
_MIXED_CASE_NAMES = ("Zulu.txt", "alpha.txt")


def _write(root: Path, relative: str, payload: bytes) -> Path:
    """Write one file under a root, creating its parents.

    Args:
        root: Tree root.
        relative: Path under the root, forward-slashed.
        payload: Bytes to write.

    Returns:
        The written path.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


class TestReadingATree:
    def test_every_file_becomes_an_entry(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        _write(tmp_path, "maps/skirmish.tmx", b"two")
        assert [entry["path"] for entry in tree_entries(tmp_path)] == [
            "game.txt",
            "maps/skirmish.tmx",
        ]

    def test_an_entry_carries_the_digest_and_length_of_its_own_bytes(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        entry = tree_entries(tmp_path)[0]
        assert entry["sha256"] == sha256(b"one").hexdigest()
        assert entry["size_bytes"] == 3

    def test_a_nested_path_is_forward_slashed_on_every_platform(self, tmp_path: Path) -> None:
        """The listing is read on the cluster, so a backslash from a Windows
        walk would name a file that is not there."""
        _write(tmp_path, "assets/maps/skirmish.tmx", b"map")
        assert tree_entries(tmp_path)[0]["path"] == "assets/maps/skirmish.tmx"

    def test_an_empty_file_is_an_entry_rather_than_an_omission(self, tmp_path: Path) -> None:
        """The real tree carries markers the game enables its builtin mods
        with. Dropping them would let two trees differing by one digest
        alike."""
        _write(tmp_path, "builtin_mods_enabled", b"")
        assert tree_entries(tmp_path)[0]["size_bytes"] == 0

    def test_a_directory_is_not_an_entry(self, tmp_path: Path) -> None:
        """A directory holds no bytes, so it carries no identity."""
        _write(tmp_path, "assets/maps/skirmish.tmx", b"map")
        (tmp_path / "empty-dir").mkdir()
        assert [entry["path"] for entry in tree_entries(tmp_path)] == ["assets/maps/skirmish.tmx"]

    def test_the_order_is_by_byte_not_by_locale(self, tmp_path: Path) -> None:
        """Ordered to match ``LC_ALL=C sort``, which is what makes the digest
        reproducible with coreutils on the cluster."""
        for name in _MIXED_CASE_NAMES:
            _write(tmp_path, name, name.encode("utf-8"))
        assert [entry["path"] for entry in tree_entries(tmp_path)] == ["Zulu.txt", "alpha.txt"]

    def test_an_absent_root_is_refused_rather_than_read_as_empty(self, tmp_path: Path) -> None:
        """ "The assets are gone" and "the assets are unchanged" must not
        produce the same value."""
        with pytest.raises(TreeIdentityError) as caught:
            tree_entries(tmp_path / "not-here")
        assert caught.value.code == "RW-TREE-001"

    def test_a_file_given_as_a_root_is_refused(self, tmp_path: Path) -> None:
        target = _write(tmp_path, "game.txt", b"one")
        with pytest.raises(TreeIdentityError) as caught:
            tree_entries(target)
        assert caught.value.code == "RW-TREE-001"

    def test_a_root_holding_no_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "assets" / "maps").mkdir(parents=True)
        with pytest.raises(TreeIdentityError) as caught:
            tree_entries(tmp_path)
        assert caught.value.code == "RW-TREE-002"


def test_a_symlink_is_refused_rather_than_followed_or_skipped(tmp_path: Path) -> None:
    """An archive resolves a link differently from a filesystem walk, so a
    digest that accepted one would depend on which of the two produced it.
    Skipping it silently would be worse still: a broken link subtracts a file
    from the identity without subtracting it from the tree.

    Not skipped on Windows. Creating a link there needs a privilege the
    operating system withholds by default, so this test states a requirement
    on the machine that runs the suite -- and failing loudly when that
    requirement is unmet is the point: the branch is otherwise unverified
    everywhere, which is exactly how a guard outlives its subject.
    """
    _write(tmp_path, "game.txt", b"one")
    (tmp_path / "link.txt").symlink_to(tmp_path / "game.txt")
    with pytest.raises(TreeIdentityError) as caught:
        tree_entries(tmp_path)
    assert caught.value.code == "RW-TREE-003"


class TestTheDigest:
    def test_the_same_bytes_in_two_places_digest_alike(self, tmp_path: Path) -> None:
        """Identity travels with the content, which is what lets a tree be
        staged: the same tree under another path is the same tree."""
        for root in ("a", "b"):
            _write(tmp_path / root, "assets/maps/skirmish.tmx", b"map")
            _write(tmp_path / root, "game.txt", b"one")
        assert digest_tree(tmp_path / "a") == digest_tree(tmp_path / "b")

    def test_one_changed_byte_changes_the_digest(self, tmp_path: Path) -> None:
        _write(tmp_path / "a", "game.txt", b"one")
        _write(tmp_path / "b", "game.txt", b"onf")
        assert digest_tree(tmp_path / "a") != digest_tree(tmp_path / "b")

    def test_a_renamed_file_changes_the_digest(self, tmp_path: Path) -> None:
        """The listing digests paths as well as contents, because a map moved
        out of ``maps/`` is a map the engine will not find."""
        _write(tmp_path / "a", "maps/skirmish.tmx", b"map")
        _write(tmp_path / "b", "maps/duel.tmx", b"map")
        assert digest_tree(tmp_path / "a") != digest_tree(tmp_path / "b")

    def test_an_added_file_changes_the_digest(self, tmp_path: Path) -> None:
        _write(tmp_path / "a", "game.txt", b"one")
        _write(tmp_path / "b", "game.txt", b"one")
        _write(tmp_path / "b", "mods/extra.rwmod", b"mod")
        assert digest_tree(tmp_path / "a") != digest_tree(tmp_path / "b")

    def test_it_is_a_lowercase_hex_sha256(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        digest = digest_tree(tmp_path)
        assert len(digest) == SHA256_HEX_LENGTH
        assert set(digest) <= set("0123456789abcdef")

    def test_digesting_no_entries_is_refused(self) -> None:
        """Digesting the empty string would hand every absent tree one shared,
        plausible-looking value."""
        with pytest.raises(TreeIdentityError) as caught:
            tree_digest(())
        assert caught.value.code == "RW-TREE-002"


class TestTheListing:
    def test_a_line_is_what_sha256sum_prints(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        entries = tree_entries(tmp_path)
        assert render_listing(entries) == (
            f"{sha256(b'one').hexdigest()}{LISTING_SEPARATOR}game.txt",
        )

    def test_the_separator_is_the_text_mode_one(self) -> None:
        """``sha256sum -b`` writes a space and an asterisk instead, and a
        listing rendered that way would not compare equal to the cluster's."""
        assert LISTING_SEPARATOR == "  "

    def test_the_digest_is_the_digest_of_the_listing(self, tmp_path: Path) -> None:
        """Stated as the composition it is, so the property the docstring
        advertises -- reproducible with coreutils -- is the property tested."""
        _write(tmp_path, "game.txt", b"one")
        _write(tmp_path, "maps/skirmish.tmx", b"map")
        entries = tree_entries(tmp_path)
        listing = "".join(f"{line}\n" for line in render_listing(entries))
        assert tree_digest(entries) == sha256(listing.encode("utf-8")).hexdigest()


#: A fixed three-file tree and what GNU coreutils computes for it.
#:
#: The value is not derived from this package. It was produced on 2026-08-29
#: by running the command :mod:`rw_bot.tree_identity` publishes, against a
#: real tree of exactly these bytes::
#:
#:     find . -type f -printf '%P\n' | LC_ALL=C sort \
#:         | xargs -d '\n' sha256sum --text | sha256sum
#:
#: Recorded here rather than re-run because the tools it needs -- ``sh``,
#: ``sha256sum``, ``xargs`` -- are not on the PATH the test runner uses on
#: this workstation, and a check that quietly skipped would be the one thing
#: worse than no check. Pinning the answer keeps the assertion real
#: everywhere: any drift in the ordering, the separator or the trailing
#: newline moves this digest.
#:
#: ``--text`` in that command is load-bearing. GNU coreutils defaults to text
#: mode on Linux and to BINARY mode on Windows, and the two spell the
#: separator differently -- two spaces against a space and an asterisk -- so
#: the same tree digests to two values depending on which machine checked it.
#: Written without it first, and this comparison is what caught that.
COREUTILS_TREE = (
    ("game.txt", b"one"),
    ("assets/maps/skirmish.tmx", b"map"),
    ("builtin_mods_enabled", b""),
)
COREUTILS_DIGEST = "b879ba31eb5526db7219265237080ea5b9005d32c220761d7d978be139d3c26a"


def test_the_digest_is_the_one_coreutils_computed(tmp_path: Path) -> None:
    """The claim the module's docstring makes, held against a real answer.

    A record only this package can verify is a record nobody checks, so the
    value on the right came out of ``sha256sum`` rather than out of this code.
    If the ordering, the separator or the trailing newline ever drifts, this
    is what says so -- not a reading of the source.
    """
    for relative, payload in COREUTILS_TREE:
        _write(tmp_path, relative, payload)
    assert digest_tree(tmp_path) == COREUTILS_DIGEST


class TestTheEntryCodec:
    def test_it_round_trips(self) -> None:
        entry = TreeEntry(path="assets/maps/skirmish.tmx", sha256="a" * 64, size_bytes=12)
        assert decode_tree_entry(encode_tree_entry(entry)) == entry

    def test_a_zero_length_entry_is_accepted(self) -> None:
        """Where the staging contract refuses one: an empty marker file is a
        fact about a game tree, not a failed transfer."""
        entry = TreeEntry(path="builtin_mods_enabled", sha256="b" * 64, size_bytes=0)
        assert decode_tree_entry(encode_tree_entry(entry))["size_bytes"] == 0

    def test_a_negative_length_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_tree_entry({"path": "a.txt", "sha256": "c" * 64, "size_bytes": -1})

    def test_a_value_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_tree_entry(["not", "an", "object"])

    @pytest.mark.parametrize(
        "digest",
        ["A" * 64, "a" * 63, "a" * 65, "g" * 64],
        ids=["uppercase", "too-short", "too-long", "non-hex"],
    )
    def test_a_malformed_digest_is_refused(self, digest: str) -> None:
        """A re-cased or truncated digest no longer names the bytes it came
        from, so comparing against it would pass on the wrong file."""
        with pytest.raises(JSONTypeError):
            decode_tree_entry({"path": "a.txt", "sha256": digest, "size_bytes": 1})

    @pytest.mark.parametrize(
        "path",
        ["", "/etc/passwd", "assets\\maps\\x.tmx", "../outside.txt"],
        ids=["empty", "absolute", "backslashed", "escaping"],
    )
    def test_a_path_that_does_not_describe_this_tree_is_refused(self, path: str) -> None:
        """Something unpacking a listing joins these onto a root, so an
        absolute or escaping entry would write outside it."""
        with pytest.raises(JSONTypeError):
            decode_tree_entry({"path": path, "sha256": "d" * 64, "size_bytes": 1})


class TestThePublishedRecord:
    """The shape every staged tree's record shares: some headers a person
    reads, the archive's own digest -- which is what a stage manifest is
    checked against -- and the listing that says which file moved."""

    def test_the_archive_line_comes_before_the_listing(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        record = digest_record(("# a header",), "tree.tar", "a" * 64, tree_entries(tmp_path))
        assert record[0] == "# a header"
        assert record[1] == f"{'a' * 64}{LISTING_SEPARATOR}tree.tar"

    def test_every_entry_is_listed_after_it(self, tmp_path: Path) -> None:
        _write(tmp_path, "game.txt", b"one")
        _write(tmp_path, "maps/duel.tmx", b"map")
        entries = tree_entries(tmp_path)
        record = digest_record((), "tree.tar", "b" * 64, entries)
        assert record[1:] == render_listing(entries)

    def test_headers_are_the_callers_because_only_they_differ(self, tmp_path: Path) -> None:
        """A game tree names depots; a payload names a commit. The listing and
        the archive line are the same either way, which is why they are here
        and the headers are not."""
        _write(tmp_path, "game.txt", b"one")
        record = digest_record(
            ("# rw_bot payload at commit abc123", "# tree deadbeef"),
            "rw-payload.tar",
            "c" * 64,
            tree_entries(tmp_path),
        )
        assert record[:2] == ("# rw_bot payload at commit abc123", "# tree deadbeef")

    def test_a_record_naming_no_entry_is_refused(self) -> None:
        """It would vouch for nothing that could be staged, while looking
        like a record."""
        with pytest.raises(TreeIdentityError) as caught:
            digest_record(("# a header",), "tree.tar", "d" * 64, ())
        assert caught.value.code == "RW-TREE-002"
