"""What the staged game tree is, as a document rather than as a directory.

The load-bearing test is the last class: the digest record written here is
handed to ``hpc3``'s OWN reader and its OWN refusal, because a record only
this side had checked would be rejected at staging time -- after 311 MB had
crossed the network.
"""

from __future__ import annotations

import pathlib

import pytest
from hpc3.contracts.stage import decode_stage_manifest
from hpc3.core.expected import check_expected, read_expected_digests
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str

from rw_bot.harness.clone import VOLATILE_FILES
from rw_bot.staged_tree import (
    APP_ID,
    CONTENT_DEPOT,
    LINUX_DEPOT,
    NO_SOURCE,
    ORIGIN_ASSEMBLED,
    ORIGIN_MIRRORED,
    ORIGIN_PINNED_STATE,
    ORIGIN_RENAMED,
    ORIGIN_VERIFIED,
    ORIGINS,
    ROLE_CONTENT,
    ROLE_LINUX_RUNTIME,
    ClassifiedEntry,
    DepotRef,
    StagedTreeError,
    StagedTreeSpec,
    classify_entries,
    count_origins,
    decode_classified_entry,
    decode_depot,
    decode_staged_tree_spec,
    encode_classified_entry,
    encode_depot,
    encode_staged_tree_spec,
    render_digest_record,
)
from rw_bot.tree_identity import TreeEntry, tree_digest

_ARCHIVE = "rusted-warfare-linux.tar"
_ARCHIVE_DIGEST = "f" * 64
_DESTINATION = "/pub/wagnera3/rusted/staging"

#: Digests for the sample entries. Distinct constants rather than computed,
#: so a test that expected the wrong classification cannot be rescued by
#: accidentally matching.
_JAR = "a" * 64
_MAP = "b" * 64
_RUNTIME = "c" * 64


def _entry(path: str, digest: str, size: int = 12) -> TreeEntry:
    """Build one tree entry.

    Args:
        path: Its path under the root.
        digest: Its sha256.
        size: Its length.

    Returns:
        The entry.
    """
    return TreeEntry(path=path, sha256=digest, size_bytes=size)


#: A tree carrying one of each origin: a file the reference also has at that
#: path, a shell-safe copy of a reference file under another name, and a
#: runtime file the reference does not have at all.
_TREE = (
    _entry("assets/maps/[p2]duel_lake.tmx", _MAP),
    _entry("game-lib.jar", _JAR),
    _entry("jvm-linux/bin/java", _RUNTIME),
)
_REFERENCE = (
    _entry("assets/maps/[p2]Lake (2p).tmx", _MAP),
    _entry("game-lib.jar", _JAR),
)

#: Paths the game rewrites on every boot, as
#: :data:`rw_bot.harness.clone.VOLATILE_FILES` names them. Passed in rather
#: than known by the module under test, so the one exception to the
#: same-path-must-agree rule has a single owner.
_REWRITTEN = VOLATILE_FILES


def _classified() -> tuple[ClassifiedEntry, ...]:
    """Classify the sample tree against the sample reference.

    Returns:
        One entry per file of :data:`_TREE`, in the order given.
    """
    return classify_entries(_TREE, _REFERENCE, rewritten=_REWRITTEN)


def _spec(entries: tuple[ClassifiedEntry, ...] | None = None) -> StagedTreeSpec:
    """Build a described tree.

    Args:
        entries: Its classified entries. ``None`` classifies the sample tree.

    Returns:
        The spec, its stated identity computed from its own entries.
    """
    classified = _classified() if entries is None else entries
    return StagedTreeSpec(
        app_id=APP_ID,
        build_id="9902063",
        depots=[
            DepotRef(depot_id=CONTENT_DEPOT, manifest="9090535937117498741", role=ROLE_CONTENT),
            DepotRef(depot_id=LINUX_DEPOT, manifest="223921525878913700", role=ROLE_LINUX_RUNTIME),
        ],
        reference=r"C:\Program Files (x86)\Steam\steamapps\common\Rusted Warfare",
        entries=list(classified),
        tree_sha256=tree_digest(classified),
        archive_name=_ARCHIVE,
        archive_sha256=_ARCHIVE_DIGEST,
        archive_size_bytes=326_000_000,
    )


class TestClassifyingATree:
    def test_a_file_the_reference_has_at_that_path_is_verified(self) -> None:
        """Content depot 647961 declares no oslist, so a match against Steam's
        own installed copy is an independent second opinion."""
        found = {entry["path"]: entry for entry in _classified()}
        assert found["game-lib.jar"]["origin"] == ORIGIN_VERIFIED
        assert found["game-lib.jar"]["source"] == "game-lib.jar"

    def test_the_same_filename_elsewhere_is_mirrored_rather_than_renamed(self) -> None:
        """Sixty-four of these, almost all vendor data the two bundled JVMs
        both ship. Verified as firmly as a same-path match; the path differs
        only because the platform's runtime directory does."""
        classified = classify_entries(
            (_entry("jvm-linux/COPYRIGHT", _JAR),),
            (_entry("jvm/COPYRIGHT", _JAR),),
            rewritten=_REWRITTEN,
        )
        assert classified[0]["origin"] == ORIGIN_MIRRORED
        assert classified[0]["source"] == "jvm/COPYRIGHT"

    def test_an_empty_file_is_never_attributed_to_another_empty_file(self) -> None:
        """Every empty file shares one digest, so attribution by digest says
        nothing about one. The first run of this called a JVM security file a
        copy of a map's demo marker -- a confident wrong answer of exactly the
        shape a provenance document exists to prevent."""
        classified = classify_entries(
            (_entry("jvm-linux/lib/security/trusted.libraries", _JAR, size=0),),
            (_entry("assets/maps/[p2]Fire Bridge (2p)_demo", _JAR, size=0),),
            rewritten=_REWRITTEN,
        )
        assert classified[0]["origin"] == ORIGIN_ASSEMBLED
        assert classified[0]["source"] == NO_SOURCE

    def test_an_empty_file_at_the_same_path_is_still_verified(self) -> None:
        """The refusal is only about attributing across paths: a marker file
        the reference also has AT THAT PATH is checked the ordinary way."""
        classified = classify_entries(
            (_entry("assets/builtin_mods_enabled", _JAR, size=0),),
            (_entry("assets/builtin_mods_enabled", _JAR, size=0),),
            rewritten=_REWRITTEN,
        )
        assert classified[0]["origin"] == ORIGIN_VERIFIED

    def test_a_byte_copy_under_another_name_is_a_rename_and_names_its_source(self) -> None:
        """Seven of these, and finding them is why the module exists. Every
        sweep plays ``[p2]duel_lake.tmx``, which is a copy of a Steam map --
        rebuilding the tree "cleanly" from Steam would have deleted it."""
        found = {entry["path"]: entry for entry in _classified()}
        renamed = found["assets/maps/[p2]duel_lake.tmx"]
        assert renamed["origin"] == ORIGIN_RENAMED
        assert renamed["source"] == "assets/maps/[p2]Lake (2p).tmx"

    def test_a_file_with_no_counterpart_is_stated_as_assembled(self) -> None:
        """The Linux runtime. There is no second local copy, so the document
        says where it came from rather than claiming it was checked."""
        found = {entry["path"]: entry for entry in _classified()}
        assert found["jvm-linux/bin/java"]["origin"] == ORIGIN_ASSEMBLED
        assert found["jvm-linux/bin/java"]["source"] == NO_SOURCE

    def test_the_order_is_the_one_it_was_given(self) -> None:
        assert [entry["path"] for entry in _classified()] == [entry["path"] for entry in _TREE]

    def test_one_path_with_two_different_bytes_stops_the_document(self) -> None:
        """The case no later check would catch: each digest matches the file
        it came from, and the transfer matches the digest, so a tree carrying
        the wrong version of a file stages clean and runs a different
        simulation."""
        with pytest.raises(StagedTreeError) as caught:
            classify_entries(
                (_entry("game-lib.jar", _JAR),),
                (_entry("game-lib.jar", _RUNTIME),),
                rewritten=_REWRITTEN,
            )
        assert caught.value.code == "RW-TREE-101"
        assert "game-lib.jar" in str(caught.value)

    def test_a_file_the_game_rewrites_is_pinned_state_rather_than_a_disagreement(self) -> None:
        """``preferences.ini`` is rewritten on every boot, so the two copies
        differing says nothing. It is staged anyway, because the runner resets
        every worker's copy from it and a tree without it fails on the node."""
        settings = VOLATILE_FILES[0]
        classified = classify_entries(
            (_entry(settings, _JAR),), (_entry(settings, _RUNTIME),), rewritten=_REWRITTEN
        )
        assert classified[0]["origin"] == ORIGIN_PINNED_STATE
        assert classified[0]["source"] == NO_SOURCE

    def test_the_exception_covers_only_the_paths_the_caller_named(self) -> None:
        """One owner for the rule, so the allowance cannot grow here. The same
        file is a refusal the moment it is not declared rewritten."""
        settings = VOLATILE_FILES[0]
        with pytest.raises(StagedTreeError) as caught:
            classify_entries((_entry(settings, _JAR),), (_entry(settings, _RUNTIME),), rewritten=())
        assert caught.value.code == "RW-TREE-101"

    def test_an_empty_tree_is_refused(self) -> None:
        with pytest.raises(StagedTreeError) as caught:
            classify_entries((), _REFERENCE, rewritten=_REWRITTEN)
        assert caught.value.code == "RW-TREE-103"

    def test_a_reference_holding_one_file_twice_names_one_source(self) -> None:
        """The reference arrives byte-sorted, so a duplicated file always
        names the same source and two runs of this produce one document."""
        reference = (_entry("a.tmx", _MAP), _entry("z.tmx", _MAP))
        classified = classify_entries((_entry("copy.tmx", _MAP),), reference, rewritten=_REWRITTEN)
        assert classified[0]["source"] == "a.tmx"


class TestTheTally:
    def test_it_counts_each_origin(self) -> None:
        counts = count_origins(_classified())
        assert counts == {
            ORIGIN_VERIFIED: 1,
            ORIGIN_MIRRORED: 0,
            ORIGIN_RENAMED: 1,
            ORIGIN_ASSEMBLED: 1,
            ORIGIN_PINNED_STATE: 0,
        }

    def test_an_origin_with_no_entries_is_reported_as_zero(self) -> None:
        """An absent key and a zero read alike, and only one of them means the
        tree has no such file."""
        counts = count_origins(
            classify_entries((_entry("game-lib.jar", _JAR),), _REFERENCE, rewritten=_REWRITTEN)
        )
        assert set(counts) == set(ORIGINS)
        assert counts[ORIGIN_ASSEMBLED] == 0


class TestTheCodec:
    def test_a_depot_round_trips(self) -> None:
        depot = DepotRef(depot_id=CONTENT_DEPOT, manifest="9090535937117498741", role=ROLE_CONTENT)
        assert decode_depot(encode_depot(depot)) == depot

    @pytest.mark.parametrize("field", ["depot_id", "manifest", "role"])
    def test_a_depot_field_left_empty_is_refused(self, field: str) -> None:
        """A depot with no manifest names a moving target, which is the one
        thing the document must not do."""
        raw = encode_depot(DepotRef(depot_id=CONTENT_DEPOT, manifest="909053", role=ROLE_CONTENT))
        with pytest.raises(JSONTypeError):
            decode_depot({**raw, field: ""})

    def test_an_entry_round_trips(self) -> None:
        entry = _classified()[0]
        assert decode_classified_entry(encode_classified_entry(entry)) == entry

    def test_an_entry_naming_an_unknown_origin_is_refused(self) -> None:
        """A reader tallying origins would silently drop it."""
        raw = encode_classified_entry(_classified()[0])
        with pytest.raises(StagedTreeError) as caught:
            decode_classified_entry({**raw, "origin": "probably-fine"})
        assert caught.value.code == "RW-TREE-102"

    def test_the_document_round_trips(self) -> None:
        spec = _spec()
        assert decode_staged_tree_spec(encode_staged_tree_spec(spec)) == spec

    def test_it_round_trips_through_real_json(self) -> None:
        """Written to disk and read back by whoever audits the staging."""
        spec = _spec()
        text = dump_json_str(encode_staged_tree_spec(spec), indent=2)
        assert decode_staged_tree_spec(load_json_str(text)) == spec

    def test_a_document_whose_listing_and_identity_disagree_is_refused(self) -> None:
        """The load-bearing decode check. The digest is what everything
        downstream compares against, so a document stating one while carrying
        the entries of another describes two different trees."""
        raw = encode_staged_tree_spec(_spec())
        with pytest.raises(StagedTreeError) as caught:
            decode_staged_tree_spec({**raw, "tree_sha256": "d" * 64})
        assert caught.value.code == "RW-TREE-103"

    def test_a_document_naming_no_depot_is_refused(self) -> None:
        raw = encode_staged_tree_spec(_spec())
        empty: list[JSONValue] = []
        with pytest.raises(JSONTypeError):
            decode_staged_tree_spec({**raw, "depots": empty})

    def test_a_document_carrying_no_entry_is_refused(self) -> None:
        raw = encode_staged_tree_spec(_spec())
        empty: list[JSONValue] = []
        with pytest.raises(JSONTypeError):
            decode_staged_tree_spec({**raw, "entries": empty})

    def test_a_zero_length_archive_is_refused(self) -> None:
        raw = encode_staged_tree_spec(_spec())
        with pytest.raises(JSONTypeError):
            decode_staged_tree_spec({**raw, "archive_size_bytes": 0})


class TestTheRecordHpc3Reads:
    """The record is written by this act and read by another, which is the
    whole point: ``hpc3.core.expected`` exists because a manifest emitted
    alongside its own files always agrees with them."""

    def _record(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Write the digest record to a real file.

        Args:
            tmp_path: Directory to write into.

        Returns:
            The record's path.
        """
        path = tmp_path / "rusted-tree-digests.txt"
        path.write_text("\n".join(render_digest_record(_spec())) + "\n", encoding="utf-8")
        return path

    def _manifest(self, digest: str = _ARCHIVE_DIGEST) -> JSONValue:
        """Build the stage manifest this tree would be staged with.

        Args:
            digest: The archive digest it names. Given so the negative case
                can name one the record does not vouch for.

        Returns:
            The manifest document, decoded by hpc3's own decoder so a shape
            it would refuse cannot pass here.
        """
        spec = _spec()
        return {
            "destination": _DESTINATION,
            "files": [
                {
                    "name": spec["archive_name"],
                    "sha256": digest,
                    "size_bytes": spec["archive_size_bytes"],
                }
            ],
            "provenance": {
                "app_id": spec["app_id"],
                "build_id": spec["build_id"],
                "tree_sha256": spec["tree_sha256"],
            },
        }

    def test_hpc3_finds_the_archive_digest_in_it(self, tmp_path: pathlib.Path) -> None:
        assert _ARCHIVE_DIGEST in read_expected_digests(self._record(tmp_path))

    def test_hpc3_admits_the_manifest_this_tree_would_be_staged_with(
        self, tmp_path: pathlib.Path
    ) -> None:
        """End to end against the real refusal: a record this side had only
        checked itself would be rejected at staging, after 311 MB had already
        crossed the network."""
        record = self._record(tmp_path)
        manifest = decode_stage_manifest(self._manifest())
        check_expected(manifest, read_expected_digests(record), source=record)

    def test_hpc3_refuses_an_archive_the_record_does_not_name(self, tmp_path: pathlib.Path) -> None:
        """The negative half. Without it the test above would pass against a
        record that vouched for everything."""
        record = self._record(tmp_path)
        with pytest.raises(AppError):
            check_expected(
                decode_stage_manifest(self._manifest("e" * 64)),
                read_expected_digests(record),
                source=record,
            )

    def test_it_names_the_depots_and_the_tally_for_a_reader(self) -> None:
        """Read by a person as often as by the tool -- "which files were
        actually checked" is the question the document exists to answer."""
        text = "\n".join(render_digest_record(_spec()))
        assert "depot 647961 manifest 9090535937117498741 (content)" in text
        assert "depot 647963 manifest 223921525878913700 (linux-runtime)" in text
        assert f"1 {ORIGIN_VERIFIED}" in text
        assert f"1 {ORIGIN_ASSEMBLED}" in text

    def test_every_file_is_listed_so_a_moved_one_can_be_found(self) -> None:
        """The tree digest says something changed; these lines say what."""
        text = "\n".join(render_digest_record(_spec()))
        for entry in _TREE:
            assert f"{entry['sha256']}  {entry['path']}" in text

    def test_a_spec_with_no_entries_vouches_for_nothing(self) -> None:
        empty = _spec()
        empty["entries"] = []
        with pytest.raises(StagedTreeError) as caught:
            render_digest_record(empty)
        assert caught.value.code == "RW-TREE-103"
