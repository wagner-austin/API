"""Freezing the code a batch runs, packing it, and vouching for it.

Driven against a REAL filesystem with the real hooks, unlike most of this
suite. That is not a preference, it is what the subject is: the freeze is a
filesystem copy, the archive's bytes are what gets staged, and the digest is
taken off what actually landed. Driving it against the in-memory host would
freeze into the fake and then digest a directory that does not exist -- which
is exactly what the first version of this file did.

The freeze runs against a miniature repository rather than this one, because a
test that copied the real ``src/rw_bot`` would change its own answer every time
the package did.
"""

from __future__ import annotations

import hashlib
import re
import runpy
import sys
from pathlib import Path

import pytest
from hpc3.contracts.stage import StagedFile, decode_stage_manifest
from platform_core.json_utils import (
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_list,
)
from scripts.stage_payload import (
    AGENT_BYTECODE,
    CLASS_FILE_OFFSET,
    EXIT_OK,
    REQUIRED_FLAGS,
    freeze_config,
    main,
)

from rw_bot.harness.agent_build import JAVA_RELEASE
from rw_bot.harness.launch import (
    CATALOGUE,
    FROZEN_CATALOGUE,
    FROZEN_TYPE_DUMP,
    TYPE_DUMP,
)
from rw_bot.harness.runner import FROZEN_ENTRIES, TREE_MARKER, TREE_SOURCES
from rw_bot.harness.sweep import SweepError
from rw_bot.tree_identity import tree_digest, tree_entries

_COMMIT = "722da55a6bd3dfaa3b110c769b294af753f832c2"

#: Where the archive is staged to. An absolute cluster path, because that is
#: what the manifest's ``destination`` is required to be.
_DESTINATION = "/pub/wagnera3/rusted/staging"


def _repository(root: Path) -> None:
    """Write the miniature repository a freeze copies from.

    Built FROM :data:`~rw_bot.harness.runner.TREE_SOURCES` rather than from a
    hand-written list beside it. The list drifted: the two registry dumps
    were added to the freeze and this planted neither, so every test here
    would have gone on passing against a repository shaped like the old one
    -- which is the same shape of miss that let a member reach the cluster
    reading a catalogue that had never been staged.

    A source with a suffix is a file and one without is a directory, which is
    a property of the real list rather than a convention invented here.

    Args:
        root: Directory to build it in.
    """
    (root / "src" / "rw_bot" / "policy" / "__pycache__").mkdir(parents=True)
    (root / "src" / "rw_bot" / "__init__.py").write_bytes(b"package")
    (root / "src" / "rw_bot" / "policy" / "doom.py").write_bytes(b"policy")
    (root / "src" / "rw_bot" / "policy" / "__pycache__" / "doom.pyc").write_bytes(b"bytecode")
    for source in TREE_SOURCES:
        entry = root / source
        if entry.suffix:
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.write_bytes(source.encode("utf-8"))
        else:
            entry.mkdir(parents=True)
            (entry / f"a.{source}").write_bytes(source.encode("utf-8"))


def _argv(root: Path, tree: str = "frozen") -> list[str]:
    """Build the command line the freezer is run with.

    Args:
        root: The working directory.
        tree: Where the snapshot goes, relative to it.

    Returns:
        The arguments after the program name.
    """
    return [
        "--tree",
        str(root / tree),
        "--commit",
        _COMMIT,
        "--archive",
        str(root / "out" / "rw-payload.tar"),
        "--out",
        str(root / "provenance" / "payload-tree.json"),
        "--digests",
        str(root / "provenance" / "payload-tree-digests.txt"),
        "--manifest",
        str(root / "provenance" / "stage-payload-tree.json"),
        "--destination",
        _DESTINATION,
    ]


def _planted(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the miniature repository and enter it.

    Entering is load-bearing: the freeze reads its sources from paths relative
    to the process, which is precisely the property that makes it impossible
    on a compute node and the reason this staging exists at all.

    Args:
        root: Where to build it.
        monkeypatch: Used to change directory.
    """
    _repository(root)
    monkeypatch.chdir(root)


def _document(root: Path) -> dict[str, JSONValue]:
    """Read back the document the freezer wrote.

    Args:
        root: The working directory.

    Returns:
        The decoded document.

    Raises:
        JSONTypeError: When what was written is not a JSON object.
    """
    text = (root / "provenance" / "payload-tree.json").read_text(encoding="utf-8")
    return narrow_json_to_dict(load_json_str(text))


class TestTheStageManifestItWrites:
    def test_it_names_the_archive_this_run_actually_wrote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Composed, never typed. The manifest here was hand-written and went
        stale the moment the tree gained a file: the payload grew from 407
        entries to 409, the archive changed digest and length, and the
        manifest went on naming the old ones -- which stages either the wrong
        tree or nothing at all, and reads as correct either way."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        manifest = decode_stage_manifest(
            load_json_str(
                (tmp_path / "provenance" / "stage-payload-tree.json").read_text(encoding="utf-8")
            )
        )
        archive = tmp_path / "out" / "rw-payload.tar"
        assert manifest["destination"] == _DESTINATION
        assert manifest["files"] == [
            StagedFile(
                name="rw-payload.tar",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                size_bytes=archive.stat().st_size,
            )
        ]

    def test_its_provenance_carries_the_tree_the_document_describes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The digests prove the bytes on the cluster are the bytes named
        here; only the provenance says what "here" is."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        manifest = decode_stage_manifest(
            load_json_str(
                (tmp_path / "provenance" / "stage-payload-tree.json").read_text(encoding="utf-8")
            )
        )
        document = _document(tmp_path)
        assert manifest["provenance"]["tree_sha256"] == document["tree_sha256"]
        assert manifest["provenance"]["git_commit"] == _COMMIT
        assert manifest["provenance"]["entries"] == str(len(require_list(document, "entries")))

    def test_the_bytecode_claim_is_read_from_the_builder(self) -> None:
        """The number IS the claim: the Linux depot's JRE reads class-file
        versions up to 52, and a jar one release higher is a FATAL ERROR in
        native method before the game starts, on every member."""
        assert AGENT_BYTECODE.startswith(f"java {JAVA_RELEASE} ")
        assert f"class {int(JAVA_RELEASE) + CLASS_FILE_OFFSET}" in AGENT_BYTECODE


class TestTheConfigItFreezesFrom:
    def test_the_tree_is_the_only_thing_it_really_says(self) -> None:
        config = freeze_config("/tmp/frozen")
        assert config["tree"] == "/tmp/frozen"
        assert config["out_dir"] == "/tmp/frozen"

    def test_every_other_field_is_a_real_value(self) -> None:
        """The decoder refuses empties, and a payload of placeholders is
        harder to read later than one that says what it is."""
        config = freeze_config("/tmp/frozen")
        assert config["workers"] == 1
        assert config["source_game_dir"] == ".game"


class TestTheFrozenTree:
    def test_it_carries_everything_a_match_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        assert main(_argv(tmp_path)) == EXIT_OK
        for name in (TREE_MARKER, *FROZEN_ENTRIES):
            assert (tmp_path / "frozen" / name).exists(), name

    def test_the_agent_jar_lands_flat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy lands under its own basename, so ``agent/build/rw-agent.jar``
        arrives as ``rw-agent.jar`` -- which is where the launcher looks."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        landed = tmp_path / "frozen" / "rw-agent.jar"
        assert landed.read_bytes() == b"agent/build/rw-agent.jar"
        assert not (tmp_path / "frozen" / "agent").exists()

    def test_the_registry_dumps_the_planner_reads_land_flat_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The planner reads the catalogue and the type dump by path at
        startup, and the launcher's defaults for both are repository-relative
        -- true only where a repository is. A compute node starts in a home
        directory, so the first member to reach the planner died on
        ``FileNotFoundError: 'wiki/sources/m0-probe/printunits.log'`` having
        already patched the engine, seeded it and held the world at frame
        one (job 55663569, 2026-08-30)."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        frozen = tmp_path / "frozen"
        assert (frozen / FROZEN_CATALOGUE).read_bytes() == CATALOGUE.encode("utf-8")
        assert (frozen / FROZEN_TYPE_DUMP).read_bytes() == TYPE_DUMP.encode("utf-8")
        assert not (frozen / "wiki").exists()

    def test_the_job_files_travel_with_the_doctrines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A compute node reads its job file out of the payload, and the file
        naming which arms and seeds a batch played is as much the experiment
        as the doctrines are."""
        assert "sweeps" in TREE_SOURCES
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        assert (tmp_path / "frozen" / "sweeps" / "a.sweeps").exists()

    def test_no_bytecode_is_carried(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``.pyc`` embeds the source's timestamp, so carrying one would
        make two freezes of identical source digest differently."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        assert list((tmp_path / "frozen").rglob("__pycache__")) == []

    def test_an_absent_source_stops_the_freeze_naming_the_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The agent jar is built by ``make agent`` and is not in the
        repository, so a clean checkout has no jar to freeze. The copy itself
        refuses, naming the path -- nothing is packed and nothing is staged."""
        _planted(tmp_path, monkeypatch)
        (tmp_path / "agent" / "build" / "rw-agent.jar").unlink()
        with pytest.raises(FileNotFoundError, match=re.escape("rw-agent.jar")):
            main(_argv(tmp_path))
        assert not (tmp_path / "out" / "rw-payload.tar").exists()

    def test_a_tree_frozen_without_the_jar_is_refused_before_packing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half, and the one that matters on a node: a tree whose
        marker says the freeze finished but which is missing something a match
        reads. The marker certifies the copies BEFORE it and nothing else."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        (tmp_path / "frozen" / "rw-agent.jar").unlink()
        (tmp_path / "out" / "rw-payload.tar").unlink()
        with pytest.raises(SweepError) as caught:
            main(_argv(tmp_path))
        assert caught.value.code == "RW-SWEEP-006"
        assert "rw-agent.jar" in str(caught.value)
        assert not (tmp_path / "out" / "rw-payload.tar").exists()


class TestTheDocumentItWrites:
    def test_it_pins_the_commit_it_was_told(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given rather than read from git: the tree frozen is the WORKING
        one, and a command that stamped HEAD onto a dirty tree would print a
        reassuring lie."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        assert _document(tmp_path)["git_commit"] == _COMMIT

    def test_the_identity_is_the_tree_it_packed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        assert _document(tmp_path)["tree_sha256"] == tree_digest(tree_entries(tmp_path / "frozen"))

    def test_the_archive_it_names_is_the_one_it_wrote(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        packed = tmp_path / "out" / "rw-payload.tar"
        assert _document(tmp_path)["archive_size_bytes"] == packed.stat().st_size

    def test_the_output_directory_is_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The archive's parent did not exist, and a tool that made the caller
        create it first would be one more thing to get right."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        assert (tmp_path / "out" / "rw-payload.tar").exists()

    def test_the_record_names_the_archive_digest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It is what ``hpc3-stage --expect-from`` is actually checked
        against."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        record = (tmp_path / "provenance" / "payload-tree-digests.txt").read_text(encoding="utf-8")
        assert str(_document(tmp_path)["archive_sha256"]) in record

    def test_the_record_names_the_commit_for_a_reader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path))
        record = (tmp_path / "provenance" / "payload-tree-digests.txt").read_text(encoding="utf-8")
        assert _COMMIT in record


class TestTheEntryPoint:
    def test_two_freezes_of_one_source_pack_alike(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The property that makes the payload a pinned artifact rather than
        a directory each run assembled for itself."""
        _planted(tmp_path, monkeypatch)
        main(_argv(tmp_path, tree="first"))
        main(_argv(tmp_path, tree="second"))
        assert tree_digest(tree_entries(tmp_path / "first")) == tree_digest(
            tree_entries(tmp_path / "second")
        )

    def test_a_missing_flag_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        # Named from the flag table rather than written out, so adding a
        # required flag does not leave this asserting about the old last one.
        with pytest.raises(ValueError, match=f"{REQUIRED_FLAGS[-1]} is required"):
            main(_argv(tmp_path)[:-2])

    def test_it_reads_the_process_arguments_when_given_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        # Swapped and restored rather than patched: ``monkeypatch.setattr``
        # is banned here, and this hook is the one seam the file cannot drive
        # through the in-memory host -- everything else about it is real
        # filesystem work.
        before = sys.argv
        sys.argv = ["stage_payload", *_argv(tmp_path)]
        try:
            assert main(None) == EXIT_OK
        finally:
            sys.argv = before

    def test_the_module_guard_runs_main(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _planted(tmp_path, monkeypatch)
        before = sys.argv
        sys.argv = ["stage_payload", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as caught:
                runpy.run_module("scripts.stage_payload", run_name="__main__")
            assert caught.value.code == EXIT_OK
        finally:
            sys.argv = before
