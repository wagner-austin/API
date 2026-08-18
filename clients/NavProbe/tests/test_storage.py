"""Tests for persisting and reloading run records.

Two layers are exercised deliberately. The hook-level tests rebind
:mod:`navprobe._test_hooks` to an in-memory filesystem so the call sequence is
checked without touching a disk. The real-filesystem tests leave the production
hooks in place and write under ``tmp_path``, because those hooks are production
code and a suite that only ever ran fakes would never execute them.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from navprobe import _test_hooks
from navprobe.codecs.observation import OBSERVATION_BANNER
from navprobe.codecs.run import RUN_BANNER
from navprobe.codecs.trial import TRIAL_BANNER
from navprobe.records import (
    ObservationRecord,
    RunRecord,
    RunSpec,
    StepRecord,
    TrialRecord,
    TrialSpec,
)
from navprobe.storage import (
    load_observation_record,
    load_run_record,
    load_trial_record,
    save_observation_record,
    save_run_record,
    save_trial_record,
)
from navprobe.wireformat import WireFormatError


def _record() -> RunRecord:
    """Build a valid run record.

    Returns:
        A record with two contiguous steps.
    """
    return RunRecord(
        spec=RunSpec(label="same-process", seed=7, step_count=2, world_count=3),
        steps=(
            StepRecord(step_index=0, digest="aa"),
            StepRecord(step_index=1, digest="bb"),
        ),
        digest="cc",
    )


def _observation() -> ObservationRecord:
    """Build a valid observation record.

    Returns:
        A record whose values span negative, zero and inexact-in-decimal.
    """
    return ObservationRecord(label="wsl-cuda", seed=7, step_count=200, values=(-1.5, 0.0, 0.055))


def _trial() -> TrialRecord:
    """Build a valid trial record.

    Returns:
        A record reporting a deterministic three-repetition trial.
    """
    return TrialRecord(
        spec=TrialSpec(seed=7, step_count=4, repetitions=3),
        world_count=2,
        reference_digest="cc",
        deterministic=True,
        first_divergent_step=None,
    )


class MemoryFiles:
    """An in-memory filesystem satisfying the storage hooks.

    Real implementations of the three hook Protocols, backed by a dict instead
    of a disk. Records which parents were created so the ordering contract can
    be asserted rather than assumed.
    """

    def __init__(self) -> None:
        self.contents: dict[Path, str] = {}
        self.created_parents: list[Path] = []

    def read_text(self, path: Path) -> str:
        """Read a stored file.

        Args:
            path: File to read.

        Returns:
            The stored contents.

        Raises:
            FileNotFoundError: When nothing was stored at ``path``.
        """
        if path not in self.contents:
            raise FileNotFoundError(str(path))
        return self.contents[path]

    def write_text(self, path: Path, text: str) -> None:
        """Store a file's contents.

        Args:
            path: File to write.
            text: Contents to store.
        """
        self.contents[path] = text

    def make_parent_dirs(self, path: Path) -> None:
        """Record that a path's parents were created.

        Args:
            path: File whose parents should exist.
        """
        self.created_parents.append(path.parent)


@pytest.fixture()
def memory_files() -> Generator[MemoryFiles, None, None]:
    """Rebind the storage hooks to an in-memory filesystem.

    Yields:
        The in-memory filesystem the hooks are bound to.
    """
    files = MemoryFiles()
    original_read = _test_hooks.read_text
    original_write = _test_hooks.write_text
    original_parents = _test_hooks.make_parent_dirs
    _test_hooks.read_text = files.read_text
    _test_hooks.write_text = files.write_text
    _test_hooks.make_parent_dirs = files.make_parent_dirs
    try:
        yield files
    finally:
        _test_hooks.read_text = original_read
        _test_hooks.write_text = original_write
        _test_hooks.make_parent_dirs = original_parents


class TestSaveRunRecord:
    """Tests for :func:`save_run_record`."""

    def test_writes_to_the_requested_path(self, memory_files: MemoryFiles) -> None:
        """The record lands at the path given."""
        save_run_record(Path("runs/a.txt"), _record())
        assert sorted(memory_files.contents) == [Path("runs/a.txt")]

    def test_creates_the_parent_directory(self, memory_files: MemoryFiles) -> None:
        """Missing parents are created before the write."""
        save_run_record(Path("runs/nested/a.txt"), _record())
        assert memory_files.created_parents == [Path("runs/nested")]

    def test_writes_the_versioned_banner(self, memory_files: MemoryFiles) -> None:
        """The stored text is the wire format, starting with its banner."""
        save_run_record(Path("a.txt"), _record())
        assert memory_files.contents[Path("a.txt")].startswith(f"{RUN_BANNER}\n")

    def test_output_is_byte_stable_for_equal_records(self, memory_files: MemoryFiles) -> None:
        """Equal records produce identical files, so files can be compared."""
        save_run_record(Path("a.txt"), _record())
        save_run_record(Path("b.txt"), _record())
        assert memory_files.contents[Path("b.txt")] == memory_files.contents[Path("a.txt")]


class TestLoadRunRecord:
    """Tests for :func:`load_run_record`."""

    def test_round_trips_through_the_hooks(self, memory_files: MemoryFiles) -> None:
        """A saved record reloads equal to what was written."""
        save_run_record(Path("a.txt"), _record())
        assert load_run_record(Path("a.txt")) == _record()

    def test_propagates_a_missing_file(self, memory_files: MemoryFiles) -> None:
        """An absent file raises rather than returning an empty record."""
        with pytest.raises(FileNotFoundError):
            load_run_record(Path("missing.txt"))

    def test_rejects_a_file_that_is_not_a_run_record(self, memory_files: MemoryFiles) -> None:
        """Readable text without the banner is refused by the codec."""
        memory_files.contents[Path("other.txt")] = "hello\n"
        with pytest.raises(WireFormatError) as caught:
            load_run_record(Path("other.txt"))
        assert caught.value.code == "NP-WIRE-009"


class TestTrialRecordStorage:
    """Tests for :func:`save_trial_record` and :func:`load_trial_record`."""

    def test_writes_the_trial_banner(self, memory_files: MemoryFiles) -> None:
        """A trial file identifies itself as a trial."""
        save_trial_record(Path("t.txt"), _trial())
        assert memory_files.contents[Path("t.txt")].startswith(f"{TRIAL_BANNER}\n")

    def test_creates_the_parent_directory(self, memory_files: MemoryFiles) -> None:
        """Trials go through the same create-then-write path as runs."""
        save_trial_record(Path("runs/nested/t.txt"), _trial())
        assert memory_files.created_parents == [Path("runs/nested")]

    def test_round_trips_through_the_hooks(self, memory_files: MemoryFiles) -> None:
        """A saved trial reloads equal to what was written."""
        save_trial_record(Path("t.txt"), _trial())
        assert load_trial_record(Path("t.txt")) == _trial()

    def test_refuses_to_load_a_run_record_as_a_trial(self, memory_files: MemoryFiles) -> None:
        """Per-type banners stop one record being read as another.

        Both files are valid records, so only the banner distinguishes them.
        Without that, a run record's first header line would decode as a trial's
        seed and the loader would return a trial that never happened.
        """
        save_run_record(Path("a.txt"), _record())
        with pytest.raises(WireFormatError) as caught:
            load_trial_record(Path("a.txt"))
        assert caught.value.code == "NP-WIRE-009"

    def test_refuses_to_load_a_trial_record_as_a_run(self, memory_files: MemoryFiles) -> None:
        """The same protection holds in the other direction."""
        save_trial_record(Path("t.txt"), _trial())
        with pytest.raises(WireFormatError) as caught:
            load_run_record(Path("t.txt"))
        assert caught.value.code == "NP-WIRE-009"


class TestObservationRecordStorage:
    """Tests for :func:`save_observation_record` and :func:`load_observation_record`.

    This is the record that exists so two environments which cannot share a
    process — two MuJoCo-Warp devices, for instance — can still be compared by
    magnitude rather than only by bit-equality.
    """

    def test_writes_the_observation_banner(self, memory_files: MemoryFiles) -> None:
        """An observation file identifies itself as one."""
        save_observation_record(Path("o.txt"), _observation())
        assert memory_files.contents[Path("o.txt")].startswith(f"{OBSERVATION_BANNER}\n")

    def test_round_trips_through_the_hooks(self, memory_files: MemoryFiles) -> None:
        """A saved observation reloads equal to what was written."""
        save_observation_record(Path("o.txt"), _observation())
        assert load_observation_record(Path("o.txt")) == _observation()

    def test_values_survive_the_round_trip_exactly(self, memory_files: MemoryFiles) -> None:
        """Exactness is the point: a magnitude is computed from these values."""
        save_observation_record(Path("o.txt"), _observation())
        assert load_observation_record(Path("o.txt"))["values"] == (-1.5, 0.0, 0.055)

    def test_refuses_to_load_a_run_record_as_an_observation(
        self, memory_files: MemoryFiles
    ) -> None:
        """Per-type banners stop one record being read as another."""
        save_run_record(Path("a.txt"), _record())
        with pytest.raises(WireFormatError) as caught:
            load_observation_record(Path("a.txt"))
        assert caught.value.code == "NP-WIRE-009"


class TestProductionHooks:
    """The real hooks, exercised against a real filesystem.

    Without this the production implementations of ``read_text``,
    ``write_text``, and ``make_parent_dirs`` would never run under test, and a
    fault in them would be invisible behind the in-memory fakes.
    """

    def test_round_trips_through_a_real_file(self, tmp_path: Path) -> None:
        """A record survives a real write and read."""
        destination = tmp_path / "nested" / "run.txt"
        save_run_record(destination, _record())
        assert load_run_record(destination) == _record()

    def test_creates_missing_parent_directories_on_disk(self, tmp_path: Path) -> None:
        """The parent directory is created rather than assumed."""
        destination = tmp_path / "deep" / "deeper" / "run.txt"
        save_run_record(destination, _record())
        assert destination.parent.is_dir()

    def test_overwrites_an_existing_file(self, tmp_path: Path) -> None:
        """A second write replaces the first rather than appending."""
        destination = tmp_path / "run.txt"
        save_run_record(destination, _record())
        save_run_record(destination, _record())
        assert load_run_record(destination) == _record()

    def test_survives_a_separate_read_of_the_same_bytes(self, tmp_path: Path) -> None:
        """The file on disk is what the decoder reads.

        Reads the text back independently and decodes it, so the assertion does
        not depend on the loader having cached anything.
        """
        destination = tmp_path / "run.txt"
        save_run_record(destination, _record())
        assert destination.read_text(encoding="utf-8").startswith(f"{RUN_BANNER}\n")

    def test_round_trips_a_trial_through_a_real_file(self, tmp_path: Path) -> None:
        """The trial path uses the same production hooks, so it is run too."""
        destination = tmp_path / "nested" / "trial.txt"
        save_trial_record(destination, _trial())
        assert load_trial_record(destination) == _trial()
