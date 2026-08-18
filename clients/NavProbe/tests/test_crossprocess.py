"""Tests for comparing rollouts that did not share a process.

The real-filesystem tests are the point of this module: an exchange between two
processes that never touched a disk would not be an exchange. They use
``tmp_path`` and the production storage hooks throughout.

The genuinely-separate-process case is covered by
``TestActualSeparateProcess``, which spawns a real interpreter. Everything
weaker than that shares module state with the test, which is the exact condition
this module exists to escape.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from navprobe.codecs.run import encode_run_record
from navprobe.comparison import ComparisonError
from navprobe.crossprocess import (
    RUN_FILENAME_TEMPLATE,
    TRIAL_FILENAME,
    CrossProcessError,
    compare_recorded_runs,
    compare_recordings,
    record_trial,
    run_record_path,
    trial_record_path,
)
from navprobe.experiment import ProbeService
from navprobe.records import TrialSpec
from navprobe.rollout import roll_out
from navprobe.storage import load_trial_record, save_run_record
from tests.factories import DriftingSimulatorFactory, LinearSimulatorFactory
from tests.simulators import DriftingSimulator, LinearSimulator

#: The trial design every test here records.
SPEC = TrialSpec(seed=7, step_count=6, repetitions=3)


class TestPaths:
    """Tests for the recording layout."""

    def test_run_records_are_numbered_by_repetition(self, tmp_path: Path) -> None:
        """Each repetition gets its own file, named by index."""
        assert run_record_path(tmp_path, 2).name == RUN_FILENAME_TEMPLATE.format(index=2)

    def test_trial_record_has_a_fixed_name(self, tmp_path: Path) -> None:
        """The summary is findable without knowing the repetition count."""
        assert trial_record_path(tmp_path).name == TRIAL_FILENAME

    def test_rejects_a_negative_repetition_index(self, tmp_path: Path) -> None:
        """A negative index names a file belonging to no repetition."""
        with pytest.raises(CrossProcessError) as caught:
            run_record_path(tmp_path, -1)
        assert caught.value.code == "NP-XPROC-001"


class TestRecordTrial:
    """Tests for :func:`record_trial`."""

    def test_writes_one_record_per_repetition(self, tmp_path: Path) -> None:
        """Every repetition is persisted, not only the reference."""
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), tmp_path, SPEC)
        assert sorted(p.name for p in tmp_path.glob("run-*.txt")) == [
            "run-0.txt",
            "run-1.txt",
            "run-2.txt",
        ]

    def test_writes_the_trial_summary(self, tmp_path: Path) -> None:
        """The verdict is persisted alongside the runs."""
        returned = record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), tmp_path, SPEC)
        assert load_trial_record(trial_record_path(tmp_path)) == returned

    def test_creates_a_missing_directory(self, tmp_path: Path) -> None:
        """A recording directory does not have to exist beforehand."""
        destination = tmp_path / "nested" / "recording"
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), destination, SPEC)
        assert trial_record_path(destination).is_file()

    def test_persisted_runs_carry_their_step_digests(self, tmp_path: Path) -> None:
        """A later process needs the steps to localise a divergence."""
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), tmp_path, SPEC)
        text = run_record_path(tmp_path, 0).read_text(encoding="utf-8")
        assert text.count("\nstep\t") == SPEC["step_count"]


class TestCompareRecordedRuns:
    """Tests for :func:`compare_recorded_runs`."""

    def test_identical_recordings_agree(self, tmp_path: Path) -> None:
        """Two recordings of the same rollout compare as matching."""
        left = tmp_path / "a.txt"
        right = tmp_path / "b.txt"
        save_run_record(left, roll_out(LinearSimulator(world_count=2), "a", 7, 6))
        save_run_record(right, roll_out(LinearSimulator(world_count=2), "b", 7, 6))
        assert compare_recorded_runs(left, right)["digests_match"] is True

    def test_a_genuine_divergence_is_localised_across_files(self, tmp_path: Path) -> None:
        """Two recordings that part ways mid-rollout are localised to the step.

        Both records are internally consistent — each was produced by a real
        rollout — so this exercises localisation rather than the tamper check
        below.
        """
        left = tmp_path / "a.txt"
        right = tmp_path / "b.txt"
        save_run_record(
            left,
            roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=0), "a", 7, 6),
        )
        save_run_record(
            right,
            roll_out(DriftingSimulator(world_count=2, diverge_at_step=3, offset=1), "b", 7, 6),
        )
        assert compare_recorded_runs(left, right)["first_divergent_step"] == 3

    def test_a_tampered_record_cannot_pass_as_agreement(self, tmp_path: Path) -> None:
        """Editing a step digest on disk is refused, not silently accepted.

        A file whose run digest no longer follows from its steps describes no
        real rollout. Comparing it would report agreement on the run digest
        while the steps disagree, so the integrity check refuses it instead —
        which is exactly the protection persistence needs, because a file is
        the one artefact in this package that something outside it can edit.
        """
        record = roll_out(LinearSimulator(world_count=2), "a", 7, 6)
        left = tmp_path / "a.txt"
        right = tmp_path / "b.txt"
        save_run_record(left, record)
        lines = encode_run_record(record).strip("\n").split("\n")
        lines[6 + 3] = "step\t3\tdeadbeef"
        right.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ComparisonError) as caught:
            compare_recorded_runs(left, right)
        assert caught.value.code == "NP-COMPARE-002"

    def test_rejects_recordings_at_different_seeds(self, tmp_path: Path) -> None:
        """Different seeds cannot produce evidence about determinism."""
        left = tmp_path / "a.txt"
        right = tmp_path / "b.txt"
        save_run_record(left, roll_out(LinearSimulator(world_count=2), "a", 7, 6))
        save_run_record(right, roll_out(LinearSimulator(world_count=2), "b", 8, 6))
        with pytest.raises(ComparisonError) as caught:
            compare_recorded_runs(left, right)
        assert caught.value.code == "NP-COMPARE-001"

    def test_propagates_a_missing_recording(self, tmp_path: Path) -> None:
        """An absent file fails rather than comparing against nothing."""
        left = tmp_path / "a.txt"
        save_run_record(left, roll_out(LinearSimulator(world_count=2), "a", 7, 6))
        with pytest.raises(FileNotFoundError):
            compare_recorded_runs(left, tmp_path / "missing.txt")


class TestCompareRecordings:
    """Tests for :func:`compare_recordings`."""

    def test_matching_environments_agree(self, tmp_path: Path) -> None:
        """Two recordings from equivalent conditions match at repetition zero."""
        left = tmp_path / "left"
        right = tmp_path / "right"
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), left, SPEC)
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), right, SPEC)
        assert compare_recordings(left, right, 0)["digests_match"] is True

    def test_differing_environments_are_localised(self, tmp_path: Path) -> None:
        """A recording from a diverging simulator is caught and localised.

        This is the cross-environment case in miniature: two recordings made
        under conditions that differ, compared entirely through files.
        """
        left = tmp_path / "left"
        right = tmp_path / "right"
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), left, SPEC)
        record_trial(
            ProbeService(DriftingSimulatorFactory(world_count=2, diverge_at_step=0)),
            right,
            SPEC,
        )
        assert compare_recordings(left, right, 0)["digests_match"] is False

    def test_compares_the_requested_repetition(self, tmp_path: Path) -> None:
        """A repetition other than the reference can be compared."""
        left = tmp_path / "left"
        right = tmp_path / "right"
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), left, SPEC)
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), right, SPEC)
        assert compare_recordings(left, right, 2)["left_label"] == "repetition-2"

    def test_rejects_a_negative_repetition_index(self, tmp_path: Path) -> None:
        """The index check applies through this entry point too."""
        with pytest.raises(CrossProcessError) as caught:
            compare_recordings(tmp_path, tmp_path, -1)
        assert caught.value.code == "NP-XPROC-001"


class TestActualSeparateProcess:
    """The fresh-process condition, measured with an actual fresh process.

    Every other test in this suite shares an interpreter with the code under
    test, which is precisely the condition this module exists to escape. Here a
    real child interpreter records a trial, and the parent — which never saw
    that process's memory — compares its own recording against it through the
    files alone.
    """

    @staticmethod
    def _record_in_child(destination: Path) -> None:
        """Record a trial in a freshly spawned interpreter.

        Args:
            destination: Directory the child writes its recording into.

        Raises:
            AssertionError: When the child exits non-zero, which would leave
                the comparison reading a partial recording.
        """
        source_root = Path(__file__).resolve().parents[1] / "src"
        program = (
            "from pathlib import Path\n"
            "from navprobe.crossprocess import record_trial\n"
            "from navprobe.experiment import ProbeService\n"
            "from navprobe.records import TrialSpec\n"
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
            "from tests.factories import LinearSimulatorFactory\n"
            "record_trial(\n"
            "    ProbeService(LinearSimulatorFactory(world_count=2)),\n"
            f"    Path({str(destination)!r}),\n"
            f"    TrialSpec(seed={SPEC['seed']}, step_count={SPEC['step_count']},"
            f" repetitions={SPEC['repetitions']}),\n"
            ")\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
            env={"PYTHONPATH": str(source_root), "PATH": ""},
        )
        assert completed.returncode == 0, completed.stderr

    def test_a_rollout_survives_a_process_restart(self, tmp_path: Path) -> None:
        """A recording made by another interpreter matches one made here.

        The two rollouts share no module state, no import cache, and no
        allocator history. Agreement between them is the fresh-process
        condition, and it is the only claim in this package that a
        single-process test cannot make.
        """
        child = tmp_path / "child"
        parent = tmp_path / "parent"
        self._record_in_child(child)
        record_trial(ProbeService(LinearSimulatorFactory(world_count=2)), parent, SPEC)
        assert compare_recordings(child, parent, 0)["digests_match"] is True

    def test_the_child_recording_is_complete(self, tmp_path: Path) -> None:
        """The child wrote every repetition and the summary.

        Asserted separately so a partial recording is diagnosed as a partial
        recording rather than as a determinism failure.
        """
        child = tmp_path / "child"
        self._record_in_child(child)
        assert sorted(p.name for p in child.iterdir()) == [
            "run-0.txt",
            "run-1.txt",
            "run-2.txt",
            "trial.txt",
        ]
