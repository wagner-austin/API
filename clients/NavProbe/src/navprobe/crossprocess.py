"""Compare rollouts that did not share a process.

:class:`navprobe.experiment.ProbeService` compares repetitions inside one
interpreter. That is the weaker condition: repetitions there share module state,
the JIT cache, and allocator history, so agreement between them says nothing
about whether a rollout survives a restart, a different machine, or a different
backend.

The stronger condition needs two processes, and two processes can only exchange
a rollout through a file. This module is that exchange: one process records a
trial, another loads the records and compares them. It is the reason
:mod:`navprobe.storage` exists, and without it that module's persistence has no
consumer.

The comparison is deliberately *not* a method on the service. A cross-process
comparison has no simulator and no factory — by the time it runs, the thing that
produced the numbers is gone. Modelling it as a free function over two paths
keeps that honest: there is nothing to inject, because there is nothing left
running.
"""

from __future__ import annotations

from pathlib import Path

from navprobe import NavProbeError
from navprobe.comparison import compare_runs
from navprobe.experiment import ProbeService
from navprobe.records import ComparisonRecord, TrialRecord, TrialSpec
from navprobe.storage import load_run_record, save_run_record, save_trial_record

#: Filename of the trial summary within a recording directory.
TRIAL_FILENAME = "trial.txt"

#: Filename pattern for one repetition's run record.
RUN_FILENAME_TEMPLATE = "run-{index}.txt"


class CrossProcessError(NavProbeError):
    """A recording could not be laid out or read back.

    Args:
        code: Stable identifier in the ``NP-XPROC-<NNN>`` range.
        message: Human-readable description of what went wrong.
    """


def run_record_path(directory: Path, index: int) -> Path:
    """Locate one repetition's run record within a recording directory.

    Args:
        directory: The recording directory.
        index: Zero-based repetition number.

    Returns:
        The path that repetition's record is written to and read from.

    Raises:
        CrossProcessError: When ``index`` is negative. A negative index would
            produce a filename that reads as a valid record and belongs to no
            repetition.
    """
    if index < 0:
        raise CrossProcessError(
            "NP-XPROC-001", f"repetition index must be zero or greater, got {index}"
        )
    return directory / RUN_FILENAME_TEMPLATE.format(index=index)


def trial_record_path(directory: Path) -> Path:
    """Locate the trial summary within a recording directory.

    Args:
        directory: The recording directory.

    Returns:
        The path the trial record is written to and read from.
    """
    return directory / TRIAL_FILENAME


def record_trial(service: ProbeService, directory: Path, spec: TrialSpec) -> TrialRecord:
    """Run a trial and persist every repetition alongside its summary.

    Every repetition is written, not just the reference. A later process
    comparing against this recording needs the step digests to localise a
    divergence, and the summary alone carries only the verdict.

    Args:
        service: The service to run the trial with.
        directory: Directory to write the records into. Created if missing.
        spec: The trial design.

    Returns:
        The trial record, which is also written to ``directory``.

    Raises:
        TrialError: When the trial design is unusable.
        RolloutError: When a simulator reports an unusable world count.
        ComparisonError: When two repetitions cannot be compared.
        CanonicalEncodingError: When an observation cannot be encoded.
        OSError: When a record cannot be written.
    """
    runs = service.roll_out_repetitions(spec)
    for index, run in enumerate(runs):
        save_run_record(run_record_path(directory, index), run)
    record = service.summarise(spec, runs)
    save_trial_record(trial_record_path(directory), record)
    return record


def compare_recorded_runs(left: Path, right: Path) -> ComparisonRecord:
    """Compare two run records read back from disk.

    Neither rollout needs to have been produced by this process, this machine,
    or this backend — only at the same seed. That is the whole point: it is the
    only comparison in the package whose two sides can come from different
    executions.

    Args:
        left: Path to the first run record.
        right: Path to the second run record.

    Returns:
        The verdict, including where the two first stopped agreeing.

    Raises:
        OSError: When either file cannot be read.
        WireFormatError: When either file is not a valid run record.
        ComparisonError: When the two rollouts were produced under different
            seeds, or a record's digest contradicts its own steps.
    """
    return compare_runs(load_run_record(left), load_run_record(right))


def compare_recordings(left_directory: Path, right_directory: Path, index: int) -> ComparisonRecord:
    """Compare one repetition across two recording directories.

    Args:
        left_directory: The first recording.
        right_directory: The second recording.
        index: Which repetition to compare. Repetition zero is the reference
            both trials measured themselves against, so it is the meaningful
            default for a cross-environment comparison.

    Returns:
        The verdict for that repetition.

    Raises:
        CrossProcessError: When ``index`` is negative.
        OSError: When either file cannot be read.
        WireFormatError: When either file is not a valid run record.
        ComparisonError: When the two rollouts were produced under different
            seeds, or a record's digest contradicts its own steps.
    """
    return compare_recorded_runs(
        run_record_path(left_directory, index), run_record_path(right_directory, index)
    )


__all__ = [
    "RUN_FILENAME_TEMPLATE",
    "TRIAL_FILENAME",
    "CrossProcessError",
    "compare_recorded_runs",
    "compare_recordings",
    "record_trial",
    "run_record_path",
    "trial_record_path",
]
