"""Tests for scripts/optimize/run_records.py - the workspace RunRecord an
optimisation run emits beside its history row.

The point of the module under test is that a ``RunFingerprint`` sitting
inside this package's private ``UnifiedHistoryEntry`` cannot be read by
``platform_core.run_record.compare_run_records``. So these tests assert on
the comparability the record buys, not only on its fields: two records that
ran under one configuration must actually subtract.

Strict typing only: no Any, casts, or type: ignore.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.environment_record import PackageVersion, host_record
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.run_record import compare_run_records, decode_run_record
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import OptimizationHistory, UnifiedHistoryEntry
from scripts.optimize.run_records import (
    OPTIMIZATION_EXPERIMENT,
    append_optimization_record,
    optimization_label,
    optimization_observations,
    optimization_payload_digest,
    optimization_record_path,
    optimization_run_record,
)


def _make_fingerprint(numpy_version: str = "2.3.5") -> RunFingerprint:
    """Build a fingerprint with no host or GPU probing.

    Constructed from the shapes directly rather than through
    ``benchmark_fingerprint``, which reads the real machine: a test that
    captured this box would assert on whatever box it ran on.

    Args:
        numpy_version: The resolved numpy version to record, so a caller can
            produce two fingerprints that differ on exactly one axis.

    Returns:
        A fingerprint stating an unpinned determinism stack, which is what
        ``scripts/optimize`` honestly reports.
    """
    return RunFingerprint(
        image_digest="sha256:feedface",
        gpu_model="",
        driver_version="",
        determinism=determinism_record(UNPINNED_STACK, {}),
        host=host_record(platform="Linux-6.1", machine="x86_64", logical_cores=8),
        packages=(PackageVersion(name="numpy", version=numpy_version),),
    )


def _make_entry(
    backend: str = "cleargbm",
    dataset: DatasetName = "taiwan",
    feature_preset: FeaturePreset = "full",
    best_val_auc: float = 0.78,
    duration_seconds: float = 12.5,
    n_trials: int = 40,
    n_samples: int = 30000,
    n_features: int = 23,
    best_trial_number: int = 17,
    timestamp: str = "2026-09-04T00:00:00+00:00",
    fingerprint: RunFingerprint | None = None,
) -> UnifiedHistoryEntry:
    """Build a history entry for testing.

    Args:
        backend: Backend name.
        dataset: Dataset name.
        feature_preset: Feature preset name.
        best_val_auc: The claim.
        duration_seconds: Wall clock.
        n_trials: Trials that completed.
        n_samples: Rows the search saw.
        n_features: Columns the search saw.
        best_trial_number: Index of the winning trial.
        timestamp: ISO timestamp.
        fingerprint: The configuration, or None to build the pre-2026-08-28
            shape that states it has none.

    Returns:
        The entry.
    """
    return UnifiedHistoryEntry(
        timestamp=timestamp,
        backend=backend,
        dataset=dataset,
        feature_preset=feature_preset,
        n_trials=n_trials,
        n_samples=n_samples,
        n_features=n_features,
        best_val_auc=best_val_auc,
        best_trial_number=best_trial_number,
        duration_seconds=duration_seconds,
        fingerprint=fingerprint,
    )


class TestOptimizationLabel:
    """The label naming which run within the experiment."""

    def test_joins_backend_dataset_preset_and_timestamp(self) -> None:
        """All four varying axes appear, in that order."""
        label = optimization_label(_make_entry())
        assert label == "cleargbm-taiwan-full-2026-09-04T00:00:00+00:00"

    def test_two_runs_of_one_configuration_get_distinct_labels(self) -> None:
        """The timestamp is what stops a re-run colliding with its predecessor.

        This experiment exists to track progression, so two runs of the same
        backend on the same dataset are the common case and must not share a
        label.
        """
        first = optimization_label(_make_entry(timestamp="2026-09-04T00:00:00+00:00"))
        second = optimization_label(_make_entry(timestamp="2026-09-04T01:00:00+00:00"))
        assert first != second

    def test_backend_is_in_the_label_not_the_experiment(self) -> None:
        """Which model wins is the question, so backends must pair.

        If the backend were part of the experiment name, two backends'
        records would be declared incomparable outright and the comparison
        the sweep exists to make could never be expressed.
        """
        cleargbm = optimization_run_record(
            _make_entry(backend="cleargbm", fingerprint=_make_fingerprint())
        )
        lightgbm = optimization_run_record(
            _make_entry(backend="lightgbm", fingerprint=_make_fingerprint())
        )
        assert cleargbm["experiment"] == lightgbm["experiment"] == OPTIMIZATION_EXPERIMENT
        assert cleargbm["label"] != lightgbm["label"]


class TestOptimizationObservations:
    """The named numbers the record carries."""

    def test_names_the_claim_the_wall_clock_and_the_trial_count(self) -> None:
        """Exactly three observations, and they are these three."""
        observations = optimization_observations(_make_entry())
        assert {o["name"] for o in observations} == {
            "best_val_auc",
            "duration_seconds",
            "trials_completed",
        }

    def test_carries_the_entrys_values(self) -> None:
        """Values pass through unchanged, with the trial count widened."""
        observations = optimization_observations(
            _make_entry(best_val_auc=0.91, duration_seconds=3.25, n_trials=7)
        )
        by_name = {o["name"]: o["value"] for o in observations}
        assert by_name["best_val_auc"] == 0.91
        assert by_name["duration_seconds"] == 3.25
        assert by_name["trials_completed"] == 7.0

    def test_omits_the_best_trial_index(self) -> None:
        """An index into the search is not a quantity anyone subtracts."""
        observations = optimization_observations(_make_entry(best_trial_number=17))
        assert "best_trial_number" not in {o["name"] for o in observations}


class TestOptimizationPayloadDigest:
    """The digest covering run detail the comparability layer cannot read."""

    def test_is_the_digest_of_the_canonical_encoding(self) -> None:
        """Computed over the JSON, not over any file."""
        entry = _make_entry(n_samples=100, n_features=5, best_trial_number=3)
        expected = sha256(
            dump_json_str({"n_samples": 100, "n_features": 5, "best_trial_number": 3}).encode(
                "utf-8"
            )
        ).hexdigest()
        assert optimization_payload_digest(entry) == expected

    def test_two_runs_of_one_shape_digest_alike(self) -> None:
        """The timestamp is not payload, so it must not move the digest."""
        first = _make_entry(timestamp="2026-09-04T00:00:00+00:00")
        second = _make_entry(timestamp="2026-09-04T09:00:00+00:00")
        assert optimization_payload_digest(first) == optimization_payload_digest(second)

    def test_a_differing_winning_trial_moves_the_digest(self) -> None:
        """The index the observations dropped still distinguishes two runs."""
        first = _make_entry(best_trial_number=3)
        second = _make_entry(best_trial_number=4)
        assert optimization_payload_digest(first) != optimization_payload_digest(second)

    def test_a_differing_dataset_shape_moves_the_digest(self) -> None:
        """Same AUC on a different number of columns is not the same run."""
        first = _make_entry(n_features=23)
        second = _make_entry(n_features=42)
        assert optimization_payload_digest(first) != optimization_payload_digest(second)


class TestOptimizationRunRecord:
    """Building the record itself."""

    def test_builds_a_record_carrying_the_entrys_fingerprint(self) -> None:
        """The fingerprint is the entry's, not a reconstruction."""
        fingerprint = _make_fingerprint()
        record = optimization_run_record(_make_entry(fingerprint=fingerprint))
        assert record["fingerprint"] == fingerprint
        assert record["experiment"] == OPTIMIZATION_EXPERIMENT

    def test_orders_observations_canonically(self) -> None:
        """``run_record`` sorts by name, so two records list one order."""
        record = optimization_run_record(_make_entry(fingerprint=_make_fingerprint()))
        names = [o["name"] for o in record["observations"]]
        assert names == sorted(names)

    def test_refuses_an_entry_that_states_no_fingerprint(self) -> None:
        """The pre-2026-08-28 rows cannot become records, and must not.

        A synthesised fingerprint would claim a configuration nobody
        observed; an empty one would make every such row compare equal to
        every other. Refusing is the only answer that stays true.
        """
        with pytest.raises(ValueError, match="states no fingerprint"):
            optimization_run_record(_make_entry(fingerprint=None))

    def test_the_refusal_names_the_run_it_refused(self) -> None:
        """A refusal that does not say which row is not actionable."""
        entry = _make_entry(backend="lstm", fingerprint=None)
        with pytest.raises(ValueError, match="lstm-taiwan-full"):
            optimization_run_record(entry)


class TestComparability:
    """What the record actually buys: two runs that subtract."""

    def test_two_runs_under_one_configuration_compare(self) -> None:
        """The whole point -- and it is checked, not assumed."""
        left = optimization_run_record(
            _make_entry(backend="cleargbm", best_val_auc=0.70, fingerprint=_make_fingerprint())
        )
        right = optimization_run_record(
            _make_entry(backend="lightgbm", best_val_auc=0.75, fingerprint=_make_fingerprint())
        )
        comparison = compare_run_records(left, right, ())
        assert comparison["kind"] == "compared"
        deltas = {d["name"]: d["difference"] for d in comparison["deltas"]}
        assert deltas["best_val_auc"] == pytest.approx(0.05)

    def test_two_runs_under_differing_configurations_do_not_compare(self) -> None:
        """A record that compares across a numpy change would be worse than none."""
        left = optimization_run_record(
            _make_entry(fingerprint=_make_fingerprint(numpy_version="2.3.5"))
        )
        right = optimization_run_record(
            _make_entry(fingerprint=_make_fingerprint(numpy_version="2.4.0"))
        )
        comparison = compare_run_records(left, right, ())
        assert comparison["kind"] != "compared"


class TestOptimizationRecordPath:
    """Where the records land."""

    def test_sits_beside_the_history_it_describes(self, tmp_path: Path) -> None:
        """Same directory, name derived from the history's own stem."""
        history = tmp_path / "optimization_history.jsonl"
        assert optimization_record_path(history) == (
            tmp_path / "optimization_history.runrecords.jsonl"
        )

    def test_carries_a_job_name_suffix_through(self, tmp_path: Path) -> None:
        """Per-member history files must get per-member record files.

        Under the HPC3 farm the history name is suffixed with the job name
        so concurrent sweep members never append to one file. Records that
        collapsed back onto a shared name would reintroduce exactly the
        cross-node append the suffix exists to avoid.
        """
        history = tmp_path / "optimization_history-sweep7.jsonl"
        assert optimization_record_path(history) == (
            tmp_path / "optimization_history-sweep7.runrecords.jsonl"
        )

    def test_the_path_matches_what_the_history_manager_uses(self, tmp_path: Path) -> None:
        """The manager's own path is what the records are named from."""
        history = OptimizationHistory(tmp_path / "optimization_history.jsonl")
        assert optimization_record_path(history.path).parent == tmp_path


class TestAppendOptimizationRecord:
    """Writing records out."""

    def test_writes_one_decodable_line(self, tmp_path: Path) -> None:
        """A record written must read back as the record written."""
        history = tmp_path / "optimization_history.jsonl"
        entry = _make_entry(fingerprint=_make_fingerprint())
        append_optimization_record(history, entry)

        lines = optimization_record_path(history).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        decoded = decode_run_record(narrow_json_to_dict(load_json_str(lines[0])))
        assert decoded == optimization_run_record(entry)

    def test_appends_rather_than_replacing(self, tmp_path: Path) -> None:
        """The record file grows with the history, one line per run."""
        history = tmp_path / "optimization_history.jsonl"
        append_optimization_record(
            history,
            _make_entry(timestamp="2026-09-04T00:00:00+00:00", fingerprint=_make_fingerprint()),
        )
        append_optimization_record(
            history,
            _make_entry(timestamp="2026-09-04T01:00:00+00:00", fingerprint=_make_fingerprint()),
        )

        lines = optimization_record_path(history).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        labels = [
            decode_run_record(narrow_json_to_dict(load_json_str(line)))["label"] for line in lines
        ]
        assert labels[0] != labels[1]

    def test_refuses_a_fingerprintless_entry_and_writes_nothing(self, tmp_path: Path) -> None:
        """The refusal happens before the file is touched.

        A record file created empty by a refused write would read as "this
        run emitted no records" rather than "this run could not".
        """
        history = tmp_path / "optimization_history.jsonl"
        with pytest.raises(ValueError, match="states no fingerprint"):
            append_optimization_record(history, _make_entry(fingerprint=None))
        assert not optimization_record_path(history).exists()
