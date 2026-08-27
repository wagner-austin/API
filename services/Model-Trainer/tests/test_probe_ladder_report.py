"""The ladder report, over records written to a real directory.

The records here are synthesised rather than measured, and that is the right
call: what is under test is the reading -- which rung is named as the
threshold, in which order, and whether a rung one card skipped is reported
instead of counted as agreement. Computing eight real forward passes per test
would exercise the probe, which has its own tests, and would still leave the
interesting cases unreachable, because a CPU cannot produce two cards.

The one thing NOT synthesised is the rung labels. Those come from
``probe_shapes``, so a rung renamed there breaks this file rather than
quietly producing a report in which every rung is "not reported by every run".
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import dump_json_str
from platform_core.run_record import (
    Observation,
    RunAgreement,
    RunRecord,
    agree_across_runs,
    encode_run_record,
    run_record,
)

from model_trainer.cli import probe_ladder
from model_trainer.cli import probe_ladder_report as report_cli
from model_trainer.core.services.model.probe_shapes import (
    PROBE_AXES,
    PROBE_SHAPES,
    probe_label,
    require_probe_shape,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})

#: The gate probe's measured value on three cards, which agreed exactly.
GATE_VALUE = 6.250983715057373


def label_of(rung: str) -> str:
    """Return one rung's label."""
    return probe_label(require_probe_shape(rung))


def ladder_record(gpu: str, values: dict[str, float]) -> RunRecord:
    """Build a ladder record for one card.

    Args:
        gpu: The card the run reports.
        values: Rung NAME to value; the record carries rung LABELS.

    Returns:
        The record.
    """
    return run_record(
        experiment=probe_ladder.LADDER_EXPERIMENT,
        label=probe_ladder.ladder_label(tuple(label_of(rung) for rung in values)),
        fingerprint=RunFingerprint(
            image_digest="sha256:1112dbb1",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(
            Observation(name=label_of(rung), value=value) for rung, value in values.items()
        ),
        payload_digest="",
    )


def write_records(directory: pathlib.Path, records: dict[str, RunRecord]) -> pathlib.Path:
    """Write records into a directory under the given filenames.

    Args:
        directory: Directory to create and fill.
        records: Filename stem to record.

    Returns:
        The directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for stem, record in records.items():
        (directory / f"{stem}.json").write_text(
            dump_json_str(encode_run_record(record)), encoding="utf-8"
        )
    return directory


def full_ladder(gpu: str, *, break_at: str | None = None, offset: float = 1e-7) -> RunRecord:
    """Build a record covering every declared rung.

    Args:
        gpu: The card the run reports.
        break_at: Rung whose value is nudged, or None for the shared values.
        offset: How far to nudge it.

    Returns:
        The record.
    """
    values = {rung: GATE_VALUE + index for index, rung in enumerate(PROBE_SHAPES)}
    if break_at is not None:
        values[break_at] += offset
    return ladder_record(gpu, values)


class TestReadingRecords:
    def test_it_reads_every_record_in_filename_order(self, tmp_path: pathlib.Path) -> None:
        # Filename order, not directory order, so the value columns line up
        # the same way every time the report is run.
        directory = write_records(
            tmp_path / "ladder",
            {"v100": full_ladder("Tesla V100"), "a100": full_ladder("NVIDIA A100 80GB PCIe")},
        )

        named = report_cli.read_run_records(directory)

        assert [name for name, _ in named] == ["a100.json", "v100.json"]
        assert [record["fingerprint"]["gpu_model"] for _, record in named] == [
            "NVIDIA A100 80GB PCIe",
            "Tesla V100",
        ]

    def test_a_missing_directory_is_refused_by_name(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.read_run_records(tmp_path / "absent")

    def test_a_directory_holding_no_records_is_refused(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "empty").mkdir()

        with pytest.raises(FileNotFoundError, match=r"no \.json records"):
            report_cli.read_run_records(tmp_path / "empty")


class TestFindingTheThreshold:
    def _agreement(self, *, break_at: str | None) -> RunAgreement:
        """Compute agreement between two cards, optionally broken at a rung."""
        return agree_across_runs(
            (full_ladder("Tesla V100"), full_ladder("NVIDIA A100 80GB PCIe", break_at=break_at))
        )

    def test_a_ladder_that_agrees_everywhere_names_no_threshold(self) -> None:
        agreement = self._agreement(break_at=None)

        assert [report_cli.first_disagreement(agreement, axis) for axis in PROBE_AXES] == [
            None,
            None,
        ]

    def test_the_threshold_is_the_first_rung_in_axis_order_not_label_order(self) -> None:
        # THE test in this file. A record sorts its observations by name, and
        # the labels sort alphabetically -- large, medium, small, tiny, xl --
        # so reading a record's own order would name "large" as the first
        # size rung and report a threshold at random. "small" is the second
        # rung on the size axis and the third alphabetically.
        agreement = self._agreement(break_at="small")
        size_axis = next(axis for axis in PROBE_AXES if axis["name"] == "model-size")

        assert report_cli.first_disagreement(agreement, size_axis) == "small"

    def test_a_break_on_one_axis_leaves_the_other_axis_clean(self) -> None:
        # What makes the ladder worth walking: the axis that broke is named.
        agreement = self._agreement(break_at="tiny-len256")

        by_axis = {
            axis["name"]: report_cli.first_disagreement(agreement, axis) for axis in PROBE_AXES
        }

        assert by_axis == {"model-size": None, "sequence-length": "tiny-len256"}

    def test_a_break_at_the_shared_origin_shows_on_both_axes(self) -> None:
        # The gate rung starts both axes, so a card that disagrees there
        # disagrees about everything downstream too.
        agreement = self._agreement(break_at="tiny")

        assert [report_cli.first_disagreement(agreement, axis) for axis in PROBE_AXES] == [
            "tiny",
            "tiny",
        ]

    def test_a_rung_one_run_skipped_is_not_counted_as_agreement(self) -> None:
        partial = ladder_record("NVIDIA A30", {"tiny": GATE_VALUE})
        agreement = agree_across_runs((full_ladder("Tesla V100"), partial))
        size_axis = next(axis for axis in PROBE_AXES if axis["name"] == "model-size")

        assert report_cli.first_disagreement(agreement, size_axis) is None
        assert report_cli.rung_agreement(agreement, "small") is None

    def test_a_shared_rung_reports_its_values(self) -> None:
        agreement = self._agreement(break_at=None)

        entry = report_cli.rung_agreement(agreement, "tiny")

        if entry is None:
            raise AssertionError("the gate rung is shared by both runs")
        assert entry["values"] == (GATE_VALUE, GATE_VALUE)
        assert entry["distinct"] == 1

    def test_an_undeclared_rung_is_refused(self) -> None:
        with pytest.raises(KeyError, match="unknown probe rung 'gigantic'"):
            report_cli.rung_agreement(self._agreement(break_at=None), "gigantic")


class TestTheConfigurationSection:
    """Whether the runs differ only in card is computed, not left to the eye."""

    def _named(self, *records: tuple[str, RunRecord]) -> tuple[tuple[str, RunRecord], ...]:
        """Pair records with filenames for the report."""
        return records

    def test_the_first_run_is_the_reference(self) -> None:
        lines = report_cli.configuration_lines(
            self._named(
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe")),
            )
        )

        assert lines[0] == "  [0] v100.json  (reference)"

    def test_a_card_difference_is_named_as_the_axis_it_is(self) -> None:
        lines = report_cli.configuration_lines(
            self._named(
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe")),
            )
        )

        assert lines[1] == "  [1] a100.json  differs on: gpu_model"

    def test_two_runs_on_one_configuration_say_so(self) -> None:
        # Not a silent empty list. "These are the same configuration" is a
        # fact worth reading -- it means the comparison is a repeat, not a
        # cross-card measurement.
        lines = report_cli.configuration_lines(
            self._named(
                ("first.json", full_ladder("Tesla V100")),
                ("second.json", full_ladder("Tesla V100")),
            )
        )

        assert lines[1] == "  [1] second.json  differs on: identical to the reference"

    def test_a_differing_image_is_called_out_and_not_merely_listed(self) -> None:
        # THE case this section exists for: the local 3090 Ti runs carry no
        # image digest, and dropped into a directory of cluster records they
        # would produce a confident-looking cross-card answer that confounds
        # image with card.
        elsewhere = ladder_record("NVIDIA GeForce RTX 3090 Ti", {"tiny": GATE_VALUE})
        moved: RunRecord = {
            **elsewhere,
            "fingerprint": RunFingerprint(
                image_digest="",
                gpu_model=elsewhere["fingerprint"]["gpu_model"],
                driver_version=elsewhere["fingerprint"]["driver_version"],
                determinism=PINNED,
            ),
        }
        lines = report_cli.configuration_lines(
            self._named(("v100.json", full_ladder("Tesla V100")), ("local.json", moved))
        )

        assert "image_digest" in lines[1]
        assert any("do not all share one image_digest" in line for line in lines)

    def test_a_card_difference_alone_raises_no_warning(self) -> None:
        # The warning has to stay rare or it stops being read. A different
        # card is the POINT of the measurement, not a confound.
        lines = report_cli.configuration_lines(
            self._named(
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe")),
            )
        )

        assert not any(line.strip().startswith("!!") for line in lines)

    def test_one_confounded_axis_is_reported_once_however_many_runs_carry_it(self) -> None:
        stray = ladder_record("NVIDIA A30", {"tiny": GATE_VALUE})
        moved: RunRecord = {
            **stray,
            "fingerprint": RunFingerprint(
                image_digest="sha256:different",
                gpu_model=stray["fingerprint"]["gpu_model"],
                driver_version=stray["fingerprint"]["driver_version"],
                determinism=PINNED,
            ),
        }
        lines = report_cli.configuration_lines(
            self._named(
                ("v100.json", full_ladder("Tesla V100")),
                ("a.json", moved),
                ("b.json", moved),
            )
        )

        assert sum("do not all share one image_digest" in line for line in lines) == 1

    def test_a_pair_that_each_recorded_no_image_is_not_called_identical_and_left_there(
        self,
    ) -> None:
        # The hole differencing is structurally unable to see: an unrecorded
        # axis compares EQUAL to the same gap in another run, so two runs that
        # each failed to record their image read as one configuration. The
        # local 3090 Ti ladder pair is exactly this shape.
        local: RunRecord = {
            **ladder_record("NVIDIA GeForce RTX 3090 Ti", {"tiny": GATE_VALUE}),
            "fingerprint": RunFingerprint(
                image_digest="",
                gpu_model="NVIDIA GeForce RTX 3090 Ti",
                driver_version="591.86",
                determinism=PINNED,
            ),
        }
        lines = report_cli.configuration_lines(
            self._named(("run1.json", local), ("run2.json", local))
        )

        # Still reported as matching on every recorded axis, which is true...
        assert lines[1] == "  [1] run2.json  differs on: identical to the reference"
        # ...and the gap that makes that reading worthless is named for BOTH.
        assert sum("recorded no image_digest" in line for line in lines) == 2
        # NOT the "do not all share one image_digest" warning: they DO share
        # it, both being absent. Saying otherwise would be a second, false
        # claim stacked on a true one.
        assert not any("do not all share" in line for line in lines)

    def test_one_run_recording_an_image_and_another_not_is_a_real_difference(self) -> None:
        # The case the empty-axis path must NOT swallow: empty against
        # non-empty is a genuine disagreement and the differencing loop sees
        # it, so this set gets both warnings and deserves them.
        unknown: RunRecord = {
            **ladder_record("NVIDIA GeForce RTX 3090 Ti", {"tiny": GATE_VALUE}),
            "fingerprint": RunFingerprint(
                image_digest="",
                gpu_model="NVIDIA GeForce RTX 3090 Ti",
                driver_version="591.86",
                determinism=PINNED,
            ),
        }
        lines = report_cli.configuration_lines(
            self._named(("v100.json", full_ladder("Tesla V100")), ("local.json", unknown))
        )

        assert "image_digest" in lines[1]
        assert any("recorded no image_digest" in line for line in lines)
        assert any("do not all share one image_digest" in line for line in lines)

    def test_a_fully_recorded_pair_raises_no_unrecorded_warning(self) -> None:
        lines = report_cli.configuration_lines(
            self._named(
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe")),
            )
        )

        assert not any("recorded no" in line for line in lines)

    def test_the_section_appears_in_the_report(self) -> None:
        lines = report_cli.report_lines(
            (
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe")),
            )
        )

        assert "configuration" in lines
        assert "  [1] a100.json  differs on: gpu_model" in lines


class TestTheReport:
    def _lines(self, *, break_at: str | None) -> tuple[str, ...]:
        """Render a two-card report."""
        return report_cli.report_lines(
            (
                ("v100.json", full_ladder("Tesla V100")),
                ("a100.json", full_ladder("NVIDIA A100 80GB PCIe", break_at=break_at)),
            )
        )

    def test_the_header_counts_the_runs_and_names_the_experiment(self) -> None:
        expected = f"2 runs, experiment {probe_ladder.LADDER_EXPERIMENT}"

        assert self._lines(break_at=None)[0] == expected

    def test_every_run_is_listed_with_the_card_it_ran_on(self) -> None:
        # Printed rather than checked: whether the runs differ ONLY in card is
        # the reader's judgement, and it cannot be made from a report that
        # hides the fingerprints.
        lines = self._lines(break_at=None)

        assert "[0] v100.json" in lines[1]
        assert "Tesla V100" in lines[1]
        assert "[1] a100.json" in lines[2]
        assert "NVIDIA A100 80GB PCIe" in lines[2]

    def test_each_axis_gets_a_section_and_a_verdict(self) -> None:
        lines = self._lines(break_at=None)

        for axis in PROBE_AXES:
            assert f"axis {axis['name']}" in lines
        assert lines.count("  -> every shared rung agreed exactly") == len(PROBE_AXES)

    def test_a_break_is_reported_as_the_rung_it_happened_at(self) -> None:
        assert "  -> agreement breaks at rung 'small'" in self._lines(break_at="small")

    def test_two_values_differing_in_the_last_digit_print_differently(self) -> None:
        # Seventeen significant digits is what it takes to write a double so
        # it reads back unchanged. Anything shorter prints these two
        # identically, and the report would show a disagreement as a pair of
        # matching numbers -- the exact mistake this command exists to
        # prevent. The default %g would print six.
        nudged = GATE_VALUE + 1e-15
        lines = report_cli.report_lines(
            (
                ("v100.json", ladder_record("Tesla V100", {"tiny": GATE_VALUE})),
                ("a100.json", ladder_record("NVIDIA A100 80GB PCIe", {"tiny": nudged})),
            )
        )
        row = next(line for line in lines if line.startswith("  tiny "))

        assert f"{GATE_VALUE:.17g}" in row
        assert f"{nudged:.17g}" in row
        assert f"{GATE_VALUE:.17g}" != f"{nudged:.17g}"
        assert f"{GATE_VALUE:g}" == f"{nudged:g}"

    def test_a_rung_not_every_run_reported_is_named_in_its_axis_section(self) -> None:
        lines = report_cli.report_lines(
            (
                ("v100.json", full_ladder("Tesla V100")),
                ("a30.json", ladder_record("NVIDIA A30", {"tiny": GATE_VALUE})),
            )
        )

        assert any("(not reported by every run)" in line for line in lines)

    def test_observations_missing_from_some_run_are_listed_at_the_end(self) -> None:
        lines = report_cli.report_lines(
            (
                ("v100.json", full_ladder("Tesla V100")),
                ("a30.json", ladder_record("NVIDIA A30", {"tiny": GATE_VALUE})),
            )
        )

        assert "observations not reported by every run:" in lines
        assert f"  {label_of('xl')}" in lines

    def test_a_complete_pair_lists_nothing_as_missing(self) -> None:
        assert "observations not reported by every run:" not in self._lines(break_at=None)


class TestTheCommandLine:
    def test_main_reports_and_returns_zero_even_when_the_cards_disagree(
        self, tmp_path: pathlib.Path
    ) -> None:
        # A disagreement is the measurement, not a failure. A non-zero exit
        # would make a shell treat the answer as an error.
        directory = write_records(
            tmp_path / "ladder",
            {
                "v100": full_ladder("Tesla V100"),
                "a100": full_ladder("NVIDIA A100 80GB PCIe", break_at="small"),
            },
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_an_absent_dir_flag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--dir"):
            report_cli.main([])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--axis"):
            report_cli.main(["--dir", str(tmp_path), "--axis", "model-size"])

    def test_a_single_record_is_refused_because_one_run_always_agrees(
        self, tmp_path: pathlib.Path
    ) -> None:
        directory = write_records(tmp_path / "ladder", {"v100": full_ladder("Tesla V100")})

        with pytest.raises(ValueError, match="at least two runs, got 1"):
            report_cli.main(["--dir", str(directory)])

    def test_running_the_module_as_main_actually_reports(self, tmp_path: pathlib.Path) -> None:
        # Same regression the ladder and the gate probe carry a guard for:
        # without `if __name__ == "__main__"`, `python -m ...` imports the
        # module, runs nothing and exits 0. This command is the one someone
        # runs by hand after a job lands, so a silent no-op here reads as
        # "the records are not there yet".
        directory = write_records(
            tmp_path / "ladder",
            {"v100": full_ladder("Tesla V100"), "a100": full_ladder("NVIDIA A100 80GB PCIe")},
        )
        module_name = "model_trainer.cli.probe_ladder_report"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-probe-ladder-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0

    def test_the_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "ladder",
            {"v100": full_ladder("Tesla V100"), "a100": full_ladder("NVIDIA A100 80GB PCIe")},
        )
        saved = sys.argv
        sys.argv = ["modeltrainer-probe-ladder-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                report_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0
