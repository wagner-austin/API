"""The sdpa-backend command and its report.

The command runs for real on the CPU: every attention call is issued, every
backend forced, and the record written and read back. The REPORT's records are
synthesised, for the reason the other report tests give -- what is under test
there is the reading, and a CPU cannot produce two cards.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.determinism_record import TRUE, determinism_record
from platform_core.json_utils import dump_json_str, load_json_str
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    decode_run_record,
    encode_run_record,
    run_record,
)
from platform_core.testing import sample_run_fingerprint

from model_trainer.cli import sdpa_probe as probe_cli
from model_trainer.cli import sdpa_probe_report as report_cli
from model_trainer.core.services.model.sdpa_probe import SdpaMeasurement, probe_sdpa
from model_trainer.core.services.model.sdpa_shapes import (
    AVAILABLE_SUFFIX,
    DEFAULT_KEY,
    DIGEST_SUFFIX,
    ELIGIBLE_SUFFIX,
    FALSE_VALUE,
    SDPA_EXPERIMENT,
    TRUE_VALUE,
    sdpa_label,
    sdpa_shape_for,
    sdpa_shapes,
)

PINNED = determinism_record("torch", {"cudnn_deterministic": TRUE})
TINY = sdpa_shape_for("tiny")


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "sdpa.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path))]


def selection_record(
    gpu: str,
    *,
    chosen: str | None = "efficient",
    eligible: str | None = "efficient",
    ran: tuple[str, ...] = ("math", "efficient"),
) -> RunRecord:
    """Build a one-shape record where a named backend matched the default.

    Args:
        gpu: The card the run reports.
        chosen: Backend whose forced digest equals the default's, or None for
            a record where nothing matched.
        eligible: Backend torch called eligible, or None for none.
        ran: Backends whose forcing produced a result.

    Returns:
        The record.
    """
    default = 111.0
    observations = [Observation(name=sdpa_label(TINY, DEFAULT_KEY, DIGEST_SUFFIX), value=default)]
    for index, name in enumerate(("math", "flash", "efficient", "cudnn")):
        available = name in ran
        observations.append(
            Observation(
                name=sdpa_label(TINY, name, AVAILABLE_SUFFIX),
                value=TRUE_VALUE if available else FALSE_VALUE,
            )
        )
        if available:
            observations.append(
                Observation(
                    name=sdpa_label(TINY, name, DIGEST_SUFFIX),
                    value=default if name == chosen else 200.0 + index,
                )
            )
    for name in ("flash", "efficient", "cudnn"):
        observations.append(
            Observation(
                name=sdpa_label(TINY, name, ELIGIBLE_SUFFIX),
                value=TRUE_VALUE if name == eligible else FALSE_VALUE,
            )
        )
    return run_record(
        experiment=SDPA_EXPERIMENT,
        label=probe_cli.SDPA_LABEL,
        fingerprint=sample_run_fingerprint(
            image_digest="sha256:a274e4ee",
            gpu_model=gpu,
            driver_version="580.82.07",
            determinism=PINNED,
        ),
        observations=tuple(observations),
        payload_digest="",
    )


def write_records(directory: pathlib.Path, records: dict[str, RunRecord]) -> pathlib.Path:
    """Write records into a directory under the given filenames."""
    directory.mkdir(parents=True, exist_ok=True)
    for stem, record in records.items():
        (directory / f"{stem}.json").write_text(
            dump_json_str(encode_run_record(record)), encoding="utf-8"
        )
    return directory


class TestWhatOneRecordCarries:
    def test_it_reports_availability_for_every_backend_of_every_shape(self) -> None:
        record = probe_cli.sdpa_run_record("cpu")
        names = {o["name"] for o in record["observations"]}
        wanted = {
            sdpa_label(shape, name, AVAILABLE_SUFFIX)
            for shape in sdpa_shapes()
            for name in ("math", "flash", "efficient", "cudnn")
        }

        assert wanted <= names

    def test_a_backend_that_could_not_run_contributes_no_digest(self) -> None:
        # The asymmetry is deliberate: an absent digest is reported as an
        # unmatched observation, so a backend available on one card and not
        # another shows up as a structural difference.
        record = probe_cli.sdpa_run_record("cpu")
        values = {o["name"]: o["value"] for o in record["observations"]}

        assert values[sdpa_label(TINY, "efficient", AVAILABLE_SUFFIX)] == FALSE_VALUE
        assert sdpa_label(TINY, "efficient", DIGEST_SUFFIX) not in values

    def test_it_carries_the_split_k_condition(self) -> None:
        # Forcing the math backend routes through cuBLAS, so the two
        # questions meet and a record that could not name its arm would be
        # unreadable for the same reason the trace's would.
        record = probe_cli.sdpa_run_record("cpu")

        assert "cublaslt_workspace_size" in {o["name"] for o in record["observations"]}

    def test_it_declares_its_own_experiment(self) -> None:
        record = probe_cli.sdpa_run_record("cpu")

        assert (record["experiment"], record["label"]) == (
            SDPA_EXPERIMENT,
            probe_cli.SDPA_LABEL,
        )

    def test_it_carries_no_payload_digest(self) -> None:
        assert probe_cli.sdpa_run_record("cpu")["payload_digest"] == NO_PAYLOAD


class TestNamingOneMeasurement:
    def test_it_emits_a_digest_for_the_unforced_call(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        observations = probe_cli.measurement_observations(TINY, measured)
        names = {o["name"] for o in observations}

        assert sdpa_label(TINY, DEFAULT_KEY, DIGEST_SUFFIX) in names

    def test_it_emits_torchs_opinion_beside_the_measurement(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        names = {o["name"] for o in probe_cli.measurement_observations(TINY, measured)}

        assert sdpa_label(TINY, "flash", ELIGIBLE_SUFFIX) in names
        assert sdpa_label(TINY, "flash", AVAILABLE_SUFFIX) in names


class TestDerivingTheSelection:
    def test_the_matching_backend_is_named(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        assert probe_cli.selected_backend(measured) != ()

    def test_nothing_matching_is_reported_as_nothing(self) -> None:
        measured = probe_sdpa(TINY, "cpu")
        detached = SdpaMeasurement(
            default_digest=measured["default_digest"],
            digests={name: value + 1.0 for name, value in measured["digests"].items()},
            available=measured["available"],
            eligible=measured["eligible"],
        )

        assert probe_cli.selected_backend(detached) == ()

    def test_several_matching_are_all_reported(self) -> None:
        # Two backends agreeing bit for bit is a real outcome of this method
        # and the probe must not pick one of them.
        measured = probe_sdpa(TINY, "cpu")
        tied = SdpaMeasurement(
            default_digest=measured["default_digest"],
            digests=dict.fromkeys(measured["digests"], measured["default_digest"]),
            available=measured["available"],
            eligible=measured["eligible"],
        )

        assert probe_cli.selected_backend(tied) == tuple(measured["digests"])


class TestTheReport:
    def test_it_names_the_backend_each_card_selected(self) -> None:
        lines = report_cli.report_lines(
            (
                ("a100.json", selection_record("NVIDIA A100 80GB PCIe")),
                ("v100.json", selection_record("Tesla V100", chosen="math")),
            )
        )
        # Split on whitespace: "tiny-len128" also startswith "tiny".
        row = [line for line in lines if line.split()[:1] == ["tiny"]]

        assert len(row) == 1
        assert "[0] efficient" in row[0]
        assert "[1] math" in row[0]

    def test_nothing_matching_is_said_rather_than_left_blank(self) -> None:
        record = selection_record("Tesla V100", chosen=None)

        assert report_cli.describe_selection(report_cli.selection_for(record, TINY)) == (
            report_cli.NONE_MATCHED
        )

    def test_an_ambiguous_result_shows_both_rather_than_picking(self) -> None:
        assert report_cli.describe_selection(("math", "efficient")) == "math=efficient"

    def test_a_record_without_the_shape_selects_nothing(self) -> None:
        empty = run_record(
            experiment=SDPA_EXPERIMENT,
            label=probe_cli.SDPA_LABEL,
            fingerprint=selection_record("A30")["fingerprint"],
            observations=(Observation(name="cublaslt_workspace_size", value=-1.0),),
            payload_digest="",
        )

        assert report_cli.selection_for(empty, TINY) == ()

    def test_eligible_but_no_kernel_is_flagged(self) -> None:
        record = selection_record("Tesla V100", chosen="math", eligible="cudnn", ran=("math",))

        assert report_cli.disagreements(record, TINY) == (
            "cudnn: torch says eligible, forcing it found no kernel",
        )

    def test_ineligible_yet_it_ran_is_flagged(self) -> None:
        record = selection_record("Tesla V100", eligible=None, ran=("math", "flash"))

        assert report_cli.disagreements(record, TINY) == (
            "flash: torch says ineligible, yet forcing it ran",
        )

    def test_agreement_between_opinion_and_reality_is_silent(self) -> None:
        record = selection_record("A30", eligible="efficient", ran=("math", "efficient"))

        assert report_cli.disagreements(record, TINY) == ()

    def test_a_record_carrying_no_opinion_is_not_flagged(self) -> None:
        empty = run_record(
            experiment=SDPA_EXPERIMENT,
            label=probe_cli.SDPA_LABEL,
            fingerprint=selection_record("A30")["fingerprint"],
            observations=(Observation(name="cublaslt_workspace_size", value=-1.0),),
            payload_digest="",
        )

        assert report_cli.disagreements(empty, TINY) == ()

    def test_the_disagreement_line_names_the_run_it_came_from(self) -> None:
        lines = report_cli.shape_lines(
            (
                ("a100.json", selection_record("A100")),
                ("v100.json", selection_record("V100", eligible="cudnn")),
            ),
            TINY,
        )

        assert any("!! [1] v100.json" in line for line in lines)

    def test_it_says_which_cards_produced_the_same_output(self) -> None:
        # Two cards selecting the SAME backend and still disagreeing bitwise
        # is a different finding from two cards selecting different ones, and
        # a selection table alone cannot tell them apart.
        agreeing = (
            ("a100.json", selection_record("A100")),
            ("a30.json", selection_record("A30")),
        )

        assert report_cli.output_agreement(agreeing, TINY) == "0,1"

    def test_differing_outputs_name_the_odd_run(self) -> None:
        odd = selection_record("V100")
        for observation in odd["observations"]:
            if observation["name"] == sdpa_label(TINY, DEFAULT_KEY, DIGEST_SUFFIX):
                observation["value"] = 999.0

        pairs = (
            ("a100.json", selection_record("A100")),
            ("a30.json", selection_record("A30")),
            ("v100.json", odd),
        )

        assert report_cli.output_agreement(pairs, TINY) == "0,1|2"

    def test_a_run_missing_the_shape_is_said_rather_than_grouped(self) -> None:
        empty = run_record(
            experiment=SDPA_EXPERIMENT,
            label=probe_cli.SDPA_LABEL,
            fingerprint=selection_record("A30")["fingerprint"],
            observations=(Observation(name="cublaslt_workspace_size", value=-1.0),),
            payload_digest="",
        )
        pairs = (("a100.json", selection_record("A100")), ("a30.json", empty))

        assert report_cli.output_agreement(pairs, TINY) == "not reported"

    def test_the_row_carries_the_output_agreement(self) -> None:
        lines = report_cli.shape_lines(
            (("a100.json", selection_record("A100")), ("a30.json", selection_record("A30"))),
            TINY,
        )

        assert lines[0].endswith("outputs=0,1")

    def test_one_run_alone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least two runs"):
            report_cli.report_lines((("a100.json", selection_record("A100")),))


class TestTheCommandLines:
    def test_main_writes_a_record_that_decodes_back(self, tmp_path: pathlib.Path) -> None:
        assert probe_cli.main(_argv(tmp_path)) == 0

        written = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert written["experiment"] == SDPA_EXPERIMENT

    def test_main_creates_the_parent_directory(self, tmp_path: pathlib.Path) -> None:
        probe_cli.main(_argv(tmp_path))

        assert _out_path(tmp_path).is_file()

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            probe_cli.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--backend"):
            probe_cli.main([*_argv(tmp_path), "--backend", "math"])

    def test_the_report_reads_a_directory(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "sdpa",
            {"a100": selection_record("A100"), "v100": selection_record("V100", chosen="math")},
        )

        assert report_cli.main(["--dir", str(directory)]) == 0

    def test_the_report_refuses_a_missing_directory(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError, match="no such directory"):
            report_cli.main(["--dir", str(tmp_path / "absent")])

    def test_the_probe_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-sdpa-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                probe_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_the_report_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "sdpa", {"a": selection_record("A100"), "b": selection_record("V100")}
        )
        saved = sys.argv
        sys.argv = ["modeltrainer-sdpa-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                report_cli.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0

    def test_running_the_probe_as_main_actually_probes(self, tmp_path: pathlib.Path) -> None:
        module_name = "model_trainer.cli.sdpa_probe"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-sdpa-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert _out_path(tmp_path).is_file()

    def test_running_the_report_as_main_actually_reports(self, tmp_path: pathlib.Path) -> None:
        directory = write_records(
            tmp_path / "sdpa", {"a": selection_record("A100"), "b": selection_record("V100")}
        )
        module_name = "model_trainer.cli.sdpa_probe_report"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-sdpa-report", "--dir", str(directory)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
