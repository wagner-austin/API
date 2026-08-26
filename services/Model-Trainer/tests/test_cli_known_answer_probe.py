"""The environment known-answer probe, exercised as real code on the CPU.

Nothing here is faked. The probe needs no network and no GPU -- that is most
of its point -- so every test runs the production function and asserts on
what it actually returned.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import load_json_str
from platform_core.known_answer import KnownAnswer, check_known_answer
from platform_core.run_record import decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import known_answer_probe as probe_cli
from model_trainer.core.services.model.known_answer_probe import (
    PROBE_EXPERIMENT,
    PROBE_LABEL,
    PROBE_MODEL_SIZE,
    PROBE_OBSERVATION,
    PROBE_SEQUENCE_LEN,
    PROBE_VOCAB_SIZE,
    probe_forward_loss,
)
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "probe.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line for the probe."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path))]


class TestTheProbeItself:
    """The forward pass, run for real on the CPU."""

    def test_it_reproduces_itself_exactly_across_two_calls(self) -> None:
        # The property the whole known-answer mechanism rests on. If a probe
        # cannot reproduce itself in one process it can never establish an
        # expected value for anything.
        first = probe_forward_loss("cpu")
        second = probe_forward_loss("cpu")

        assert first == second

    def test_it_returns_a_loss_in_the_range_a_fresh_model_must_produce(self) -> None:
        # An untrained model over a vocabulary of 512 predicts uniformly, so
        # the loss must sit near ln(512) = 6.2383. Asserting the value lands
        # where the arithmetic says it must is what distinguishes this from
        # asserting that some number came back.
        loss = probe_forward_loss("cpu")

        assert loss == pytest.approx(6.24, abs=0.5)

    def test_the_label_names_every_axis_that_changes_the_number(self) -> None:
        # A label that did not name the shape would let a re-widened probe
        # register under the same name and overwrite an expected value it
        # cannot reproduce.
        assert PROBE_LABEL == "gpt2-tiny-L2-d128-h2-v512-len64-seed42"

    def test_the_label_tracks_the_shared_size_table_rather_than_restating_it(
        self,
    ) -> None:
        # The dimensions come from GPT2_MODEL_SIZES, so changing "tiny" there
        # renames this probe instead of silently redefining it under a label
        # whose expected value was measured on the old shape.
        dims = GPT2_MODEL_SIZES[PROBE_MODEL_SIZE]

        assert f"-L{dims['n_layer']}-" in PROBE_LABEL
        assert f"-d{dims['hidden_size']}-" in PROBE_LABEL
        assert f"-h{dims['n_head']}-" in PROBE_LABEL
        assert f"-v{PROBE_VOCAB_SIZE}-" in PROBE_LABEL
        assert f"-len{PROBE_SEQUENCE_LEN}-" in PROBE_LABEL


class TestTheRecordItWrites:
    """What the CLI leaves behind, which is the durable part."""

    def test_the_record_carries_the_probe_value_and_its_configuration(self) -> None:
        record = probe_cli.probe_run_record("cpu")

        assert record["experiment"] == PROBE_EXPERIMENT
        assert record["label"] == PROBE_LABEL
        assert [o["name"] for o in record["observations"]] == [PROBE_OBSERVATION]
        assert record["observations"][0]["value"] == probe_forward_loss("cpu")

    def test_a_cpu_run_records_no_card_and_no_driver(self) -> None:
        # Not cosmetic. The empty string differs from every real card, so a
        # cpu-measured number can never compare equal to a cuda-measured one.
        record = probe_cli.probe_run_record("cpu")

        assert record["fingerprint"]["gpu_model"] == ""
        assert record["fingerprint"]["driver_version"] == ""

    def test_the_fingerprint_reports_the_determinism_that_was_applied(self) -> None:
        record = probe_cli.probe_run_record("cpu")
        applied: DeterminismRecord = cli_hooks.apply_determinism_hook()

        assert record["fingerprint"]["determinism"] == applied

    def test_main_writes_a_record_that_decodes_back(self, tmp_path: pathlib.Path) -> None:
        assert probe_cli.main(_argv(tmp_path)) == 0

        written = load_json_str(_out_path(tmp_path).read_text(encoding="utf-8"))
        decoded = decode_run_record(written)

        assert decoded["label"] == PROBE_LABEL
        assert decoded["observations"][0]["value"] == probe_forward_loss("cpu")

    def test_main_creates_the_parent_directory_it_was_pointed_at(
        self, tmp_path: pathlib.Path
    ) -> None:
        # The path names a directory that does not exist yet; a job document
        # should not have to mkdir before naming an output.
        assert not _out_path(tmp_path).parent.exists()

        assert probe_cli.main(_argv(tmp_path)) == 0

        assert _out_path(tmp_path).is_file()


class TestTheGateItFeeds:
    """The record is only worth writing if it drives the check correctly."""

    def _answer(self, record_value: float, tolerance: float) -> KnownAnswer:
        """Build a known answer from a CPU probe run."""
        record = probe_cli.probe_run_record("cpu")
        return KnownAnswer(
            label=PROBE_LABEL,
            fingerprint=record["fingerprint"],
            expected=record_value,
            tolerance=tolerance,
        )

    def test_a_rerun_of_the_same_probe_matches_bit_exactly(self) -> None:
        record = probe_cli.probe_run_record("cpu")
        known = self._answer(record["observations"][0]["value"], 0.0)

        outcome = check_known_answer(
            known, record["fingerprint"], record["observations"][0]["value"]
        )

        assert outcome == {"kind": "matches", "observed": known["expected"], "deviation": 0.0}

    def test_a_drifted_value_deviates_rather_than_passing(self) -> None:
        # The check has to fire on a change this small, or a silently
        # rebuilt stack passes it.
        record = probe_cli.probe_run_record("cpu")
        observed = record["observations"][0]["value"]
        known = self._answer(observed, 0.0)

        outcome = check_known_answer(known, record["fingerprint"], observed + 1e-9)

        assert outcome["kind"] == "deviates"

    def test_a_different_card_does_not_apply_rather_than_deviating(self) -> None:
        # The measured reason this matters: this probe returns the SAME value
        # on sm_70, sm_80 and sm_86, so it cannot speak to hardware at all.
        # Reporting a card change as a deviation would condemn a working
        # image for moving nodes.
        record = probe_cli.probe_run_record("cpu")
        observed = record["observations"][0]["value"]
        known = self._answer(observed, 0.0)
        moved = RunFingerprint(
            image_digest=record["fingerprint"]["image_digest"],
            gpu_model="NVIDIA A100 80GB PCIe",
            driver_version=record["fingerprint"]["driver_version"],
            determinism=record["fingerprint"]["determinism"],
        )

        outcome = check_known_answer(known, moved, observed)

        assert outcome["kind"] == "configuration_differs"
        assert [d["axis"] for d in outcome["differences"]] == ["gpu_model"]


class TestTheCommandLine:
    """Arguments, and the refusals that keep a mistyped run from recording."""

    def test_the_console_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-known-answer-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                probe_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            probe_cli.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_out_is_refused_after_no_file_appears(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            probe_cli.main(["--device", "cpu"])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--seed"):
            probe_cli.main([*_argv(tmp_path), "--seed", "1"])

    def test_running_the_module_as_main_actually_probes(self, tmp_path: pathlib.Path) -> None:
        # The regression this exists for: without the __main__ guard,
        # `python -m model_trainer.cli.known_answer_probe` imported the module,
        # ran nothing and exited 0. HPC3 jobs 55595084 and 55595086 each
        # "succeeded" in six seconds having written no record and no stderr.
        # Asserting the guard's presence would not have caught it either --
        # only executing the module as __main__ and demanding the output does.
        module_name = "model_trainer.cli.known_answer_probe"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-known-answer-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert decoded["label"] == PROBE_LABEL
