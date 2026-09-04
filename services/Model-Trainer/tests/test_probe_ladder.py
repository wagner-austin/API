"""The probe ladder, exercised as real code on the CPU.

Nothing here is faked. The rungs run, the record is written and read back, and
the values are compared against the production forward pass.

WHAT IS DELIBERATELY SMALLER THAN PRODUCTION. The declared ladder ends at a
1.5-billion-parameter model, which is a GPU measurement and not something to
construct on a test runner. So these tests install a two-rung ladder through
``cli/_test_hooks`` and walk every line of the real code with it, and one test
asserts that the production hook returns the full table. What is NOT covered
here is arithmetic on the large rungs -- that is what the cluster runs, and no
CPU test could stand in for it anyway, since the whole question is what
different cards do.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping

import pytest
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import entry_from_record, gate_record
from platform_core.run_record import NO_PAYLOAD, decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import known_answer_probe as probe_cli
from model_trainer.cli import probe_ladder
from model_trainer.core.services.model.known_answer_probe import probe_forward_loss
from model_trainer.core.services.model.probe_shapes import (
    PROBE_EXPERIMENT,
    PROBE_SHAPES,
    ProbeShape,
    probe_label,
)

#: Two rungs that differ on the length axis and both build a tiny model. Real
#: rungs from the real table, so the labels and the walk are production ones.
CHEAP: Mapping[str, ProbeShape] = {
    "tiny": PROBE_SHAPES["tiny"],
    "tiny-len128": PROBE_SHAPES["tiny-len128"],
}


def _cheap_ladder() -> Generator[None, None, None]:
    """Install the two-rung ladder for the duration of one test.

    Written as ``pytest.fixture(impl)`` below rather than with the decorator,
    matching ``tests/conftest.py``: the decorator form returns an overloaded
    function and trips this package's ``disallow_any_decorated``.

    Yields:
        Nothing; the ladder is installed for the body of the test.
    """
    measurement_hooks.ladder_shapes = lambda: CHEAP
    try:
        yield
    finally:
        measurement_hooks.ladder_shapes = measurement_hooks._default_ladder_shapes


cheap_ladder = pytest.fixture(_cheap_ladder)


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "ladder.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line for the ladder."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path))]


class TestTheLadderLabel:
    def test_it_counts_the_rungs_and_digests_them(self) -> None:
        label = probe_ladder.ladder_label(("a", "b", "c"))

        assert label.startswith("probe-ladder-3x")
        assert len(label) == len("probe-ladder-3x") + probe_ladder.LADDER_DIGEST_CHARS

    def test_adding_a_rung_renames_the_ladder(self) -> None:
        # A shorter ladder agrees trivially over the rungs it kept, so two
        # different ladders must never share a name.
        assert probe_ladder.ladder_label(("a", "b")) != probe_ladder.ladder_label(("a", "b", "c"))

    def test_reordering_the_rungs_renames_the_ladder(self) -> None:
        # Order is part of what ran: the rungs share one process and one
        # allocator, so a ladder walked backwards is not the same measurement.
        assert probe_ladder.ladder_label(("a", "b")) != probe_ladder.ladder_label(("b", "a"))

    def test_the_same_rungs_always_produce_the_same_name(self) -> None:
        assert probe_ladder.ladder_label(("a", "b")) == probe_ladder.ladder_label(("a", "b"))

    def test_rung_labels_come_back_in_table_order(self) -> None:
        assert probe_ladder.ladder_rung_labels(CHEAP) == (
            probe_label(PROBE_SHAPES["tiny"]),
            probe_label(PROBE_SHAPES["tiny-len128"]),
        )


class TestTheRecordItBuilds:
    def test_it_reports_one_observation_per_rung_named_by_its_label(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)

        assert sorted(o["name"] for o in record["observations"]) == sorted(
            probe_ladder.ladder_rung_labels(CHEAP)
        )

    def test_every_value_is_what_the_production_forward_pass_returns(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)
        by_name = {o["name"]: o["value"] for o in record["observations"]}

        for shape in CHEAP.values():
            assert by_name[probe_label(shape)] == probe_forward_loss("cpu", shape)

    def test_the_two_rungs_do_not_return_the_same_number(self) -> None:
        # Not a formality. If they did, the ladder would be measuring one
        # thing twice and every rung would agree across cards for free.
        record = probe_ladder.ladder_run_record("cpu", CHEAP)
        values = [o["value"] for o in record["observations"]]

        assert len(set(values)) == len(values)

    def test_it_declares_its_own_experiment(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)

        assert record["experiment"] == probe_ladder.LADDER_EXPERIMENT
        assert record["experiment"] != PROBE_EXPERIMENT

    def test_it_carries_no_payload_digest(self) -> None:
        # The values ARE the output; a digest over them would restate the
        # numbers rather than add an independent check.
        assert probe_ladder.ladder_run_record("cpu", CHEAP)["payload_digest"] == NO_PAYLOAD

    def test_it_pins_what_the_gate_probe_pins_on_the_same_device(self) -> None:
        # The reason the pin is imported rather than re-derived: a ladder and
        # a gate run on one card must describe one configuration, or neither
        # says anything about the other.
        ladder = probe_ladder.ladder_run_record("cpu", CHEAP)
        gate = probe_cli.probe_run_record("cpu")

        assert ladder["fingerprint"]["determinism"] == gate["fingerprint"]["determinism"]

    def test_a_cpu_ladder_records_no_card_and_no_driver(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)

        assert record["fingerprint"]["gpu_model"] == ""
        assert record["fingerprint"]["driver_version"] == ""


class TestItCannotBeMistakenForAGateRun:
    """A ladder record offered to the registry is refused by count."""

    def test_the_registry_refuses_to_build_an_entry_from_it(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)

        with pytest.raises(ValueError, match="exactly one observation, record has 2"):
            entry_from_record(record, 0.0)

    def test_the_gate_refuses_to_check_it(self) -> None:
        record = probe_ladder.ladder_run_record("cpu", CHEAP)

        with pytest.raises(ValueError, match="exactly one observation, got 2"):
            gate_record((), record)


class TestTheCommandLine:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_ladder: None
    ) -> None:
        assert probe_ladder.main(_argv(tmp_path)) == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))

        assert decoded["label"] == probe_ladder.ladder_label(probe_ladder.ladder_rung_labels(CHEAP))
        assert len(decoded["observations"]) == len(CHEAP)

    def test_main_creates_the_parent_directory_it_was_pointed_at(
        self, tmp_path: pathlib.Path, cheap_ladder: None
    ) -> None:
        assert not _out_path(tmp_path).parent.exists()

        assert probe_ladder.main(_argv(tmp_path)) == 0

        assert _out_path(tmp_path).is_file()

    def test_the_production_hook_walks_the_whole_declared_ladder(self) -> None:
        # What the two-rung fixture would otherwise hide: that the deployed
        # command runs every rung, not the cheap pair these tests install.
        assert measurement_hooks._default_ladder_shapes() == PROBE_SHAPES

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            probe_ladder.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            probe_ladder.main(["--device", "cpu"])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--rung"):
            probe_ladder.main([*_argv(tmp_path), "--rung", "xl"])

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_ladder: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-probe-ladder", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                probe_ladder.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_walks_the_ladder(
        self, tmp_path: pathlib.Path, cheap_ladder: None
    ) -> None:
        # The regression this exists for, inherited from the gate probe:
        # without the __main__ guard, `python -m ...` imported the module, ran
        # nothing and exited 0. HPC3 jobs 55595084 and 55595086 each
        # "succeeded" in six seconds having written no record and no stderr.
        # Asserting the guard's presence would not catch it -- only executing
        # the module as __main__ and demanding the output does.
        module_name = "model_trainer.cli.probe_ladder"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-probe-ladder", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert len(decoded["observations"]) == len(CHEAP)
