"""The trace command, exercised as real code on the CPU.

Nothing is faked. The rung runs, the hooks fire, the record is written and
read back, and the loss it carries is compared against the production forward
pass.

WHAT IS DELIBERATELY SMALLER THAN PRODUCTION. The declared trace walks four
rungs ending at a 1.5-billion-parameter model and digests about a hundred and
seventy million floats -- a GPU measurement, and not something to run on a
test runner. So these install a one-rung trace through ``cli/_test_hooks``
and walk every line of the real code with it, and one test asserts that the
production hook returns the whole declared set.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import load_json_str
from platform_core.known_answer_registry import entry_from_record
from platform_core.run_record import NO_PAYLOAD, decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import probe_trace
from model_trainer.core.services.model.known_answer_probe import probe_forward_loss
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    SUM_SUFFIX,
    TRACE_EXPERIMENT,
    TRACE_RUNGS,
    parse_trace_name,
    trace_label,
    trace_loss_name,
)

#: One real rung from the real table, so the labels and the walk are
#: production ones and only the SIZE of the work is reduced.
CHEAP = ("tiny",)


def _cheap_trace() -> Generator[None, None, None]:
    """Install the one-rung trace for the duration of one test.

    Written as ``pytest.fixture(impl)`` below rather than with the decorator,
    matching ``tests/conftest.py``: the decorator form returns an overloaded
    function and trips this package's ``disallow_any_decorated``.

    Yields:
        Nothing; the trace is installed for the body of the test.
    """
    cli_hooks.trace_rungs = lambda: CHEAP
    try:
        yield
    finally:
        cli_hooks.trace_rungs = cli_hooks._default_trace_rungs


cheap_trace = pytest.fixture(_cheap_trace)


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "trace.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line for the trace."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path))]


class TestWhatOneRecordCarries:
    def test_it_reports_two_observations_for_every_traced_tensor(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)
        parsed = [parse_trace_name(o["name"]) for o in record["observations"]]
        tensors = [p for p in parsed if p is not None]
        suffixes = {p["suffix"] for p in tensors}

        assert suffixes == {DIGEST_SUFFIX, SUM_SUFFIX}
        assert len(tensors) % 2 == 0

    def test_the_loss_it_records_is_what_the_production_forward_pass_returns(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)
        values = {o["name"]: o["value"] for o in record["observations"]}

        assert values[trace_loss_name("tiny")] == probe_forward_loss("cpu", PROBE_SHAPES["tiny"])

    def test_every_observation_names_the_rung_it_came_from(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)
        names = [o["name"] for o in record["observations"]]

        assert [name for name in names if not name.startswith("tiny|")] == []

    def test_the_observations_come_back_in_execution_order(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)
        digests = [
            parsed
            for parsed in (parse_trace_name(o["name"]) for o in record["observations"])
            if parsed is not None and parsed["suffix"] == DIGEST_SUFFIX
        ]
        steps = [p["step"] for p in digests]

        assert steps == sorted(steps)

    def test_the_first_traced_tensor_is_the_token_embedding_input(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)
        digests = [
            parsed
            for parsed in (parse_trace_name(o["name"]) for o in record["observations"])
            if parsed is not None and parsed["suffix"] == DIGEST_SUFFIX
        ]

        assert (digests[0]["path"], digests[0]["kind"]) == ("transformer.wte", "in")

    def test_it_declares_its_own_experiment_and_a_derived_label(self) -> None:
        record = probe_trace.trace_run_record("cpu", CHEAP)

        assert record["experiment"] == TRACE_EXPERIMENT
        assert record["label"] == trace_label(CHEAP)

    def test_it_carries_no_payload_digest(self) -> None:
        # The digests ARE the output; a payload digest over them would restate
        # the numbers rather than add the independent check a digest is for.
        assert probe_trace.trace_run_record("cpu", CHEAP)["payload_digest"] == NO_PAYLOAD

    def test_a_cpu_trace_records_no_card_and_no_driver(self) -> None:
        fingerprint = probe_trace.trace_run_record("cpu", CHEAP)["fingerprint"]

        assert (fingerprint["gpu_model"], fingerprint["driver_version"]) == ("", "")

    def test_the_registry_refuses_to_build_an_entry_from_it(self) -> None:
        # A trace record holds thousands of observations; the registry stores
        # exactly one. It must be refused by count rather than quietly
        # registering whichever observation happened to sort first.
        with pytest.raises(ValueError, match="exactly one observation"):
            entry_from_record(probe_trace.trace_run_record("cpu", CHEAP), 0.0)


class TestNamingTracedTensors:
    def test_a_tensor_becomes_a_digest_row_and_a_sum_row(self) -> None:
        observations = probe_trace.tensor_observations(
            "tiny",
            (
                {
                    "step": 3,
                    "path": "transformer.wte",
                    "module_class": "Embedding",
                    "kind": "out",
                    "index": 0,
                    "digest": 12.0,
                    "total": -4.5,
                },
            ),
        )

        assert [o["name"] for o in observations] == [
            "tiny|00003|out|0|Embedding|transformer.wte|digest48",
            "tiny|00003|out|0|Embedding|transformer.wte|sum",
        ]
        assert [o["value"] for o in observations] == [12.0, -4.5]

    def test_no_traced_tensors_means_no_observations(self) -> None:
        assert probe_trace.tensor_observations("tiny", ()) == ()


class TestRefusals:
    def test_an_unknown_rung_is_refused_and_nothing_is_traced(self) -> None:
        with pytest.raises(KeyError, match="unknown probe rung 'enormous'"):
            probe_trace.trace_run_record("cpu", ("enormous",))

    def test_a_repeated_rung_is_refused_before_any_work(self) -> None:
        with pytest.raises(ValueError, match="cannot walk one rung twice"):
            probe_trace.trace_run_record("cpu", ("tiny", "tiny"))


class TestTheCommandLine:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, cheap_trace: None
    ) -> None:
        assert probe_trace.main(_argv(tmp_path)) == 0

        written = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert written["experiment"] == TRACE_EXPERIMENT
        assert written["label"] == trace_label(CHEAP)

    def test_main_creates_the_parent_directory_it_was_pointed_at(
        self, tmp_path: pathlib.Path, cheap_trace: None
    ) -> None:
        assert not _out_path(tmp_path).parent.exists()

        probe_trace.main(_argv(tmp_path))

        assert _out_path(tmp_path).is_file()

    def test_the_production_hook_walks_the_whole_declared_set(self) -> None:
        assert cli_hooks._default_trace_rungs() == TRACE_RUNGS

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            probe_trace.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            probe_trace.main(["--device", "cpu"])

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--rung"):
            probe_trace.main([*_argv(tmp_path), "--rung", "tiny"])

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, cheap_trace: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-probe-trace", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                probe_trace.entrypoint()
        finally:
            sys.argv = saved

        assert raised.value.code == 0
        assert _out_path(tmp_path).is_file()

    def test_running_the_module_as_main_actually_traces(
        self, tmp_path: pathlib.Path, cheap_trace: None
    ) -> None:
        # Without the __main__ guard the module imports, runs nothing and
        # exits 0 -- which is how two cluster jobs "succeeded" in six seconds
        # having written no record and no stderr. Asserting the guard's
        # presence would not catch it; only executing the module as __main__
        # and demanding the output does.
        module_name = "model_trainer.cli.probe_trace"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-probe-trace", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0

        written = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert written["label"] == trace_label(CHEAP)
