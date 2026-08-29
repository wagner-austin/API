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
from platform_ml.determinism import (
    ATTENTION_MATH_ONLY,
    ATTENTION_SETTING,
    SPLIT_K_REMOVED,
    SPLIT_K_SETTING,
)

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import probe_trace
from model_trainer.core.services.model.known_answer_probe import probe_forward_loss
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    SUM_SUFFIX,
    TRACE_EXPERIMENT,
    TRACE_RUNGS,
    WORKSPACE_NAME,
    WORKSPACE_UNSET,
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


def _no_workspace() -> Generator[None, None, None]:
    """Pin the split-K condition to "not set" for the duration of one test.

    Pinned rather than inherited: a test runner that happened to have
    ``CUBLASLT_WORKSPACE_SIZE`` exported would otherwise change what these
    tests assert, which is the opposite of what a test is for.

    Yields:
        Nothing; the condition is pinned for the body of the test.
    """
    cli_hooks.env_cublaslt_workspace = lambda: None
    try:
        yield
    finally:
        cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace


no_workspace = pytest.fixture(_no_workspace)


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "trace.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line for the trace."""
    return ["--device", "cpu", "--out", str(_out_path(tmp_path)), "--controls", "none"]


class TestWhatOneRecordCarries:
    def test_it_reports_two_observations_for_every_traced_tensor(self) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        parsed = [parse_trace_name(o["name"]) for o in record["observations"]]
        tensors = [p for p in parsed if p is not None]
        suffixes = {p["suffix"] for p in tensors}

        assert suffixes == {DIGEST_SUFFIX, SUM_SUFFIX}
        assert len(tensors) % 2 == 0

    def test_the_loss_it_records_is_what_the_production_forward_pass_returns(self) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        values = {o["name"]: o["value"] for o in record["observations"]}

        assert values[trace_loss_name("tiny")] == probe_forward_loss("cpu", PROBE_SHAPES["tiny"])

    def test_every_measurement_names_the_rung_it_came_from(self, no_workspace: None) -> None:
        # Everything except the one RECORD-level observation, which describes
        # the process rather than any rung and so carries no rung prefix.
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        names = [o["name"] for o in record["observations"]]

        assert [name for name in names if not name.startswith("tiny|")] == [WORKSPACE_NAME]

    def test_the_observations_come_back_in_execution_order(self) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        digests = [
            parsed
            for parsed in (parse_trace_name(o["name"]) for o in record["observations"])
            if parsed is not None and parsed["suffix"] == DIGEST_SUFFIX
        ]
        steps = [p["step"] for p in digests]

        assert steps == sorted(steps)

    def test_the_first_traced_tensor_is_the_token_embedding_input(self) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        digests = [
            parsed
            for parsed in (parse_trace_name(o["name"]) for o in record["observations"])
            if parsed is not None and parsed["suffix"] == DIGEST_SUFFIX
        ]

        assert (digests[0]["path"], digests[0]["kind"]) == ("transformer.wte", "in")

    def test_it_declares_its_own_experiment_and_a_derived_label(self) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )

        assert record["experiment"] == TRACE_EXPERIMENT
        assert record["label"] == trace_label(CHEAP)

    def test_it_carries_no_payload_digest(self) -> None:
        # The digests ARE the output; a payload digest over them would restate
        # the numbers rather than add the independent check a digest is for.
        assert (
            probe_trace.trace_run_record("cpu", CHEAP, remove_split_k=False, math_attention=False)[
                "payload_digest"
            ]
            == NO_PAYLOAD
        )

    def test_a_cpu_trace_records_no_card_and_no_driver(self) -> None:
        fingerprint = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )["fingerprint"]

        assert (fingerprint["gpu_model"], fingerprint["driver_version"]) == ("", "")

    def test_the_registry_refuses_to_build_an_entry_from_it(self) -> None:
        # A trace record holds thousands of observations; the registry stores
        # exactly one. It must be refused by count rather than quietly
        # registering whichever observation happened to sort first.
        with pytest.raises(ValueError, match="exactly one observation"):
            entry_from_record(
                probe_trace.trace_run_record(
                    "cpu", CHEAP, remove_split_k=False, math_attention=False
                ),
                0.0,
            )


class TestRecordingTheSplitKCondition:
    def test_an_unset_variable_records_the_sentinel(self, no_workspace: None) -> None:
        assert probe_trace.workspace_observation() == {
            "name": WORKSPACE_NAME,
            "value": WORKSPACE_UNSET,
        }

    def test_the_production_reader_returns_what_the_environment_holds(self) -> None:
        from platform_core.config import config_test_hooks

        original = config_test_hooks.get_env

        def _fake_get_env(key: str) -> str | None:
            return "0" if key == "CUBLASLT_WORKSPACE_SIZE" else None

        config_test_hooks.get_env = _fake_get_env
        try:
            assert cli_hooks._default_env_cublaslt_workspace() == "0"
        finally:
            config_test_hooks.get_env = original

    def test_the_production_reader_treats_an_empty_variable_as_unset(self) -> None:
        # cuBLASLt ignores an empty value, so a record claiming a condition
        # its library did not apply would be worse than one saying nothing.
        # The rule lives in the reader, which is where it is exercised.
        from platform_core.config import config_test_hooks

        original = config_test_hooks.get_env

        def _empty_get_env(key: str) -> str | None:
            return "" if key == "CUBLASLT_WORKSPACE_SIZE" else None

        config_test_hooks.get_env = _empty_get_env
        try:
            assert cli_hooks._default_env_cublaslt_workspace() is None
        finally:
            config_test_hooks.get_env = original

    def test_the_intervention_records_as_zero_not_as_unset(self) -> None:
        # The whole experiment is zero versus unset. If these collapsed to
        # one value the record could not name its own arm.
        cli_hooks.env_cublaslt_workspace = lambda: "0"
        try:
            assert probe_trace.workspace_observation()["value"] == 0.0
        finally:
            cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace

        assert WORKSPACE_UNSET != 0.0

    def test_a_size_is_recorded_as_the_number_it_is(self) -> None:
        cli_hooks.env_cublaslt_workspace = lambda: "4194304"
        try:
            assert probe_trace.workspace_observation()["value"] == 4194304.0
        finally:
            cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace

    def test_a_non_integer_stops_the_run_before_it_spends_a_gpu(self) -> None:
        cli_hooks.env_cublaslt_workspace = lambda: "lots"
        try:
            with pytest.raises(ValueError, match="which is not an integer"):
                probe_trace.trace_run_record(
                    "cpu", CHEAP, remove_split_k=False, math_attention=False
                )
        finally:
            cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace

    def test_the_record_carries_the_condition_beside_the_tensors(self, no_workspace: None) -> None:
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        named = {o["name"]: o["value"] for o in record["observations"]}

        assert named[WORKSPACE_NAME] == WORKSPACE_UNSET

    def test_the_condition_is_not_mistaken_for_a_traced_tensor(self) -> None:
        assert parse_trace_name(WORKSPACE_NAME) is None


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
            probe_trace.trace_run_record(
                "cpu", ("enormous",), remove_split_k=False, math_attention=False
            )

    def test_a_repeated_rung_is_refused_before_any_work(self) -> None:
        with pytest.raises(ValueError, match="cannot walk one rung twice"):
            probe_trace.trace_run_record(
                "cpu", ("tiny", "tiny"), remove_split_k=False, math_attention=False
            )


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
            probe_trace.main(["--out", str(_out_path(tmp_path)), "--controls", "none"])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            probe_trace.main(["--device", "cpu", "--controls", "none"])

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


class TestTheControlArms:
    """The flag exists so the instrument can reach the treated condition.

    Split-K had an environment escape and the attention pin has none -- it is
    four `torch.backends.cuda` calls -- so before this flag a trace could only
    ever observe attention untreated. An instrument that cannot reach a
    condition cannot measure it.
    """

    def test_every_arm_names_a_distinct_posture(self) -> None:
        # Four arms because the two controls are disjoint: split-K governs
        # cuBLASLt matmuls, the math pin governs attention. The single-control
        # arms are what make attribution a run rather than a code change.
        assert probe_trace.CONTROL_ARMS == {
            "none": (False, False),
            "split-k": (True, False),
            "attention": (False, True),
            "both": (True, True),
        }

    def test_it_resolves_each_arm(self) -> None:
        assert probe_trace.require_control_arm("none") == (False, False)
        assert probe_trace.require_control_arm("both") == (True, True)

    def test_an_unknown_arm_is_refused_by_name(self) -> None:
        # Refused rather than defaulted: a trace whose arm was guessed is a
        # trace whose record names a condition it may not have run.
        with pytest.raises(ValueError, match="must be one of attention, both, none, split-k"):
            probe_trace.require_control_arm("splitk")

    def test_the_applied_arm_reaches_the_determinism_record(self, tmp_path: pathlib.Path) -> None:
        # The record is how a reader knows which arm produced a trace, and
        # `apply_determinism` writes these keys ONLY when it applied them --
        # so their presence is evidence, not a restatement of the flag.
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=True, math_attention=True
        )
        settings = dict(record["fingerprint"]["determinism"]["settings"])

        assert settings[SPLIT_K_SETTING] == SPLIT_K_REMOVED
        assert settings[ATTENTION_SETTING] == ATTENTION_MATH_ONLY

    def test_the_none_arm_leaves_the_record_as_it_was(self, tmp_path: pathlib.Path) -> None:
        # The arm every other measurement command is fixed at. It must add no
        # setting, so a trace taken under it stays comparable with every trace
        # taken before the flag existed.
        record = probe_trace.trace_run_record(
            "cpu", CHEAP, remove_split_k=False, math_attention=False
        )
        settings = dict(record["fingerprint"]["determinism"]["settings"])

        assert SPLIT_K_SETTING not in settings
        assert ATTENTION_SETTING not in settings

    def test_the_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--controls"):
            probe_trace.main(["--device", "cpu", "--out", str(_out_path(tmp_path))])
