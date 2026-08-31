"""The train-step probe: one backward pass and one update, digested exactly.

The forward measurements have a whole suite each; what is new here is the
backward walk -- gradients digested before the update, the update applied at
the declared step size, a parameter without a gradient refused -- and the
command line that records it. Everything runs the REAL arithmetic on the CPU
at the tiny rung, per this package's no-mocks rule; only the walked shape
table is small.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.run_record import NO_PAYLOAD, decode_run_record

from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import train_step_probe as step_cli
from model_trainer.core.services.model.deterministic_gemm import CUBLAS_ARM, RANK1_ARM
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES, require_probe_shape
from model_trainer.core.services.model.tensor_digest import describe_tensor
from model_trainer.core.services.model.trace_plan import WORKSPACE_NAME, WORKSPACE_UNSET
from model_trainer.core.services.model.train_step_plan import (
    GRAD_KIND,
    TRAIN_STEP_EXPERIMENT,
    TRAIN_STEP_LR,
    TRAIN_STEP_RUNGS,
    UPDATED_KIND,
    require_train_rungs,
    train_loss_name,
    train_step_label,
    train_tensor_name,
)
from model_trainer.core.services.model.train_step_probe import (
    TrainTensor,
    digest_step_tensors,
    require_step_reproduced,
    train_step_identity,
    train_step_once,
)
from model_trainer.core.types import NamedParameter, TracedLMModelProto

#: The one rung cheap enough to step repeatedly on a CPU.
TINY = require_probe_shape("tiny")


def _no_workspace() -> Generator[None, None, None]:
    """Pin the split-K condition to "not set" for the duration of one test.

    Pinned rather than inherited, for the reason ``test_probe_trace`` pins
    it: a test runner that happened to have ``CUBLASLT_WORKSPACE_SIZE``
    exported would otherwise change what these tests assert.

    Yields:
        Nothing; the pin holds for the body of the test.
    """
    cli_hooks.env_cublaslt_workspace = lambda: None
    try:
        yield
    finally:
        cli_hooks.env_cublaslt_workspace = cli_hooks._default_env_cublaslt_workspace


no_workspace = pytest.fixture(_no_workspace)


def _stepped_tiny() -> TracedLMModelProto:
    """Build the tiny rung on the CPU and run its backward pass.

    The real model through the real builder rather than a hand-made module,
    per this package's no-mocks rule -- and because the walk order under test
    is ``named_parameters`` order of the model the cluster actually steps.

    Returns:
        The model, gradients populated and update not yet applied.
    """
    model, ids = probe_model_and_input("cpu", TINY)
    torch.autograd.backward([model.forward(input_ids=ids, labels=ids).loss])
    return model


def _first_parameter(model: TracedLMModelProto) -> tuple[str, NamedParameter, torch.Tensor]:
    """Return the first walked parameter and its gradient.

    Raises:
        RuntimeError: If backward left it no gradient -- the fixture would be
            broken, not the code under test.
    """
    path, param = next(iter(model.named_parameters()))
    grad = param.grad
    if grad is None:
        raise RuntimeError(f"backward left {path} no gradient; the fixture is broken")
    return path, param, grad


class TestThePlan:
    def test_rungs_parse_in_order_and_strip_spaces(self) -> None:
        assert require_train_rungs("tiny, medium,large") == ("tiny", "medium", "large")

    def test_an_empty_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--rungs"):
            require_train_rungs("")

    def test_an_empty_item_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--rungs"):
            require_train_rungs("tiny,,large")

    def test_a_repeated_rung_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tiny"):
            require_train_rungs("tiny,tiny")

    def test_the_label_carries_count_controls_and_kernel(self) -> None:
        label = train_step_label(("tiny", "large"), "both", CUBLAS_ARM)

        assert label.startswith("train-step-2x")
        assert label.endswith("-both-cublas")

    def test_two_rung_sets_never_share_a_label(self) -> None:
        assert train_step_label(("tiny",), "both", CUBLAS_ARM) != train_step_label(
            ("large",), "both", CUBLAS_ARM
        )

    def test_a_tensor_name_carries_rung_kind_path_and_measurement(self) -> None:
        assert (
            train_tensor_name("large", GRAD_KIND, "transformer.wte.weight", "digest48")
            == "large|grad|transformer.wte.weight|digest48"
        )

    def test_a_loss_name_is_two_fields(self) -> None:
        assert train_loss_name("large") == "large|loss"

    def test_the_default_rungs_are_declared_ladder_rungs_without_xl(self) -> None:
        # xl's gradients double its parameter memory past a 16 GB V100; the
        # contrast the experiment needs -- an agreeing rung and the breaking
        # rung -- is present without it. See the plan module's docstring.
        assert TRAIN_STEP_RUNGS == ("tiny", "medium", "large")
        assert [rung for rung in TRAIN_STEP_RUNGS if rung not in PROBE_SHAPES] == []


class TestDigestStepTensors:
    def test_the_walk_interleaves_gradient_then_update_in_parameter_order(self) -> None:
        model = _stepped_tiny()
        walked = [path for path, _ in model.named_parameters()]

        tensors = digest_step_tensors(model)

        expected = [(kind, path) for path in walked for kind in (GRAD_KIND, UPDATED_KIND)]
        assert [(t["kind"], t["path"]) for t in tensors] == expected

    def test_gradients_are_digested_before_the_update_and_values_after(self) -> None:
        model = _stepped_tiny()
        path, param, grad = _first_parameter(model)
        expected_grad = describe_tensor(grad)
        expected_updated = describe_tensor(param.detach().clone().add_(grad, alpha=-TRAIN_STEP_LR))

        tensors = digest_step_tensors(model)

        assert tensors[0] == TrainTensor(
            kind=GRAD_KIND, path=path, digest=expected_grad[0], total=expected_grad[1]
        )
        assert tensors[1] == TrainTensor(
            kind=UPDATED_KIND, path=path, digest=expected_updated[0], total=expected_updated[1]
        )
        # And the update really landed in the model, not only in the record.
        assert describe_tensor(param.detach()) == expected_updated

    def test_a_parameter_without_a_gradient_is_refused_by_name(self) -> None:
        # A model whose backward never ran has no gradients at all -- the
        # realest way this arm fires: digesting before stepping. Nothing may
        # be recorded, because the record would claim a step nobody took.
        model, _ = probe_model_and_input("cpu", TINY)
        path, _ = next(iter(model.named_parameters()))

        with pytest.raises(ValueError, match="no gradient after backward") as excinfo:
            digest_step_tensors(model)

        # The refusal names the parameter, not just the condition.
        assert path in str(excinfo.value)


class TestTheStep:
    def test_one_step_digests_two_tensors_per_parameter(self) -> None:
        tensors, loss = train_step_once("cpu", TINY, kernel=CUBLAS_ARM)

        parameters = {t["path"] for t in tensors}
        assert len(tensors) == 2 * len(parameters)
        assert sorted({t["kind"] for t in tensors}) == [GRAD_KIND, UPDATED_KIND]
        assert loss > 0.0

    def test_the_step_reproduces_itself_on_cpu(self) -> None:
        assert train_step_identity("cpu", TINY, kernel=CUBLAS_ARM) == train_step_once(
            "cpu", TINY, kernel=CUBLAS_ARM
        )

    def test_a_treated_arm_steps_the_same_parameters(self) -> None:
        # The rank-one arm swaps every Conv1D and the lm_head, holding the
        # ORIGINAL parameters by reference -- so the walk must see the same
        # paths and the backward must reach every one of them.
        treated, _ = train_step_once("cpu", TINY, kernel=RANK1_ARM)
        untreated, untreated_loss = train_step_once("cpu", TINY, kernel=CUBLAS_ARM)

        assert [(t["kind"], t["path"]) for t in treated] == [
            (t["kind"], t["path"]) for t in untreated
        ]
        assert untreated_loss > 0.0

    def test_an_unknown_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kernel must be one of"):
            train_step_once("cpu", TINY, kernel="triton")


class TestTheReproductionGuard:
    def test_matching_runs_pass_through(self) -> None:
        tensors = (TrainTensor(kind=GRAD_KIND, path="w", digest=1.0, total=2.0),)

        assert require_step_reproduced(tensors, tensors, 3.0, 3.0, "cpu") == (tensors, 3.0)

    def test_a_differing_tensor_is_refused(self) -> None:
        first = (TrainTensor(kind=GRAD_KIND, path="w", digest=1.0, total=2.0),)
        second = (TrainTensor(kind=GRAD_KIND, path="w", digest=9.0, total=2.0),)

        with pytest.raises(RuntimeError, match="did not reproduce itself on cpu"):
            require_step_reproduced(first, second, 3.0, 3.0, "cpu")

    def test_a_differing_loss_is_refused(self) -> None:
        tensors = (TrainTensor(kind=GRAD_KIND, path="w", digest=1.0, total=2.0),)

        with pytest.raises(RuntimeError, match="did not reproduce itself on cpu"):
            require_step_reproduced(tensors, tensors, 3.0, 4.0, "cpu")


def _out_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the record path used by the CLI tests."""
    return tmp_path / "records" / "train-step.json"


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Return a complete CPU command line for the tiny rung."""
    return [
        "--device",
        "cpu",
        "--rungs",
        "tiny",
        "--out",
        str(_out_path(tmp_path)),
        "--controls",
        "none",
        "--kernel",
        CUBLAS_ARM,
    ]


class TestTheRecord:
    def test_it_carries_workspace_loss_and_two_observations_per_tensor(
        self, no_workspace: None
    ) -> None:
        record = step_cli.train_step_run_record(
            "cpu", ("tiny",), controls="none", kernel=CUBLAS_ARM
        )
        tensors, loss = train_step_once("cpu", TINY, kernel=CUBLAS_ARM)

        assert record["experiment"] == TRAIN_STEP_EXPERIMENT
        assert record["label"] == train_step_label(("tiny",), "none", CUBLAS_ARM)
        assert record["payload_digest"] == NO_PAYLOAD
        assert len(record["observations"]) == 1 + 2 * len(tensors) + 1

        by_name = {o["name"]: o["value"] for o in record["observations"]}
        assert by_name[WORKSPACE_NAME] == WORKSPACE_UNSET
        assert by_name[train_loss_name("tiny")] == loss
        first = tensors[0]
        assert (
            by_name[train_tensor_name("tiny", first["kind"], first["path"], "digest48")]
            == first["digest"]
        )
        assert (
            by_name[train_tensor_name("tiny", first["kind"], first["path"], "sum")]
            == (first["total"])
        )

    def test_an_undeclared_rung_is_refused_before_anything_computes(
        self, no_workspace: None
    ) -> None:
        with pytest.raises(KeyError, match="huge"):
            step_cli.train_step_run_record("cpu", ("huge",), controls="none", kernel=CUBLAS_ARM)

    def test_an_unknown_control_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--controls"):
            step_cli.train_step_run_record("cpu", ("tiny",), controls="all", kernel=CUBLAS_ARM)


class TestTheCommandLine:
    def test_main_writes_a_record_that_decodes_back(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        assert step_cli.main(_argv(tmp_path)) == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))

        assert decoded["label"] == train_step_label(("tiny",), "none", CUBLAS_ARM)
        assert decoded["experiment"] == TRAIN_STEP_EXPERIMENT

    def test_an_absent_device_is_refused_and_nothing_is_written(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="--device"):
            step_cli.main(["--out", str(_out_path(tmp_path))])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_rungs_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--rungs"):
            step_cli.main(
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(_out_path(tmp_path)),
                    "--controls",
                    "none",
                    "--kernel",
                    CUBLAS_ARM,
                ]
            )

    def test_a_repeated_rung_is_refused_before_anything_computes(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ValueError, match="tiny"):
            step_cli.main([*_argv(tmp_path)[:3], "tiny,tiny", *_argv(tmp_path)[4:]])

        assert not _out_path(tmp_path).exists()

    def test_an_absent_controls_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--controls"):
            step_cli.main(
                [
                    "--device",
                    "cpu",
                    "--rungs",
                    "tiny",
                    "--out",
                    str(_out_path(tmp_path)),
                    "--kernel",
                    CUBLAS_ARM,
                ]
            )

    def test_an_absent_kernel_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--kernel"):
            step_cli.main(
                [
                    "--device",
                    "cpu",
                    "--rungs",
                    "tiny",
                    "--out",
                    str(_out_path(tmp_path)),
                    "--controls",
                    "none",
                ]
            )

    def test_an_absent_out_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--out"):
            step_cli.main(
                ["--device", "cpu", "--rungs", "tiny", "--controls", "none", "--kernel", CUBLAS_ARM]
            )

    def test_the_entry_point_carries_the_exit_code(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-train-step-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                step_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_steps(
        self, tmp_path: pathlib.Path, no_workspace: None
    ) -> None:
        module_name = "model_trainer.cli.train_step_probe"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-train-step-probe", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0

        decoded = decode_run_record(load_json_str(_out_path(tmp_path).read_text(encoding="utf-8")))
        assert decoded["experiment"] == TRAIN_STEP_EXPERIMENT
