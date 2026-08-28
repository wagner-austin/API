"""The two library entry points `lm_head` can take, and the record of both.

The open question this instrument exists for: with split-K removed the A100
and A30 agree on 1,017 of `xl`'s 1,018 traced tensors, and the survivor is
`lm_head` -- a bias-free ``Linear`` that never reaches cuBLASLt, so the
workspace variable governs nothing about it.

These tests run on the CPU, where both arms take the same host path and agree
trivially. What a CPU runner CANNOT establish is the answer, and it does not
try: the finding is a cross-card comparison of the two digests. What it CAN
establish is that the two arms compute the same product, that the record
carries both plus the within-card control, and that the shapes are the output
projections rather than something else.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
import torch
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import legacy_gemm_probe as probe_cli
from model_trainer.core.services.model.gemm_probe import gemm_operands
from model_trainer.core.services.model.gemm_shapes import (
    DIGEST_SUFFIX,
    SUM_SUFFIX,
    GemmShape,
)
from model_trainer.core.services.model.legacy_gemm_probe import (
    ARMS,
    EPILOGUE_ARM,
    LEGACY_ARM,
    LM_HEAD_SHAPES,
    arm_identity,
    arm_outputs,
    arms_agree,
)

#: The cheapest declared shape, for the tests that only need one.
SMALL = LM_HEAD_SHAPES[0]


class TestTheShapes:
    def test_the_ladder_rows_are_the_probe_vocabulary(self) -> None:
        # `rows` is the OUTPUT WIDTH, which for an output projection is the
        # vocabulary. The ladder runs at the probe's 512-token vocabulary, so
        # its rungs are narrower than they are deep at every size above tiny
        # -- the opposite of a real model, which is exactly why the
        # real-vocabulary row below exists rather than being assumed
        # equivalent.
        ladder = [s["rows"] for s in LM_HEAD_SHAPES if s["origin"].endswith("-lm-head")]

        assert ladder == [512, 512, 512, 512]

    def test_the_summed_dimension_spans_the_ladder(self) -> None:
        # K is what a split reduction partitions, so it is the axis the
        # experiment varies. The gate rung's 128 is included deliberately:
        # the trace found `lm_head` AGREEING there, so a rung where the
        # effect is absent is what separates "K is the variable" from "this
        # path always differs".
        ladder = [s["inner"] for s in LM_HEAD_SHAPES if s["origin"].endswith("-lm-head")]

        assert ladder == [128, 1024, 1280, 1600]

    def test_one_row_carries_the_real_vocabulary(self) -> None:
        # No ladder rung uses it and every real model does. If the effect
        # depends on output width rather than on K, this is the row that
        # shows it.
        real = [s for s in LM_HEAD_SHAPES if s["origin"] == "gpt2-real-vocab"]

        assert [s["rows"] for s in real] == [50257]

    def test_no_two_shapes_share_an_origin(self) -> None:
        origins = [s["origin"] for s in LM_HEAD_SHAPES]

        assert len(set(origins)) == len(origins)


class TestTheTwoArms:
    def test_they_compute_the_same_product(self) -> None:
        # A zero bias is exact in IEEE-754, so these are the same product in
        # real arithmetic. Asserted with a tolerance rather than bitwise
        # because bitwise IS the measurement, and a test that required it
        # would be asserting the answer instead of the setup.
        outputs = arm_outputs(SMALL, "cpu")

        assert torch.allclose(outputs["legacy"], outputs["epilogue"], rtol=0.0, atol=1e-3)

    def test_the_epilogue_arm_adds_nothing(self) -> None:
        # If the zero bias were not zero, every cross-card comparison would
        # be measuring the bias rather than the reduction order.
        outputs = arm_outputs(SMALL, "cpu")

        assert torch.equal(outputs["legacy"], torch.mm(*_operands(SMALL)))

    def test_both_arms_have_the_output_shape(self) -> None:
        outputs = arm_outputs(SMALL, "cpu")

        assert tuple(outputs["legacy"].shape) == (SMALL["cols"], SMALL["rows"])
        assert tuple(outputs["epilogue"].shape) == (SMALL["cols"], SMALL["rows"])

    def test_on_one_host_path_the_arms_agree_bitwise(self) -> None:
        # The control's other direction: on the CPU both arms take one
        # reduction, so a disagreement here would mean the operands differ
        # rather than the kernels.
        assert arms_agree(arm_outputs(SMALL, "cpu")) is True

    def test_the_identity_describes_both_arms(self) -> None:
        legacy, epilogue = arm_identity(SMALL, "cpu")

        assert legacy == epilogue

    def test_the_two_arms_are_named_distinctly(self) -> None:
        assert ARMS == (LEGACY_ARM, EPILOGUE_ARM)
        assert LEGACY_ARM != EPILOGUE_ARM


def _operands(shape: GemmShape) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild the two multiplicands for one shape.

    Args:
        shape: The shape to build for.

    Returns:
        ``(x, w)``, matching what the probe multiplies.
    """
    _, x, w = gemm_operands(shape, "cpu")
    return x, w


class TestTheRecord:
    def test_it_declares_its_own_experiment(self) -> None:
        # A record carrying two arms per shape, differenced against the gemm
        # probe's single-arm record, would report every observation unmatched.
        record = probe_cli.legacy_run_record("cpu")

        assert record["experiment"] == "lm-head-entry-point"

    def test_it_carries_a_digest_and_a_sum_for_every_arm(self) -> None:
        record = probe_cli.legacy_run_record("cpu")
        names = {o["name"] for o in record["observations"]}

        for shape in LM_HEAD_SHAPES:
            for arm in ARMS:
                assert probe_cli.arm_label(shape["origin"], arm, DIGEST_SUFFIX) in names
                assert probe_cli.arm_label(shape["origin"], arm, SUM_SUFFIX) in names

    def test_it_carries_the_within_card_control_for_every_shape(self) -> None:
        record = probe_cli.legacy_run_record("cpu")
        names = {o["name"] for o in record["observations"]}

        for shape in LM_HEAD_SHAPES:
            assert f"{shape['origin']}|{probe_cli.SAME_SUFFIX}" in names

    def test_the_control_reads_as_agreement_on_the_cpu(self) -> None:
        record = probe_cli.legacy_run_record("cpu")
        values = {o["name"]: o["value"] for o in record["observations"]}

        controls = [values[f"{s['origin']}|{probe_cli.SAME_SUFFIX}"] for s in LM_HEAD_SHAPES]
        assert controls == [probe_cli.ARMS_AGREE] * len(LM_HEAD_SHAPES)

    def test_an_observation_name_says_shape_arm_and_quantity(self) -> None:
        assert probe_cli.arm_label("xl-lm-head", LEGACY_ARM, DIGEST_SUFFIX) == (
            "xl-lm-head|mm|digest48"
        )


class TestTheCommandLine:
    def test_it_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "nested" / "lmhead.json"

        assert probe_cli.main(["--device", "cpu", "--out", str(out)]) == 0

        written = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert written["label"] == probe_cli.LEGACY_GEMM_LABEL

    def test_running_it_as_a_module_actually_probes(self, tmp_path: pathlib.Path) -> None:
        out = tmp_path / "lmhead.json"
        module_name = "model_trainer.cli.legacy_gemm_probe"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", "--device", "cpu", "--out", str(out)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert raised.value.code == 0
        assert out.is_file()
