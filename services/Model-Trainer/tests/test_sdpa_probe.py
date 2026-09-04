"""The attention-backend probe, run for real on the CPU.

Nothing is faked. Every case here issues a real
`scaled_dot_product_attention` call through the production functions.

WHAT A CPU RUNNER CAN AND CANNOT ESTABLISH. It cannot answer the question the
probe exists for -- which kernel a V100 picks and whether an A100 picks the
same one -- because that needs two cards. What it CAN establish is that the
instrument is sound, and the CPU dispatcher happens to exercise every arm:
measured 2026-08-27 on torch 2.6.0+cu124, forcing `math` and `flash` on cpu
RUNS, while forcing `efficient` and `cudnn` makes torch refuse for want of a
kernel. So both the available and unavailable paths are real calls here, not
constructions.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention import SDPBackend

from model_trainer.core import _test_hooks
from model_trainer.core._hook_defaults_cuda import _default_sdpa_cuda_eligibility
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.sdpa_probe import (
    BACKENDS,
    NO_KERNEL_MESSAGES,
    default_digest,
    forced_digest,
    forced_sdpa_output,
    is_no_kernel_refusal,
    probe_sdpa,
    sdpa_eligibility,
    sdpa_operands,
    sdpa_output,
)
from model_trainer.core.services.model.sdpa_shapes import (
    BACKEND_KEYS,
    ELIGIBLE_KEYS,
    SDPA_EXPERIMENT,
    SdpaShape,
    sdpa_label,
    sdpa_shape_for,
    sdpa_shapes,
)

TINY = sdpa_shape_for("tiny")


class TestTheShapeTable:
    def test_every_shape_is_the_call_its_rung_actually_makes(self) -> None:
        # Derived from the ladder rather than restated, so a rung reshaped
        # there changes what this probes instead of leaving it measuring a
        # shape nothing runs.
        assert TINY == {"rung": "tiny", "heads": 2, "head_dim": 128 // 2, "sequence_len": 64}

    def test_the_xl_rung_folds_1600_into_25_heads_of_64(self) -> None:
        assert sdpa_shape_for("xl") == {
            "rung": "xl",
            "heads": 25,
            "head_dim": 64,
            "sequence_len": 64,
        }

    def test_every_gpt2_size_has_the_same_head_width(self) -> None:
        # Which is what makes the size axis a head-COUNT axis. If this ever
        # stops holding, a rung differs on two things at once.
        assert {shape["head_dim"] for shape in sdpa_shapes()} == {64}

    def test_the_length_axis_moves_only_the_sequence(self) -> None:
        lengths = [s["sequence_len"] for s in sdpa_shapes() if s["rung"].startswith("tiny")]

        assert lengths == [64, 128, 256, 512]

    def test_it_covers_every_ladder_rung(self) -> None:
        assert tuple(s["rung"] for s in sdpa_shapes()) == tuple(PROBE_SHAPES)

    def test_an_unknown_rung_is_refused(self) -> None:
        with pytest.raises(KeyError):
            sdpa_shape_for("enormous")


class TestTheOperands:
    def test_they_carry_the_layout_split_heads_produces(self) -> None:
        # GPT-2 permutes rather than reshaping, leaving a NON-contiguous
        # tensor, and transformers only forces contiguity when there is an
        # attention mask -- which this path has not. Backend eligibility
        # depends on strides, so a contiguous probe could measure a
        # different selection than the model gets.
        query, key, value = sdpa_operands(TINY, "cpu")

        assert tuple(query.shape) == (1, 2, 64, 64)
        assert not query.is_contiguous()
        assert not key.is_contiguous()
        assert not value.is_contiguous()

    def test_query_key_and_value_are_three_different_tensors(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        assert not torch.equal(query, key)
        assert not torch.equal(key, value)

    def test_they_are_the_same_bits_every_time(self) -> None:
        # Generated on the CPU under a fixed seed and moved, so two cards
        # attend over identical inputs and a difference is about the cards.
        first = sdpa_operands(TINY, "cpu")
        second = sdpa_operands(TINY, "cpu")

        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


class TestRecognisingARefusal:
    def test_the_cuda_wording_is_a_refusal(self) -> None:
        assert is_no_kernel_refusal("No available kernel. Aborting execution.")

    def test_the_cpu_wording_is_a_refusal(self) -> None:
        assert is_no_kernel_refusal("No viable backend for scaled_dot_product_attention was found.")

    def test_an_unrelated_failure_is_not_a_refusal(self) -> None:
        # An out-of-memory recorded as "this backend is unavailable" would be
        # a false fact about the hardware.
        assert not is_no_kernel_refusal("CUDA out of memory. Tried to allocate 2.00 GiB")

    def test_both_wordings_are_declared(self) -> None:
        assert len(NO_KERNEL_MESSAGES) == 2


class TestForcingABackend:
    def test_a_backend_with_a_kernel_returns_a_result(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        out = forced_sdpa_output(query, key, value, SDPBackend.MATH)

        if out is None:
            raise AssertionError("forcing math on cpu must produce a result")
        assert tuple(out.shape) == (1, 2, 64, 64)

    def test_a_backend_without_one_returns_none_rather_than_raising(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        assert forced_sdpa_output(query, key, value, SDPBackend.EFFICIENT_ATTENTION) is None

    def test_a_failure_that_is_not_a_refusal_propagates(self) -> None:
        # Real call, real error: mismatched dtypes are not a statement about
        # which kernels exist, so recording them as unavailability would be
        # recording something false.
        query, key, value = sdpa_operands(TINY, "cpu")

        with pytest.raises(RuntimeError, match="same dtype"):
            forced_sdpa_output(query, key.double(), value, SDPBackend.MATH)

    def test_the_unforced_call_matches_one_of_the_forced_ones(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")
        default = sdpa_output(query, key, value)
        matches = [
            name
            for name in BACKEND_KEYS
            for out in [forced_sdpa_output(query, key, value, BACKENDS[name])]
            if out is not None and torch.equal(out, default)
        ]

        assert matches != []


class TestDigestingACall:
    def test_the_unforced_digest_is_a_number_not_an_optional(self) -> None:
        # The dispatcher always has the math fallback, so this call has no
        # refusal to report -- and typing it as optional would create an arm
        # no input could reach.
        query, key, value = sdpa_operands(TINY, "cpu")

        assert default_digest(query, key, value, "attention", "cpu") > 0.0

    def test_a_forced_digest_is_absent_when_the_backend_has_no_kernel(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        assert (
            forced_digest(query, key, value, SDPBackend.EFFICIENT_ATTENTION, "attention", "cpu")
            is None
        )

    def test_a_forced_digest_is_present_when_it_does(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        digest = forced_digest(query, key, value, SDPBackend.MATH, "attention", "cpu")

        if digest is None:
            raise AssertionError("forcing math on cpu must produce a digest")
        assert digest > 0.0

    def test_two_backends_that_ran_do_not_share_a_digest_here(self) -> None:
        # If they did, the selection method could not separate them, and the
        # report would have to say so rather than pick.
        query, key, value = sdpa_operands(TINY, "cpu")
        math = forced_digest(query, key, value, SDPBackend.MATH, "attention", "cpu")
        flash = forced_digest(query, key, value, SDPBackend.FLASH_ATTENTION, "attention", "cpu")

        assert math != flash


class TestEligibility:
    def test_it_answers_for_every_declared_key(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        assert tuple(sorted(sdpa_eligibility(query, key, value))) == tuple(sorted(ELIGIBLE_KEYS))

    def test_no_cuda_backend_is_eligible_for_cpu_tensors(self) -> None:
        query, key, value = sdpa_operands(TINY, "cpu")

        assert sdpa_eligibility(query, key, value) == {
            "flash": False,
            "efficient": False,
            "cudnn": False,
        }

    def test_cpu_operands_never_reach_the_cuda_eligibility_apis(self) -> None:
        """torch 2.7's can_use_cudnn_attention initialises CUDA even for CPU
        operands, so on a driverless host the consultation itself is the
        crash. Measured 2026-09-04: image build 55747880 (torch 2.7.1+cu128)
        died with "CUDA driver version is insufficient" inside this exact
        call, on a CPU probe that torch 2.6 answered quietly."""
        consulted: list[str] = []

        def raising_eligibility(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> dict[str, bool]:
            consulted.append("consulted")
            raise AssertionError("CUDA eligibility consulted for CPU operands")

        original = _test_hooks.sdpa_cuda_eligibility
        _test_hooks.sdpa_cuda_eligibility = raising_eligibility
        try:
            query, key, value = sdpa_operands(TINY, "cpu")
            answered = sdpa_eligibility(query, key, value)
        finally:
            _test_hooks.sdpa_cuda_eligibility = original

        assert consulted == []
        assert answered == dict.fromkeys(ELIGIBLE_KEYS, False)

    def test_cuda_operands_are_answered_by_the_consultation(self) -> None:
        """The gate's other arm, on the GPU this suite runs beside: CUDA
        operands reach the consultation hook, and its answer passes through
        unchanged."""
        sentinel = {"flash": True, "efficient": False, "cudnn": True}
        seen: list[tuple[bool, bool, bool]] = []

        def recording_eligibility(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> dict[str, bool]:
            seen.append((query.is_cuda, key.is_cuda, value.is_cuda))
            return dict(sentinel)

        original = _test_hooks.sdpa_cuda_eligibility
        _test_hooks.sdpa_cuda_eligibility = recording_eligibility
        try:
            query, key, value = sdpa_operands(TINY, "cuda")
            answered = sdpa_eligibility(query, key, value)
        finally:
            _test_hooks.sdpa_cuda_eligibility = original

        assert seen == [(True, True, True)]
        assert answered == sentinel

    def test_the_production_consultation_answers_false_for_cpu_operands(self) -> None:
        """The gate does not change the answer, only who computes it: asked
        directly (as CUDA operands would ask it), torch itself rules every
        fused backend out for CPU tensors on a host whose driver works."""
        query, key, value = sdpa_operands(TINY, "cpu")

        assert _default_sdpa_cuda_eligibility(query, key, value) == dict.fromkeys(
            ELIGIBLE_KEYS, False
        )


class TestOneWholeMeasurement:
    def test_it_records_availability_for_every_backend(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        assert tuple(sorted(measured["available"])) == tuple(sorted(BACKEND_KEYS))

    def test_it_records_a_digest_only_for_the_backends_that_ran(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        assert set(measured["digests"]) == {
            name for name, ok in measured["available"].items() if ok
        }

    def test_on_cpu_the_fused_cuda_backends_are_unavailable(self) -> None:
        measured = probe_sdpa(TINY, "cpu")

        assert measured["available"]["efficient"] is False
        assert measured["available"]["cudnn"] is False

    def test_it_reproduces_itself_exactly(self) -> None:
        assert probe_sdpa(TINY, "cpu") == probe_sdpa(TINY, "cpu")

    def test_the_default_digest_is_one_of_the_forced_digests(self) -> None:
        # The whole method: the backend whose forced output is bit-identical
        # to the unforced one is the backend the dispatcher chose.
        measured = probe_sdpa(TINY, "cpu")

        assert measured["default_digest"] in set(measured["digests"].values())

    def test_two_different_shapes_do_not_produce_one_digest(self) -> None:
        assert (
            probe_sdpa(TINY, "cpu")["default_digest"]
            != probe_sdpa(sdpa_shape_for("tiny-len128"), "cpu")["default_digest"]
        )


class TestTheBackendMapping:
    def test_it_covers_exactly_the_declared_keys(self) -> None:
        # A key declared with no backend -- or the reverse -- would silently
        # drop a column from every record written.
        assert tuple(sorted(BACKENDS)) == tuple(sorted(BACKEND_KEYS))

    def test_every_entry_is_a_distinct_backend(self) -> None:
        assert len(set(BACKENDS.values())) == len(BACKENDS)

    def test_it_maps_only_the_four_kernel_backends(self) -> None:
        # SDPBackend also carries ERROR and OVERRIDEABLE, which are not
        # kernels a call can land on; asserting the mapped set whole is what
        # keeps either of them from being added.
        assert set(BACKENDS.values()) == {
            SDPBackend.MATH,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
        }


class TestLabels:
    def test_a_label_carries_the_dimensions_and_not_only_the_rung(self) -> None:
        assert (
            sdpa_label(TINY, "efficient", "digest48") == "sdpa-tiny-h2-d64-s64|efficient|digest48"
        )

    def test_two_shapes_of_one_rung_name_cannot_collide(self) -> None:
        reshaped = SdpaShape(rung="tiny", heads=2, head_dim=64, sequence_len=128)

        assert sdpa_label(TINY, "math", "digest48") != sdpa_label(reshaped, "math", "digest48")

    def test_the_experiment_is_its_own(self) -> None:
        assert SDPA_EXPERIMENT == "sdpa-backend-selection"
