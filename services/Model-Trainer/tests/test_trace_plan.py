"""The trace's naming scheme, and the rungs it declares.

Pure functions over strings, so everything here is exercised directly. The one
test that reaches outside checks :data:`TRACE_RUNGS` against the ladder's own
shape table -- a rung named here that the ladder does not declare would fail
at run time on a compute node, an hour into a job, and this is the cheapest
place to find out instead.
"""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trace_plan import (
    DIGEST_SUFFIX,
    STEP_DIGITS,
    SUM_SUFFIX,
    TRACE_EXPERIMENT,
    TRACE_RUNGS,
    TraceName,
    parse_trace_name,
    trace_label,
    trace_loss_name,
    trace_tensor_name,
)

SAMPLE = TraceName(
    rung="tiny",
    step=42,
    kind="out",
    index=0,
    module_class="Conv1D",
    path="transformer.h.0.attn.c_attn",
    suffix=DIGEST_SUFFIX,
)


class TestNamingATracedTensor:
    def test_it_renders_every_field_in_order(self) -> None:
        assert (
            trace_tensor_name(SAMPLE)
            == "tiny|00042|out|0|Conv1D|transformer.h.0.attn.c_attn|digest48"
        )

    def test_the_step_is_zero_padded_so_names_sort_in_execution_order(self) -> None:
        early = trace_tensor_name({**SAMPLE, "step": 9})
        late = trace_tensor_name({**SAMPLE, "step": 100})

        assert sorted((late, early)) == [early, late]
        assert f"{0:0{STEP_DIGITS}d}" == "00000"

    def test_two_tensors_of_one_step_differ_by_index(self) -> None:
        assert trace_tensor_name(SAMPLE) != trace_tensor_name({**SAMPLE, "index": 1})

    def test_the_two_measurements_of_one_tensor_differ_by_suffix(self) -> None:
        assert trace_tensor_name({**SAMPLE, "suffix": SUM_SUFFIX}).endswith("|sum")
        assert trace_tensor_name(SAMPLE).endswith("|digest48")

    def test_the_module_class_is_part_of_the_name(self) -> None:
        # Two cards that ran different attention classes must not pair their
        # observations. The class being in the name is what prevents it.
        assert trace_tensor_name(SAMPLE) != trace_tensor_name(
            {**SAMPLE, "module_class": "GPT2SdpaAttention"}
        )

    def test_a_name_reads_back_into_the_fields_it_was_built_from(self) -> None:
        assert parse_trace_name(trace_tensor_name(SAMPLE)) == SAMPLE


class TestNamingALoss:
    def test_it_is_the_rung_and_the_word_loss(self) -> None:
        assert trace_loss_name("xl") == "xl|loss"

    def test_a_loss_name_does_not_parse_as_a_tensor(self) -> None:
        assert parse_trace_name(trace_loss_name("xl")) is None


class TestParsingRefusals:
    def test_a_name_with_the_wrong_field_count_is_not_a_tensor(self) -> None:
        assert parse_trace_name("tiny|00042|out|0|Conv1D|path") is None

    def test_a_non_numeric_step_is_not_a_tensor(self) -> None:
        assert parse_trace_name("tiny|early|out|0|Conv1D|path|digest48") is None

    def test_a_non_numeric_index_is_not_a_tensor(self) -> None:
        assert parse_trace_name("tiny|00042|out|first|Conv1D|path|digest48") is None

    def test_an_unrelated_observation_name_is_not_a_tensor(self) -> None:
        assert parse_trace_name("cloze_accuracy") is None


class TestLabellingATrace:
    def test_it_counts_the_rungs_and_digests_them(self) -> None:
        label = trace_label(("tiny", "xl"))

        assert label.startswith("forward-trace-2x")
        assert len(label) == len("forward-trace-2x") + 12

    def test_the_same_rungs_always_produce_the_same_label(self) -> None:
        assert trace_label(("tiny", "xl")) == trace_label(("tiny", "xl"))

    def test_reordering_the_rungs_relabels_the_trace(self) -> None:
        assert trace_label(("tiny", "xl")) != trace_label(("xl", "tiny"))

    def test_adding_a_rung_relabels_the_trace(self) -> None:
        assert trace_label(("tiny",)) != trace_label(("tiny", "xl"))

    def test_a_repeated_rung_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match=r"cannot walk one rung twice: \['tiny'\]"):
            trace_label(("tiny", "xl", "tiny"))


class TestTheDeclaredRungs:
    def test_every_declared_rung_is_one_the_ladder_defines(self) -> None:
        assert [rung for rung in TRACE_RUNGS if rung not in PROBE_SHAPES] == []

    def test_it_declares_the_four_the_contrast_needs(self) -> None:
        # tiny is broken by removing split-K, xl is not fixed by it, large IS
        # fixed by it, and medium never moves. Dropping any one of the four
        # would leave a mechanism unfalsifiable, so the set is asserted whole.
        assert TRACE_RUNGS == ("tiny", "medium", "large", "xl")

    def test_no_rung_is_traced_twice(self) -> None:
        assert len(set(TRACE_RUNGS)) == len(TRACE_RUNGS)

    def test_it_declares_its_own_experiment(self) -> None:
        assert TRACE_EXPERIMENT == "forward-trace-attribution"
