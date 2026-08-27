"""The forward trace, run for real on the CPU at the smallest rung.

Nothing is faked: these build the production model through the production
constructor and run the production hooks over it. What is deliberately absent
is any cross-card claim -- a CPU cannot produce two cards, and the whole
question the trace exists for is what different cards do. What CAN be
established here is that the instrument is sound: that it observes every
boundary it claims to, that the boundaries it reports are the ones the model
actually has, that it does not perturb the number it is explaining, and that
it leaves nothing behind.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.services.model.forward_trace import (
    ForwardTrace,
    install_hooks,
    tensors_in,
    traced_forward,
)
from model_trainer.core.services.model.known_answer_probe import (
    probe_forward_loss,
    probe_model_and_input,
)
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.trace_plan import INPUT_KIND, OUTPUT_KIND

TINY = PROBE_SHAPES["tiny"]


def one(value: float) -> torch.Tensor:
    """Build a one-element tensor.

    A helper rather than ``torch.tensor([1.0])`` at each call site, which
    types its list literal as ``list[Any]`` and is forbidden here.
    """
    values: list[float] = [value]
    return torch.tensor(values, dtype=torch.float32)


class TestFindingTensorsInAHookArgument:
    def test_a_tensor_is_itself(self) -> None:
        tensor = one(1.0)

        assert tensors_in(tensor) == (tensor,)

    def test_a_tuple_yields_its_tensors_in_order(self) -> None:
        first = one(1.0)
        second = one(2.0)

        assert tensors_in((first, second)) == (first, second)

    def test_a_tuple_drops_its_non_tensors(self) -> None:
        # GPT2Block returns (hidden_states, present) and present is None when
        # the cache is off, which is the case the probe runs in.
        tensor = one(1.0)

        assert tensors_in((tensor, None)) == (tensor,)

    def test_a_mapping_result_yields_nothing(self) -> None:
        # GPT2Model returns a ModelOutput, which is a mapping. Every tensor
        # inside it is also some module's output and is recorded there.
        assert tensors_in({"logits": one(1.0)}) == ()

    def test_a_bare_value_yields_nothing(self) -> None:
        assert tensors_in(7) == ()


class TestTheTraceCollector:
    def test_it_numbers_calls_not_tensors(self) -> None:
        module, _ = probe_model_and_input("cpu", TINY)
        trace = ForwardTrace()

        trace.record(module, "a", OUTPUT_KIND, (one(1.0), one(2.0)))
        trace.record(module, "b", OUTPUT_KIND, (one(3.0),))

        assert [t["step"] for t in trace.tensors] == [0, 0, 1]
        assert [t["index"] for t in trace.tensors] == [0, 1, 0]

    def test_a_call_carrying_no_tensor_still_consumes_a_step(self) -> None:
        # Otherwise a module whose output shape changed would renumber every
        # step after it, and two records could not be compared step by step.
        module, _ = probe_model_and_input("cpu", TINY)
        trace = ForwardTrace()

        trace.record(module, "empty", OUTPUT_KIND, ())
        trace.record(module, "full", OUTPUT_KIND, (one(1.0),))

        assert [t["step"] for t in trace.tensors] == [1]

    def test_it_records_the_module_class_not_the_path_twice(self) -> None:
        module, _ = probe_model_and_input("cpu", TINY)
        trace = ForwardTrace()

        trace.record(module, "some.path", INPUT_KIND, (one(1.0),))

        assert trace.tensors[0]["module_class"] == "GPT2LMHeadModel"
        assert trace.tensors[0]["path"] == "some.path"
        assert trace.tensors[0]["kind"] == INPUT_KIND


class TestWhatOneTinyForwardTraces:
    def test_the_traced_loss_is_the_untraced_loss_to_the_last_bit(self) -> None:
        # The control. If the hooks perturbed the arithmetic, nothing else in
        # a trace record could be read against the ladder that motivated it.
        _, loss = traced_forward("cpu", TINY)

        assert loss == probe_forward_loss("cpu", TINY)

    def test_the_steps_never_go_backwards(self) -> None:
        traced, _ = traced_forward("cpu", TINY)
        steps = [t["step"] for t in traced]

        assert steps == sorted(steps)

    def test_it_starts_at_the_token_embedding_input(self) -> None:
        traced, _ = traced_forward("cpu", TINY)

        assert (traced[0]["path"], traced[0]["kind"]) == ("transformer.wte", INPUT_KIND)

    def test_it_ends_at_the_output_projection(self) -> None:
        traced, _ = traced_forward("cpu", TINY)

        assert (traced[-1]["path"], traced[-1]["kind"]) == ("lm_head", OUTPUT_KIND)

    def test_the_output_projection_is_a_bias_free_linear(self) -> None:
        # Which is why disabling cuBLASLt's split-K cannot reach it: a Linear
        # with no bias takes the legacy cublasSgemm path, not cuBLASLt.
        traced, _ = traced_forward("cpu", TINY)
        head = [t for t in traced if t["path"] == "lm_head"]

        assert [t["module_class"] for t in head] == ["Linear", "Linear"]

    def test_it_captures_the_attention_output_as_the_projection_input(self) -> None:
        # scaled_dot_product_attention is a function call inside the attention
        # module, not a submodule, so its result is observable ONLY as the
        # input of the projection that consumes it. This is the boundary the
        # whole leaf-input rule exists for.
        traced, _ = traced_forward("cpu", TINY)
        matching = [
            t
            for t in traced
            if t["path"] == "transformer.h.0.attn.c_proj" and t["kind"] == INPUT_KIND
        ]

        assert [t["module_class"] for t in matching] == ["Conv1D"]

    def test_the_attention_module_is_the_sdpa_implementation(self) -> None:
        # Recorded rather than assumed: transformers 4.46 chooses the
        # attention class at construction, and which one ran is part of what
        # a difference between two cards would have to be explained by.
        traced, _ = traced_forward("cpu", TINY)
        classes = {t["module_class"] for t in traced if t["path"] == "transformer.h.0.attn"}

        assert classes == {"GPT2SdpaAttention"}

    def test_both_layers_of_the_tiny_model_are_traced(self) -> None:
        traced, _ = traced_forward("cpu", TINY)
        layers = {t["path"].split(".")[2] for t in traced if t["path"].startswith("transformer.h.")}

        assert layers == {"0", "1"}

    def test_every_leaf_that_ran_contributed_an_input_and_an_output(self) -> None:
        traced, _ = traced_forward("cpu", TINY)
        kinds: dict[str, set[str]] = {}
        for tensor in traced:
            kinds.setdefault(tensor["path"], set()).add(tensor["kind"])
        leaves = {path for path, seen in kinds.items() if INPUT_KIND in seen}

        assert leaves == {path for path in leaves if kinds[path] == {INPUT_KIND, OUTPUT_KIND}}

    def test_a_container_module_contributes_an_output_and_no_input(self) -> None:
        traced, _ = traced_forward("cpu", TINY)
        block = {t["kind"] for t in traced if t["path"] == "transformer.h.0"}

        assert block == {OUTPUT_KIND}

    def test_the_root_is_never_traced(self) -> None:
        # Its hook could not fire: the probe calls forward directly, not
        # __call__, and forward hooks fire from __call__.
        traced, _ = traced_forward("cpu", TINY)

        assert [t for t in traced if t["path"] == ""] == []

    def test_the_trace_reproduces_itself_exactly(self) -> None:
        first, first_loss = traced_forward("cpu", TINY)
        second, second_loss = traced_forward("cpu", TINY)

        assert first == second
        assert first_loss == second_loss


class TestTheHooksAreRemoved:
    def test_a_second_trace_records_the_same_count_as_the_first(self) -> None:
        # Handles left installed would double the second run's records, and
        # would go on instrumenting whatever the process ran next.
        first, _ = traced_forward("cpu", TINY)
        second, _ = traced_forward("cpu", TINY)

        assert len(first) == len(second)

    def test_removing_the_handles_stops_the_recording(self) -> None:
        model, ids = probe_model_and_input("cpu", TINY)
        trace = ForwardTrace()
        handles = install_hooks(model, trace)
        for handle in handles:
            handle.remove()

        with torch.no_grad():
            outputs = model.forward(input_ids=ids, labels=ids)

        # The forward still computes what it always computed -- removing the
        # handles takes away the observation, not the arithmetic.
        assert float(outputs.loss.item()) == probe_forward_loss("cpu", TINY)
        assert trace.tensors == []

    def test_it_installs_a_handle_for_every_module_and_every_leaf(self) -> None:
        model, _ = probe_model_and_input("cpu", TINY)
        trace = ForwardTrace()
        modules = [name for name, _ in model.named_modules() if name != ""]
        leaves = [
            name
            for name, module in model.named_modules()
            if name != "" and next(module.children(), None) is None
        ]

        handles = install_hooks(model, trace)
        for handle in handles:
            handle.remove()

        assert len(handles) == len(modules) + len(leaves)


class TestRefusals:
    def test_a_shape_longer_than_its_vocabulary_is_refused_before_anything_runs(self) -> None:
        with pytest.raises(ValueError, match="exceeds vocab_size"):
            traced_forward("cpu", {"model_size": "tiny", "sequence_len": 8, "vocab_size": 4})
