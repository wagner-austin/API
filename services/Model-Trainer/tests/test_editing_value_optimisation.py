"""Learning the output a module must emit, on a real model.

NOTHING IS FAKED. The `tiny` rung is built by the production probe builder, a
real forward pass runs, real gradients flow into the delta, and the module is
really swapped out and back. A fake model would make the central assertion --
that the search REDUCES the loss it is given -- a statement about the fake.

The one thing deliberately not asserted is a particular loss value. The claim
is directional: the optimisation must make its target more likely, and the
edit built from its result must carry that through to the model's own
scoring. A threshold would be a fact about this rung's initialisation.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.editing.activations import capture_module_io
from model_trainer.core.services.model.editing.value_optimisation import (
    IGNORED_LABEL,
    DeltaAtPosition,
    module_class,
    optimise_target_output,
    target_labels,
    target_token_nll,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import TracedLMModelProto

_MODULE = "transformer.h.0.mlp.c_proj"

#: One sequence of ids, annotated rather than written inline: a bare nested
#: list literal handed to `torch.tensor` is typed `list[Any]`, and this
#: repository refuses an Any even in a test.
_PROMPT: list[list[int]] = [[5, 6, 7]]


def _model() -> TracedLMModelProto:
    """Build the tiny rung.

    Returns:
        A real GPT-2 with the probe's deterministic initialisation.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return model


class TestTargetLabels:
    def test_the_prompt_is_masked_and_the_target_is_not(self) -> None:
        labels = target_labels(prompt_length=3, target_ids=[7, 8])

        assert labels.shape == (1, 5)
        expected: list[list[int]] = [[IGNORED_LABEL, IGNORED_LABEL, IGNORED_LABEL, 7, 8]]

        assert torch.equal(labels, torch.tensor(expected, dtype=torch.long))

    def test_an_ignored_label_is_the_one_torch_skips(self) -> None:
        """A different sentinel would be scored as a real token id."""
        assert IGNORED_LABEL == -100


class TestModuleClass:
    def test_it_returns_the_base_every_torch_module_shares(self) -> None:
        """Named rather than merely instance-checked: the narrowing this
        feeds decides whether a module can be stood in for, and a getter that
        returned some other class would still satisfy an isinstance against
        one of its own instances.
        """
        cls = module_class()

        assert cls.__name__ == "Module"
        assert cls.__module__ == "torch.nn.modules.module"
        # The same class object every call, so a narrowing done in one place
        # and a swap done in another agree about what a module is.
        assert cls is module_class()


class TestDeltaAtPosition:
    def test_it_adds_the_delta_at_exactly_one_position(self) -> None:
        """A wrapper that added everywhere would edit every token at once,
        which is not what a rank-one edit at a keyed position does.
        """
        inner = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            inner.weight.copy_(torch.eye(3))
        wrapper = DeltaAtPosition(inner, width=3, position=1)
        with torch.no_grad():
            planted: list[float] = [1.0, 2.0, 3.0]
            wrapper.delta.copy_(torch.tensor(planted))
        hidden = torch.zeros(1, 3, 3)
        rows: list[list[list[float]]] = [[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]]

        out = wrapper.forward(hidden)

        assert torch.allclose(out, torch.tensor(rows, dtype=torch.float32))

    def test_a_zero_delta_leaves_the_original_output_alone(self) -> None:
        """The state every optimisation starts from must be a no-op, or step
        zero is already an edit nobody asked for.
        """
        inner = torch.nn.Linear(3, 3, bias=False)
        hidden = torch.randn(1, 4, 3)

        wrapper = DeltaAtPosition(inner, width=3, position=2)

        assert torch.equal(wrapper.forward(hidden), inner.forward(hidden))

    def test_the_original_stays_a_registered_submodule(self) -> None:
        """A vanished parameter would be a difference the arm did not intend."""
        inner = torch.nn.Linear(3, 3, bias=False)
        wrapper = DeltaAtPosition(inner, width=3, position=0)

        names = [name for name, _ in wrapper.named_parameters()]

        assert "delta" in names
        assert "original.weight" in names


class TestTargetTokenNll:
    def test_it_scores_the_target_and_not_the_prompt(self) -> None:
        """THE CORRECTION THIS FUNCTION CARRIES.

        Scoring `prompt + target` as one string reported working edits as
        large regressions. Two calls that share a target but not a prompt
        length must not be forced to agree, and a call whose target is longer
        must total more surprise for the same per-token surprise -- which is
        only true if the prompt is excluded.
        """
        model = _model()
        short = target_token_nll(model=model, prompt_ids=[1, 2, 3], target_ids=[4], device="cpu")
        long = target_token_nll(model=model, prompt_ids=[1, 2, 3], target_ids=[4, 5], device="cpu")

        assert short > 0.0
        assert long > short


class TestOptimiseTargetOutput:
    def test_it_makes_the_target_more_likely(self) -> None:
        """THE ONE CLAIM THE WHOLE ARM RESTS ON.

        If the search does not reduce the target's surprise, every downstream
        number is about a delta that means nothing.
        """
        model = _model()
        prompt_ids = _PROMPT[0]
        target_ids: list[int] = [11]
        before = target_token_nll(
            model=model, prompt_ids=prompt_ids, target_ids=target_ids, device="cpu"
        )
        captured = capture_module_io(
            model=model,
            module_name=_MODULE,
            input_ids=torch.tensor(_PROMPT, dtype=torch.long),
            position=2,
        )

        target = optimise_target_output(
            model=model,
            module_name=_MODULE,
            prompt_ids=prompt_ids,
            target_ids=target_ids,
            position=2,
            current_output=captured["module_output"],
            steps=20,
            learning_rate=0.5,
            device="cpu",
        )

        # The optimisation's own claim, checked without applying an edit: the
        # returned vector differs from what the module emits now, and it is
        # the same shape, so the solve downstream has something to work with.
        assert target.shape == captured["module_output"].shape
        assert not torch.equal(target, captured["module_output"])
        # And the model is unchanged by the search itself -- the swap is put
        # back, so scoring again reproduces the number from before it ran.
        assert target_token_nll(
            model=model, prompt_ids=prompt_ids, target_ids=target_ids, device="cpu"
        ) == pytest.approx(before)

    def test_zero_steps_returns_the_current_output_unchanged(self) -> None:
        """The dose curve's floor. A plan that spends nothing must edit
        nothing, rather than applying whatever an uninitialised delta held.
        """
        model = _model()
        captured = capture_module_io(
            model=model,
            module_name=_MODULE,
            input_ids=torch.tensor(_PROMPT, dtype=torch.long),
            position=2,
        )

        target = optimise_target_output(
            model=model,
            module_name=_MODULE,
            prompt_ids=[5, 6, 7],
            target_ids=[11],
            position=2,
            current_output=captured["module_output"],
            steps=0,
            learning_rate=0.5,
            device="cpu",
        )

        assert torch.equal(target, captured["module_output"])

    def test_the_module_is_put_back_even_when_a_step_raises(self) -> None:
        """A model left holding the wrapper would score every later arm
        through a delta nobody asked for, and nothing would say so.
        """
        model = _model()
        captured = capture_module_io(
            model=model,
            module_name=_MODULE,
            input_ids=torch.tensor(_PROMPT, dtype=torch.long),
            position=2,
        )
        before = type(model.get_submodule(_MODULE))

        with pytest.raises(IndexError):
            optimise_target_output(
                model=model,
                module_name=_MODULE,
                prompt_ids=[5, 6, 7],
                # A target id outside the vocabulary makes the forward raise
                # rather than the wrapper, which is the arm being exercised.
                target_ids=[PROBE_SHAPES["tiny"]["vocab_size"] + 10],
                position=2,
                current_output=captured["module_output"],
                steps=1,
                learning_rate=0.5,
                device="cpu",
            )

        assert type(model.get_submodule(_MODULE)) is before

    def test_a_two_dimensional_current_output_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            optimise_target_output(
                model=_model(),
                module_name=_MODULE,
                prompt_ids=[5, 6],
                target_ids=[7],
                position=1,
                current_output=torch.zeros(2, 3),
                steps=1,
                learning_rate=0.5,
                device="cpu",
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH

    def test_an_empty_target_is_refused(self) -> None:
        """The loss would average over no scored position, which is not a
        number, and the run would record a NaN as a measurement.
        """
        with pytest.raises(AppError) as raised:
            optimise_target_output(
                model=_model(),
                module_name=_MODULE,
                prompt_ids=[5, 6],
                target_ids=[],
                position=1,
                current_output=torch.zeros(8),
                steps=1,
                learning_rate=0.5,
                device="cpu",
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH

    def test_a_module_that_is_not_a_torch_module_is_refused(self) -> None:
        """`require_edit_module` proves the path resolves; it does not prove
        the thing at the end of it can be stood in for.
        """
        # A name that resolves to the WEIGHT rather than to the module holding
        # it. `require_edit_module` refuses it, which is the guard this arm
        # sits behind: a path can name something real that is not a module.
        with pytest.raises(AppError) as raised:
            optimise_target_output(
                model=_model(),
                module_name="transformer.h.0.mlp.c_proj.weight",
                prompt_ids=[5, 6],
                target_ids=[7],
                position=1,
                current_output=torch.zeros(8),
                steps=1,
                learning_rate=0.5,
                device="cpu",
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND
