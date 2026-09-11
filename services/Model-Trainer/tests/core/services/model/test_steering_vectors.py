"""The published intervention, on a real module graph.

WHY THESE RUN AGAINST A REAL MODEL rather than a stand-in. Every claim here is
about what a forward pass DOES -- that a direction read at a site can be added
back at that site, that removing the hook leaves nothing behind, that a module
returning a tuple cannot be perturbed. A fake module would answer each of
those by construction, which is to say it would answer none of them.

THE ONE REFUSAL THAT NEEDS NO FAKE AT ALL is the one most likely to be met: a
cartridge-wrapped model is not steerable, and passing one is the exact mistake
the narrowing exists to catch. It is a real object from this package, so the
test is the mistake rather than a picture of it.
"""

from __future__ import annotations

import math

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.services.model.cartridge_measurement import fresh_cartridge
from model_trainer.core.services.model.cartridge_scoring import TraitPair, base_loss
from model_trainer.core.services.model.editing.activations import capture_module_io
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.steering_vectors import (
    attach_steering,
    compose_directions,
    extract_steering_vector,
    require_steerable,
    steered_trait_losses,
    unit_direction,
)
from model_trainer.core.types import SteerableLMProto

#: A tensor-valued site in the tiny probe model. The projection that writes a
#: block's result back into the residual stream, which is where the edit
#: literature already keys its interventions.
_SITE = "transformer.h.0.mlp.c_proj"

#: A site whose output is a TUPLE -- the whole block -- used to prove the
#: refusal rather than to steer.
_TUPLE_SITE = "transformer.h.0"


def _model() -> SteerableLMProto:
    """Build the tiny probe model, narrowed to steerable.

    Returns:
        The model.
    """
    built, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_steerable(built)


def _pair(seed: int) -> TraitPair:
    """Build one pair whose members differ in every position.

    Args:
        seed: Seed for the draw, so pairs differ from each other.

    Returns:
        The pair, inside the probe model's vocabulary and sequence budget.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    expressing = torch.randint(0, 256, (1, 8), generator=generator, dtype=torch.long)
    neutral = torch.randint(256, 512, (1, 8), generator=generator, dtype=torch.long)
    return TraitPair(expressing=expressing, neutral=neutral)


def _pairs(count: int = 4) -> list[TraitPair]:
    """Build several pairs.

    Args:
        count: How many.

    Returns:
        The pairs.
    """
    return [_pair(seed) for seed in range(count)]


def _vector(values: tuple[float, ...]) -> torch.Tensor:
    """Build a one-dimensional tensor from chosen numbers.

    Allocated and filled rather than built from a list literal, for the reason
    the source modules give: ``torch.tensor([...])`` is typed as returning
    Any, and this package refuses an unchecked value even in a test.

    Args:
        values: The numbers, in order.

    Returns:
        The tensor.
    """
    built = torch.zeros(len(values))
    for index, value in enumerate(values):
        built[index] = value
    return built


def _norm(vector: torch.Tensor) -> float:
    """Measure a vector's length.

    Spelled as the source spells it rather than through ``Tensor.norm``, which
    this stub set types as Any.

    Args:
        vector: The vector.

    Returns:
        Its Euclidean length.
    """
    return math.sqrt(float((vector * vector).sum().item()))


class TestWhatMayBeSteered:
    """A steering arm needs both a module graph and a loss."""

    def test_a_real_model_narrows(self) -> None:
        """The ordinary case, and the same object comes back."""
        built, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        assert require_steerable(built) is built

    def test_a_cartridge_wrapped_model_is_refused(self) -> None:
        """A steering vector is an ALTERNATIVE to a prefix, not an addition.

        A cartridge model presents no module graph, so there is no site to
        read a direction at -- and steering through a prefix would measure the
        pair of them while reporting as the steering arm.
        """
        wrapped = fresh_cartridge(_model(), num_slots=2, seed=1)
        with pytest.raises(AppError) as excinfo:
            require_steerable(wrapped)
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


class TestReadingTheDirection:
    """The vector is the mean activation difference, and nothing else."""

    def test_one_pair_reproduces_the_capture_directly(self) -> None:
        """Computed here from the capture rather than through the extractor.

        The two agree only if the extractor reads the member its docstring
        names, at the position it names. Reading the neutral member first
        would flip the sign and every downstream number with it.
        """
        model = _model()
        pair = _pair(11)
        expected = (
            capture_module_io(
                model=model, module_name=_SITE, input_ids=pair["expressing"], position=-1
            )["module_output"]
            - capture_module_io(
                model=model, module_name=_SITE, input_ids=pair["neutral"], position=-1
            )["module_output"]
        )
        actual = extract_steering_vector(model, [pair], module_name=_SITE)
        assert torch.allclose(actual, expected)

    def test_the_direction_is_one_dimensional(self) -> None:
        """A steering vector is a direction, not a batch of them."""
        vector = extract_steering_vector(_model(), _pairs(), module_name=_SITE)
        assert vector.dim() == 1

    def test_extraction_from_no_pairs_is_refused(self) -> None:
        """A zero vector steers nothing while reporting as a steering arm."""
        with pytest.raises(AppError) as excinfo:
            extract_steering_vector(_model(), [], module_name=_SITE)
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE

    def test_an_absent_site_is_refused(self) -> None:
        """A plan naming a module this architecture lacks fails at the site."""
        with pytest.raises(AppError) as excinfo:
            extract_steering_vector(_model(), _pairs(1), module_name="transformer.h.99.mlp")
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


class TestNormalisingAndComposing:
    """Unit length is what makes the declared strength the only knob."""

    def test_a_direction_is_scaled_to_unit_norm(self) -> None:
        """Two traits' raw vectors are not comparable; their directions are."""
        vector = extract_steering_vector(_model(), _pairs(), module_name=_SITE)
        assert _norm(unit_direction(vector)) == pytest.approx(1.0)

    def test_a_zero_vector_has_no_direction(self) -> None:
        """The site's output did not move, so it cannot carry this trait.

        Named rather than smoothed with an epsilon: dividing by zero would put
        NaNs into the record under an arm's name.
        """
        with pytest.raises(AppError) as excinfo:
            unit_direction(torch.zeros(4))
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE

    def test_composing_one_direction_returns_it(self) -> None:
        """The solo steering arm must be the direction itself, re-normalised."""
        direction = unit_direction(_vector((3.0, 4.0)))
        assert torch.allclose(compose_directions([direction]), direction)

    def test_composing_is_the_sum_re_normalised(self) -> None:
        """Summing is the scheme the published composition measurements use.

        Re-normalised so a composed arm is applied at the same strength as a
        solo one; without that, n4 would steer at up to four times the
        magnitude and the arm would measure the strength.
        """
        composed = compose_directions(
            [unit_direction(_vector((1.0, 0.0))), unit_direction(_vector((0.0, 1.0)))]
        )
        assert _norm(composed) == pytest.approx(1.0)
        assert float(composed[0].item()) == pytest.approx(float(composed[1].item()))

    def test_composing_nothing_is_refused(self) -> None:
        """A composed arm over zero traits is the plain base under a new name."""
        with pytest.raises(AppError) as excinfo:
            compose_directions([])
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE

    def test_directions_that_cancel_are_named_not_smoothed(self) -> None:
        """Two traits whose directions oppose really do compose to nothing.

        That is a property of the traits, so it is reported as a refusal
        rather than replaced with a small nudge that would let a meaningless
        arm into the record.
        """
        direction = unit_direction(_vector((1.0, 0.0)))
        with pytest.raises(AppError) as excinfo:
            compose_directions([direction, -direction])
        assert excinfo.value.code is ModelTrainerErrorCode.TRAIT_CORPUS_UNUSABLE


class TestApplyingTheDirection:
    """Attaching changes the answer; removing leaves nothing behind."""

    def test_steering_changes_the_loss_and_removal_restores_it_exactly(self) -> None:
        """The residue question, and it is asked with exact equality.

        A steered model that outlived its own measurement would steer every
        later arm and each would look like a result, so "close enough" is the
        wrong bar: the loss after removal must be the number from before.
        """
        model = _model()
        item = _pair(3)["expressing"]
        before = base_loss(model, item)
        direction = unit_direction(extract_steering_vector(model, _pairs(), module_name=_SITE))

        handle = attach_steering(model, direction, module_name=_SITE, strength=10.0)
        during = base_loss(model, item)
        handle.remove()
        after = base_loss(model, item)

        assert during != before
        assert after == before

    def test_a_tuple_valued_site_is_refused_when_it_runs(self) -> None:
        """A block returns its hidden state and its cache together.

        Reaching in and perturbing element zero would be easy and wrong: the
        same site has to be READABLE by the capture that extracts the
        direction, and that refuses anything but a tensor.
        """
        model = _model()
        handle = attach_steering(model, torch.zeros(4), module_name=_TUPLE_SITE, strength=1.0)
        with pytest.raises(AppError) as excinfo:
            base_loss(model, _pair(4)["expressing"])
        handle.remove()
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_ACTIVATION_NOT_CAPTURED

    def test_attaching_to_an_absent_site_is_refused(self) -> None:
        """The refusal is at attach time, not at the next forward pass."""
        with pytest.raises(AppError) as excinfo:
            attach_steering(_model(), torch.zeros(4), module_name="nowhere", strength=1.0)
        assert excinfo.value.code is ModelTrainerErrorCode.EDIT_MODULE_NOT_FOUND


class TestScoringASteeredArm:
    """The control pass and the steered pass, on one object."""

    def test_the_control_losses_are_the_plain_models_own(self) -> None:
        """Computed here from ``base_loss`` directly, with no hook attached.

        If the control pass ran with the hook on, both arms would be steered
        and the difference of differences would be exactly zero -- a clean
        null that measures nothing.
        """
        model = _model()
        pairs = _pairs(3)
        direction = unit_direction(extract_steering_vector(model, pairs, module_name=_SITE))
        losses = steered_trait_losses(model, pairs, direction, module_name=_SITE, strength=10.0)
        assert [item["base_expressing"] for item in losses] == [
            base_loss(model, pair["expressing"]) for pair in pairs
        ]
        assert [item["base_neutral"] for item in losses] == [
            base_loss(model, pair["neutral"]) for pair in pairs
        ]

    def test_the_steered_losses_differ_from_the_control(self) -> None:
        """An arm identical to its control is an arm that did not run."""
        model = _model()
        pairs = _pairs(3)
        direction = unit_direction(extract_steering_vector(model, pairs, module_name=_SITE))
        losses = steered_trait_losses(model, pairs, direction, module_name=_SITE, strength=10.0)
        assert all(item["arm_expressing"] != item["base_expressing"] for item in losses)
        assert all(item["arm_neutral"] != item["base_neutral"] for item in losses)

    def test_the_hook_is_gone_when_it_returns(self) -> None:
        """Measured by scoring again: the model must answer as it did before."""
        model = _model()
        pairs = _pairs(2)
        item = pairs[0]["expressing"]
        before = base_loss(model, item)
        direction = unit_direction(extract_steering_vector(model, pairs, module_name=_SITE))
        steered_trait_losses(model, pairs, direction, module_name=_SITE, strength=10.0)
        assert base_loss(model, item) == before

    def test_the_indices_run_in_pair_order(self) -> None:
        """The order is the pairing; an index has to name the same pair."""
        model = _model()
        pairs = _pairs(4)
        direction = unit_direction(extract_steering_vector(model, pairs, module_name=_SITE))
        losses = steered_trait_losses(model, pairs, direction, module_name=_SITE, strength=10.0)
        assert [item["index"] for item in losses] == [0, 1, 2, 3]
