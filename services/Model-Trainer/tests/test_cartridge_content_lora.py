"""Crowd-invariance distillation: the objective, the freeze, the draws.

Real PEFT over the real tiny GPT-2 for the training-loop contracts -- the
adapter seam and the teacher/student split are precisely what this lever
depends on. The loss itself is pure and tested directly: zero exactly at
identical distributions, positive off them, gradients through the student
side only. The loop contracts mirror the LM-objective suite's: gradients
land in the LoRA and nowhere else (base, pool AND the separate teacher
byte-identical after training), training is a pure function of its seed,
a different seed trains a different adapter, and the refusals refuse.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.services.finetuning.strategies import _test_hooks as strategy_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import (
    measure_geometry,
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    initialise_slots,
)
from model_trainer.core.services.model.cartridge_content_lora import (
    _require_logits,
    invariance_loss,
    train_composition_lora_invariant,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import CacheCapableLMProto

_VOCAB = PROBE_SHAPES["tiny"]["vocab_size"]


def _adapted_base() -> CacheCapableLMProto:
    """Wrap a fresh tiny GPT-2 with a real rank-2 LoRA on its attention.

    Returns:
        The PEFT-wrapped base, cache-capable.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(
        strategy_hooks.Hooks.create_peft_model(
            model,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=("c_attn",),
            bias="none",
        )
    )


def _teacher_base() -> CacheCapableLMProto:
    """Load a fresh, un-adapted tiny GPT-2 for the teacher side.

    Returns:
        The plain base, cache-capable.
    """
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(model)


def _pool(model: CacheCapableLMProto, *seeds: int) -> tuple[CartridgeSlots, ...]:
    """Draw one untrained pool member per seed, cut for the model.

    Args:
        model: The base whose geometry the members take.
        seeds: One draw seed per member.

    Returns:
        The pool.
    """
    geometry = measure_geometry(model, num_slots=2)
    return tuple(initialise_slots(geometry, seed=seed) for seed in seeds)


def _corpus(seed: int, rows: int) -> list[torch.Tensor]:
    """Draw a small deterministic corpus of 8-token windows.

    Args:
        seed: Seed for the draw.
        rows: How many windows.

    Returns:
        One (1, 8) id tensor per window.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [
        torch.randint(0, _VOCAB, (1, 8), generator=generator, dtype=torch.long) for _ in range(rows)
    ]


def _named_bytes(model: CacheCapableLMProto) -> dict[str, torch.Tensor]:
    """Clone every named parameter for later exact comparison.

    Args:
        model: The model to snapshot.

    Returns:
        Detached clones, keyed by parameter name.
    """
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


class TestInvarianceLoss:
    def test_identical_distributions_cost_exactly_zero(self) -> None:
        """The objective's fixed point: a student that predicts behind the
        crowd what the teacher predicts alone has nothing left to learn."""
        logits = torch.randn(1, 5, _VOCAB, generator=torch.Generator().manual_seed(3))
        assert float(invariance_loss(logits, logits.clone()).item()) == 0.0

    def test_different_distributions_cost_something(self) -> None:
        generator = torch.Generator().manual_seed(4)
        student = torch.randn(1, 5, _VOCAB, generator=generator)
        teacher = torch.randn(1, 5, _VOCAB, generator=generator)
        assert float(invariance_loss(student, teacher).item()) > 0.0

    def test_the_loss_is_differentiable_through_the_student_side(self) -> None:
        """Distillation must pull the student toward the teacher: a nonzero
        gradient must reach the student's logits. (The teacher side is
        protected structurally -- the trainer detaches it -- and the loop
        suite asserts the teacher's bytes never move.)"""
        generator = torch.Generator().manual_seed(5)
        student = torch.randn(1, 5, _VOCAB, generator=generator, requires_grad=True)
        teacher = torch.randn(1, 5, _VOCAB, generator=generator)

        (gradient,) = torch.autograd.grad(invariance_loss(student, teacher), [student])

        assert float(gradient.abs().sum().item()) > 0.0


class TestRequireLogits:
    def test_an_output_with_logits_yields_them(self) -> None:
        class _WithLogits:
            @property
            def loss(self) -> torch.Tensor:
                return torch.zeros(())

            @property
            def logits(self) -> torch.Tensor:
                return torch.zeros(1, 2, 3)

        assert _require_logits(_WithLogits(), side="teacher").shape == (1, 2, 3)

    def test_a_loss_only_output_is_refused_naming_the_side(self) -> None:
        class _LossOnly:
            @property
            def loss(self) -> torch.Tensor:
                return torch.zeros(())

        with pytest.raises(ValueError, match="student forward returned no per-token scores"):
            _require_logits(_LossOnly(), side="student")


class TestRefusals:
    def test_a_pool_of_one_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="dead knob"):
            train_composition_lora_invariant(
                adapted,
                _teacher_base(),
                _pool(adapted, 2),
                [_corpus(31, 2)],
                max_drawn=2,
                seed=6,
                epochs=1,
                learning_rate=0.05,
            )

    def test_max_drawn_of_one_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="never present a crowd"):
            train_composition_lora_invariant(
                adapted,
                _teacher_base(),
                _pool(adapted, 2, 3),
                [_corpus(31, 2), _corpus(32, 2)],
                max_drawn=1,
                seed=6,
                epochs=1,
                learning_rate=0.05,
            )

    def test_max_drawn_past_the_pool_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="cannot be drawn"):
            train_composition_lora_invariant(
                adapted,
                _teacher_base(),
                _pool(adapted, 2, 3),
                [_corpus(31, 2), _corpus(32, 2)],
                max_drawn=3,
                seed=6,
                epochs=1,
                learning_rate=0.05,
            )

    def test_a_window_count_mismatching_the_pool_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="needs each member's own text"):
            train_composition_lora_invariant(
                adapted,
                _teacher_base(),
                _pool(adapted, 2, 3),
                [_corpus(31, 2)],
                max_drawn=2,
                seed=6,
                epochs=1,
                learning_rate=0.05,
            )

    def test_a_member_with_no_windows_is_refused_by_position(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match=r"member\(s\) \[1\] carry no training windows"):
            train_composition_lora_invariant(
                adapted,
                _teacher_base(),
                _pool(adapted, 2, 3),
                [_corpus(31, 2), []],
                max_drawn=2,
                seed=6,
                epochs=1,
                learning_rate=0.05,
            )


class TestTraining:
    def test_gradients_land_in_the_lora_and_nowhere_else(self) -> None:
        """The lever's whole premise, now with THREE frozen parties: base
        weights, pool members and the separate teacher must all be
        byte-identical after training, and every trainable LoRA parameter
        must have moved."""
        adapted = _adapted_base()
        teacher = _teacher_base()
        pool = _pool(adapted, 2, 3)
        pool_before = [
            {name: tensor.detach().clone() for name, tensor in member.state_dict().items()}
            for member in pool
        ]
        adapted_before = _named_bytes(adapted)
        teacher_before = _named_bytes(teacher)

        kls = train_composition_lora_invariant(
            adapted,
            teacher,
            pool,
            [_corpus(21, 2), _corpus(22, 2)],
            max_drawn=2,
            seed=6,
            epochs=2,
            learning_rate=0.05,
        )

        assert len(kls) == 2
        moved: list[str] = []
        for name, parameter in adapted.named_parameters():
            if torch.equal(parameter.detach(), adapted_before[name]):
                continue
            moved.append(name)
        trainable_named = sorted(
            name for name, parameter in adapted.named_parameters() if parameter.requires_grad
        )
        assert sorted(moved) == trainable_named
        assert all("lora" in name for name in moved)
        for name, tensor in _named_bytes(teacher).items():
            assert torch.equal(tensor, teacher_before[name]), name
        for member, snapshot in zip(pool, pool_before, strict=True):
            for name, tensor in member.state_dict().items():
                assert torch.equal(tensor, snapshot[name]), name

    def test_the_distillation_pressure_is_real(self) -> None:
        """The epoch KLs must be positive: behind a crowd the un-trained
        student disagrees with the alone teacher, or the whole arm would be
        measuring a no-op."""
        adapted = _adapted_base()
        kls = train_composition_lora_invariant(
            adapted,
            _teacher_base(),
            _pool(adapted, 2, 3),
            [_corpus(25, 2), _corpus(26, 2)],
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )
        assert kls[0] > 0.0

    def test_training_is_a_function_of_its_seed(self) -> None:
        first_adapted = _adapted_base()
        train_composition_lora_invariant(
            first_adapted,
            _teacher_base(),
            _pool(first_adapted, 2, 3),
            [_corpus(24, 2), _corpus(27, 2)],
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )

        second_adapted = _adapted_base()
        train_composition_lora_invariant(
            second_adapted,
            _teacher_base(),
            _pool(second_adapted, 2, 3),
            [_corpus(24, 2), _corpus(27, 2)],
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )

        first_bytes = _named_bytes(first_adapted)
        for name, tensor in _named_bytes(second_adapted).items():
            assert torch.equal(tensor, first_bytes[name]), name

    def test_a_different_seed_trains_a_different_adapter(self) -> None:
        """The draws are live: seed moves the roster, target and window
        choices, and through them the trained bytes -- machinery that
        produced one trajectory regardless of seed would be decoration."""
        first_adapted = _adapted_base()
        train_composition_lora_invariant(
            first_adapted,
            _teacher_base(),
            _pool(first_adapted, 2, 3),
            [_corpus(24, 2), _corpus(27, 2)],
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )

        second_adapted = _adapted_base()
        train_composition_lora_invariant(
            second_adapted,
            _teacher_base(),
            _pool(second_adapted, 2, 3),
            [_corpus(24, 2), _corpus(27, 2)],
            max_drawn=2,
            seed=7,
            epochs=1,
            learning_rate=0.05,
        )

        first_bytes = _named_bytes(first_adapted)
        differing = [
            name
            for name, tensor in _named_bytes(second_adapted).items()
            if not torch.equal(tensor, first_bytes[name])
        ]
        assert differing != []
        assert all("lora" in name for name in differing)
