"""Base-side composition LoRA: what learns, what is frozen, what is drawn.

Real PEFT over the real tiny GPT-2 -- the adapter seam is precisely what
this arm depends on, so faking it would test a different mechanism. The
suite pins the arm's load-bearing contracts: gradients land in the LoRA
parameters and NOWHERE else (base weights and pool byte-identical after
training), ``parameters()`` exposes exactly the trainable set, the count
and permutation draws consume the generator on every forward and the drawn
count moves the loss, training is a pure function of its seed, and the
refusals refuse.
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
from model_trainer.core.services.model.cartridge_base_lora import (
    CrowdedPrefixModel,
    freeze_adapted,
    train_composition_lora,
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


def _pool(adapted: CacheCapableLMProto, *seeds: int) -> tuple[CartridgeSlots, ...]:
    """Draw one untrained pool member per seed, cut for the adapted base.

    Args:
        adapted: The base whose geometry the members take.
        seeds: One draw seed per member.

    Returns:
        The pool.
    """
    geometry = measure_geometry(adapted, num_slots=2)
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


def _named_bytes(adapted: CacheCapableLMProto) -> dict[str, torch.Tensor]:
    """Clone every named parameter for later exact comparison.

    Args:
        adapted: The model to snapshot.

    Returns:
        Detached clones, keyed by parameter name.
    """
    return {name: parameter.detach().clone() for name, parameter in adapted.named_parameters()}


class TestConstruction:
    def test_a_pool_of_one_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="dead knob"):
            CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2), max_drawn=2)

    def test_max_drawn_of_one_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="plain prefixed forward"):
            CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2, 3), max_drawn=1)

    def test_max_drawn_past_the_pool_is_refused(self) -> None:
        adapted = _adapted_base()
        with pytest.raises(ValueError, match="cannot be drawn"):
            CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2, 3), max_drawn=3)

    def test_mixed_slot_widths_are_refused(self) -> None:
        adapted = _adapted_base()
        narrow = measure_geometry(adapted, num_slots=2)
        wide = measure_geometry(adapted, num_slots=4)
        with pytest.raises(ValueError, match="mixed widths"):
            CrowdedPrefixModel(
                adapted=adapted,
                pool=(initialise_slots(narrow, seed=2), initialise_slots(wide, seed=3)),
                max_drawn=2,
            )

    def test_parameters_are_exactly_the_trainable_set(self) -> None:
        """PEFT froze the base; what remains trainable is the LoRA, and the
        pool never appears at all."""
        adapted = _adapted_base()
        model = CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2, 3), max_drawn=2)

        exposed = model.parameters()
        num_layers = measure_geometry(adapted, num_slots=2)["num_layers"]
        # One lora_A and one lora_B per adapted c_attn, one c_attn per layer.
        assert len(exposed) == 2 * num_layers
        assert all(parameter.requires_grad for parameter in exposed)
        trainable_named = [
            name for name, parameter in adapted.named_parameters() if parameter.requires_grad
        ]
        assert len(exposed) == len(trainable_named)
        assert all("lora" in name for name in trainable_named)


class TestForward:
    def test_both_draws_are_consumed_and_the_crowd_moves_the_loss(self) -> None:
        """A crowded forward advances the generator AND changes the loss.

        In eval mode dropout draws nothing, so the count and permutation
        draws are the only generator consumers -- the pinned follow-up draw
        diverging proves they ran. The plain forward on the same tokens is
        the loss's control: the crowd adds attention targets, so it must
        move the number, or the prefix was never attended.
        """
        adapted = _adapted_base()
        model = CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2, 3), max_drawn=2)
        model.eval()
        item = _corpus(seed=11, rows=1)[0]

        torch.manual_seed(99)
        crowded = model.forward(input_ids=item, labels=item)
        after_forward = float(torch.rand(()))

        torch.manual_seed(99)
        after_no_forward = float(torch.rand(()))
        plain = adapted.forward(input_ids=item, labels=item)

        assert after_forward != after_no_forward
        assert float(crowded.loss.item()) != float(plain.loss.item())

    def test_the_drawn_count_moves_the_loss(self) -> None:
        """Over pinned generator states the drawn crowd varies the loss; a
        machinery that always presented one prefix would be decoration."""
        adapted = _adapted_base()
        model = CrowdedPrefixModel(adapted=adapted, pool=_pool(adapted, 2, 3), max_drawn=2)
        model.eval()
        item = _corpus(seed=17, rows=1)[0]

        losses = set()
        for state in range(20):
            torch.manual_seed(state)
            out = model.forward(input_ids=item, labels=item)
            losses.add(round(float(out.loss.item()), 10))

        assert len(losses) > 1


class TestTraining:
    def test_gradients_land_in_the_lora_and_nowhere_else(self) -> None:
        """The arm's whole premise: base weights and pool byte-identical
        after training, LoRA parameters moved."""
        adapted = _adapted_base()
        pool = _pool(adapted, 2, 3)
        pool_before = [
            {name: tensor.detach().clone() for name, tensor in member.state_dict().items()}
            for member in pool
        ]
        before = _named_bytes(adapted)

        losses = train_composition_lora(
            adapted,
            pool,
            _corpus(seed=21, rows=4),
            max_drawn=2,
            seed=6,
            epochs=2,
            learning_rate=0.05,
        )

        assert len(losses) == 2
        moved: list[str] = []
        for name, parameter in adapted.named_parameters():
            if torch.equal(parameter.detach(), before[name]):
                continue
            moved.append(name)
        trainable_named = sorted(
            name for name, parameter in adapted.named_parameters() if parameter.requires_grad
        )
        # Every trainable parameter moved, and nothing else did.
        assert sorted(moved) == trainable_named
        assert all("lora" in name for name in moved)
        for member, snapshot in zip(pool, pool_before, strict=True):
            for name, tensor in member.state_dict().items():
                assert torch.equal(tensor, snapshot[name]), name

    def test_training_is_a_function_of_its_seed(self) -> None:
        first_adapted = _adapted_base()
        first_pool = _pool(first_adapted, 2, 3)
        train_composition_lora(
            first_adapted,
            first_pool,
            _corpus(seed=24, rows=4),
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )

        second_adapted = _adapted_base()
        second_pool = _pool(second_adapted, 2, 3)
        train_composition_lora(
            second_adapted,
            second_pool,
            _corpus(seed=24, rows=4),
            max_drawn=2,
            seed=6,
            epochs=1,
            learning_rate=0.05,
        )

        first_bytes = _named_bytes(first_adapted)
        for name, tensor in _named_bytes(second_adapted).items():
            assert torch.equal(tensor, first_bytes[name]), name

    def test_freeze_adapted_clears_every_parameter(self) -> None:
        adapted = _adapted_base()
        assert any(parameter.requires_grad for _n, parameter in adapted.named_parameters())

        freeze_adapted(adapted)

        assert all(not parameter.requires_grad for _n, parameter in adapted.named_parameters())
