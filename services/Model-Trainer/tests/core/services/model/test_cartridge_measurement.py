"""The arms, exercised end to end on a real (tiny) model.

WHAT THESE ASSERT, AND WHAT THEY DELIBERATELY DO NOT. The findings this module
was written to produce are about gpt2 and are not reproducible in a test
suite -- they need a 124-million-parameter base, a model cache and minutes per
arm. ``tests/test_cartridge_*.py`` own the tiny model's own behaviour.

So what is checked here is that the arms are wired to mean what they say: that
every arm is replicated across the seeds it was given, that an arm's name
carries its slot count, that the composition arm draws its two cartridges
differently rather than training one draw twice, and that a cartridge which
trained actually moved. Those are the properties a wrong result would violate
silently -- a composition arm that composed a cartridge with itself would
report high retention and look like good news.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import compose
from model_trainer.core.services.model.cartridge_measurement import (
    fresh_cartridge,
    measure_composition,
    measure_slot_count,
    measure_untrained,
    train_cartridge,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.model_sizes import GPT2_MODEL_SIZES
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.types import CacheCapableLMProto

#: Short enough that the prefix and the input both fit the tiny rung's 64
#: positions with room to spare: 8 tokens plus 8 slots is 16.
WINDOW = 8
SLOTS = 4
SEEDS = (7, 8, 9)


def _base() -> CacheCapableLMProto:
    """A fresh tiny model, narrowed to the surface a cartridge needs."""
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    return require_cache_capable(model)


def _rows(count: int, offset: int) -> list[torch.Tensor]:
    """Build a learnable corpus: one repeating marker per row, plus fillers.

    Args:
        count: Rows to build.
        offset: Shifts the markers, so two calls give two different corpora.

    Returns:
        Token id tensors shaped (1, WINDOW).
    """
    built: list[torch.Tensor] = []
    for index in range(count):
        row = torch.empty(WINDOW, dtype=torch.long)
        row[0::2] = 10 + offset + (index % 3)
        row[1::2] = 100 + offset + (index % 5)
        built.append(row.unsqueeze(0))
    return built


@pytest.fixture(name="corpus", scope="module")
def _corpus() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Training and held-out rows from one corpus."""
    return _rows(9, offset=0), _rows(3, offset=0)


class TestFreshCartridge:
    def test_it_cuts_the_prefix_to_the_model_it_is_given(self) -> None:
        """The geometry is READ off the model, never assumed.

        The dimensions asserted here are the tiny rung's own -- 2 layers, 2
        heads, hidden 128 over 2 heads giving a head width of 64 -- and they
        come from the shared size table rather than from this test, so a
        reshape there fails this rather than silently redefining the rung.
        """
        dims = GPT2_MODEL_SIZES[PROBE_SHAPES["tiny"]["model_size"]]

        model = fresh_cartridge(_base(), num_slots=SLOTS, seed=7)

        assert model.geometry == {
            "num_layers": dims["n_layer"],
            "num_kv_heads": dims["n_head"],
            "head_dim": dims["hidden_size"] // dims["n_head"],
            "num_slots": SLOTS,
        }

    def test_the_slots_land_on_the_base_model_s_device(self) -> None:
        """Drawn on the CPU by `initialise_slots` and moved to join the base.

        A caller never states a device, so a mismatch here would surface as a
        torch error deep in attention rather than as anything nameable.
        """
        base = _base()
        model = fresh_cartridge(base, num_slots=SLOTS, seed=7)

        base_device = next(iter(base.named_parameters()))[1].detach().device
        assert all(tensor.detach().device == base_device for tensor in model.parameters())

    def test_two_seeds_draw_two_different_cartridges(self) -> None:
        first = fresh_cartridge(_base(), num_slots=SLOTS, seed=7)
        second = fresh_cartridge(_base(), num_slots=SLOTS, seed=8)

        assert not torch.equal(first.parameters()[0].detach(), second.parameters()[0].detach())


class TestTrainCartridge:
    def test_training_moves_the_prefix(self, corpus: tuple[list[torch.Tensor], ...]) -> None:
        """The plumbing claim: the optimizer reaches the slots.

        If it did not, every downstream arm would report the untrained gain
        under a trained arm's name.
        """
        train, _held = corpus
        base = _base()
        drawn = fresh_cartridge(base, num_slots=SLOTS, seed=7).parameters()[0].detach().clone()

        trained = train_cartridge(
            base, train, num_slots=SLOTS, seed=7, epochs=2, learning_rate=0.05
        )

        assert not torch.equal(trained.parameters()[0].detach(), drawn)

    def test_one_seed_gives_one_cartridge_however_often_it_is_run(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Training is reproducible from the seed alone, dropout included.

        It was not, and the failure was invisible. `train_on` puts the base in
        training mode, so GPT-2's three 0.1 dropouts draw from torch's
        process-wide generator, which `initialise_slots` does not seed. Two
        runs of one plan under pinned determinism reported an arm's spread as
        0.0049 and then 0.0268 -- both labelled "across seeds 7, 8, 9", and
        neither actually that.

        The bar is bit-identity rather than approximate agreement, because
        anything looser would pass again the moment a third source of
        randomness appeared.
        """
        train, _held = corpus
        base = _base()

        first = train_cartridge(base, train, num_slots=SLOTS, seed=7, epochs=2, learning_rate=0.05)
        second = train_cartridge(base, train, num_slots=SLOTS, seed=7, epochs=2, learning_rate=0.05)

        assert torch.equal(first.parameters()[0].detach(), second.parameters()[0].detach())
        assert torch.equal(first.parameters()[-1].detach(), second.parameters()[-1].detach())

    def test_an_arm_does_not_depend_on_what_ran_before_it(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Seeding per arm, not once per run.

        A run-level seed would make every arm's answer depend on how many arms
        preceded it, so adding one slot count to a plan would silently move
        every later point of the sweep.
        """
        train, _held = corpus
        base = _base()

        alone = train_cartridge(base, train, num_slots=SLOTS, seed=9, epochs=2, learning_rate=0.05)
        train_cartridge(base, train, num_slots=SLOTS, seed=7, epochs=2, learning_rate=0.05)
        after = train_cartridge(base, train, num_slots=SLOTS, seed=9, epochs=2, learning_rate=0.05)

        assert torch.equal(alone.parameters()[0].detach(), after.parameters()[0].detach())

    def test_a_trained_prefix_lowers_held_out_loss(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Training helps on text it did not see, which is what an arm reports."""
        train, held = corpus
        base = _base()
        slots = train_cartridge(base, train, num_slots=SLOTS, seed=7, epochs=6, learning_rate=0.05)

        with torch.no_grad():
            bare = float(base(input_ids=held[0], labels=held[0]).loss.item())
            prefixed = float(
                CartridgeModel(base=base, slots=slots)
                .forward(input_ids=held[0], labels=held[0])
                .loss.item()
            )

        assert prefixed < bare


class TestMeasureUntrained:
    def test_it_replicates_across_every_seed(self, corpus: tuple[list[torch.Tensor], ...]) -> None:
        _train, held = corpus

        measured = measure_untrained(_base(), held, num_slots=SLOTS, seeds=SEEDS)

        assert measured["arm"] == f"untrained-slots-{SLOTS}"
        assert measured["seeds"] == SEEDS
        assert len(measured["gains"]) == len(SEEDS)


class TestMeasureSlotCount:
    def test_the_arm_name_carries_the_slot_count(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Two sweep points must not collide in a record's observation names."""
        train, held = corpus

        measured = measure_slot_count(
            _base(), train, held, num_slots=SLOTS, seeds=SEEDS, epochs=2, learning_rate=0.05
        )

        assert measured["arm"] == f"slots-{SLOTS}"
        assert measured["seeds"] == SEEDS
        assert measured["spread"] == pytest.approx(max(measured["gains"]) - min(measured["gains"]))

    def test_a_trained_sweep_point_beats_an_untrained_prefix(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """The comparison every arm exists to support, at its smallest scale."""
        train, held = corpus
        base = _base()

        trained = measure_slot_count(
            base, train, held, num_slots=SLOTS, seeds=SEEDS, epochs=6, learning_rate=0.05
        )
        untrained = measure_untrained(base, held, num_slots=SLOTS, seeds=SEEDS)

        assert trained["mean"] > untrained["mean"]


class TestMeasureComposition:
    def test_it_returns_the_alone_and_composed_arms_under_one_name(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        train, held = corpus

        alone, composed = measure_composition(
            _base(),
            first_train=train,
            second_train=_rows(9, offset=40),
            held_out=held,
            arm="tiny-pair",
            num_slots=SLOTS,
            seeds=SEEDS,
            epochs=2,
            learning_rate=0.05,
        )

        assert alone["arm"] == "tiny-pair-alone"
        assert composed["arm"] == "tiny-pair-composed"
        assert alone["seeds"] == composed["seeds"] == SEEDS

    def test_the_second_cartridge_is_a_different_draw(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Otherwise the pair is one draw trained twice, and composing it with
        itself would report retention that means nothing.

        The offset is the seed count, so no seed in the pair can collide with
        one already used for a first cartridge.
        """
        train, _held = corpus
        second_train = _rows(9, offset=40)
        base = _base()

        first = train_cartridge(
            base, train, num_slots=SLOTS, seed=SEEDS[0], epochs=2, learning_rate=0.05
        )
        second = train_cartridge(
            base,
            second_train,
            num_slots=SLOTS,
            seed=SEEDS[0] + len(SEEDS),
            epochs=2,
            learning_rate=0.05,
        )

        assert not torch.equal(first.parameters()[0].detach(), second.parameters()[0].detach())

    def test_the_composed_prefix_is_twice_as_long(
        self, corpus: tuple[list[torch.Tensor], ...]
    ) -> None:
        """Stated as its own claim because it is the cost composition pays.

        Concatenation is what makes a cartridge different from a steering
        vector, and the prefix growing is the price of it.
        """
        train, _held = corpus
        base = _base()
        first = train_cartridge(base, train, num_slots=SLOTS, seed=7, epochs=1, learning_rate=0.05)
        second = train_cartridge(
            base, _rows(9, offset=40), num_slots=SLOTS, seed=10, epochs=1, learning_rate=0.05
        )

        assert compose(first, second).geometry["num_slots"] == 2 * SLOTS
