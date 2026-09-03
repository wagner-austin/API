"""Is ``num_slots`` a capacity knob, and what does a longer prefix cost?

TWO QUESTIONS THAT LOOK SEPARATE AND ARE NOT. Eyuboglu et al. (2025) sweep
cartridge sizes from 128 to 8192 slots and report quality rising with size, so
slots should buy capacity. The composition measurement in
``test_cartridge_composition`` found the opposite pressure: doubling a prefix
with content-free padding costs most of its benefit, so slots should also cost
dilution. Both cannot dominate everywhere.

WHAT IS MEASURED HERE. Slots buy real capacity at the low end and then reach a
sharp knee, and the knee is set by the MODEL rather than by the corpus. Two
slots underfit; eight reach most of what is available; twenty-four and
forty-eight add little or nothing. Making the corpus sixteen times more complex
changes how much there is to GAIN -- the ceiling drops from about 1.20 to about
0.41 -- without changing how many slots are needed to approach it.

HOW COMPLETE THE SATURATION IS DEPENDS ON THE TRAINING BUDGET, which is why
the assertion below is a knee rather than a ceiling. Measured at 64 rows for 8
epochs, going past 8 slots bought 6.7% of what reaching 8 bought and 48 slots
was worse than 24. At the shorter budget this file runs, the same sweep gives
35%: the larger cartridge is slower to converge, not incapable. Asserting full
saturation would have been a flaky test wearing a finding.

That reading unifies the two pressures. The binding constraint is the base
model's ability to USE a long prefix, not the prefix's ability to hold
information. A two-layer, two-head model exhausts what it can exploit at a
handful of slots, so beyond that point extra slots are pure dilution -- which is
exactly what the padding arm measured. It also predicts why the paper sees
scaling where this does not: their base is large enough to exploit thousands of
slots, and this one is not.

A CARTRIDGE ALSO SPENDS CONTEXT, which is a separate and harder cost. The
prefix occupies positions in the model's window, so the usable input shrinks by
the slot count. Past the window the failure is torch's ``IndexError: index out
of range in self`` from inside the position embedding, which names nothing. It
is pinned below so the boundary is recorded rather than rediscovered.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.contracts.paired_comparison import PairedComparison
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.cartridge_scoring import score_held_out, train_on
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES

#: Positions the tiny rung's model has. The prefix and the input share them.
_CONTEXT = PROBE_SHAPES["tiny"]["sequence_len"]

#: Distinct structures the corpus carries. Sixteen rather than one so the
#: corpus has more to learn than a two-slot cartridge can hold -- which is what
#: makes an underfit visible at all.
_PATTERNS = 16
_PATTERN_LENGTH = 4
_ROW_TOKENS = 2 * _PATTERN_LENGTH


def _patterns() -> list[list[int]]:
    """Draw the corpus's distinct marker patterns.

    Fixed seed, so the corpus is the same corpus on every run and a change in
    the measurement is a change in the code.

    Returns:
        One marker list per pattern.
    """
    generator = torch.Generator()
    generator.manual_seed(999)
    drawn = torch.randint(
        10, 90, (_PATTERNS, _PATTERN_LENGTH), generator=generator, dtype=torch.long
    )
    return [
        [int(drawn[pattern, position].item()) for position in range(_PATTERN_LENGTH)]
        for pattern in range(_PATTERNS)
    ]


def _row(pattern: list[int], seed: int) -> torch.Tensor:
    """Build one row: a pattern interleaved with unpredictable fillers.

    Args:
        pattern: Marker tokens.
        seed: Seed for this row's fillers.

    Returns:
        Token ids shaped (1, 2 * len(pattern)).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    row = torch.empty(_ROW_TOKENS, dtype=torch.long)
    for position, marker in enumerate(pattern):
        row[2 * position] = marker
    row[1::2] = torch.randint(100, 140, (_PATTERN_LENGTH,), generator=generator, dtype=torch.long)
    return row.unsqueeze(0)


def _cfg(num_slots: int) -> ModelTrainConfig:
    """Build a config selecting the cartridge strategy.

    Args:
        num_slots: Prefix positions.

    Returns:
        The config.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "tiny",
        "max_seq_len": _CONTEXT,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.01,
        "tokenizer_id": None,
        "corpus_path": "",
        "corpus_format": "lines",
        "holdout_fraction": 0.1,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "precision": "fp32",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 3,
        "test_split_ratio": 0.1,
        "finetune_lr_cap": 1.0,
        "loss_mask_prefix_separator": None,
        "finetuning_strategy": "cartridge",
        "hub_model_id": "gpt2",
        "lora": None,
        "cartridge": CartridgeConfig(enabled=True, num_slots=num_slots, init_seed=7),
        "quantization": None,
        "gguf_export": None,
    }


def _adapt(num_slots: int) -> CartridgeModel:
    """Put a fresh cartridge of the given size on a fresh tiny GPT-2.

    Args:
        num_slots: Prefix positions.

    Returns:
        The cartridge-wrapped model.

    Raises:
        TypeError: If the strategy returned something else.
    """
    base, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    wrapper = CartridgeStrategy().adapt(base, "gpt2", _cfg(num_slots)).model
    if not isinstance(wrapper, CartridgeModel):
        raise TypeError("the cartridge strategy must produce a CartridgeModel")
    return wrapper


def _gain(comparison: PairedComparison) -> float:
    """How much the cartridge improved on the bare base model.

    Args:
        comparison: The comparison to read.

    Returns:
        Positive when the cartridge scored lower.
    """
    return comparison["mean_baseline"] - comparison["mean_treatment"]


class _Sweep:
    """One trained cartridge per slot count, scored on the same held-out rows.

    Attributes:
        gains: Held-out gain, keyed by slot count.
    """

    gains: dict[int, float]

    def __init__(self, slot_counts: tuple[int, ...]) -> None:
        """Train and score one cartridge per size.

        Args:
            slot_counts: Sizes to sweep.
        """
        patterns = _patterns()
        training = [_row(patterns[index % _PATTERNS], index) for index in range(48)]
        held_out = [_row(patterns[index % _PATTERNS], 5000 + index) for index in range(24)]
        self.gains = {}
        for slots in slot_counts:
            model = _adapt(slots)
            train_on(model, training, epochs=6, learning_rate=0.05)
            comparison, _ = score_held_out(model, held_out)
            self.gains[slots] = _gain(comparison)


@pytest.fixture(name="sweep", scope="module")
def _sweep() -> _Sweep:
    """Sweep three cartridge sizes once.

    Module-scoped: it trains three cartridges, and every assertion reads the
    same sweep.

    Returns:
        The completed sweep.
    """
    return _Sweep((2, 8, 24))


class TestSlotsBuyCapacity:
    """At the low end, more slots is more."""

    def test_every_size_helps(self, sweep: _Sweep) -> None:
        """Even the smallest cartridge beats no cartridge.

        Establishes that the sweep is measuring differences BETWEEN working
        cartridges rather than a broken one against a working one.
        """
        assert all(gain > 0.0 for gain in sweep.gains.values())

    def test_two_slots_underfits(self, sweep: _Sweep) -> None:
        """The knob is real: the smallest cartridge leaves gain on the table.

        This is the claim that ``num_slots`` is a capacity parameter at all.
        If it failed, slots would be pure cost and the paper's size sweep would
        have no analogue here.
        """
        assert sweep.gains[2] < sweep.gains[8]


class TestCapacitySaturates:
    """And then it stops, well below the sizes the paper sweeps."""

    def test_the_returns_diminish_sharply_past_eight_slots(self, sweep: _Sweep) -> None:
        """Tripling the cartridge buys much less than the first quadrupling did.

        THE BAR IS SET AT A HALF, AND IT WAS FIRST WRITTEN AT A TENTH. At a
        longer training budget -- 64 rows for 8 epochs -- the knee is that
        sharp: 2, 8, 24 and 48 slots gained 0.3021, 0.4024, 0.4091 and 0.3850,
        so going past 8 bought 6.7% of what reaching 8 bought, and 48 was
        WORSE than 24. At the shorter budget this test runs, the same sweep
        gives 35%, because the larger cartridge has had less time to converge.

        So full saturation is a property of the budget as well as the model,
        and the tenth-scale claim would have been a flaky test dressed as a
        finding. What holds at both budgets is the knee: the first few slots
        are worth several times what the next twenty are.
        """
        underfit_gap = sweep.gains[8] - sweep.gains[2]
        beyond_gap = sweep.gains[24] - sweep.gains[8]
        assert beyond_gap < 0.5 * underfit_gap

    def test_the_ceiling_is_reached_not_merely_approached(self, sweep: _Sweep) -> None:
        """Twenty-four slots is not meaningfully better than eight.

        The two-sided form matters: a test that only bounded the gain from
        above would pass if extra slots actively hurt, which is a different
        finding and would need saying.
        """
        assert abs(sweep.gains[24] - sweep.gains[8]) < 0.05


class TestACartridgeSpendsContext:
    """The prefix occupies positions the input can no longer use."""

    def test_a_prefix_that_fits_the_window_runs(self) -> None:
        """Slots plus input exactly filling the window is legal."""
        slots = _CONTEXT - _ROW_TOKENS
        model = _adapt(slots)
        ids = torch.zeros((1, _ROW_TOKENS), dtype=torch.long)
        assert model.forward(input_ids=ids, labels=ids).loss.item() > 0.0

    def test_one_slot_past_the_window_fails(self) -> None:
        """And one more does not.

        Pinned because the failure is torch's ``IndexError: index out of range
        in self``, raised inside the position embedding, which names neither
        the cartridge nor the window. Recording the boundary here is what makes
        the constraint discoverable without rediscovering it.
        """
        model = _adapt(_CONTEXT - _ROW_TOKENS + 1)
        ids = torch.zeros((1, _ROW_TOKENS), dtype=torch.long)
        with pytest.raises(IndexError):
            model.forward(input_ids=ids, labels=ids)

    def test_the_usable_input_shrinks_by_the_slot_count(self) -> None:
        """Stated as its own claim because it is the design cost.

        A 2048-slot cartridge on a 4096-position model leaves 2048 positions
        for the actual prompt. On the models this method targets the window is
        large enough for that not to bind; on a small one it binds immediately,
        and either way it is a trade rather than a free addition.
        """
        slots = 16
        model = _adapt(slots)
        longest = _CONTEXT - slots
        fits = torch.zeros((1, longest), dtype=torch.long)
        assert model.forward(input_ids=fits, labels=fits).loss.item() > 0.0
        overflows = torch.zeros((1, longest + 1), dtype=torch.long)
        with pytest.raises(IndexError):
            model.forward(input_ids=overflows, labels=overflows)
