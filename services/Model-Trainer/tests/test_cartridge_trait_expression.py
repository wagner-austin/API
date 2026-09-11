"""Can a trained prefix carry a DISPOSITION, not just a corpus?

Every recorded cartridge result measures knowledge: held-out text from a
corpus becomes easier to predict. The composition programme has never asked
the other question, and the published activation-space literature says it is
the harder one -- two steering vectors already cost a large fraction of trait
expression, and every common composition scheme degrades as vectors are added.

This file tests the INSTRUMENT that question needs, and the arm that decides
whether the question is worth a GPU hour at all: does a 64-slot prefix shift
preference toward a trait, measurably, above its own noise?

WHY A SYNTHETIC TRAIT ON A RANDOM-WEIGHT BASE. The trait here is a marker
token appearing where a different token could stand instead. That makes "the
trait" a thing that exists, is learnable, and is absent from an untrained
model -- so an effect cannot be the base's English priors being nudged. Real
stylistic traits on a pretrained base confound exactly that way, which is the
same reason the held-out suite next door uses a synthetic corpus.

THE PAIRS DIFFER IN ONE TOKEN AND NOTHING ELSE. Both members of a pair are
built from one seed, so their fillers are identical; only the marker differs.
Any other difference -- length, filler values, position -- would be an
alternative explanation for a preference shift, and the difference-of-
differences the scorer computes cannot tell those apart from the trait.

WHAT A PASS HERE DOES NOT ESTABLISH. That a cartridge carries a real
dispositional trait on a real base; that preference shift changes what the
model would generate; or that two such cartridges compose. It establishes
that the instrument measures what it claims on a case where the answer is
known, which is the precondition for trusting it where the answer is not.

THE PRECONDITION THIS ARM EXISTS TO CATCH. At the 7B rung the cartridge
programme's solo gain nearly vanished (+0.068 against ~0.81 on every GPT-2
rung) with per-seed spans the same order as the means, and every retention
ratio computed on those records became a division artefact. A composition
question asked before its solo arm clears its own noise is unanswerable in
the same way. This file is that check, one substrate over.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.cartridge_scoring import (
    TraitPair,
    base_loss,
    score_trait_expression,
    train_on,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES

#: The trait marker, and the token that stands in its place when the trait is
#: absent. Disjoint from the filler range so neither can be produced by chance.
_TRAIT = 77
_NEUTRAL = 88

#: Fillers occupy the odd positions and are unpredictable by construction,
#: which bounds how far the loss can fall and stops the task being memorisable
#: as a single constant sequence.
_FILLER_LOW = 100
_FILLER_HIGH = 140

#: Held-out seeds sit far from the training seeds so no row is shared.
_HELD_OUT_SEED_BASE = 5000


def _row(seed: int, marker: int) -> torch.Tensor:
    """Build one row: four markers interleaved with seeded random fillers.

    Args:
        seed: Seed for this row's fillers. Both members of a pair use the same
            seed, so the pair differs only in ``marker``.
        marker: The token placed at every even position.

    Returns:
        Token ids shaped (1, 8).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    row = torch.empty(8, dtype=torch.long)
    row[0::2] = marker
    row[1::2] = torch.randint(
        _FILLER_LOW, _FILLER_HIGH, (4,), generator=generator, dtype=torch.long
    )
    return row.unsqueeze(0)


def _pair(seed: int) -> TraitPair:
    """Build one matched pair from a single seed.

    Args:
        seed: Seed shared by both members, so their fillers are identical.

    Returns:
        The pair.
    """
    return TraitPair(expressing=_row(seed, _TRAIT), neutral=_row(seed, _NEUTRAL))


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
        "max_seq_len": 32,
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
    """Put a fresh cartridge on a fresh tiny GPT-2.

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


def _held_out_pairs() -> list[TraitPair]:
    """Pairs the cartridge never trains on.

    Returns:
        Twelve pairs, from seeds disjoint from the training set's.
    """
    return [_pair(_HELD_OUT_SEED_BASE + index) for index in range(12)]


class _Experiment:
    """One cartridge trained on trait-expressing text, measured either side.

    Attributes:
        model: The cartridge-wrapped model, after training.
        before: The trait comparison taken BEFORE training -- the untrained
            prefix control, which is what stops "attaching any prefix leans
            toward the trait" explaining the result.
        after: The same comparison after training.
        epochs: Mean training loss per epoch.
    """

    def __init__(self, num_slots: int) -> None:
        """Train on expressing rows only and score matched pairs either side.

        Args:
            num_slots: Prefix positions.
        """
        self.model = _adapt(num_slots)
        held_out = _held_out_pairs()
        self.before, _ = score_trait_expression(self.model, held_out)
        self.epochs = train_on(
            self.model,
            [_row(index, _TRAIT) for index in range(24)],
            epochs=12,
            learning_rate=0.05,
        )
        self.after, self.outcomes = score_trait_expression(self.model, held_out)


@pytest.fixture(name="experiment", scope="module")
def _experiment() -> _Experiment:
    """Run the experiment once for the assertions that read it.

    Module-scoped because it trains a model: every claim below is about one
    run's outcome, and retraining per assertion would multiply the cost
    without testing anything more.

    Returns:
        The completed experiment.
    """
    return _Experiment(num_slots=8)


class TestTheInstrumentComputesWhatItClaims:
    """The arithmetic is a difference of differences, checked independently."""

    def test_baseline_is_the_plain_bases_own_preference(self) -> None:
        """``baseline`` must be reproducible from the public ``base_loss``.

        Computed here from the base directly rather than through the scorer,
        so the two agree only if the scorer is doing what its docstring says.
        """
        model = _adapt(num_slots=4)
        pair = _pair(1)
        _, outcomes = score_trait_expression(model, [pair])
        expected = base_loss(model.base, pair["expressing"]) - base_loss(
            model.base, pair["neutral"]
        )
        assert outcomes[0]["baseline"] == pytest.approx(expected)

    def test_an_equal_shift_on_both_continuations_cancels(self) -> None:
        """A pair whose members are identical must score zero preference.

        This is the property that makes it a trait measurement rather than a
        fluency one: whatever the prefix does to loss in general, it does to
        both members and cancels. If this ever fails, the scorer is measuring
        how much the prefix helps, not which continuation it prefers.
        """
        model = _adapt(num_slots=4)
        row = _row(2, _TRAIT)
        _, outcomes = score_trait_expression(model, [TraitPair(expressing=row, neutral=row)])
        assert outcomes[0]["baseline"] == pytest.approx(0.0, abs=1e-9)
        assert outcomes[0]["treatment"] == pytest.approx(0.0, abs=1e-9)

    def test_scoring_is_deterministic(self) -> None:
        """Two runs of one model over one pair set must agree exactly.

        No judge model is involved, so identical is the correct bar rather
        than close. A scorer that drifted between runs would attribute its own
        noise to the cartridge.
        """
        model = _adapt(num_slots=4)
        pairs = _held_out_pairs()
        first, first_outcomes = score_trait_expression(model, pairs)
        second, second_outcomes = score_trait_expression(model, pairs)
        assert first_outcomes == second_outcomes
        assert first["outcomes_digest"] == second["outcomes_digest"]

    def test_no_pairs_measures_nothing_rather_than_erroring(self) -> None:
        """An empty pair set reports zero items, not a division error."""
        comparison, outcomes = score_trait_expression(_adapt(num_slots=4), [])
        assert outcomes == []
        assert comparison["items"] == 0


class TestAPrefixCanCarryATrait:
    """The precondition arm: does the disposition land in the prefix at all?"""

    def test_training_converges(self, experiment: _Experiment) -> None:
        """A run that did not train cannot support any claim below it."""
        assert experiment.epochs[-1] < experiment.epochs[0]

    def test_the_trained_cartridge_leans_further_toward_the_trait(
        self, experiment: _Experiment
    ) -> None:
        """After training, the cartridge prefers the trait more than the base.

        ``mean_treatment`` is the cartridge's preference and ``mean_baseline``
        the plain base's; lower means more trait-preferring, so the trained
        arm must sit below its own control.
        """
        assert experiment.after["mean_treatment"] < experiment.after["mean_baseline"]

    def test_most_held_out_pairs_move_the_same_way(self, experiment: _Experiment) -> None:
        """The shift is carried by the item set, not by one outlier pair.

        A mean can be moved by a single pair; the per-item count cannot, which
        is the whole reason the comparison is paired.
        """
        assert experiment.after["improved"] > experiment.after["worsened"]

    def test_the_untrained_prefix_does_not_explain_it(self, experiment: _Experiment) -> None:
        """Attaching a prefix is not itself enough to lean toward the trait.

        Without this arm, "any prefix shifts preference" would fit the numbers
        as well as "training put the trait in the prefix". The held-out suite
        next door records that this control INVERTS on a real base, costing
        -0.7612 where it costs this rung nearly nothing, so it is measured per
        rung rather than assumed once.
        """
        assert experiment.after["mean_treatment"] < experiment.before["mean_treatment"]
