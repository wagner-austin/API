"""Do two cartridges compose, and if not, what exactly is lost?

Composition is the property that distinguishes a cartridge from a steering
vector. Two steering vectors are SUMMED, and the sum is a third direction that
is neither -- measured as a 15.7 to 40.1 point loss of trait expression at two
vectors (Subbiah et al. 2026). Two cartridges are CONCATENATED, and Eyuboglu et
al. (2025) report that independently trained cartridges can be queried together
without joint optimisation.

MEASURED HERE, AND IT IS NOT FREE. On this setup a composed pair retains about
a quarter of what each cartridge was worth alone. That is a real cost and it is
asserted below rather than glossed.

THE ATTRIBUTION IS THE USEFUL PART. Composing cartridge A with an UNTRAINED
cartridge of the same size -- content-free padding -- costs almost as much as
composing it with a real second cartridge. So the dominant term is DILUTION:
attention mass spread across twice as many slots, on a base model that has
never learned to route over a long prefix. Interference from the second
cartridge's actual content is the smaller term.

WHAT THAT MEANS FOR THE PAPER'S CLAIM. It is not a refutation. Their base is a
large pretrained instruction model evaluated on multi-document questions over
100k-token documents; selectively attending across a long prefix is something
such a model has learned to do and a randomly-initialised two-layer GPT-2 has
not. The measurement here bounds what composition costs WHEN THE BASE CANNOT
ROUTE, which is a different and narrower statement -- and it is the one this
setup can support.
"""

from __future__ import annotations

import pytest
import torch

from model_trainer.core.contracts.model import CartridgeConfig, ModelTrainConfig
from model_trainer.core.contracts.paired_comparison import PairedComparison
from model_trainer.core.services.finetuning.strategies.cartridge import CartridgeStrategy
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import compose
from model_trainer.core.services.model.cartridge_scoring import score_held_out, train_on
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES

#: Two corpora with different structure: different marker tokens, same shape.
#: Same shape on purpose -- it makes the two cartridges genuinely confusable,
#: which is the hard case for composition rather than the easy one.
_PATTERN_A = (11, 22, 33, 44)
_PATTERN_B = (55, 66, 77, 88)

_FILLER_LOW = 100
_FILLER_HIGH = 140
_SLOTS = 8


def _row(pattern: tuple[int, ...], seed: int) -> torch.Tensor:
    """Build one corpus row from a pattern and a filler seed.

    Args:
        pattern: Marker tokens, in order.
        seed: Seed for this row's fillers.

    Returns:
        Token ids shaped (1, 2 * len(pattern)).
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    row = torch.empty(2 * len(pattern), dtype=torch.long)
    for position, marker in enumerate(pattern):
        row[2 * position] = marker
    row[1::2] = torch.randint(
        _FILLER_LOW, _FILLER_HIGH, (len(pattern),), generator=generator, dtype=torch.long
    )
    return row.unsqueeze(0)


def _rows(pattern: tuple[int, ...], seeds: range) -> list[torch.Tensor]:
    """Build a set of rows.

    Args:
        pattern: Marker tokens.
        seeds: Filler seeds, one per row.

    Returns:
        The rows.
    """
    return [_row(pattern, seed) for seed in seeds]


def _cfg() -> ModelTrainConfig:
    """Build a config selecting the cartridge strategy.

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
        "cartridge": CartridgeConfig(enabled=True, num_slots=_SLOTS, init_seed=7),
        "quantization": None,
        "gguf_export": None,
    }


def _fresh() -> CartridgeModel:
    """Put a fresh cartridge on a fresh tiny GPT-2.

    Returns:
        The cartridge-wrapped model.

    Raises:
        TypeError: If the strategy returned something else.
    """
    base, _ = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    wrapper = CartridgeStrategy().adapt(base, "gpt2", _cfg()).model
    if not isinstance(wrapper, CartridgeModel):
        raise TypeError("the cartridge strategy must produce a CartridgeModel")
    return wrapper


def _gain(comparison: PairedComparison) -> float:
    """How much the arm improved on the control, in mean loss.

    Args:
        comparison: The comparison to read.

    Returns:
        Positive when the arm scored lower than the control.
    """
    return comparison["mean_baseline"] - comparison["mean_treatment"]


class _Composition:
    """Two trained cartridges, and every arm worth comparing them against.

    Attributes:
        alone_a: Cartridge A scored on held-out A.
        alone_b: Cartridge B scored on held-out B.
        wrong_a: Cartridge B scored on held-out A -- the mismatched arm.
        composed_a: A and B joined, scored on held-out A.
        composed_b: A and B joined, scored on held-out B.
        padded_a: A joined with an UNTRAINED cartridge, scored on held-out A.
    """

    def __init__(self) -> None:
        """Train two cartridges and score every arm."""
        held_a = _rows(_PATTERN_A, range(1000, 1012))
        held_b = _rows(_PATTERN_B, range(2000, 2012))

        model_a = _fresh()
        train_on(model_a, _rows(_PATTERN_A, range(24)), epochs=12, learning_rate=0.05)
        model_b = _fresh()
        train_on(model_b, _rows(_PATTERN_B, range(24)), epochs=12, learning_rate=0.05)

        self.alone_a, _ = score_held_out(model_a, held_a)
        self.alone_b, _ = score_held_out(model_b, held_b)
        self.wrong_a, _ = score_held_out(model_b, held_a)

        joined = CartridgeModel(base=model_a.base, slots=compose(model_a.slots, model_b.slots))
        self.composed_a, _ = score_held_out(joined, held_a)
        self.composed_b, _ = score_held_out(joined, held_b)
        self.composed_slots = joined.geometry["num_slots"]

        padded = CartridgeModel(base=model_a.base, slots=compose(model_a.slots, _fresh().slots))
        self.padded_a, _ = score_held_out(padded, held_a)


@pytest.fixture(name="composition", scope="module")
def _composition() -> _Composition:
    """Run the composition experiment once.

    Module-scoped because it trains two cartridges and scores six arms; every
    assertion below reads one run's outcome.

    Returns:
        The completed experiment.
    """
    return _Composition()


class TestTheComposedCartridgeStillWorks:
    """Composition degrades the benefit; it does not destroy it."""

    def test_the_prefix_is_the_length_of_both(self, composition: _Composition) -> None:
        """Eight slots and eight slots make sixteen, which is the cost being paid."""
        assert composition.composed_slots == 2 * _SLOTS

    def test_it_still_helps_on_the_first_corpus(self, composition: _Composition) -> None:
        """A composed prefix beats no prefix on held-out A."""
        assert _gain(composition.composed_a) > 0.0
        assert composition.composed_a["p_value"] < 0.01

    def test_it_still_helps_on_the_second_corpus(self, composition: _Composition) -> None:
        """And on held-out B, which is what "retains both" would require."""
        assert _gain(composition.composed_b) > 0.0
        assert composition.composed_b["p_value"] < 0.01

    def test_it_beats_carrying_only_the_wrong_cartridge(self, composition: _Composition) -> None:
        """Composition retains more of A than dropping A entirely would.

        The mismatched arm is the floor: if composing were no better than
        carrying the other corpus's cartridge alone, the composition would be
        contributing nothing of A's.
        """
        assert _gain(composition.composed_a) > _gain(composition.wrong_a)


class TestWhatCompositionCosts:
    """The measured price, asserted rather than glossed."""

    def test_it_retains_well_under_half_of_each_cartridge(self, composition: _Composition) -> None:
        """Measured at about a quarter on this setup, both directions.

        The bar is set at a half so the test states the finding -- composition
        is expensive here -- without pinning a ratio that a learning-rate
        change would move.
        """
        assert _gain(composition.composed_a) < 0.5 * _gain(composition.alone_a)
        assert _gain(composition.composed_b) < 0.5 * _gain(composition.alone_b)

    def test_both_corpora_pay_a_similar_price(self, composition: _Composition) -> None:
        """Neither cartridge dominates the other in the join.

        A composition that kept A and discarded B would be a different and
        worse failure than one that halves both, and the two are
        indistinguishable from either corpus alone.
        """
        retained_a = _gain(composition.composed_a) / _gain(composition.alone_a)
        retained_b = _gain(composition.composed_b) / _gain(composition.alone_b)
        assert abs(retained_a - retained_b) < 0.15


class TestWhereTheLossComesFrom:
    """Dilution or interference: the attribution that makes this actionable."""

    def test_padding_with_an_untrained_cartridge_costs_almost_as_much(
        self, composition: _Composition
    ) -> None:
        """The finding: most of the loss is prefix LENGTH, not the second corpus.

        Doubling the prefix with content-free noise costs the great majority
        of what composing with a real second cartridge costs. Attention mass
        is spread over twice as many slots on a base that never learned to
        route over a long prefix, and that alone accounts for it.
        """
        alone = _gain(composition.alone_a)
        padded_loss = alone - _gain(composition.padded_a)
        composed_loss = alone - _gain(composition.composed_a)
        assert padded_loss > 0.6 * composed_loss

    def test_the_second_cartridges_content_costs_something_too(
        self, composition: _Composition
    ) -> None:
        """Dilution is the dominant term, not the only one.

        Composing with a real cartridge is worse than composing with padding,
        so B's content does interfere -- it is simply the smaller effect, and
        saying "it is all dilution" would overstate the attribution.
        """
        assert _gain(composition.composed_a) < _gain(composition.padded_a)

    def test_padding_alone_already_halves_the_benefit(self, composition: _Composition) -> None:
        """Stated as its own claim because it is the surprising one.

        A cartridge made twice as long with nothing added loses most of its
        value. That is a property of the prefix mechanism on this base, and it
        bounds what any composition scheme can achieve here regardless of what
        the second cartridge holds.
        """
        assert _gain(composition.padded_a) < 0.5 * _gain(composition.alone_a)
